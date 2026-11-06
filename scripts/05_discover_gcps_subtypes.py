"""Step 05. NMF subtype discovery (k = 4-6), model selection and the frozen gCPS labels."""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, squareform
from scipy.stats import chi2
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, confusion_matrix,
                             davies_bouldin_score, silhouette_score)
from sklearn.model_selection import KFold, StratifiedKFold

from gcps_utils import CLINICAL, COLORS, JOBS, SEED, SUBTYPES, bh, load_matrix, outdir, sample_id, save_fig, \
    style_matplotlib

K_RANGE = [4, 5, 6]
K_FINAL = 4
NMF_ARGS = dict(init="nndsvda", solver="mu", beta_loss="kullback-leibler", max_iter=150, tol=1e-3)
N_CONSENSUS, N_SUBSAMPLE, SUBSAMPLE_FRAC = 20, 200, 0.8
N_REPEATS, N_FOLDS, N_PERM = 10, 5, 2000
EPS = 1e-6

out = outdir("05_subtypes")


# ---------- cohort ----------
x = load_matrix()
x.index = [sample_id(i) for i in x.index]
x = x[~x.index.duplicated()]
clin = pd.read_csv(CLINICAL, sep="\t", low_memory=False)
clin["sid"] = clin.iloc[:, 0].map(sample_id)
clin = clin.dropna(subset=["sid", "OS_MONTHS", "OS_STATUS"]).query("OS_MONTHS >= 0").drop_duplicates("sid").set_index("sid")
samples = [s for s in x.index if s in clin.index]
X = (x.loc[samples].fillna(0).values > 0).astype(np.float32)
os_time = clin.loc[samples, "OS_MONTHS"].to_numpy(float)
os_event = clin.loc[samples, "OS_STATUS"].to_numpy(int)
N = len(samples)
HAM = cdist(X, X, metric="hamming").astype(np.float32)
print(f"{N} tumors x {X.shape[1]} features, {os_event.sum()} deaths")


# ---------- helpers ----------
def relabel_map(labels, ref, k):
    cm = confusion_matrix(ref, labels, labels=list(range(k)))
    r, c = linear_sum_assignment(-cm)
    return {int(b): int(a) for a, b in zip(r, c)}


def relabel(labels, mp):
    return np.array([mp.get(int(v), int(v)) for v in labels])


def jaccard(a, b, c):
    u = np.sum((a == c) | (b == c))
    return np.sum((a == c) & (b == c)) / u if u else np.nan


def logrank_multi(t, e, g):
    groups = np.unique(g)
    O, E, V = np.zeros(len(groups)), np.zeros(len(groups)), np.zeros((len(groups),) * 2)
    for tt in np.unique(t[e == 1]):
        risk, ev = t >= tt, (t == tt) & (e == 1)
        n, d = risk.sum(), ev.sum()
        gr = np.array([(risk & (g == q)).sum() for q in groups], float)
        O += np.array([(ev & (g == q)).sum() for q in groups])
        E += d * gr / n
        if n > 1:
            p = gr / n
            V += d * (n - d) / (n - 1) * (np.diag(p) - np.outer(p, p))
    diff = (O - E)[:-1]
    return float(chi2.sf(diff @ np.linalg.pinv(V[:-1, :-1]) @ diff, len(groups) - 1))


def fit_single(k, train, seed, test=None):
    """One NMF fit; the held-out tumors take the component with the highest projected activation."""
    m = NMF(n_components=k, random_state=int(seed), **NMF_ARGS)
    w = m.fit_transform(train + EPS)
    held = None if test is None else np.argmax(m.transform(test + EPS), 1)
    return np.argmax(w, 1), held


def fit_best_of(k, train, seed, n_start=8):
    rng = np.random.default_rng(seed)
    best, err = None, np.inf
    for s in rng.integers(0, 99999, n_start):
        m = NMF(n_components=k, random_state=int(s), **NMF_ARGS)
        w = m.fit_transform(train + EPS)
        if len(np.unique(w.argmax(1))) >= 2 and m.reconstruction_err_ < err:
            best, err = w.argmax(1), m.reconstruction_err_
    return best


def crossfit(k, mat, repeat, strata=None):
    """Five-fold cross-fitting; fold labels are aligned to fold 0 using training tumors only."""
    seed = SEED + repeat
    if strata is not None:
        folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(mat, strata))
    else:
        folds = list(KFold(N_FOLDS, shuffle=True, random_state=seed).split(mat))
    labels = np.full(len(mat), -1)
    for f, (tr, te) in enumerate(folds):
        train_lab, held = fit_single(k, mat[tr], SEED + repeat * N_FOLDS + f, mat[te])
        if f == 0:
            ref_idx, ref_lab = tr, train_lab
        else:
            _, a, b = np.intersect1d(tr, ref_idx, return_indices=True)
            if len(a) >= k:
                held = relabel(held, relabel_map(train_lab[a], ref_lab[b], k))
        labels[te] = held
    assert (labels >= 0).all()
    return labels


def consensus(k, mat, n_restarts):
    rng = np.random.default_rng(SEED)
    C, ok = np.zeros((len(mat), len(mat))), 0
    for _ in range(n_restarts):
        m = NMF(n_components=k, init="random", solver="mu", beta_loss="kullback-leibler",
                max_iter=NMF_ARGS["max_iter"], tol=NMF_ARGS["tol"], random_state=int(rng.integers(0, 2**31 - 1)))
        lab = m.fit_transform(mat + EPS).argmax(1)
        if len(np.unique(lab)) < 2:
            continue
        C += lab[:, None] == lab[None, :]
        ok += 1
    return C / ok


def cophenetic_dispersion(C):
    n = C.shape[0]
    rho = float(np.sum(4 * (C - 0.5) ** 2) / n**2)
    d = 1 - C
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    cond = squareform(d, checks=False)
    return float(cophenet(linkage(cond, "average"), cond)[0]), rho


def internal_metrics(lab, mat, ham):
    return dict(silhouette_hamming=silhouette_score(ham, lab, metric="precomputed"),
                calinski_harabasz=calinski_harabasz_score(mat, lab),
                davies_bouldin=davies_bouldin_score(mat, lab))


def resample(k, ref):
    def once(r):
        rng = np.random.default_rng(SEED * 1000 + r)
        idx = rng.choice(N, int(round(SUBSAMPLE_FRAC * N)), replace=False)
        lab = fit_best_of(k, X[idx], int(rng.integers(0, 2**31 - 1)))
        if lab is None:
            return None
        lab = relabel(lab, relabel_map(lab, ref[idx], k))
        return adjusted_rand_score(ref[idx], lab), [jaccard(lab, ref[idx], c) for c in range(k)]

    res = [r for r in Parallel(n_jobs=JOBS, prefer="processes")(delayed(once)(r) for r in range(N_SUBSAMPLE)) if r]
    return np.array([r[0] for r in res]), np.array([r[1] for r in res])


# ---------- model selection ----------
print("cross-fitted labels, consensus statistics, null matrix")
labels_k, rows, null_rows = {}, [], []
rng = np.random.default_rng(SEED)
X_null = X.copy()
for j in range(X_null.shape[1]):
    X_null[:, j] = X_null[rng.permutation(N), j]
HAM_null = cdist(X_null, X_null, metric="hamming").astype(np.float32)

for k in K_RANGE:
    lab = crossfit(k, X, 0, os_event)
    labels_k[k] = lab
    coph, disp = cophenetic_dispersion(consensus(k, X, N_CONSENSUS))
    sizes = np.bincount(lab, minlength=k)
    rows.append(dict(k=k, **internal_metrics(lab, X, HAM), cophenetic=coph, dispersion=disp,
                     logrank_p_OS=logrank_multi(os_time, os_event, lab), smallest_cluster=int(sizes.min()),
                     sizes=json.dumps({f"gCPS{c + 1}": int(s) for c, s in enumerate(sizes)})))
    lab_n = crossfit(k, X_null, 0, os_event)
    coph_n, disp_n = cophenetic_dispersion(consensus(k, X_null, N_CONSENSUS))
    null_rows.append(dict(k=k, **internal_metrics(lab_n, X_null, HAM_null), cophenetic=coph_n, dispersion=disp_n))
    print(f"  k={k} cophenetic={coph:.3f} dispersion={disp:.3f} sizes={sizes.tolist()}")

observed = pd.DataFrame(rows)
null = pd.DataFrame(null_rows)

print("resampling reproducibility")
stab, jac_rows = [], []
for k in K_RANGE:
    ari, jac = resample(k, labels_k[k])
    stab.append(dict(k=k, ARI_mean=ari.mean(), ARI_p5=np.percentile(ari, 5), ARI_min=ari.min(),
                     min_mean_cluster_jaccard=np.nanmean(jac, 0).min(), n_ok=len(ari)))
    for c in range(k):
        jac_rows.append(dict(k=k, cluster=f"gCPS{c + 1}", n=int((labels_k[k] == c).sum()),
                             jaccard_mean=np.nanmean(jac[:, c]), jaccard_p5=np.nanpercentile(jac[:, c], 5)))
stability = pd.DataFrame(stab)
selection = observed.merge(null, on="k", suffixes=("", "_null")).merge(stability, on="k")
selection.to_csv(out / "model_selection_k4_k6.csv", index=False)
pd.DataFrame(jac_rows).to_csv(out / "cluster_jaccard.csv", index=False)

# ---------- repeated cross-fitting, k = 4 ----------
print("ten cross-fitting repetitions")
reps = Parallel(n_jobs=min(JOBS, N_REPEATS), prefer="processes")(
    delayed(crossfit)(K_FINAL, X, r, os_event) for r in range(N_REPEATS))
ref = reps[0]
unstrat = Parallel(n_jobs=min(JOBS, N_REPEATS), prefer="processes")(
    delayed(crossfit)(K_FINAL, X, r, None) for r in range(N_REPEATS))
conc = []
for scheme, rr in (("stratified", reps), ("unstratified", unstrat)):
    for r, lab in enumerate(rr):
        al = relabel(lab, relabel_map(lab, ref, K_FINAL))
        conc.append(dict(scheme=scheme, repeat=r, ARI_vs_reference=adjusted_rand_score(ref, al),
                         gCPS4_n=int((al == 3).sum())))
pd.DataFrame(conc).to_csv(out / "crossfit_repeat_concordance.csv", index=False)

# ---------- survival across k, family-wise minimum-P ----------
group_mat = np.vstack([labels_k[k] for k in K_RANGE])
p_obs = np.array([logrank_multi(os_time, os_event, g) for g in group_mat])


def min_p(b):
    idx = np.random.default_rng(SEED * 7919 + b).permutation(N)
    return min(logrank_multi(os_time[idx], os_event[idx], g) for g in group_mat)


perm = np.array(Parallel(n_jobs=JOBS, prefer="processes")(delayed(min_p)(b) for b in range(N_PERM)))
pd.DataFrame(dict(k=K_RANGE, logrank_p=p_obs, logrank_p_BH=bh(p_obs),
                  familywise_minP_p=(1 + (perm <= p_obs.min()).sum()) / (1 + len(perm)))).to_csv(
    out / "survival_across_k.csv", index=False)

# ---------- frozen labels ----------
pd.DataFrame({"sample_id": samples, "gCPS": [SUBTYPES[c] for c in ref]}).to_csv(out / "gcps_labels.csv", index=False)
print("frozen partition:", dict(zip(SUBTYPES, np.bincount(ref, minlength=4))))

# ---------- embeddings and figures ----------
import umap

pca = PCA(n_components=2, random_state=SEED)
pc = pca.fit_transform(X)
um = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.2, metric="hamming", random_state=SEED).fit_transform(X)
pd.DataFrame({"sample_id": samples, "PC1": pc[:, 0], "PC2": pc[:, 1], "UMAP1": um[:, 0], "UMAP2": um[:, 1],
              "gCPS": [SUBTYPES[c] for c in ref]}).to_csv(out / "pca_umap_coordinates.csv", index=False)

plt = style_matplotlib()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, emb, name in ((axes[0], pc, "PC"), (axes[1], um, "UMAP")):
    for c, s in enumerate(SUBTYPES):
        m = ref == c
        ax.scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.85, color=COLORS[s], edgecolor="none",
                   label=f"{s} (n={m.sum()})")
    ax.set_xlabel(f"{name}1" + (f" ({pca.explained_variance_ratio_[0]:.1%})" if name == "PC" else ""))
    ax.set_ylabel(f"{name}2" + (f" ({pca.explained_variance_ratio_[1]:.1%})" if name == "PC" else ""))
    ax.legend(frameon=False, fontsize=8, markerscale=1.5)
save_fig(fig, out / "fig2B_pca_umap")
plt.close(fig)

panels = [("cophenetic", "Cophenetic correlation"), ("dispersion", "Dispersion"),
          ("silhouette_hamming", "Hamming silhouette"), ("calinski_harabasz", "Calinski-Harabasz"),
          ("davies_bouldin", "Davies-Bouldin"), ("ARI_mean", "Resampling ARI"),
          ("min_mean_cluster_jaccard", "Minimum cluster Jaccard"), ("smallest_cluster", "Smallest cluster (n)")]
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, (col, title) in zip(axes.ravel(), panels):
    ax.plot(selection.k, selection[col], "o-", color="#0072B2", label="observed")
    if f"{col}_null" in selection:
        ax.plot(selection.k, selection[f"{col}_null"], "o--", color="#999999", label="permuted null")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(K_RANGE)
    ax.set_xlabel("k")
axes[0, 0].legend(frameon=False, fontsize=8)
save_fig(fig, out / "fig2A_model_selection")
plt.close(fig)

json.dump(dict(seed=SEED, n=N, features=int(X.shape[1]), events=int(os_event.sum()),
               sizes={s: int(n) for s, n in zip(SUBTYPES, np.bincount(ref, minlength=4))}),
          open(out / "run_summary.json", "w"), indent=1)
