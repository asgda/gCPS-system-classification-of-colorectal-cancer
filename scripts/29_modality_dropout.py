"""Step 29. Modality-dropout robustness by repeated cross-fitting (Figure S4D, Supplementary Table S8c)."""
import re
from itertools import combinations

import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold

from gcps_utils import COLORS, MODALITY_COLORS, OUT, SEED, SUBTYPES, bh, load_clinical, load_labels, outdir, \
    sample_id, save_fig, style_matplotlib

K, MAX_ITER, TOL = 4, 600, 1e-4
REPEATS, FOLDS, BOOTSTRAPS = 10, 5, 2000
ORDER = ["SNV", "CNV", "SV", "SNV+CNV", "SNV+SV", "CNV+SV", "ALL"]
ARM = re.compile(r"^CNV_(?:\d{1,2}|X|Y)[pq]$")


def read(path):
    m = pd.read_csv(path, index_col=0)
    m.index = [sample_id(i) for i in m.index]
    return m[~m.index.duplicated()]


def keep_features(frame, train):
    """Prevalence filter repeated inside the training fold; arm-level CNV features are always kept."""
    freq = frame.iloc[train].mean()
    keep = [c for c in frame.columns
            if ARM.match(c) or (0.01 if c.startswith("SV_") else 0.02) <= freq[c] <= 0.98]
    return keep if len(keep) >= K else freq.sort_values(ascending=False).index[:K].tolist()


def map_components(components, reference):
    table = np.zeros((K, K), int)
    for a, b in zip(components, reference):
        table[a, b] += 1
    rows, cols = linear_sum_assignment(-table)
    return dict(zip(rows.tolist(), cols.tolist()))


def partition_metrics(truth, pred):
    t, p = truth == 3, pred == 3
    tp, fp, fn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(ARI=adjusted_rand_score(truth, pred), NMI=normalized_mutual_info_score(truth, pred),
                gCPS4_precision=precision, gCPS4_recall=recall, gCPS4_jaccard=tp / max(tp + fp + fn, 1))


def consensus(predictions, memberships):
    """Majority vote across repeats; ties are broken by the mean mapped NMF weight."""
    out = np.empty(predictions.shape[1], int)
    for i in range(predictions.shape[1]):
        counts = np.bincount(predictions[:, i], minlength=K)
        winners = np.flatnonzero(counts == counts.max())
        out[i] = winners[np.argmax(memberships[:, i, :].mean(0)[winners])] if len(winners) > 1 else winners[0]
    return out


def survival_p(survival, pred, endpoint):
    d = survival.assign(label=pred)
    d = d[d.stage.isin([1, 2, 3])] if endpoint == "RFS" else d
    d = d.dropna(subset=[f"{endpoint}_months", f"{endpoint}_event"])
    return multivariate_logrank_test(d[f"{endpoint}_months"], d.label, d[f"{endpoint}_event"]).p_value


def main():
    out = outdir("29_modality_dropout")
    labels = load_labels()
    snv = read(OUT / "01_snv" / "snv_discovery_matrix.csv")
    cnv = read(OUT / "02_cnv" / "cnv_discovery_matrix.csv")
    sv = read(OUT / "03_sv" / "sv_discovery_matrix.csv")
    allm = read(OUT / "04_features" / "unified_discovery_matrix.csv")
    common = sorted(set(labels.index) & set(snv.index) & set(cnv.index) & set(sv.index) & set(allm.index))
    snv, cnv, sv, allm = (m.loc[common] for m in (snv, cnv, sv, allm))
    y = labels.loc[common].map({s: i for i, s in enumerate(SUBTYPES)}).values
    inputs = {"SNV": snv, "CNV": cnv, "SV": sv, "SNV+CNV": pd.concat([snv, cnv], axis=1),
              "SNV+SV": pd.concat([snv, sv], axis=1), "CNV+SV": pd.concat([cnv, sv], axis=1), "ALL": allm}
    clin = load_clinical().reindex(common)

    repeat_rows, summary_rows, predictions = [], [], pd.DataFrame({"sample_id": common, "reference": y})
    for ci, (name, frame) in enumerate(inputs.items()):
        votes, weights = [], []
        for repeat in range(REPEATS):
            seed = SEED + 10000 * (ci + 1) + repeat
            oof, membership = np.full(len(y), -1), np.zeros((len(y), K))
            for fold, (train, test) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(frame, y)):
                cols = keep_features(frame, train)
                model = NMF(n_components=K, init="nndsvda", solver="mu", beta_loss="kullback-leibler",
                            max_iter=MAX_ITER, tol=TOL, random_state=seed + fold)
                w_train = model.fit_transform(frame.iloc[train][cols].to_numpy(np.float32) + 1e-6)
                mapping = map_components(w_train.argmax(1), y[train])   # learned on training tumors only
                w_test = model.transform(frame.iloc[test][cols].to_numpy(np.float32) + 1e-6)
                oof[test] = [mapping[c] for c in w_test.argmax(1)]
                mapped = np.zeros_like(w_test)
                for component, subtype in mapping.items():
                    mapped[:, subtype] = w_test[:, component]
                membership[test] = mapped / np.maximum(mapped.sum(1, keepdims=True), 1e-12)
            votes.append(oof)
            weights.append(membership)
            repeat_rows.append(dict(input=name, repeat=repeat, **partition_metrics(y, oof),
                                    OS_logrank_P=survival_p(clin, oof, "OS"), RFS_logrank_P=survival_p(clin, oof, "RFS")))
        final = consensus(np.array(votes), np.array(weights))
        predictions[name] = final
        rng = np.random.default_rng(SEED + 90909)
        boot = pd.DataFrame([partition_metrics(y[i], final[i]) for i in
                             (rng.integers(0, len(y), len(y)) for _ in range(BOOTSTRAPS))])
        row = dict(input=name, n=len(y), **partition_metrics(y, final))
        for c in boot:
            row[f"{c}_ci_low"], row[f"{c}_ci_high"] = boot[c].quantile([0.025, 0.975])
        row["OS_logrank_P"], row["RFS_logrank_P"] = survival_p(clin, final, "OS"), survival_p(clin, final, "RFS")
        summary_rows.append(row)
        print(f"{name}: ARI {row['ARI']:.3f}, gCPS4 Jaccard {row['gCPS4_jaccard']:.3f}")

    summary = pd.DataFrame(summary_rows)
    for endpoint in ("OS", "RFS"):
        summary[f"{endpoint}_logrank_P_adj"] = bh(summary[f"{endpoint}_logrank_P"])
    summary.to_csv(out / "modality_dropout_summary.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(out / "modality_dropout_repeats.csv", index=False)
    predictions.to_csv(out / "modality_dropout_predictions.csv", index=False)

    # paired contrasts between the seven inputs on identical bootstrap samples
    rng = np.random.default_rng(SEED + 70707)
    draws = [rng.integers(0, len(y), len(y)) for _ in range(BOOTSTRAPS)]
    jaccard = lambda t, p: ((t == 3) & (p == 3)).sum() / max(((t == 3) | (p == 3)).sum(), 1)
    contrasts = []
    for a, b in combinations(ORDER, 2):
        for metric, fn in (("ARI", adjusted_rand_score), ("gCPS4_jaccard", jaccard)):
            observed = fn(y, predictions[a].values) - fn(y, predictions[b].values)
            diff = np.array([fn(y[i], predictions[a].values[i]) - fn(y[i], predictions[b].values[i]) for i in draws])
            p = min(1.0, 2 * min((np.sum(diff <= 0) + 1) / (BOOTSTRAPS + 1), (np.sum(diff >= 0) + 1) / (BOOTSTRAPS + 1)))
            contrasts.append(dict(metric=metric, input_a=a, input_b=b, difference=observed,
                                  ci_low=np.quantile(diff, 0.025), ci_high=np.quantile(diff, 0.975), P=p))
    contrasts = pd.DataFrame(contrasts)
    contrasts["P_adj"] = contrasts.groupby("metric")["P"].transform(bh)
    contrasts.to_csv(out / "modality_dropout_paired_contrasts.csv", index=False)

    plt = style_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    s = summary.set_index("input").reindex(ORDER)
    ax.bar(ORDER, s.ARI, color=[MODALITY_COLORS.get(o, "#777777") for o in ORDER])
    ax.errorbar(range(len(ORDER)), s.ARI, yerr=[s.ARI - s.ARI_ci_low, s.ARI_ci_high - s.ARI], fmt="none", ecolor="black")
    ax.set_ylabel("Cross-fitted ARI against the gCPS labels")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, out / "modality_dropout_ARI")
    plt.close(fig)


if __name__ == "__main__":
    main()
