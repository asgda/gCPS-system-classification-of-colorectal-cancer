"""Step 31. Agreement of the gCPS subtypes with CMS, CRPS and iCMS, and added prognostic value (Figure 7, Tables 2, S2, S10).

Tumors without a call in an expression-based system are excluded from the concordance statistics and from the nested
models; they are listed in the per-tumor table and shown as a separate block in the alluvial plot.
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2, chi2_contingency
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from gcps_utils import COLORS, DATA, SUBTYPES, bh, load_clinical, load_labels, outdir, sample_id, save_fig, style_matplotlib

SYSTEMS = {"CMS": "CMS Tumour", "CRPS": "CRPS Tumour", "iCMS": "iCMS Tumour"}
out = outdir("31_expression_systems")
labels = load_labels()

nunes = pd.read_excel(DATA / "Supplementary_Table_01.xlsx")
nunes.index = nunes["Sample ID"].map(sample_id)
nunes = nunes[~nunes.index.duplicated() & nunes.index.notna()]
calls = pd.DataFrame({k: nunes[v].astype(str) for k, v in SYSTEMS.items()}).reindex(labels.index)
calls = calls.replace({"nan": "Undefined", "NA": "Undefined"})
calls.insert(0, "gCPS", labels)
calls.to_csv(out / "per_tumor_labels.csv")

concordance, tables = [], {}
for system in SYSTEMS:
    d = calls[calls[system] != "Undefined"]
    ct = pd.crosstab(d.gCPS, d[system]).reindex(SUBTYPES)
    chi, p, dof, _ = chi2_contingency(ct)
    concordance.append(dict(system=system, n_with_call=len(d), classes=ct.shape[1], ARI=adjusted_rand_score(d.gCPS, d[system]),
                            NMI=normalized_mutual_info_score(d.gCPS, d[system]), chi2=chi, dof=dof, P=p,
                            cramers_v=np.sqrt(chi / len(d) / (min(ct.shape) - 1))))
    full = pd.crosstab(calls.gCPS, calls[system]).reindex(SUBTYPES)
    pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1).add_suffix(" (% of classified)")
    pd.concat([full, pct], axis=1).to_csv(out / f"gCPS_by_{system}.csv")
pd.DataFrame(concordance).to_csv(out / "concordance_metrics.csv", index=False)
print(pd.DataFrame(concordance).round(3).to_string(index=False))

# nested stage-stratified Cox models: what gCPS adds to an expression system, and the reverse
clin = load_clinical().join(labels.rename("gCPS"))
clin = clin.join(calls[list(SYSTEMS)])


def design(frame, factors):
    parts = [frame[["age", "sex_m", "msi", "pretreated"]]]
    parts += [pd.get_dummies(frame[f], prefix=f, drop_first=True).astype(float) for f in factors]
    return pd.concat(parts, axis=1)


def loglik(frame, factors, endpoint):
    x = pd.concat([design(frame, factors), frame[[f"{endpoint}_months", f"{endpoint}_event", "stage"]]], axis=1)
    cph = CoxPHFitter().fit(x, f"{endpoint}_months", f"{endpoint}_event", strata=["stage"])
    return cph.log_likelihood_, design(frame, factors).shape[1]


rows = []
for endpoint in ("OS", "RFS"):
    base = clin.dropna(subset=[f"{endpoint}_months", f"{endpoint}_event", "stage", "age", "sex_m", "msi", "pretreated"])
    base = base[base[f"{endpoint}_months"] > 0]
    if endpoint == "RFS":
        base = base[base.stage.isin([1, 2, 3])]
    for system in SYSTEMS:
        d = base[base[system] != "Undefined"]
        for added, kept, label in (("gCPS", [system], "gCPS added to clinical + expression system"),
                                   (system, ["gCPS"], "expression system added to clinical + gCPS")):
            full_ll, k_full = loglik(d, [system, "gCPS"], endpoint)
            reduced_ll, k_red = loglik(d, kept, endpoint)
            stat = 2 * (full_ll - reduced_ll)
            rows.append(dict(endpoint=endpoint, system=system, test=label, n=len(d), events=int(d[f"{endpoint}_event"].sum()),
                             LRT_stat=stat, df=k_full - k_red, P=chi2.sf(stat, k_full - k_red)))
tests = pd.DataFrame(rows)
tests.to_csv(out / "added_prognostic_value.csv", index=False)
print(tests[tests.endpoint == "OS"].round(5).to_string(index=False))

# Figure 7B: iCMS composition of each subtype
plt = style_matplotlib()
d = calls[calls.iCMS != "Undefined"]
share = pd.crosstab(d.gCPS, d.iCMS, normalize="index").reindex(SUBTYPES) * 100
fig, ax = plt.subplots(figsize=(5, 4.5))
share.plot.bar(stacked=True, ax=ax, color=["#0072B2", "#D55E00"], rot=0)
ax.set_ylabel("Tumors with an iCMS call (%)")
save_fig(fig, out / "fig7B_icms_share")
plt.close(fig)
