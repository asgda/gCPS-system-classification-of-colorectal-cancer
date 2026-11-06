"""Step 30. Predicted drug sensitivity, PARP inhibitors and immune context by subtype (Figure 6H, Tables S7j, S9d-e).

Predictions come from cell-line models (oncoPredict, GDSC2) and are hypotheses, not clinical evidence.
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from gcps_utils import COLORS, OUT, SUBTYPES, bh, cliffs_delta_ci, load_labels, mwu_p, outdir, sample_id, save_fig, \
    style_matplotlib

PARP = ["Olaparib_1017", "Niraparib_1177", "Talazoparib_1259"]
MECHANISM = {
    "EGFR/HER2 inhibitor": ["Afatinib", "Erlotinib", "Gefitinib", "Lapatinib", "Sapitinib", "CP724714", "Osimertinib"],
    "MAPK inhibitor": ["Trametinib", "Selumetinib", "PD0325901", "Dabrafenib", "Refametinib", "SCH772984", "Ulixertinib"],
    "PI3K/AKT/mTOR inhibitor": ["BEZ235", "Pictilisib", "Alpelisib", "MK-2206", "Ipatasertib", "Temsirolimus", "Rapamycin",
                                "AZD8055", "GSK2126458", "Omipalisib", "Taselisib", "GNE-317", "AZD2014"],
    "PARP/DDR inhibitor": ["Olaparib", "Talazoparib", "Niraparib", "Rucaparib", "AZD6738", "AZD7762", "VE-822", "AZD1775"],
    "Antimetabolite": ["5-Fluorouracil", "Gemcitabine", "Methotrexate", "Pemetrexed", "Cytarabine", "Fludarabine"],
    "Topoisomerase inhibitor": ["Irinotecan", "SN-38", "Topotecan", "Etoposide", "Doxorubicin", "Mitoxantrone", "Teniposide"],
    "Taxane/microtubule": ["Paclitaxel", "Docetaxel", "Vinorelbine", "Vinblastine", "Vincristine"],
    "CDK inhibitor": ["Palbociclib", "Ribociclib", "Abemaciclib", "Dinaciclib", "Flavopiridol", "Roscovitine"],
    "HDAC/proteasome inhibitor": ["Bortezomib", "Vorinostat", "Panobinostat", "Belinostat", "Romidepsin"],
}
FORCED = {"gCPS1": ["Osimertinib"], "gCPS2": ["Alpelisib", "Trametinib"], "gCPS3": ["Fludarabine", "5-Fluorouracil"],
          "gCPS4": ["GNE-317", "Pictilisib", "Rapamycin"]}
PER_SUBTYPE = 7

out = outdir("30_drug_sensitivity")
labels = load_labels()


def mechanism(drug):
    name = re.sub(r"_\d+$", "", drug)
    return next((m for m, names in MECHANISM.items() if name in names), "Other")


raw = pd.read_csv(OUT / "26_oncopredict" / "DrugPredictions_lnIC50.csv", index_col=0)
raw.index = [sample_id(i) for i in raw.index]
raw = raw[~raw.index.duplicated() & raw.index.notna()]
drugs = raw.join(labels.rename("gCPS"), how="inner")
cols = [c for c in drugs.columns if c != "gCPS"]

# omnibus test for every compound, then one-vs-rest effect sizes
stats = pd.DataFrame([dict(drug=c, **{f"median_{s}": drugs.loc[drugs.gCPS == s, c].median() for s in SUBTYPES},
                           kruskal_P=kruskal(*[drugs.loc[drugs.gCPS == s, c].dropna() for s in SUBTYPES])[1]) for c in cols])
stats["P_adj"] = bh(stats.kruskal_P)
stats["mechanism"] = stats.drug.map(mechanism)
stats.to_csv(out / "drug_kruskal_wallis.csv", index=False)
print(f"{(stats.P_adj < 0.05).sum()} of {len(stats)} compounds differ across subtypes (P-adj < 0.05)")

z = drugs[cols].groupby(drugs.gCPS).median().reindex(SUBTYPES)
z = (z - z.mean()) / z.std().replace(0, np.nan)
significant = set(stats.drug[stats.P_adj < 0.05])
selected = []
for s in SUBTYPES:
    ranked = [d for d in z.loc[s].sort_values().index if d in significant]
    forced = [d for d in cols if re.sub(r"_\d+$", "", d) in FORCED[s]]
    picks = forced + [d for d in ranked if d not in forced and mechanism(d) != "Other"]
    selected += [d for d in picks[:PER_SUBTYPE] if d not in selected]
pd.DataFrame({"drug": selected, "mechanism": [mechanism(d) for d in selected]}).to_csv(out / "selected_drugs.csv", index=False)

plt = style_matplotlib()
fig, ax = plt.subplots(figsize=(6, 0.28 * len(selected) + 2))
ax.imshow(z[selected].T.values, cmap="RdBu", vmin=-1.6, vmax=1.6, aspect="auto")      # blue = more sensitive
ax.set_yticks(range(len(selected)))
ax.set_yticklabels([f"{re.sub(r'_[0-9]+$', '', d)}  [{mechanism(d)}]" for d in selected], fontsize=8)
ax.set_xticks(range(4))
ax.set_xticklabels(SUBTYPES)
save_fig(fig, out / "fig6H_predicted_sensitivity")
plt.close(fig)

# PARP inhibitors, each subtype against the rest
rows = []
for drug in PARP:
    kw = kruskal(*[drugs.loc[drugs.gCPS == s, drug].dropna() for s in SUBTYPES])[1]
    for s in SUBTYPES:
        a, b = drugs.loc[drugs.gCPS == s, drug], drugs.loc[drugs.gCPS != s, drug]
        delta, lo, hi = cliffs_delta_ci(a, b)
        rows.append(dict(drug=drug.split("_")[0], subtype=s, kruskal_P=kw, cliffs_delta=delta, ci_low=lo, ci_high=hi, P=mwu_p(a, b)))
parp = pd.DataFrame(rows)
parp["P_adj"] = parp.groupby("subtype")["P"].transform(bh)
parp.to_csv(out / "parp_inhibitor_sensitivity.csv", index=False)

# immune context: interferon response (GSVA), CD8 / M1 / NK / CD4 fractions (CIBERSORT)
gsva = pd.read_csv(OUT / "22_gsva" / "HALLMARK_gsva_scores.csv", index_col=0)
gsva.index = [sample_id(i) for i in gsva.index]
immune = gsva[["HALLMARK_INTERFERON_GAMMA_RESPONSE", "HALLMARK_INTERFERON_ALPHA_RESPONSE"]]
immune = immune[~immune.index.duplicated()]
rows = []
for f in immune.columns:
    d = immune[[f]].join(labels.rename("gCPS"), how="inner")
    for s in SUBTYPES:
        a, b = d.loc[d.gCPS == s, f], d.loc[d.gCPS != s, f]
        delta, lo, hi = cliffs_delta_ci(a, b)
        rows.append(dict(feature=f, subtype=s, cliffs_delta=delta, ci_low=lo, ci_high=hi, P=mwu_p(a, b)))
immune_table = pd.DataFrame(rows)
immune_table["P_adj"] = bh(immune_table.P)
immune_table.to_csv(out / "immune_context_by_subtype.csv", index=False)
