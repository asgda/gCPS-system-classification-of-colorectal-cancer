"""Step 25. EMT/stemness/proliferation, ESTIMATE, CIBERSORT and TIDE features by subtype (Figure 6B-E, Table S7c-g)."""
import os
import re

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import kruskal

from gcps_utils import DATA, OUT, SUBTYPES, bh, cliffs_delta, load_clinical, load_labels, mwu_p, outdir, sample_id, \
    save_fig, style_matplotlib

TPM = os.environ.get("GCPS_TPM", str(DATA / "CRC.SW.mRNA.symbol.TPM.txt"))
NUNES_TABLE29 = os.environ.get("GCPS_NUNES_TABLE29", str(DATA / "Supplementary_Table_29.xlsx"))
SIGNATURES = {
    "EMT_SCORE": (["VIM", "ZEB1", "ZEB2", "SNAI1", "SNAI2", "TWIST1", "TWIST2", "FN1", "CDH2"], ["CDH1", "EPCAM"]),
    "STEMNESS_SCORE": (["SOX2", "NANOG", "POU5F1", "KLF4", "MYC"], []),
    "PROLIFERATION_SCORE": (["MKI67", "PCNA", "TOP2A", "CDK1", "CCNB1", "CCNB2"], []),
}
TIDE_FEATURES = ["TIDE", "IFNG", "MSI Expr Sig", "CD274", "CD8", "Dysfunction", "Exclusion", "MDSC", "CAF", "TAM M2"]

out = outdir("25_immune_signature_tests")
labels = load_labels()


def tumor_matrix():
    x = pd.read_csv(TPM, sep="\t", index_col=0)
    x = x.loc[:, [c.endswith((".T", ".T2")) for c in x.columns]].clip(lower=0)
    z = np.log2(x + 1)
    z = z.sub(z.mean(axis=1), axis=0).div(z.std(axis=1), axis=0).fillna(0)     # gene-wise z-score
    z.columns = [sample_id(c) for c in z.columns]
    z = z.T
    return z[~z.index.duplicated()]


def signature_scores(z):
    d = pd.DataFrame(index=z.index)
    for name, (pos, neg) in SIGNATURES.items():
        score = z[[g for g in pos if g in z]].mean(axis=1)
        d[name] = score - z[[g for g in neg if g in z]].mean(axis=1) if neg else score
    return d


def cibersort():
    rows = list(load_workbook(NUNES_TABLE29, read_only=True, data_only=True).active.iter_rows(values_only=True))
    header, data = rows[3], [r for r in rows[4:] if r[0]]
    d = pd.DataFrame([r[:27] for r in data], columns=["Mixture"] + list(header[1:27]))
    d.index = d.Mixture.map(sample_id)
    d = d[d.index.notna() & ~d.index.duplicated()]
    return d.drop(columns=["Mixture", "pValue", "Correlation", "RMSE", "Absolute score (sig.score)"], errors="ignore").apply(pd.to_numeric)


def estimate():
    e = pd.read_csv(OUT / "23_estimate" / "estimate_scores.gct", sep="\t", skiprows=2).drop(columns="Description")
    e = e.set_index("NAME").T
    e.index = [sample_id(i) for i in e.index]
    return e[~e.index.duplicated()].apply(pd.to_numeric)


def tide():
    t = pd.read_csv(OUT / "24_tide" / "tide_output.csv", sep=None, engine="python")
    t.columns = t.columns.astype(str).str.replace('"', "").str.strip()
    t = t.rename(columns={t.columns[0]: "sid", "MSI Score": "MSI Expr Sig"})
    t.index = t.sid.map(sample_id)
    t = t[t.index.notna() & ~t.index.duplicated()]
    return t[[c for c in TIDE_FEATURES if c in t]].apply(pd.to_numeric)


def kruskal_table(panel, name, subtypes=SUBTYPES, subset=None):
    d = panel.join(labels.rename("gCPS"), how="inner")
    if subset is not None:
        d = d[d.index.isin(subset)]
        d = d[d.gCPS.isin(subtypes)]
    rows = []
    for f in panel.columns:
        groups = [d.loc[d.gCPS == s, f].dropna() for s in subtypes]
        rows.append(dict(panel=name, feature=f, **{f"mean_{s}": g.mean() for s, g in zip(subtypes, groups)},
                         kruskal_P=kruskal(*groups)[1]))
    t = pd.DataFrame(rows)
    t["P_adj"] = bh(t.kruskal_P)
    return t


z = tumor_matrix()
panels = {"signature scores": signature_scores(z), "ESTIMATE": estimate(), "CIBERSORT": cibersort(), "TIDE": tide()}
tables = pd.concat([kruskal_table(p, n) for n, p in panels.items()])
tables.to_csv(out / "subtype_tests_by_panel.csv", index=False)

# MSI status: how much of the TIDE pattern is the MSI-high status of gCPS3
msi = load_clinical()["msi"]
mss = msi.index[msi == 0]
stratified = kruskal_table(panels["TIDE"], "TIDE, MSS tumors only", ["gCPS1", "gCPS2", "gCPS4"], subset=mss)
rows = []
for f in panels["TIDE"].columns:
    t = panels["TIDE"][f].dropna()
    a, b = t[t.index.isin(msi.index[msi == 1])], t[t.index.isin(mss)]
    rows.append(dict(feature=f, cliffs_delta_MSI_vs_MSS=cliffs_delta(a, b), P=mwu_p(a, b)))
msi_vs_mss = pd.DataFrame(rows)
msi_vs_mss["P_adj"] = bh(msi_vs_mss.P)
pd.concat([stratified, msi_vs_mss]).to_csv(out / "tide_msi_stratified.csv", index=False)

# gCPS3 microsatellite instability counts
counts = labels.to_frame("gCPS").join(msi).groupby("gCPS")["msi"].agg(["sum", "count"]).reindex(SUBTYPES)
counts.to_csv(out / "msi_counts_by_subtype.csv")
print(tables[tables.panel.isin(["signature scores", "ESTIMATE"])].round(5).to_string(index=False))

plt = style_matplotlib()
fig, axes = plt.subplots(1, 4, figsize=(18, 6))
for ax, (name, panel) in zip(axes, panels.items()):
    t = tables[tables.panel == name]
    m = t.set_index("feature")[[f"mean_{s}" for s in SUBTYPES]]
    m.columns = SUBTYPES
    zs = m.sub(m.mean(axis=1), axis=0).div(m.std(axis=1).replace(0, np.nan), axis=0).fillna(0)
    zs = zs[t.set_index("feature").P_adj < 0.05] if name == "CIBERSORT" else zs
    ax.imshow(zs.values, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
    ax.set_yticks(range(len(zs)))
    ax.set_yticklabels(zs.index, fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(SUBTYPES, rotation=45)
    ax.set_title(name)
save_fig(fig, out / "fig6B-E_scores_heatmaps")
plt.close(fig)
