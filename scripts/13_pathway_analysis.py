"""Step 13. Pathway co-alteration within subtypes and convergent CNV-loss plus SV dual hits."""
import math
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

from gcps_utils import COLORS, OUT, SUBTYPES, bh, load_labels, load_matrix, outdir, sample_id, save_fig, style_matplotlib

PATHWAY_GENES = {
    "WNT": "APC APC2 CTNNB1 AXIN1 AXIN2 RNF43 ZNRF3 TCF7L2 LRP6 LRP5 AMER1 BCL9 BCL9L WNT5B WNT7B KREMEN1 NKD1 FBXW7",
    "RAS_MAPK": "KRAS NRAS HRAS BRAF ARAF RAF1 EGFR ERBB2 ERBB3 MET MAPK1 MAPK3 MAP2K1 MAP2K2 MAP2K4 DUSP16 NF1 GNAS",
    "TGFb": "SMAD2 SMAD3 SMAD4 TGFBR1 TGFBR2 ACVR1B ACVR2A BMPR1A TGFB1 TGFB2",
    "PI3K": "PIK3CA PIK3CB PIK3R1 PIK3R2 PTEN AKT1 AKT2 AKT3 MTOR RICTOR STK11 TSC1 TSC2",
    "p53_DDR": "TP53 ATM ATR CHEK1 CHEK2 BRCA1 BRCA2 BAX BBC3 MDM2 MDM4 PARP4",
    "MMR_POL": "MLH1 MSH2 MSH3 MSH6 PMS1 PMS2 POLE POLD1 EXO1",
    "CHROMATIN": "ARID1A ARID1B SMARCA4 PBRM1 EP300 CREBBP KMT2C KMT2D CHD8 BCOR NCOR2",
}
PATHWAY_GENES = {k: set(v.split()) for k, v in PATHWAY_GENES.items()}
ORDER = list(PATHWAY_GENES) + ["CIN_ARM", "SV_COMPLEX"]

# pathways carrying the documented gCPS4-enriched structural variants
DUAL_HIT_GENES = {
    "WNT": "APC CTNNB1 AXIN1 AXIN2 RNF43 TCF7L2 KREMEN1 DVL2 DVL3 WNT7B WNT5B LRP6 FBXW7 AMER1 BCL9 BCL9L PRKD1",
    "TGFb": "SMAD2 SMAD3 SMAD4 TGFBR1 TGFBR2 ACVR2A ACVR1B TGFB1 STRAP",
    "PI3K": "PIK3CA PIK3R1 PIK3R2 PIK3R5 PTEN AKT1 AKT2 AKT3 STK11 TSC1 TSC2 TULP3",
}
DUAL_HIT_GENES = {k: v.split() for k, v in DUAL_HIT_GENES.items()}
ARM = re.compile(r"^CNV_(?:\d{1,2}|X|Y)[pq]$")

out = outdir("13_pathways")
labels = load_labels()
x = load_matrix()
x.index = [sample_id(i) for i in x.index]
x = x.loc[labels.index].fillna(0).astype(int)


def gene_of(feature):
    return re.sub(r"_(GAIN|LOSS|LOH|HOMDEL)$", "", re.sub(r"^(SNV_|CNV_|SV_)", "", feature)).upper()


def pathway_matrix():
    m = pd.DataFrame(0, index=x.index, columns=ORDER)
    sv_burden = x[[c for c in x if c.startswith("SV_")]].sum(axis=1)
    m["SV_COMPLEX"] = (sv_burden > sv_burden.median()).astype(int)    # genome-wide, independent of gene flags
    for f in x.columns:
        if ARM.match(f):
            m["CIN_ARM"] = np.maximum(m["CIN_ARM"], x[f])
            continue
        for pathway, genes in PATHWAY_GENES.items():
            if gene_of(f) in genes:
                m[pathway] = np.maximum(m[pathway], x[f])
    return m


def pair_test(a, b):
    n11, n10 = int(((a == 1) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    n01, n00 = int(((a == 0) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())
    odds, p = fisher_exact([[n11, n10], [n01, n00]])
    lor = math.log((n11 + .5) * (n00 + .5) / ((n10 + .5) * (n01 + .5)))
    return dict(n11=n11, n10=n10, n01=n01, n00=n00, odds_ratio=odds, log2OR=lor / math.log(2), P=p)


pm = pathway_matrix()
pm.assign(gCPS=labels).to_csv(out / "pathway_matrix.csv")
pm.groupby(labels).mean().T.reindex(columns=SUBTYPES).mul(100).to_csv(out / "pathway_prevalence_by_subtype.csv")

rows = []
for s in SUBTYPES:
    sub = pm[labels == s]
    for i, a in enumerate(ORDER):
        for b in ORDER[i + 1:]:
            rows.append(dict(subtype=s, pathway_a=a, pathway_b=b, **pair_test(sub[a], sub[b])))
tests = pd.DataFrame(rows)
tests["P_adj"] = tests.groupby("subtype")["P"].transform(bh)
tests["relation"] = np.where(tests.log2OR > 0, "co-occurrence", "mutual exclusion")
tests.to_csv(out / "pathway_pairwise_tests.csv", index=False)

# convergent dual hit: a CNV loss, LOH or homozygous deletion plus an SV in the same pathway
cnv = pd.read_csv(OUT / "02_cnv" / "cnv_directional_discovery_matrix.csv", index_col=0)
cnv.index = [sample_id(i) for i in cnv.index]
cnv = cnv.reindex(labels.index).fillna(0)
sv = pd.read_csv(OUT / "03_sv" / "sv_discovery_matrix.csv", index_col=0)
sv.index = [sample_id(i) for i in sv.index]
sv = sv.reindex(labels.index).fillna(0)

dual = pd.DataFrame(index=labels.index)
for pathway, genes in DUAL_HIT_GENES.items():
    loss_cols = [f"CNV_{g}_{k}" for g in genes for k in ("LOSS", "LOH", "HOMDEL") if f"CNV_{g}_{k}" in cnv]
    sv_cols = [f"SV_{g}" for g in genes if f"SV_{g}" in sv]
    dual[pathway] = ((cnv[loss_cols].sum(axis=1) > 0) & (sv[sv_cols].sum(axis=1) > 0)).astype(int)
dual["n_pathways"] = dual.sum(axis=1)
dual["any"] = (dual.n_pathways > 0).astype(int)
dual["gCPS"] = labels
dual.to_csv(out / "dual_hit_per_tumor.csv")


def compare(mask_a, mask_b, name):
    a, b = int(dual.loc[mask_a, "any"].sum()), int(dual.loc[mask_b, "any"].sum())
    na, nb = int(mask_a.sum()), int(mask_b.sum())
    odds, p = fisher_exact([[a, na - a], [b, nb - b]])
    return dict(test=name, hits_a=a, n_a=na, pct_a=100 * a / na, hits_b=b, n_b=nb, pct_b=100 * b / nb, odds_ratio=odds, P=p)


is4, is1 = dual.gCPS == "gCPS4", dual.gCPS == "gCPS1"
stats = [compare(is4, is1, "gCPS4 vs gCPS1"), compare(is4, ~is4, "gCPS4 vs other subtypes")]
stats.append(dict(test="four-subtype omnibus (chi-square)", P=chi2_contingency(pd.crosstab(dual.gCPS, dual["any"]))[1]))
u, p = mannwhitneyu(dual.loc[is4, "n_pathways"], dual.loc[is1, "n_pathways"])
stats.append(dict(test="pathways hit, gCPS4 vs gCPS1 (Cliff's delta)", odds_ratio=2 * u / (is4.sum() * is1.sum()) - 1, P=p))
stats = pd.DataFrame(stats)
stats["P_adj"] = bh(stats.P)
stats.to_csv(out / "dual_hit_statistics.csv", index=False)
print(stats[["test", "pct_a", "pct_b", "odds_ratio", "P", "P_adj"]].round(5).to_string(index=False))

plt = style_matplotlib()
fig, ax = plt.subplots(figsize=(6.5, 4.5))
groups = ["WNT", "TGFb", "PI3K", "any"]
w = 0.19
for i, s in enumerate(SUBTYPES):
    ax.bar(np.arange(4) + (i - 1.5) * w, [100 * dual.loc[dual.gCPS == s, g].mean() for g in groups], w * 0.94,
           color=COLORS[s], edgecolor="black", linewidth=0.6, label=s)
ax.set_xticks(range(4))
ax.set_xticklabels(["WNT", "TGF-β", "PI3K", "Any pathway"])
ax.set_ylabel("Tumors with a CNV-loss + SV double hit (%)")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
save_fig(fig, out / "dual_hit_summary")
plt.close(fig)
