"""Step 10. One-vs-rest enrichment of SNV, CNV and SV features in each gCPS subtype (Figures 4A, 5A-D)."""
import math
import re

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from gcps_utils import COLORS, MODALITY_COLORS, OUT, SUBTYPES, bh, load_labels, load_matrix, outdir, sample_id, \
    save_fig, style_matplotlib

ARM_RE = re.compile(r"^(?:\d{1,2}|X|Y)[pq]$", re.I)
MIN_CARRIERS = 3
DOMINANT_SHARE = 0.6

out = outdir("10_enrichment")
labels = load_labels()
x = load_matrix()
x.index = [sample_id(i) for i in x.index]
x = (x.loc[labels.index].fillna(0) > 0).astype(int)

direction = pd.read_csv(OUT / "02_cnv" / "cnv_directional_discovery_matrix.csv", index_col=0)
direction.index = [sample_id(i) for i in direction.index]
direction = direction.reindex(labels.index).fillna(0).astype(bool)


def fisher_one_vs_rest(y, in_group):
    a, b = int((y & in_group).sum()), int((~y & in_group).sum())
    c, d = int((y & ~in_group).sum()), int((~y & ~in_group).sum())
    odds, p = fisher_exact([[a, b], [c, d]])
    lor = math.log((a + .5) * (d + .5) / ((b + .5) * (c + .5)))        # Haldane-Anscombe correction
    se = math.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (d + .5))
    return dict(carriers_in=a, n_in=a + b, carriers_rest=c, n_rest=c + d,
                prev_in=100 * a / (a + b), prev_rest=100 * c / (c + d), odds_ratio=odds, log2OR=lor / math.log(2),
                log2OR_low=(lor - 1.96 * se) / math.log(2), log2OR_high=(lor + 1.96 * se) / math.log(2),
                P=p, direction="enriched" if lor > 0 else "depleted")


def dominant_direction(subtype, feature):
    """Majority copy-number direction among carriers: loss (incl. homozygous), gain or copy-neutral LOH."""
    gene = feature.replace("CNV_", "")
    members = (labels == subtype).values
    col = lambda s: direction[f"CNV_{gene}_{s}"].values & members if f"CNV_{gene}_{s}" in direction else np.zeros(len(members), bool)
    loss, gain, loh = col("LOSS") | col("HOMDEL"), col("GAIN"), col("LOH")
    counts = {"loss": loss.sum(), "gain": gain.sum(), "loh": (loh & ~loss & ~gain).sum()}
    total = sum(counts.values())
    if total == 0:
        return None
    top = max(counts, key=counts.get)
    return top if counts[top] / total >= DOMINANT_SHARE else "mixed"


rows = []
for s in SUBTYPES:
    in_group = (labels == s).values
    for f in x.columns:
        if x[f].sum() < MIN_CARRIERS:
            continue
        gene = f.split("_", 1)[1]
        rows.append(dict(subtype=s, feature=f, gene_or_arm=gene, modality=f.split("_")[0], is_arm=bool(ARM_RE.match(gene)),
                         **fisher_one_vs_rest(x[f].values.astype(bool), in_group)))
enrich = pd.DataFrame(rows)
enrich["P_adj"] = enrich.groupby("subtype")["P"].transform(bh)
enrich["cnv_direction"] = [dominant_direction(r.subtype, r.feature) if r.modality == "CNV" else None
                           for r in enrich.itertuples()]
enrich.sort_values(["subtype", "P_adj", "P"]).to_csv(out / "all_feature_enrichment.csv", index=False)

sig = enrich[(enrich.P_adj < 0.05) & (enrich.direction == "enriched")]
summary = sig.groupby(["subtype", "modality"]).size().unstack(fill_value=0).reindex(SUBTYPES)
summary.to_csv(out / "enriched_feature_counts_by_modality.csv")
print(summary)

plt = style_matplotlib()
fig, ax = plt.subplots(figsize=(6, 4.5))
bottom = np.zeros(len(SUBTYPES))
for m in ("SNV", "CNV", "SV"):
    vals = summary.get(m, pd.Series(0, index=SUBTYPES)).values
    ax.bar(SUBTYPES, vals, bottom=bottom, color=MODALITY_COLORS[m], label={"SNV": "Short variants"}.get(m, m))
    bottom += vals
ax.set_ylabel("Enriched features (P-adj < 0.05)")
ax.legend(frameon=False)
save_fig(fig, out / "fig4A_enriched_feature_counts")
plt.close(fig)

# Figure 5A-D: 20 strongest enriched features per subtype
fig, axes = plt.subplots(1, 4, figsize=(20, 6.5))
for ax, s in zip(axes, SUBTYPES):
    top = sig[sig.subtype == s].nlargest(20, "log2OR").sort_values("log2OR")
    y = np.arange(len(top))
    ax.errorbar(top.log2OR, y, xerr=[top.log2OR - top.log2OR_low, top.log2OR_high - top.log2OR], fmt="none",
                ecolor="#888888", lw=1)
    ax.scatter(top.log2OR, y, color=COLORS[s], zorder=3, s=40)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g} ({p:.0f}%)" for g, p in zip(top.gene_or_arm, top.prev_in)], fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(s, color=COLORS[s], fontweight="bold")
    ax.set_xlabel("log2 odds ratio")
save_fig(fig, out / "fig5A-D_top_enriched_features")
plt.close(fig)
