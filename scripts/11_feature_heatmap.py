"""Step 11. Most prevalent features of each subtype, by modality (Figure 5E)."""
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from gcps_utils import COLORS, OUT, SUBTYPES, load_matrix, outdir, sample_id, save_fig, style_matplotlib

N_ROWS, JACCARD_MAX, MAX_ARMS = 16, 0.8, 6
MODS = ["SNV", "CNV", "SV"]
TITLE = {"SNV": "Short variants", "CNV": "CNV", "SV": "SV"}
MARKER = {"gain": "^", "loss": "v", "loh": "D", "mixed": "o"}
FAMILY = {
    "WNT": "APC RNF43 WNT7B KREMEN1 CTNNB1 AXIN1 AXIN2 TCF7L2 LRP6 DKK1 NKD1 NKD2 NOTUM FZD7 FZD8 FZD10 SFRP2 WNT2B "
           "WNT5B WNT6 WNT7A WNT1 WNT16 BCL9 BCL9L CSNK1E DVL1 DVL2 FZD3 CTBP1 CTBP2 APC2 CTNNA1".split(),
    "RAS/MAPK": "KRAS NRAS BRAF NF1 ARAF RAF1 MAP2K3 MAP2K4 MAPK1 MAPK11 MAPK12 EGFR ERBB2 ERBB3 ERBB4 GNAS".split(),
    "TGF-β": "ACVR2A ACVR1B TGFBR2 TGFBR1 SMAD2 SMAD3 SMAD4 TGFB1 GDF15 TGIF1 BMPR1A".split(),
    "PI3K": "PTEN PIK3CA PIK3R1 PIK3R2 PIK3R5 AKT1 AKT2 AKT3 PPP2R1A STRAP".split(),
    "p53/DDR": "TP53 BAX BBC3 ATM CHEK1 CHEK2 CDKN2A CCND2 CDK8 MDM2 LATS2".split(),
    "MMR/POL": "MSH2 MSH3 MSH6 MLH1 PMS2 POLE POLD1 MUTYH MGMT MRE11".split(),
    "Chromatin": "ARID1A ARID1B EP300 KMT2C KMT2D CHD8 SMARCA4 NCOR2 BCOR BCORL1 KMT2B CREBBP".split(),
}
FAMILY_COLORS = {"WNT": "#E69F00", "RAS/MAPK": "#56B4E9", "TGF-β": "#009E73", "PI3K": "#CC79A7", "p53/DDR": "#D55E00",
                 "MMR/POL": "#0072B2", "Chromatin": "#F0E442", "Other": "#999999", "Arm-level CIN": "#000000"}
family_of = {g: f for f, genes in FAMILY.items() for g in genes}

enrich = pd.read_csv(OUT / "10_enrichment" / "all_feature_enrichment.csv")
enrich["label"] = enrich.feature.str.split("_", n=1).str[1]
enrich["key"] = enrich.label.str.upper()

x = load_matrix()
x.index = [sample_id(i) for i in x.index]
x = x.fillna(0).gt(0)


def jaccard(f1, f2):
    a, b = x[f1].values, x[f2].values
    union = (a | b).sum()
    return (a & b).sum() / union if union else 0.0


selected = {}
for mod in MODS:
    sub = enrich[enrich.modality == mod]
    candidates = {s: list(sub[(sub.subtype == s) & ~((sub.P_adj < 0.05) & (sub.direction == "depleted"))]
                          .sort_values(["prev_in", "log2OR"], ascending=False).feature) for s in SUBTYPES}
    pointer, chosen, n_arms, progress = {s: 0 for s in SUBTYPES}, [], 0, True
    while len(chosen) < N_ROWS and progress:
        progress = False
        for s in SUBTYPES:
            while pointer[s] < len(candidates[s]):
                f = candidates[s][pointer[s]]
                pointer[s] += 1
                is_arm = bool(sub[sub.feature == f].iloc[0].is_arm)
                if f in chosen or (mod == "CNV" and is_arm and n_arms >= MAX_ARMS):
                    continue
                if any(jaccard(f, g) >= JACCARD_MAX for g in chosen):
                    continue
                chosen.append(f)
                n_arms += is_arm
                progress = True
                break
            if len(chosen) == N_ROWS:
                break
    prev = sub.pivot(index="feature", columns="subtype", values="prev_in")
    info = pd.DataFrame({"feature": chosen, "top": [prev.loc[f].idxmax() for f in chosen],
                         "top_prev": [prev.loc[f].max() for f in chosen]})
    info["order"] = info.top.map({s: i for i, s in enumerate(SUBTYPES)})
    selected[mod] = info.sort_values(["order", "top_prev"], ascending=[True, False])

out = outdir("11_heatmap")
pd.concat([d.assign(modality=m) for m, d in selected.items()]).to_csv(out / "selected_features.csv", index=False)

plt = style_matplotlib()
plt.rcParams["font.size"] = 15
row_h = 0.40
fig = plt.figure(figsize=(21.5, row_h * N_ROWS + 4.3))
W, H = fig.get_size_inches()
cmap = plt.get_cmap("RdYlBu_r")
cells = enrich[enrich.feature.isin([f for m in MODS for f in selected[m].feature])]
vmax = {m: (10 * np.ceil(cells[cells.modality == m].prev_in.max() / 10) if m != "SV"
            else float(np.ceil(cells[cells.modality == m].prev_in.max()))) for m in MODS}
ax_w, label_w, gap = 3.55 / W, 2.05 / W, 1.15 / W

for mi, mod in enumerate(MODS):
    info = selected[mod]
    h = row_h * N_ROWS / H
    ax = fig.add_axes([0.012 + label_w + mi * (ax_w + label_w + gap), 1 - 1.75 / H - h, ax_w, h])
    norm = Normalize(0, vmax[mod])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(N_ROWS - 0.5, -0.5)
    for i, r in enumerate(info.itertuples()):
        for j, s in enumerate(SUBTYPES):
            m = cells[(cells.feature == r.feature) & (cells.subtype == s)]
            if m.empty:
                continue
            m = m.iloc[0]
            fc = cmap(norm(m.prev_in))
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=fc, edgecolor="white", linewidth=1))
            text_color = "white" if 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2] < 0.42 else "black"
            if m.P_adj < 0.05:
                ax.text(j, i - 0.04, "*" if m.direction == "enriched" else "−", ha="center", va="center",
                        fontsize=19, fontweight="bold", color=text_color)
            if mod == "CNV" and m.cnv_direction in MARKER:
                ax.scatter([j + .31], [i + .30], marker=MARKER[m.cnv_direction], s=34, color=text_color, linewidths=0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(SUBTYPES, fontweight="bold", fontsize=15)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(TITLE[mod], fontweight="bold", fontsize=23, pad=34)
    for i in range(1, len(info)):
        if info.top.iloc[i] != info.top.iloc[i - 1]:
            ax.axhline(i - 0.5, color="#333333", linewidth=1)
    pos = ax.get_position()
    rh = pos.height / N_ROWS
    for i, r in enumerate(info.itertuples()):
        lab = enrich[enrich.feature == r.feature].iloc[0]
        family = "Arm-level CIN" if lab.is_arm else family_of.get(lab.key, "Other")
        y = pos.y1 - (i + 1) * rh
        fig.add_artist(Rectangle((pos.x0 - 0.0155, y), 0.006, rh, transform=fig.transFigure, facecolor=COLORS[r.top]))
        fig.add_artist(Rectangle((pos.x0 - 0.009, y), 0.006, rh, transform=fig.transFigure, facecolor=FAMILY_COLORS[family]))
        fig.text(pos.x0 - 0.0195, y + rh / 2, lab.label, ha="right", va="center", fontsize=17)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=fig.add_axes([pos.x1 + 0.007, pos.y0, 0.0075, pos.height]))
    cb.set_label("Prevalence in subtype (%)", fontsize=14)

fig.text(0.5, 0.988, "Most prevalent features of each subtype, by modality", ha="center", va="top", fontsize=26,
         fontweight="bold")
fig.text(0.5, 0.952, "* significantly enriched, − significantly depleted (P-adj < 0.05, one-vs-rest); "
         "each modality has its own color scale", ha="center", va="top", fontsize=16, fontstyle="italic")
families = list(dict.fromkeys(
    "Arm-level CIN" if enrich[enrich.feature == f].iloc[0].is_arm else family_of.get(enrich[enrich.feature == f].iloc[0].key, "Other")
    for m in MODS for f in selected[m].feature))
fig.legend(handles=[Patch(facecolor=FAMILY_COLORS[f], label=f) for f in families], loc="lower center", ncol=7,
           frameon=False, title="Pathway family (inner strip, descriptive only)", fontsize=15, bbox_to_anchor=(0.5, 0.025))
fig.legend(handles=[Patch(facecolor=COLORS[s], label=s) for s in SUBTYPES], loc="lower center", ncol=4, frameon=False,
           title="Outer strip: subtype in which the feature is most prevalent", fontsize=15, bbox_to_anchor=(0.5, -0.058))
fig.legend(handles=[Line2D([], [], ls="", label="CNV direction:")] +
           [Line2D([], [], ls="", marker=MARKER[k], color="black", markersize=9, label=lab)
            for k, lab in (("gain", "gain"), ("loss", "loss"), ("loh", "LOH"), ("mixed", "mixed (no direction ≥ 60%)"))],
           loc="lower center", ncol=5, frameon=False, fontsize=15, bbox_to_anchor=(0.5, 0.125))
save_fig(fig, out / "fig5E_feature_heatmap")
plt.close(fig)
