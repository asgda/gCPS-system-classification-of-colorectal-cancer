"""Step 15. Structural-variant mechanism mix by subtype (Supplementary Figure S2B-D, Supplementary Table S5)."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal

from gcps_utils import COLORS, OUT, SUBTYPES, bh, cliffs_delta_ci, load_labels, mwu_p, outdir, sample_id, save_fig, \
    style_matplotlib

CLASSES = {"SVCLASS_deletion": "Deletion", "SVCLASS_tandem-duplication": "Tandem duplication",
           "SVCLASS_inversion": "Inversion", "SVCLASS_translocation": "Translocation"}

out = outdir("15_sv_mechanism")
labels = load_labels()
arch = pd.read_csv(OUT / "03_sv" / "sv_architecture_matrix.csv")
arch.index = [sample_id(i) for i in arch.sample_id]
counts = arch.reindex(labels.index)[list(CLASSES)].fillna(0)
frac = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).rename(columns=CLASSES)
valid = frac.notna().all(axis=1)
frac, lab = frac[valid], labels[valid]
frac.assign(gCPS=lab).to_csv(out / "class_fractions_per_tumor.csv")

omnibus = pd.DataFrame([dict(sv_class=c, kruskal_P=kruskal(*[frac.loc[lab == s, c] for s in SUBTYPES])[1],
                             **{f"mean_{s}": frac.loc[lab == s, c].mean() for s in SUBTYPES}) for c in frac])
omnibus["P_adj"] = bh(omnibus.kruskal_P)
omnibus.to_csv(out / "class_fraction_kruskal_wallis.csv", index=False)

rows = []
for c in frac:
    a, b = frac.loc[lab == "gCPS4", c], frac.loc[lab != "gCPS4", c]
    d, lo, hi = cliffs_delta_ci(a, b)
    rows.append(dict(sv_class=c, gCPS4_median=a.median(), rest_median=b.median(), cliffs_delta=d, ci_low=lo,
                     ci_high=hi, P=mwu_p(a, b)))
effect = pd.DataFrame(rows)
effect["P_adj"] = bh(effect.P)
effect.to_csv(out / "class_fraction_gCPS4_vs_rest.csv", index=False)

dominant = frac.idxmax(axis=1)
table = pd.crosstab(lab, dominant).reindex(SUBTYPES)
(table.div(table.sum(axis=1), axis=0) * 100).round(1).to_csv(out / "dominant_class_percent.csv")
chi2, p, _, _ = chi2_contingency(table.values)
v = np.sqrt(chi2 / table.values.sum() / (min(table.shape) - 1))
pd.DataFrame([dict(chi2=chi2, P=p, cramers_v=v, n=int(table.values.sum()))]).to_csv(
    out / "dominant_class_chisquare.csv", index=False)

plt = style_matplotlib()
fig, ax = plt.subplots(figsize=(7, 4.5))
bottom = np.zeros(4)
palette = dict(zip(CLASSES.values(), ["#0072B2", "#009E73", "#E69F00", "#D55E00"]))
for c in frac:
    vals = np.array([frac.loc[lab == s, c].mean() * 100 for s in SUBTYPES])
    ax.bar(SUBTYPES, vals, bottom=bottom, color=palette[c], label=c)
    bottom += vals
ax.set_ylabel("Share of SV junctions (%)")
ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
save_fig(fig, out / "sv_class_composition")
plt.close(fig)
print(effect.round(4).to_string(index=False))
