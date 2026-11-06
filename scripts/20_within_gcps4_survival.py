"""Step 20. Survival within gCPS4 by dual-hit status and by a median split of the scar score (Supplementary Table S2)."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

from gcps_utils import OUT, load_clinical, load_labels, outdir, save_fig, style_matplotlib

out = outdir("20_within_gcps4")
labels = load_labels()
clin = load_clinical().join(labels.rename("gCPS"))
c4 = clin[clin.gCPS == "gCPS4"][["OS_months", "OS_event"]].copy()
c4["dual_hit"] = pd.read_csv(OUT / "13_pathways" / "dual_hit_per_tumor.csv", index_col=0)["any"]
scar = pd.read_csv(OUT / "19_hrd_scar" / "scar_scores_per_tumor.csv", index_col=0)["HRD_score"]
c4["scar_high"] = (scar.reindex(c4.index) >= scar.reindex(c4.index).median()).astype(int)
c4 = c4.dropna()
c4 = c4[c4.OS_months > 0]
c4.to_csv(out / "gcps4_survival_data.csv")


def compare(group, name):
    hi, lo = c4[c4[group] == 1], c4[c4[group] == 0]
    lr = logrank_test(hi.OS_months, lo.OS_months, hi.OS_event, lo.OS_event)
    s = CoxPHFitter().fit(c4[["OS_months", "OS_event", group]], "OS_months", "OS_event").summary.loc[group]
    return dict(comparison=name, n_high=len(hi), n_low=len(lo), HR=np.exp(s["coef"]), lower=np.exp(s["coef lower 95%"]),
                upper=np.exp(s["coef upper 95%"]), cox_P=s["p"], logrank_P=lr.p_value)


res = pd.DataFrame([compare("dual_hit", "dual-hit positive vs negative"),
                    compare("scar_high", "scar score at or above vs below median")])
res.to_csv(out / "within_gcps4_survival.csv", index=False)
print(res.round(3).to_string(index=False))

plt = style_matplotlib()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (group, labs, cols) in zip(axes, (("dual_hit", ("Dual-hit negative", "Dual-hit positive"), ("#999999", "#D55E00")),
                                          ("scar_high", ("Scar score below median", "Scar score at or above median"),
                                           ("#999999", "#CC79A7")))):
    for value, lab, col in zip((0, 1), labs, cols):
        d = c4[c4[group] == value]
        KaplanMeierFitter().fit(d.OS_months, d.OS_event, label=f"{lab} (n={len(d)})").plot_survival_function(
            ax=ax, color=col, ci_show=True)
    ax.set_xlabel("Months")
    ax.set_ylabel("Overall survival probability")
save_fig(fig, out / "km_within_gcps4")
plt.close(fig)
