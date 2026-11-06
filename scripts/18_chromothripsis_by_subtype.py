"""Step 18. High-confidence chromothripsis calls and their distribution across gCPS subtypes."""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from gcps_utils import OUT, SUBTYPES, bh, load_labels, mwu_p, outdir, sample_id

ALPHA = 0.05
out = outdir("16_chromothripsis")
labels = load_labels()

df = pd.read_csv(out / "shatterseek_chromsummary.csv")
df["sample_id"] = df.sample_id.map(sample_id)
intra = df.number_DEL + df.number_DUP + df.number_h2hINV + df.number_t2tINV
osc = df.max_number_oscillating_CN_segments_2_states
fragment_joins = df.pval_fragment_joins < ALPHA
enriched = (df.chr_breakpoint_enrichment < ALPHA) | (df.pval_exp_chr < ALPHA)

# ShatterSeek recommended cut-offs
rule_a = (intra >= 6) & (osc >= 7) & fragment_joins & enriched
rule_b = (intra >= 3) & (df.number_TRA >= 4) & (osc >= 7) & fragment_joins
low = (intra >= 6) & osc.between(4, 6) & fragment_joins & enriched
df["high"] = (rule_a | rule_b).fillna(False)
df["candidate"] = df.high | (low.fillna(False) & ~df.high)
df.to_csv(out / "chromosome_calls.csv", index=False)

per_tumor = df.groupby("sample_id")[["high", "candidate"]].sum().reindex(labels.index).fillna(0)
per_tumor["gCPS"] = labels

rows = []
for tier in ("high", "candidate"):
    positive = per_tumor[tier] > 0
    for name, a, b in (("gCPS4 vs other subtypes", per_tumor.gCPS == "gCPS4", per_tumor.gCPS != "gCPS4"),
                       ("gCPS4 vs gCPS1", per_tumor.gCPS == "gCPS4", per_tumor.gCPS == "gCPS1")):
        table = [[positive[a].sum(), (~positive[a]).sum()], [positive[b].sum(), (~positive[b]).sum()]]
        odds, p = fisher_exact(table)
        rows.append(dict(tier=tier, contrast=name, pct_a=100 * positive[a].mean(), pct_b=100 * positive[b].mean(),
                         odds_ratio=odds, P=p, burden_P=mwu_p(per_tumor.loc[a, tier], per_tumor.loc[b, tier])))
res = pd.DataFrame(rows)
res["P_adj"] = bh(res.P)
res.to_csv(out / "chromothripsis_tests.csv", index=False)
prevalence = per_tumor.assign(positive=per_tumor.high > 0).groupby("gCPS")["positive"].mean().reindex(SUBTYPES) * 100
prevalence.to_csv(out / "high_confidence_prevalence_by_subtype.csv")
print(prevalence.round(1).to_dict())
print(res.round(4).to_string(index=False))
