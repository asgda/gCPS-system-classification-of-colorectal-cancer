"""Step 14. Genomic burden by subtype: TMB, SV junctions, copy-number gain, loss and LOH (Supplementary Table S3)."""
import itertools
import re

import pandas as pd
from scipy.stats import kruskal

from gcps_utils import DATA, OUT, SUBTYPES, bh, cliffs_delta_ci, load_labels, mwu_p, outdir, sample_id

TMB_DENOMINATOR = 3000.0
ARM = re.compile(r"^CNV_(?:\d{1,2}|X|Y)[pq]_(GAIN|LOSS)$")

out = outdir("14_burden")
labels = load_labels()


def read(rel):
    d = pd.read_csv(OUT / rel, index_col=0)
    d.index = [sample_id(i) for i in d.index]
    return d.reindex(labels.index)


nunes = pd.read_excel(DATA / "Supplementary_Table_01.xlsx")
nunes["sid"] = nunes["Sample ID"].map(sample_id)
nunes = nunes.drop_duplicates("sid").set_index("sid").reindex(labels.index)

direction = read("02_cnv/cnv_directional_discovery_matrix.csv").fillna(0)
arm_loss = [c for c in direction if ARM.match(c) and c.endswith("LOSS")]
arm_gain = [c for c in direction if ARM.match(c) and c.endswith("GAIN")]
gene_loss = [c for c in direction if c.endswith("_LOSS") and not ARM.match(c)]
gene_gain = [c for c in direction if c.endswith("_GAIN") and not ARM.match(c)]
gene_loh = [c for c in direction if c.endswith("_LOH")]
sv_burden = pd.read_csv(OUT / "03_sv" / "sv_burden.csv").assign(sid=lambda d: d.sample_id).set_index("sid")
architecture = pd.read_csv(OUT / "03_sv" / "sv_architecture_matrix.csv").set_index("sample_id")
architecture.index = [sample_id(i) for i in architecture.index]

metrics = pd.DataFrame({
    "TMB (mutations/Mb)": pd.to_numeric(nunes["Total Mutation Count"], errors="coerce") / TMB_DENOMINATOR,
    "SV junctions": sv_burden["SV_TOTAL"].reindex(labels.index),
    "SV gene-event fraction": read("03_sv/sv_discovery_matrix.csv").mean(axis=1),
    "Arm-level loss fraction": direction[arm_loss].mean(axis=1),
    "Arm-level gain fraction": direction[arm_gain].mean(axis=1),
    "Focal loss fraction": direction[gene_loss].mean(axis=1),
    "Focal gain fraction": direction[gene_gain].mean(axis=1),
    "Focal LOH fraction": direction[gene_loh].mean(axis=1),
    "Arm-altered fraction": read("02_cnv/arm_cin_matrix.csv").mean(axis=1),
})
metrics.assign(gCPS=labels).to_csv(out / "burden_per_tumor.csv")

summary = pd.DataFrame({s: metrics[labels == s].median() for s in SUBTYPES})
summary["kruskal_P"] = [kruskal(*[metrics.loc[labels == s, m].dropna() for s in SUBTYPES])[1] for m in metrics]
summary["kruskal_P_adj"] = bh(summary.kruskal_P)
summary.to_csv(out / "burden_medians_by_subtype.csv")


def effects(group_a, group_b, name):
    rows = []
    for m in metrics:
        a, b = metrics.loc[group_a, m], metrics.loc[group_b, m]
        d, lo, hi = cliffs_delta_ci(a, b)
        rows.append(dict(contrast=name, metric=m, median_a=a.median(), median_b=b.median(), cliffs_delta=d,
                         ci_low=lo, ci_high=hi, P=mwu_p(a, b)))
    r = pd.DataFrame(rows)
    r["P_adj"] = bh(r.P)
    return r


pd.concat([effects(labels == "gCPS4", labels != "gCPS4", "gCPS4 vs other subtypes"),
           effects(labels == "gCPS4", labels == "gCPS1", "gCPS4 vs gCPS1")]).to_csv(
    out / "burden_effect_sizes.csv", index=False)

pairs = [dict(metric=m, a=a, b=b, P=mwu_p(metrics.loc[labels == a, m], metrics.loc[labels == b, m]))
         for m in ("SV junctions", "TMB (mutations/Mb)") for a, b in itertools.combinations(SUBTYPES, 2)]
pairs = pd.DataFrame(pairs)
pairs["P_adj"] = pairs.groupby("metric")["P"].transform(bh)
pairs.to_csv(out / "burden_pairwise_tests.csv", index=False)
print(summary.round(3).to_string())
