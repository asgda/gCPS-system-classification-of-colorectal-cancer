"""Step 02. Gene- and arm-level copy-number features from FACETS segments."""
import gzip
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

from gcps_utils import JOBS, RAW, REF, outdir, panel_genes, sample_id

N_COHORT = 1063
MIN_OCCURRENCE = int(np.ceil(N_COHORT * 0.02))
MAX_OCCURRENCE = int(N_COHORT * 0.98)
MAX_MEDIAN_SEGMENT = 50_000_000
ARM_ALTERED_FRACTION = 0.5
PANEL = panel_genes()
PASSENGERS = {"TTN", "MUC16", "OBSCN", "DNAH5", "DNAH6", "DNAH7", "DNAH8", "DNAH9", "DNAH10", "DNAH11",
              "DNAH14", "DNAH17", "CSMD1", "CSMD3", "RYR1", "RYR2", "RYR3", "SYNE1", "SYNE2", "USH2A",
              "FLG", "PCLO", "XIRP2", "FAT3", "FAT4", "LRP1B", "MACROD2"}
ARM_RE = re.compile(r"CNV_(?:[0-9]{1,2}|X|Y)[pq]$")

genes = pd.read_csv(REF / "genes_protein_coding_hg38.bed", sep="\t", header=None).iloc[:, :4]
genes.columns = ["chr", "start", "end", "gene"]
genes["gene"] = genes["gene"].astype(str).str.upper()
genes = genes[genes.gene.isin(PANEL) & ~genes.gene.isin(PASSENGERS)]

cyto = pd.read_csv(REF / "cytoBand.txt.gz", sep="\t", header=None, names=["chr", "start", "end", "band", "stain"])
cyto["arm"] = np.where(cyto.band.str.startswith("p"), cyto.chr + "p", cyto.chr + "q")
ARMS = cyto.groupby("arm").agg(chr=("chr", "first"), start=("start", "min"), end=("end", "max"))


def read_segments(path):
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            info = dict(x.split("=", 1) for x in f[7].split(";") if "=" in x)
            try:
                rows.append((f[0], int(f[1]), int(info["END"]), float(info.get("TCN_EM", "nan")),
                             float(info.get("LCN_EM", "nan")), float(info.get("CNLR_MEDIAN", "nan"))))
            except (KeyError, ValueError):
                continue
    return pd.DataFrame(rows, columns=["chr", "start", "end", "TCN", "LCN", "log2CN"])


def process(path):
    sid = sample_id(path.split("/")[-1])
    if sid is None:
        return None
    seg = read_segments(path)
    if seg.empty:
        return None
    seg["size"] = seg.end - seg.start

    gene_rows = []
    for chrom, g_chr in genes.groupby("chr"):
        s = seg[seg.chr == chrom]
        if s.empty:
            continue
        for g in g_chr.itertuples():
            ov = s[(s.end >= g.start) & (s.start <= g.end)]
            if ov.empty:
                continue
            best = ov.loc[(ov.TCN - 2).abs().sort_values(ascending=False, kind="stable").index[0]]
            gene_rows.append((sid, f"CNV_{g.gene}", int(best.TCN != 2), best["size"], best.TCN,
                              int(best.TCN > 2), int(best.TCN < 2), int(best.TCN == 0), int(best.LCN == 0)))
    gene_df = pd.DataFrame(gene_rows, columns=["sample_id", "feature", "value", "segment_size", "TCN",
                                               "gain", "loss", "homdel", "loh"])

    arm_rows = []
    for arm, a in ARMS.iterrows():
        s = seg[(seg.chr == a.chr) & (seg.end >= a.start) & (seg.start <= a.end)]
        if s.empty:
            continue
        med = np.median(s.TCN)
        arm_rows.append((sid, f"CNV_{arm.replace('chr', '')}", int((s.TCN != 2).mean() >= ARM_ALTERED_FRACTION),
                         np.abs(s.TCN - 2).mean(), med, int(med > 2), int(med < 2)))
    arm_df = pd.DataFrame(arm_rows, columns=["sample_id", "feature", "value", "CIN_score", "median_TCN", "gain", "loss"])
    return gene_df, arm_df


def directional(gene_df, arm_df):
    parts = []
    for df, flags in ((gene_df, ["gain", "loss", "homdel", "loh"]), (arm_df, ["gain", "loss"])):
        for f in flags:
            d = df[df[f] == 1][["sample_id", "feature"]].copy()
            d["feature"] += "_" + f.upper()
            parts.append(d)
    d = pd.concat(parts).drop_duplicates()
    d["value"] = 1
    return d


def main():
    out = outdir("02_cnv")
    files = sorted(str(p) for p in (RAW / "cnv").glob("*.vcf.gz"))
    print(f"{len(files)} FACETS VCFs")
    with Pool(JOBS) as pool:
        res = [r for r in pool.map(process, files) if r is not None]
    gene_df = pd.concat([r[0] for r in res], ignore_index=True)
    arm_df = pd.concat([r[1] for r in res], ignore_index=True)

    both = pd.concat([gene_df[["sample_id", "feature", "value"]], arm_df[["sample_id", "feature", "value"]]])
    both = both.groupby(["sample_id", "feature"], as_index=False)["value"].max()
    freq = both.groupby("feature")["value"].sum()
    arms = {f for f in both.feature.unique() if ARM_RE.match(f)}
    seg_median = gene_df.groupby("feature")["segment_size"].median()
    recurrent = {f for f in freq.index
                 if f not in arms and f.replace("CNV_", "") in PANEL
                 and MIN_OCCURRENCE <= freq[f] <= MAX_OCCURRENCE
                 and seg_median.get(f, 0) <= MAX_MEDIAN_SEGMENT
                 and not any(p in f for p in ("OR", "LOC", "LINC"))}
    discovery = recurrent | arms
    panel = {f"CNV_{g}" for g in PANEL} | arms

    def wide(feats, df=both):
        return df[df.feature.isin(feats)].pivot_table(index="sample_id", columns="feature", values="value",
                                                      fill_value=0).astype(int)

    disc = wide(discovery)
    disc.to_csv(out / "cnv_discovery_matrix.csv")
    wide(panel).to_csv(out / "cnv_panel_matrix.csv")

    dire = directional(gene_df, arm_df)
    base = dire.feature.str.replace("_GAIN|_LOSS|_HOMDEL|_LOH", "", regex=True)
    dire[base.isin(discovery)].pivot_table(index="sample_id", columns="feature", values="value", fill_value=0) \
        .astype(int).reindex(disc.index, fill_value=0).to_csv(out / "cnv_directional_discovery_matrix.csv")
    dire[base.isin(panel)].pivot_table(index="sample_id", columns="feature", values="value", fill_value=0) \
        .astype(int).reindex(disc.index, fill_value=0).to_csv(out / "cnv_directional_panel_matrix.csv")
    gene_df.pivot_table(index="sample_id", columns="feature", values="TCN", aggfunc="median",
                        fill_value=2).to_csv(out / "gene_tcn_matrix.csv")
    arm_df.pivot_table(index="sample_id", columns="feature", values="CIN_score", aggfunc="median",
                       fill_value=0).to_csv(out / "arm_cin_matrix.csv")
    print(f"discovery CNV features: {disc.shape[1]} ({len(arms)} arms)")


if __name__ == "__main__":
    main()
