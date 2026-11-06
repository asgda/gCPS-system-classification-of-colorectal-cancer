"""Step 19. Genomic-scar score (LOH, TAI, LST) from FACETS segments (Supplementary Figure S5A, Table S9)."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kruskal

from gcps_utils import JOBS, OUT, RAW, REF, SUBTYPES, bh, cliffs_delta_ci, load_labels, mwu_p, outdir, sample_id

AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
MIN_LOH, MIN_LST, NOISE = 15e6, 10e6, 3e6
TELOMERE_WINDOW = 1e6


def chromosome_geometry():
    """Chromosome length and centromere midpoint from the hg38 cytoband (acen bands)."""
    bands = pd.read_csv(REF / "cytoBand.txt.gz", sep="\t", header=None, names=["chrom", "start", "end", "name", "stain"])
    geometry = {}
    for chrom, b in bands[bands.chrom.isin(AUTOSOMES)].groupby("chrom"):
        acen = b[b.stain == "acen"]
        geometry[chrom] = dict(length=int(b.end.max()), centromere=(acen.start.min() + acen.end.max()) / 2)
    return geometry


def read_segments(path):
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[0] not in AUTOSOMES:
                continue
            info = dict(x.split("=", 1) for x in f[7].split(";") if "=" in x)
            try:
                start, end = int(f[1]), int(info["END"])
                tcn, lcn = float(info["TCN_EM"]), float(info["LCN_EM"])
            except (KeyError, ValueError):
                continue
            if end > start and np.isfinite(tcn) and np.isfinite(lcn):
                rows.append((f[0], start, end, tcn, lcn))
    return sample_id(Path(path).name), pd.DataFrame(rows, columns=["chrom", "start", "end", "tcn", "lcn"])


def score(sid, seg, geometry):
    """LOH (Abkevich), TAI (Birkbak) and LST (Popova) counts for one tumor."""
    if seg.empty:
        return dict(sample_id=sid, LOH=np.nan, TAI=np.nan, LST=np.nan)
    loh = tai = lst = 0
    for chrom, g in geometry.items():
        s = seg[seg.chrom == chrom].sort_values("start")
        size = s.end - s.start
        loh += int(((s.lcn == 0) & (s.tcn > 0) & (size > MIN_LOH) & (size < g["length"] - 3e6)).sum())
        p_arm, q_arm = s[s.end <= g["centromere"]], s[s.start >= g["centromere"]]
        for arm, at_telomere in ((p_arm, lambda a: a.start.iloc[0] <= TELOMERE_WINDOW),
                                 (q_arm.sort_values("end"), lambda a: a.end.iloc[-1] >= g["length"] - TELOMERE_WINDOW)):
            imbalanced = arm[arm.tcn != 2 * arm.lcn]
            if len(imbalanced) and at_telomere(imbalanced):
                tai += 1
        for arm in (p_arm, q_arm):
            arm = arm[(arm.end - arm.start) >= NOISE]
            if len(arm) < 2:
                continue
            length = (arm.end - arm.start).values
            state = list(zip(arm.tcn, arm.lcn))
            lst += sum(length[i] >= MIN_LST and length[i + 1] >= MIN_LST and state[i] != state[i + 1]
                       for i in range(len(arm) - 1))
    return dict(sample_id=sid, LOH=loh, TAI=tai, LST=lst)


def main():
    out = outdir("19_hrd_scar")
    geometry = chromosome_geometry()
    files = sorted((RAW / "cnv").glob("*.vcf.gz"))
    parsed = Parallel(n_jobs=JOBS, prefer="processes")(delayed(read_segments)(f) for f in files)
    scores = pd.DataFrame(Parallel(n_jobs=JOBS, prefer="processes")(
        delayed(score)(sid, seg, geometry) for sid, seg in parsed)).dropna()
    scores["HRD_score"] = scores.LOH + scores.TAI + scores.LST
    labels = load_labels()
    wgd = pd.read_csv(OUT / "06_ploidy_wgd" / "per_sample_facets_summary.csv", index_col=0)["wgd"]
    df = scores.set_index("sample_id").join(labels.rename("gCPS"), how="inner").join(wgd)
    df.to_csv(out / "scar_scores_per_tumor.csv")

    parts = ["HRD_score", "LOH", "TAI", "LST"]
    medians = df.groupby("gCPS")[parts].median().reindex(SUBTYPES).T
    medians["kruskal_P"] = [kruskal(*[df.loc[df.gCPS == s, c] for s in SUBTYPES])[1] for c in parts]
    medians["P_adj"] = bh(medians.kruskal_P)
    medians.to_csv(out / "scar_by_subtype.csv")

    rows = []
    for stratum, d in (("all tumors", df), ("WGD-negative", df[df.wgd == 0])):
        for s in SUBTYPES:
            for c in parts:
                a, b = d.loc[d.gCPS == s, c], d.loc[d.gCPS != s, c]
                delta, lo, hi = cliffs_delta_ci(a, b)
                rows.append(dict(stratum=stratum, subtype=s, component=c, median=a.median(), median_rest=b.median(),
                                 cliffs_delta=delta, ci_low=lo, ci_high=hi, P=mwu_p(a, b)))
        a, b = d.loc[d.gCPS == "gCPS4", "HRD_score"], d.loc[d.gCPS == "gCPS1", "HRD_score"]
        delta, lo, hi = cliffs_delta_ci(a, b)
        rows.append(dict(stratum=stratum, subtype="gCPS4", component="HRD_score vs gCPS1", median=a.median(),
                         median_rest=b.median(), cliffs_delta=delta, ci_low=lo, ci_high=hi, P=mwu_p(a, b)))
    effects = pd.DataFrame(rows)
    effects["P_adj"] = effects.groupby(["stratum", "subtype"])["P"].transform(bh)
    effects.to_csv(out / "scar_effect_sizes.csv", index=False)
    print(medians.round(3).to_string())


if __name__ == "__main__":
    main()
