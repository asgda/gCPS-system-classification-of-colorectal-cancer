"""Step 03. Structural-variant features from BRASS calls within the CRC gene panel."""
import glob
import gzip
import re
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

from gcps_utils import JOBS, RAW, outdir, panel_genes

N_COHORT = 1063
MIN_OCCURRENCE = int(N_COHORT * 0.01)
MAX_OCCURRENCE = int(N_COHORT * 0.98)
PANEL = panel_genes()
PASSENGERS = {"TTN", "MUC16", "MUC4", "MUC17", "MUC5B", "MUC6", "OBSCN", "DNAH5", "DNAH11", "DNAH2",
              "RYR1", "RYR2", "RYR3", "CSMD1", "CSMD3", "LRP1B", "PCLO", "XIRP2", "FAT3", "FAT4"}
FRAGILE_SITES = {"FHIT", "WWOX", "PARK2", "IMMP2L", "DMD", "GRID2", "MACROD2"}


def parse_sample(path):
    sid = re.search(r"CRC-SW-((?:U|UM)\d+)-T", path.split("/")[-1]).group(1)
    genes, mech, class_counts = set(), defaultdict(set), defaultdict(int)
    fragile, seen_class = set(), {}
    rearranged = line1 = n_records = n_events = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8 or f[6] not in ("PASS", "."):
                continue
            info = dict(x.split("=", 1) for x in f[7].split(";") if "=" in x)
            n_records += 1
            svtype = info.get("SVTYPE", "UNK")
            rearranged |= svtype in ("BND", "INV", "TRA")
            line1 |= "LINE1" in f[7] or "L1" in f[7]
            svclass = info.get("SVCLASS", svtype)

            # reciprocal breakends of one junction are counted once
            mates = [m.strip() for m in info.get("MATEID", "").split(",") if m.strip() not in ("", ".")]
            rid = f[2].strip()
            if rid not in ("", ".") and mates:
                key = ("pair",) + tuple(sorted({rid, *mates}))
            elif rid not in ("", "."):
                key = ("id", rid)
            else:
                key = ("coord", f[0], f[1], f[4])
            if key not in seen_class:
                seen_class[key] = svclass
                class_counts[svclass] += 1
                n_events += 1
            elif seen_class[key] != svclass:
                raise ValueError(f"conflicting SVCLASS in mate pair {key} ({path})")

            for gene in (g.strip().upper() for g in info.get("SID", "").split(",") if g.strip()):
                if not re.match(r"^[A-Z0-9\-\._]+$", gene) or gene not in PANEL or gene in PASSENGERS:
                    continue
                if gene in FRAGILE_SITES:
                    fragile.add(gene)
                    continue
                genes.add(gene)
                mech[f"SV_{gene}_{str(svclass).upper().replace('-', '_')}"].add(sid)
    return dict(sid=sid, genes=genes, mech=mech, classes=class_counts, fragile=fragile,
                rearranged=int(rearranged), line1=int(line1), events=n_events)


def main():
    out = outdir("03_sv")
    files = sorted(glob.glob(f"{RAW}/sv/*.vcf.gz"))
    print(f"{len(files)} BRASS VCFs")
    with Pool(JOBS) as pool:
        res = pool.map(parse_sample, files)

    hits = {r["sid"]: r["genes"] for r in res}
    freq = defaultdict(int)
    mech_all = defaultdict(set)
    for r in res:
        for g in r["genes"]:
            freq[g] += 1
        for k, v in r["mech"].items():
            mech_all[k] |= v
    samples = sorted(hits)
    keep = sorted(g for g, c in freq.items() if g in PANEL and MIN_OCCURRENCE <= c <= MAX_OCCURRENCE)

    def binary(genes):
        m = pd.DataFrame(0, index=samples, columns=[f"SV_{g}" for g in genes], dtype=np.int8)
        for s in samples:
            for g in hits[s] & set(genes):
                m.loc[s, f"SV_{g}"] = 1
        return m

    binary(keep).to_csv(out / "sv_discovery_matrix.csv")
    binary(sorted(g for g in freq if g in PANEL)).to_csv(out / "sv_panel_matrix.csv")
    mech = pd.DataFrame(0, index=samples, columns=sorted(mech_all), dtype=np.int8)
    for k, v in mech_all.items():
        mech.loc[list(v), k] = 1
    mech.to_csv(out / "sv_mechanism_matrix.csv")
    pd.DataFrame({"sample_id": [r["sid"] for r in res], "SV_TOTAL": [r["events"] for r in res]}).to_csv(
        out / "sv_burden.csv", index=False)
    arch = []
    for r in res:
        row = {"sample_id": r["sid"], "SV_chromothripsis": r["rearranged"], "SV_LINE1": r["line1"],
               "SV_CFS_burden": len(r["fragile"])}
        row.update({f"SVCLASS_{k}": v for k, v in r["classes"].items()})
        arch.append(row)
    pd.DataFrame(arch).fillna(0).to_csv(out / "sv_architecture_matrix.csv", index=False)
    print(f"discovery SV features: {len(keep)}")


if __name__ == "__main__":
    main()
