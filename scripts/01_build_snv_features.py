"""Step 01. Short-variant features from VEP-annotated TNscope/TNhaplotyper calls."""
import glob
import gzip
import re
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

from gcps_utils import JOBS, RAW, outdir, panel_genes, sample_id

MIN_TVAF, MIN_TDP, MAX_NVAF = 0.05, 20, 0.02
MIN_FREQ, MAX_FREQ = 0.02, 0.98
N_COHORT = 1063

PANEL = panel_genes()
PASSENGERS = {"TTN", "MUC16", "MUC4", "MUC17", "MUC5B", "MUC6", "OBSCN", "FLG", "DNAH5", "DNAH11",
              "DNAH2", "RYR1", "RYR2", "RYR3", "CSMD1", "CSMD3", "LRP1B", "PCLO", "XIRP2", "FAT4", "FAT3"}
FUNCTIONAL = {"missense_variant", "frameshift_variant", "stop_gained", "splice_acceptor_variant",
              "splice_donor_variant", "protein_altering_variant", "inframe_insertion",
              "inframe_deletion", "start_lost", "stop_lost"}
LOF = {"frameshift_variant", "stop_gained", "splice_acceptor_variant", "splice_donor_variant", "start_lost"}


def parse_vcf(path):
    sid = re.search(r"CRC-SW-((?:U|UM)\d+)-T", path).group(1)
    genes, mech, n_mut, csq_fields = set(), defaultdict(set), 0, None
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("##INFO=<ID=CSQ"):
                m = re.search(r'Format: ([^"]+)"', line)
                csq_fields = m.group(1).strip().split("|") if m else None
                continue
            if line.startswith("#"):
                continue
            f = line.rstrip().split("\t")
            if len(f) < 8 or f[6] != "PASS":
                continue
            info = dict(x.split("=", 1) for x in f[7].split(";") if "=" in x)
            try:
                tvaf, tdp, nvaf = float(info.get("TVAF", 0)), int(info.get("TDP", 0)), float(info.get("NVAF", 0))
            except ValueError:
                continue
            if tvaf < MIN_TVAF or tdp < MIN_TDP or nvaf > MAX_NVAF or "CSQ" not in info or not csq_fields:
                continue
            for entry in info["CSQ"].split(","):
                vals = entry.split("|")
                if len(vals) < len(csq_fields):
                    continue
                d = dict(zip(csq_fields, vals))
                gene = d.get("SYMBOL", "").strip().upper()
                terms = set(d.get("Consequence", "").split("&"))
                if d.get("CANONICAL") != "YES" or not gene or not terms & FUNCTIONAL:
                    continue
                genes.add(gene)
                n_mut += 1
                mech[f"SNV_{gene}_{'LOF' if terms & LOF else 'MISSENSE'}"].add(sid)
                break
    return sid, genes, mech, n_mut


def main():
    out = outdir("01_snv")
    files = sorted(glob.glob(f"{RAW}/vep_output/*.vep.vcf.gz"))
    print(f"{len(files)} VEP files")
    with Pool(JOBS) as pool:
        res = pool.map(parse_vcf, files)

    hits, burden, freq, mech_all = {}, {}, defaultdict(int), defaultdict(set)
    for sid, genes, mech, n in res:
        genes = {g for g in genes if g not in PASSENGERS}
        hits[sid], burden[sid] = genes, n
        for g in genes:
            freq[g] += 1
        for k, v in mech.items():
            mech_all[k] |= v

    samples = sorted(hits)
    keep = sorted(g for g, c in freq.items() if g in PANEL and int(N_COHORT * MIN_FREQ) <= c <= int(N_COHORT * MAX_FREQ))
    panel_present = sorted(g for g in freq if g in PANEL)

    def binary(genes):
        m = pd.DataFrame(0, index=samples, columns=[f"SNV_{g}" for g in genes], dtype=np.int8)
        for s in samples:
            for g in hits[s] & set(genes):
                m.loc[s, f"SNV_{g}"] = 1
        return m

    binary(keep).to_csv(out / "snv_discovery_matrix.csv")
    binary(panel_present).to_csv(out / "snv_panel_matrix.csv")
    mech = pd.DataFrame(0, index=samples, columns=sorted(mech_all), dtype=np.int8)
    for k, v in mech_all.items():
        mech.loc[list(v), k] = 1
    mech.to_csv(out / "snv_mechanism_matrix.csv")
    pd.DataFrame({"sample_id": samples, "SNV_TOTAL_BURDEN": [burden[s] for s in samples]}).to_csv(
        out / "snv_burden.csv", index=False)
    print(f"discovery SNV features: {len(keep)}")


if __name__ == "__main__":
    main()
