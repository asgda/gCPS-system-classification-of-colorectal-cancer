"""Step 16. Breakpoint and copy-number tables for ShatterSeek from the BRASS and FACETS VCFs."""
import glob
import gzip
import re
from multiprocessing import Pool

import pandas as pd

from gcps_utils import JOBS, RAW, outdir, sample_id

VALID = {str(i) for i in range(1, 23)} | {"X"}
POS_TOLERANCE = 25
BND = {  # strand of the local breakend from the VCF ALT notation
    re.compile(r"^[^\[\]]+\[([^:\[\]]+):(\d+)\[$"): "+",
    re.compile(r"^[^\[\]]+\]([^:\[\]]+):(\d+)\]$"): "+",
    re.compile(r"^\]([^:\[\]]+):(\d+)\][^\[\]]+$"): "-",
    re.compile(r"^\[([^:\[\]]+):(\d+)\[[^\[\]]+$"): "-",
}
SVTYPE = {"deletion": "DEL", "tandem-duplication": "DUP", "translocation": "TRA"}


def info_dict(text):
    return {k: v for k, _, v in (x.partition("=") for x in text.split(";") if x)}


def parse_bnd(alt):
    for pattern, strand in BND.items():
        m = pattern.match(alt)
        if m:
            return strand, m.group(1), int(m.group(2))
    return None


def read_sv(path):
    sid = sample_id(path.split("/")[-1])
    records = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            info = info_dict(f[7])
            if info.get("SVTYPE") == "BND":
                records[f[2]] = dict(chrom=f[0], pos=int(f[1]), alt=f[4], mate=info.get("MATEID"),
                                     cls=(info.get("SVCLASS") or "").lower())
    rows, seen = [], set()
    for vid, r in records.items():
        if vid in seen or r["mate"] not in records:
            continue
        m = records[r["mate"]]
        seen |= {vid, r["mate"]}
        a, b = parse_bnd(r["alt"]), parse_bnd(m["alt"])
        if a is None or b is None:
            continue
        if a[1] != m["chrom"] or abs(a[2] - m["pos"]) > POS_TOLERANCE or b[1] != r["chrom"] or \
                abs(b[2] - r["pos"]) > POS_TOLERANCE:
            continue
        c1, p1, s1, c2, p2, s2 = r["chrom"], r["pos"], a[0], m["chrom"], m["pos"], b[0]
        if p1 > p2:                                   # ShatterSeek expects pos1 <= pos2, strands swapped with them
            c1, p1, s1, c2, p2, s2 = c2, p2, s2, c1, p1, s1
        cls = r["cls"] or m["cls"]
        if cls in SVTYPE:
            kind = SVTYPE[cls]
        elif cls == "inversion" and (s1, s2) == ("+", "+"):
            kind = "h2hINV"
        elif cls == "inversion" and (s1, s2) == ("-", "-"):
            kind = "t2tINV"
        else:
            continue
        c1, c2 = c1.replace("chr", ""), c2.replace("chr", "")
        if c1 in VALID and c2 in VALID:
            rows.append((sid, c1, p1, c2, p2, s1, s2, kind))
    return rows


def read_cnv(path):
    sid = sample_id(path.split("/")[-1])
    segs = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            info = info_dict(f[7])
            try:
                chrom = f[0].replace("chr", "")
                if chrom in VALID:
                    segs.append((chrom, int(f[1]), int(info["END"]), int(round(float(info["TCN_EM"])))))
            except (KeyError, ValueError):
                continue
    merged = []
    order = {c: i for i, c in enumerate([str(i) for i in range(1, 23)] + ["X"])}
    for chrom in sorted({s[0] for s in segs}, key=order.get):
        cur = None
        for c, start, end, cn in sorted(s for s in segs if s[0] == chrom):
            if cur and cn == cur[3] and start <= cur[2] + 1:      # adjacent segments with equal copy number are merged
                cur = (c, cur[1], max(cur[2], end), cn)
            else:
                if cur:
                    merged.append((sid,) + cur)
                cur = (c, start, end, cn)
        merged.append((sid,) + cur)
    return merged


def main():
    out = outdir("16_chromothripsis")
    with Pool(JOBS) as pool:
        sv = [r for rows in pool.map(read_sv, sorted(glob.glob(f"{RAW}/sv/*.SV.vcf.gz"))) for r in rows]
        cnv = [r for rows in pool.map(read_cnv, sorted(glob.glob(f"{RAW}/cnv/*.vcf.gz"))) for r in rows]
    pd.DataFrame(sv, columns=["sample_id", "chrom1", "pos1", "chrom2", "pos2", "strand1", "strand2", "SVtype"]).to_csv(
        out / "shatterseek_sv_input.csv", index=False)
    pd.DataFrame(cnv, columns=["sample_id", "chrom", "start", "end", "total_cn"]).to_csv(
        out / "shatterseek_cnv_input.csv", index=False)
    print(f"{len(sv)} junctions, {len(cnv)} copy-number segments")


if __name__ == "__main__":
    main()
