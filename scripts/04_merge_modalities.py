"""Step 04. Merge the three discovery matrices into the 371-feature binary matrix."""
import pandas as pd

from gcps_utils import OUT, outdir

PARTS = {"SNV": "01_snv/snv_discovery_matrix.csv",
         "CNV": "02_cnv/cnv_discovery_matrix.csv",
         "SV": "03_sv/sv_discovery_matrix.csv"}


def main():
    mats = {}
    for tag, rel in PARTS.items():
        m = pd.read_csv(OUT / rel, index_col=0)
        m.columns = [f"{tag}_{c.replace(tag + '_', '', 1)}" for c in m.columns]
        mats[tag] = m
    shared = sorted(set.intersection(*(set(m.index) for m in mats.values())))
    merged = pd.concat([m.loc[shared] for m in mats.values()], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged.to_csv(outdir("04_features") / "unified_discovery_matrix.csv")
    print(f"{merged.shape[0]} samples x {merged.shape[1]} features "
          f"({', '.join(f'{t} {m.shape[1]}' for t, m in mats.items())})")


if __name__ == "__main__":
    main()
