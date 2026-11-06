"""Step 24. TIDE scores for the Swedish tumors with tidepy (cancer type "Other")."""
import os
import shlex
import subprocess

import numpy as np
import pandas as pd

from gcps_utils import DATA, outdir

TIDEPY = shlex.split(os.environ.get("TIDEPY", "tidepy"))      # e.g. "/path/to/python /path/to/tidepy"
TPM = os.environ.get("GCPS_TPM", str(DATA / "CRC.SW.mRNA.symbol.TPM.txt"))

out = outdir("24_tide")
tpm = pd.read_csv(TPM, sep="\t", index_col=0)
tumors = tpm.loc[:, [c.endswith((".T", ".T2")) for c in tpm.columns]].clip(lower=0)
logged = np.log2(tumors + 1)
centered = logged.sub(logged.mean(axis=1), axis=0)          # each gene centered on its mean across the tumors
centered.to_csv(out / "tide_input.tsv", sep="\t")

result = subprocess.run([*TIDEPY, str(out / "tide_input.tsv"), "-o", str(out / "tide_output.csv"), "-c", "Other"],
                        capture_output=True, text=True)
(out / "tidepy.log").write_text(result.stdout + result.stderr)
print("tidepy exit code", result.returncode, "|", centered.shape[1], "tumors")
