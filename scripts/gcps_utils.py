"""Shared paths, labels and small statistics helpers used by all scripts."""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("GCPS_ROOT", Path(__file__).resolve().parents[1]))
RAW = Path(os.environ.get("GCPS_RAW", ROOT / "data" / "raw"))
REF = Path(os.environ.get("GCPS_REF", ROOT / "data" / "reference"))
DATA = Path(os.environ.get("GCPS_DATA", ROOT / "data"))
CLINICAL = Path(os.environ.get("GCPS_CLINICAL", DATA / "clinical_data.tsv"))
OUT = Path(os.environ.get("GCPS_OUT", ROOT / "results"))
JOBS = int(os.environ.get("GCPS_JOBS", 8))
SEED = int(os.environ.get("GCPS_SEED", 42))
PANEL = ROOT / "resources" / "crc_panel_genes.txt"

SUBTYPES = ["gCPS1", "gCPS2", "gCPS3", "gCPS4"]
COLORS = {"gCPS1": "#E69F00", "gCPS2": "#56B4E9", "gCPS3": "#009E73", "gCPS4": "#D55E00"}
MODALITY_COLORS = {"SNV": "#E69F00", "CNV": "#0072B2", "SV": "#009E73"}


def outdir(*parts):
    p = OUT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sample_id(x):
    m = re.search(r"((?:U|UM)\d+)", str(x))
    return m.group(1) if m else None


def panel_genes():
    g = pd.read_csv(PANEL, header=None)[0].astype(str).str.strip().str.upper()
    return set(g) - {"", "GENE", "NAN"}


def load_labels():
    """gCPS label per sample from step 05."""
    d = pd.read_csv(OUT / "05_subtypes" / "gcps_labels.csv")
    return d.set_index("sample_id")["gCPS"]


def load_matrix():
    return pd.read_csv(OUT / "04_features" / "unified_discovery_matrix.csv", index_col=0)


def load_clinical():
    """Cohort table from the Nunes et al. supplementary data, indexed by sample ID."""
    s = pd.read_excel(DATA / "Supplementary_Table_01.xlsx")
    s["sid"] = s["DNA Tumor Sample Barcode"].map(sample_id)
    d = pd.DataFrame({"sid": s["sid"]})
    d["age"] = pd.to_numeric(s["Age at diagnosis"], errors="coerce")
    d["sex_m"] = s["Sex"].astype(str).str.lower().str[0].map({"m": 1.0, "f": 0.0})
    d["msi"] = s["MSI Status"].astype(str).str.upper().str.contains("MSI").astype(float)
    d["pretreated"] = s["Pre-Treated"].astype(str).str.lower().isin(["treated", "yes", "1", "true"]).astype(float)
    d["stage"] = s["Tumour Stage"].astype(str).str.extract(r"Stage\s*(IV|III|II|I)")[0].map(
        {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0})
    d["site"] = s["Tumour Site"].astype(str)
    d["grade"] = s["Tumour Grade"].astype(str)
    d["purity_path"] = pd.to_numeric(s["Tumour Cell Content Pathology"], errors="coerce")
    d["OS_months"] = pd.to_numeric(s["Overall survival days"], errors="coerce") / 30.44
    d["OS_event"] = s["Vital Status"].astype(str).str.lower().map({"dead": 1.0, "alive": 0.0})
    d["RFS_months"] = pd.to_numeric(s["Recurrence free survival days"], errors="coerce") / 30.44
    d["RFS_event"] = s["Recurrence"].astype(str).map({"Yes": 1.0, "No": 0.0})
    return d.dropna(subset=["sid"]).drop_duplicates("sid").set_index("sid")


def bh(p):
    """Benjamini-Hochberg adjusted P values."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    order = np.argsort(q)
    ranked = q[order] * len(q) / (np.arange(len(q)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(q)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def cliffs_delta(a, b):
    """Cliff's delta from the Mann-Whitney U statistic; positive when a tends to exceed b."""
    from scipy.stats import rankdata
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u = rankdata(np.concatenate([a, b]))[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return 2 * u / (len(a) * len(b)) - 1


def cliffs_delta_ci(a, b, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    d = [cliffs_delta(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)]
    return cliffs_delta(a, b), np.percentile(d, 2.5), np.percentile(d, 97.5)


def mwu_p(a, b):
    from scipy.stats import mannwhitneyu
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    try:
        return mannwhitneyu(a, b, alternative="two-sided")[1]
    except ValueError:
        return 1.0


def style_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 600,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def save_fig(fig, path):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight")
