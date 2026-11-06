"""Step 07. Baseline clinical and genomic characteristics by gCPS subtype (Supplementary Table S1b)."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, kruskal

from gcps_utils import DATA, OUT, SUBTYPES, load_labels, outdir, sample_id

TMB_DENOMINATOR = 3000.0
MISSING = {"nan", "none", "", "undefined", "not available", "na", "unknown"}

s = pd.read_excel(DATA / "Supplementary_Table_01.xlsx")
s["sid"] = s["Sample ID"].map(sample_id)
sv = pd.read_csv(OUT / "03_sv" / "sv_burden.csv")
sv["sid"] = sv.sample_id
labels = load_labels().rename_axis("sid").reset_index()
df = labels.merge(s.drop_duplicates("sid"), on="sid").merge(sv[["sid", "SV_TOTAL"]], on="sid", how="left")

df["TMB"] = pd.to_numeric(df["Total Mutation Count"], errors="coerce") / TMB_DENOMINATOR
df["Age"] = pd.to_numeric(df["Age at diagnosis"], errors="coerce")
df["CN_segments"] = pd.to_numeric(df["Copy Number Segments"], errors="coerce")
df["Purity"] = pd.to_numeric(df["Tumour Cell Content Pathology"], errors="coerce")
df["Hypoxia"] = pd.to_numeric(df["Tumour Hypoxia Buffa Score"], errors="coerce")
rows = []


def fmt_iqr(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return f"{x.median():.1f} [{x.quantile(.25):.1f}-{x.quantile(.75):.1f}]" if len(x) else "-"


def continuous(col, label):
    groups = [pd.to_numeric(df.loc[df.gCPS == g, col], errors="coerce").dropna() for g in SUBTYPES]
    rows.append({"characteristic": f"{label}, median [IQR]", "Overall": fmt_iqr(df[col]),
                 **{g: fmt_iqr(df.loc[df.gCPS == g, col]) for g in SUBTYPES},
                 "p_value": kruskal(*groups)[1], "test": "Kruskal-Wallis"})


def categorical(col, label, order=None):
    v = df[col].astype(str)
    missing = v.str.lower().isin(MISSING)
    levels = sorted(v[~missing].unique())
    if order:
        levels = [x for x in order if x in levels] + [x for x in levels if x not in order]
    ct = pd.crosstab(v[~missing], df.gCPS[~missing]).reindex(index=levels, columns=SUBTYPES, fill_value=0)
    chi, p, _, exp = chi2_contingency(ct.values)
    test = "chi-square"
    if ct.shape == (2, 2) and exp.min() < 5:
        p, test = fisher_exact(ct.values)[1], "Fisher exact"
    rows.append({"characteristic": f"{label}, n (%)", "Overall": "", **{g: "" for g in SUBTYPES},
                 "p_value": p, "test": test})
    shown = levels + (["Missing/Undefined"] if missing.any() else [])
    for lv in shown:
        m = missing if lv == "Missing/Undefined" else (v == lv) & ~missing
        cells = {g: f"{(m & (df.gCPS == g)).sum()} ({100 * (m & (df.gCPS == g)).sum() / (df.gCPS == g).sum():.1f})"
                 for g in SUBTYPES}
        rows.append({"characteristic": f"    {lv}", "Overall": f"{m.sum()} ({100 * m.mean():.1f})", **cells,
                     "p_value": np.nan, "test": ""})


continuous("Age", "Age at diagnosis (years)")
categorical("Sex", "Sex", ["Female", "Male"])
categorical("Tumour Stage", "AJCC stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
categorical("Tumour Site", "Primary site", ["Right Colon", "Left Colon", "Rectum"])
categorical("Tumour Grade", "Tumor grade")
categorical("Pre-Treated", "Neoadjuvant treatment", ["Untreated", "Treated"])
categorical("MSI Status", "MSI status", ["MSS", "MSI"])
categorical("CMS Tumour", "CMS")
categorical("iCMS Tumour", "iCMS")
categorical("CRPS Tumour", "CRPS")
continuous("TMB", "TMB (mut/Mb)")
continuous("SV_TOTAL", "Structural variants (n)")
continuous("CN_segments", "Copy-number segments (n)")
continuous("Purity", "Tumor purity (%)")
continuous("Hypoxia", "Hypoxia (Buffa score)")
categorical("Recurrence", "Recurrence", ["No", "Yes"])
categorical("Vital Status", "Vital status", ["Alive", "Dead"])

out = outdir("07_baseline")
pd.DataFrame(rows)[["characteristic", "Overall", *SUBTYPES, "p_value", "test"]].to_csv(
    out / "baseline_characteristics_by_subtype.csv", index=False)
print(f"{len(df)} tumors; subtype sizes {df.gCPS.value_counts().reindex(SUBTYPES).tolist()}")
