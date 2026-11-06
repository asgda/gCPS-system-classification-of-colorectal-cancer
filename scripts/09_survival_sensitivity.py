"""Step 09. gCPS4-vs-gCPS1 survival: confounder adjustment (Table 1) and stage II-III MSS tumors (Table S2)."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from gcps_utils import load_clinical, load_labels, outdir, OUT

out = outdir("09_survival_sensitivity")
clin = load_clinical().join(load_labels().rename("gCPS"), how="inner")
facets = pd.read_csv(OUT / "06_ploidy_wgd" / "per_sample_facets_summary.csv", index_col=0)
df = clin.join(facets[["facets_purity", "facets_ploidy", "wgd"]])


def fit(data, group, covariates, name, restrict=None):
    """Stage-stratified Cox model for one binary contrast; near-constant binary covariates are dropped."""
    d = data if restrict is None else data[restrict(data)]
    cols = ["OS_months", "OS_event", group, "stage"] + covariates
    d = d[cols].dropna()
    d = d[d.OS_months > 0]
    parts = [d[["OS_months", "OS_event", group, "stage"]]]
    for c in covariates:
        parts.append(pd.get_dummies(d[c], prefix=c, drop_first=True).astype(float) if c in ("site", "grade") else d[[c]])
    m = pd.concat(parts, axis=1)
    keep = {"OS_months", "OS_event", group, "stage"}
    dropped = [c for c in m.columns if c not in keep and (
        m[c].nunique() < 2 or (m[c].isin([0, 1]).all() and min(m[c].mean(), 1 - m[c].mean()) < 0.02))]
    m = m.drop(columns=dropped)
    for pen in (0.0, 0.1, 0.5):
        try:
            cph = CoxPHFitter(penalizer=pen).fit(m, duration_col="OS_months", event_col="OS_event", strata=["stage"])
            s = cph.summary.loc[group]
            return dict(model=name, n=len(m), events=int(m.OS_event.sum()), HR=np.exp(s["coef"]),
                        lower=np.exp(s["coef lower 95%"]), upper=np.exp(s["coef upper 95%"]), P=s["p"],
                        c_index=cph.concordance_index_, dropped="+".join(dropped) or "none", penalizer=pen)
        except Exception:
            continue
    return dict(model=name, n=len(m), status="did not converge")


# gCPS4 vs gCPS1 with candidate confounders added to an age and sex baseline (Table 1)
c = df[df.gCPS.isin(["gCPS1", "gCPS4"])].copy()
c["gCPS4"] = (c.gCPS == "gCPS4").astype(float)
base = ["age", "sex_m"]
models = [
    ("age + sex", []),
    ("+ FACETS purity", ["msi", "pretreated", "facets_purity"]),
    ("+ pathology purity", ["msi", "pretreated", "purity_path"]),
    ("+ ploidy", ["msi", "pretreated", "facets_ploidy"]),
    ("+ WGD status", ["msi", "pretreated", "wgd"]),
    ("+ purity + ploidy + WGD", ["msi", "pretreated", "facets_purity", "facets_ploidy", "wgd"]),
    ("+ primary site", ["site"]),
    ("+ grade", ["grade"]),
    ("+ site + grade", ["site", "grade"]),
    ("+ MSI + pretreatment", ["msi", "pretreated"]),
    ("+ MSI + pretreatment + site + grade", ["msi", "pretreated", "site", "grade"]),
]
rows = [fit(c, "gCPS4", base + extra, name) for name, extra in models]
rows.append(fit(c, "gCPS4", base + ["msi", "pretreated"], "primary covariates, WGD-negative tumors", lambda d: d.wgd == 0))
rows.append(fit(c, "gCPS4", base + ["msi", "pretreated"], "primary covariates, WGD-positive tumors", lambda d: d.wgd == 1))
rows.append(fit(c, "gCPS4", base + ["msi", "pretreated"], "primary covariates, FACETS purity >= 0.4",
                lambda d: d.facets_purity >= 0.4))
pd.DataFrame(rows).to_csv(out / "gCPS4_vs_gCPS1_confounder_models.csv", index=False)

# Stage II-III microsatellite-stable tumors (MSI term is constant and therefore dropped)
mss = df[df.stage.isin([2, 3]) & (df.msi == 0)]
pair = mss[mss.gCPS.isin(["gCPS1", "gCPS4"])].assign(gCPS4=lambda d: (d.gCPS == "gCPS4").astype(float))
pooled = mss[mss.gCPS.isin(["gCPS1", "gCPS2", "gCPS4"])].assign(gCPS4=lambda d: (d.gCPS == "gCPS4").astype(float))
cov = ["age", "sex_m", "pretreated"]
pd.DataFrame([fit(pair, "gCPS4", cov, "gCPS4 vs gCPS1, stage II-III MSS"),
              fit(pooled, "gCPS4", cov, "gCPS4 vs pooled gCPS1 + gCPS2, stage II-III MSS")]).to_csv(
    out / "gCPS4_stage2_3_MSS.csv", index=False)
print(pd.DataFrame(rows)[["model", "n", "events", "HR", "lower", "upper", "P"]].round(3).to_string(index=False))
