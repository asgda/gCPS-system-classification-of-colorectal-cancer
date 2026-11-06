"""Step 06. FACETS purity, ploidy and whole-genome doubling; gCPS1/gCPS4 gain-loss contrast (Figure 4B)."""
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2_contingency, fisher_exact, kruskal

from gcps_utils import COLORS, JOBS, OUT, RAW, SUBTYPES, bh, cliffs_delta_ci, load_clinical, load_labels, outdir, \
    sample_id, save_fig, style_matplotlib

AUTOSOMES = {f"chr{i}" for i in range(1, 23)}
WGD_MAJOR_CN, WGD_FRACTION = 2, 0.5


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def facets_summary(path):
    """Length-weighted autosomal summaries from one FACETS VCF."""
    purity = ploidy = np.nan
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("##purity="):
                purity = num(line.strip().split("=")[1])
            elif line.startswith("##ploidy="):
                ploidy = num(line.strip().split("=")[1])
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8 or f[0] not in AUTOSOMES:
                continue
            info = dict(x.split("=", 1) for x in f[7].split(";") if "=" in x)
            try:
                start, end = int(f[1]), int(info["END"])
            except (KeyError, ValueError):
                continue
            if end > start:
                rows.append((end - start, num(info.get("TCN_EM")), num(info.get("LCN_EM"))))
    if not rows:
        return None
    length, tcn, lcn = (np.array(c, float) for c in zip(*rows))
    total = length.sum()
    has_tcn, has_both = np.isfinite(tcn), np.isfinite(tcn) & np.isfinite(lcn)
    frac = lambda m, ok=has_tcn: float(length[m & ok].sum() / total)
    major = tcn - lcn
    major_ge2 = frac(major >= WGD_MAJOR_CN, has_both)
    return dict(sid=sample_id(Path(path).name), facets_purity=purity, facets_ploidy=ploidy,
                loss_vs_diploid=frac(tcn < 2), gain_vs_diploid=frac(tcn > 2),
                loss_vs_ploidy=frac(tcn < ploidy - 0.5) if np.isfinite(ploidy) else np.nan,
                gain_vs_ploidy=frac(tcn > ploidy + 0.5) if np.isfinite(ploidy) else np.nan,
                loh=frac(lcn == 0, has_both), frac_major_ge2=major_ge2, wgd=int(major_ge2 >= WGD_FRACTION))


def main():
    out = outdir("06_ploidy_wgd")
    labels = load_labels()
    files = sorted((RAW / "cnv").glob("*.vcf.gz"))
    rows = Parallel(n_jobs=JOBS, prefer="processes")(delayed(facets_summary)(f) for f in files)
    cn = pd.DataFrame([r for r in rows if r]).drop_duplicates("sid").set_index("sid")
    clin = load_clinical()
    df = pd.DataFrame({"gCPS": labels}).join(cn).join(clin[["purity_path"]])
    df.index.name = "sample_id"
    df.to_csv(out / "per_sample_facets_summary.csv")

    # WGD prevalence and ploidy by subtype
    wgd = df.groupby("gCPS")["wgd"].agg(["sum", "count", "mean"]).reindex(SUBTYPES)
    chi_p = chi2_contingency(pd.crosstab(df.gCPS, df.wgd))[1]
    pair = pd.crosstab(df[df.gCPS.isin(["gCPS1", "gCPS4"])].gCPS, df.wgd)
    odds, fisher_p = fisher_exact(pair.values)
    wgd.assign(chi2_P=chi_p, gCPS1_vs_gCPS4_OR=odds, gCPS1_vs_gCPS4_P=fisher_p).to_csv(out / "wgd_by_subtype.csv")
    print((wgd["mean"] * 100).round(1).to_dict())

    # gCPS4 vs gCPS1 contrast against the diploid and the tumor's own ploidy, with and without WGD
    res = []
    for stratum, d in (("all", df), ("WGD-negative", df[df.wgd == 0]), ("WGD-positive", df[df.wgd == 1])):
        a, b = d[d.gCPS == "gCPS4"], d[d.gCPS == "gCPS1"]
        if len(a) < 5 or len(b) < 5:
            continue
        for m in ("loss_vs_diploid", "gain_vs_diploid", "loss_vs_ploidy", "gain_vs_ploidy", "loh"):
            delta, lo, hi = cliffs_delta_ci(a[m], b[m])
            res.append(dict(stratum=stratum, metric=m, n_gCPS4=a[m].notna().sum(), n_gCPS1=b[m].notna().sum(),
                            cliffs_delta_gCPS4_vs_gCPS1=delta, ci_low=lo, ci_high=hi))
    pd.DataFrame(res).to_csv(out / "gCPS4_vs_gCPS1_directional_contrast.csv", index=False)

    purity = [kruskal(*[g.dropna() for _, g in df.groupby("gCPS")[c]])[1] for c in ("facets_purity", "purity_path")]
    pd.DataFrame(dict(measure=["facets_purity", "pathology_purity"], kruskal_P=purity)).to_csv(
        out / "purity_by_subtype.csv", index=False)

    # Figure 4B: ploidy, arm-level gain/loss and the most subtype-specific arms
    arms = pd.read_csv(OUT / "02_cnv" / "cnv_directional_panel_matrix.csv", index_col=0)
    arms.index = [sample_id(i) for i in arms.index]
    arm_cols = [c for c in arms.columns if re.fullmatch(r"CNV_(\d{1,2}|X)[pq]_(GAIN|LOSS)", c)]
    gain = arms[[c for c in arm_cols if c.endswith("GAIN")]].reindex(labels.index).fillna(0)
    loss = arms[[c for c in arm_cols if c.endswith("LOSS")]].reindex(labels.index).fillna(0)
    arm_summary = pd.DataFrame({"gain": gain.mean(1).groupby(labels).mean(), "loss": loss.mean(1).groupby(labels).mean()})
    arm_summary.reindex(SUBTYPES).to_csv(out / "arm_gain_loss_by_subtype.csv")

    plt = style_matplotlib()
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    data = [df.loc[df.gCPS == s, "facets_ploidy"].dropna() for s in SUBTYPES]
    parts = ax[0].violinplot(data, showmedians=True)
    for body, s in zip(parts["bodies"], SUBTYPES):
        body.set_facecolor(COLORS[s]); body.set_alpha(0.8)
    ax[0].set_xticks(range(1, 5)); ax[0].set_xticklabels(SUBTYPES); ax[0].set_ylabel("FACETS ploidy")
    for i, s in enumerate(SUBTYPES, 1):
        ax[0].text(i, ax[0].get_ylim()[1] * 0.97, f"WGD {wgd.loc[s, 'mean']:.1%}", ha="center", fontsize=9)
    arm_summary.reindex(SUBTYPES).plot.bar(ax=ax[1], color=["#444444", "#000000"], width=0.7)
    ax[1].set_ylabel("Mean fraction of arms altered"); ax[1].set_xticklabels(SUBTYPES, rotation=0)

    top_g = (gain[labels == "gCPS1"].mean() - gain[labels == "gCPS4"].mean()).nlargest(10)
    top_l = (loss[labels == "gCPS4"].mean() - loss[labels == "gCPS1"].mean()).nlargest(10)
    names = [c.replace("CNV_", "").replace("_GAIN", "") for c in top_g.index] + \
            [c.replace("CNV_", "").replace("_LOSS", "") for c in top_l.index]
    vals = list(top_g.values) + list(-top_l.values)
    cols = [COLORS["gCPS1"]] * len(top_g) + [COLORS["gCPS4"]] * len(top_l)
    ax[2].barh(range(len(vals)), vals, color=cols); ax[2].set_yticks(range(len(vals))); ax[2].set_yticklabels(names)
    ax[2].axvline(0, color="black", lw=0.8)
    ax[2].set_xlabel("Prevalence difference (gCPS1 gain right, gCPS4 loss left)")
    save_fig(fig, out / "fig4B_wgd_ploidy")
    plt.close(fig)


if __name__ == "__main__":
    main()
