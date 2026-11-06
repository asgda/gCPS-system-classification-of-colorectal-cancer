"""Step 12. Mutational-process (etiology) composition by gCPS subtype (Figures 4C and 4D)."""
import re

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from gcps_utils import DATA, SEED, SUBTYPES, bh, load_labels, outdir, sample_id, save_fig, style_matplotlib

N_BOOT = 2000
QC = ["Total Mutations", "Cosine Similarity", "L1 Norm", "L1 Norm %", "L2 Norm", "L2 Norm %", "KL Divergence",
      "Correlation"]

ETIOLOGY = {
    "Clock-like (age-related)": ["SBS1", "SBS5"],
    "APOBEC": ["SBS2", "SBS13"],
    "HRD (homologous recombination deficiency)": ["SBS3", "ID6"],
    "UV exposure": ["SBS7b", "SBS7c"],
    "POLE proofreading deficiency": ["SBS10a", "SBS10b", "SBS28"],
    "MMR deficiency / MSI": ["SBS6", "SBS14", "SBS15", "SBS20", "SBS21", "SBS26", "SBS44", "DBS7", "ID1", "ID2", "ID7"],
    "Oxidative damage / BER deficiency": ["SBS18", "SBS30"],
    "Treatment-associated": ["SBS17a", "SBS17b", "SBS32", "SBS86", "SBS90", "DBS5"],
    "Colibactin (pks+ E. coli)": ["SBS88", "ID18"],
    "DSB repair / NHEJ": ["ID8", "DBS8"],
    "Occupational exposure (haloalkanes)": ["SBS42"],
}
CATEGORY = {sig: cat for cat, sigs in ETIOLOGY.items() for sig in sigs}
DE_NOVO = re.compile(r"^(SBS96|DBS78|ID83)[A-Z]+$")


def read_blocks(path):
    """The workbook holds three side-by-side blocks (SBS, DBS, ID), each starting at a 'Total Mutations' column."""
    raw = pd.read_excel(path, header=None)
    names = [str(v) if pd.notna(v) else None for v in raw.iloc[2]]
    data = raw.iloc[3:].reset_index(drop=True)
    ids = data.iloc[:, 0].astype(str).map(sample_id)
    starts = [i for i, n in enumerate(names) if n == "Total Mutations"]
    assert len(starts) == 3, "unexpected workbook layout"
    blocks = {}
    for tag, s, e in zip(("SBS", "DBS", "ID"), starts, starts[1:] + [len(names)]):
        sub = data.iloc[:, s:e].apply(pd.to_numeric, errors="coerce")
        sub.columns = [f"{tag}_{n}" if n in QC else n for n in names[s:e]]
        sub.index = ids
        sub = sub[sub.index.notna() & ~sub.index.duplicated()]
        blocks[tag] = sub
    return blocks


def etiology_fractions(blocks):
    parts = []
    for tag, df in blocks.items():
        total = df[f"{tag}_Total Mutations"].replace(0, np.nan)
        sigs = [c for c in df.columns if isinstance(c, str) and not c.startswith(f"{tag}_") and "-CRC" not in c
                and not DE_NOVO.match(c)]
        cats = [CATEGORY.get(c, "Unknown aetiology") for c in sigs]
        parts.append((df[sigs].div(total, axis=0)).T.groupby(cats).sum().T)
    combined = parts[0]
    for p in parts[1:]:
        combined = combined.add(p, fill_value=0)
    return combined.div(combined.sum(axis=1).replace(0, np.nan), axis=0)


def cliffs_delta(x, y):
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    return 2 * u / (len(x) * len(y)) - 1, p


def stars(p):
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def main():
    out = outdir("12_signatures")
    frac = etiology_fractions(read_blocks(DATA / "SBS_signatures.xlsx"))
    df = frac.join(load_labels().rename("gCPS"), how="inner")
    cats = list(frac.columns)
    df.to_csv(out / "etiology_fractions_per_tumor.csv")

    # omnibus test across the four subtypes
    rows = []
    for c in cats:
        groups = [df.loc[df.gCPS == s, c].dropna() for s in SUBTYPES]
        means = df.groupby("gCPS")[c].mean()
        rows.append(dict(category=c, P=kruskal(*groups)[1], effect=means.max() - means.min(),
                         lowest=means.idxmin(), highest=means.idxmax()))
    omni = pd.DataFrame(rows)
    omni["P_adj"] = bh(omni.P)
    omni.sort_values("P_adj").to_csv(out / "etiology_kruskal_wallis.csv", index=False)

    # gCPS4 against the other tumors
    rng = np.random.default_rng(SEED)
    is4 = (df.gCPS == "gCPS4").values
    res = []
    for c in cats:
        v = df[c].values
        x, y = v[is4], v[~is4]
        delta, p = cliffs_delta(x, y)
        boot = [cliffs_delta(rng.choice(x[~np.isnan(x)], (~np.isnan(x)).sum()),
                             rng.choice(y[~np.isnan(y)], (~np.isnan(y)).sum()))[0] for _ in range(N_BOOT)]
        res.append(dict(category=c, gCPS4_median=np.nanmedian(x), rest_median=np.nanmedian(y), cliffs_delta=delta,
                        ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5), P=p))
    eff = pd.DataFrame(res)
    eff["P_adj"] = bh(eff.P)
    eff.sort_values("cliffs_delta", ascending=False).to_csv(out / "etiology_gCPS4_vs_rest.csv", index=False)

    plt = style_matplotlib()
    # Figure 4C: subtype means, z-scored within each category
    mean = df.groupby("gCPS")[cats].mean().T.reindex(columns=SUBTYPES)
    z = mean.sub(mean.mean(1), axis=0).div(mean.std(1).replace(0, np.nan), axis=0).fillna(0)
    order = omni.sort_values("P_adj").category
    z = z.loc[order]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
    adj = omni.set_index("category").P_adj
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels([f"{c}  {stars(adj[c])}" for c in z.index], fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(SUBTYPES, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Subtype z-score (relative contribution)", fraction=0.04)
    save_fig(fig, out / "fig4C_etiology_heatmap")
    plt.close(fig)

    # Figure 4D: Cliff's delta, gCPS4 versus the rest
    e = eff.sort_values("cliffs_delta")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, r in enumerate(e.itertuples()):
        col = "#D55E00" if r.cliffs_delta > 0 else "#0072B2"
        ax.errorbar(r.cliffs_delta, i, xerr=[[r.cliffs_delta - r.ci_low], [r.ci_high - r.cliffs_delta]], fmt="o",
                    color=col, capsize=3, markeredgecolor="black")
        if r.P_adj < 0.05:
            ax.text(1.02, i, "*", va="center", transform=ax.get_yaxis_transform())
    ax.axvline(0, color="grey", ls="--")
    ax.set_yticks(range(len(e)))
    ax.set_yticklabels(e.category, fontsize=9)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Cliff's delta, gCPS4 vs other subtypes (bootstrap 95% CI)")
    save_fig(fig, out / "fig4D_etiology_effect_sizes")
    plt.close(fig)
    print(omni.sort_values("P_adj")[["category", "P_adj"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
