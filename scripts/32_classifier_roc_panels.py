"""Step 32. Cross-validation and held-out ROC panels for the eight classifiers (Supplementary Figure S4A-B)."""
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from gcps_utils import OUT, SUBTYPES, outdir, save_fig, style_matplotlib

COLORS = {"RFC": "#009E73", "SVM": "#D55E00", "NaiveBayes": "#E69F00", "KNN": "#56B4E9", "XGBoost": "#CC79A7",
          "LR": "#0072B2", "MLP": "#F0E442", "ResNet": "#999999"}
LABELS = {"NaiveBayes": "Naive Bayes"}


def macro_roc(y, prob):
    """One-vs-rest ROC curves averaged over the four subtypes."""
    onehot = label_binarize(y, classes=range(len(SUBTYPES)))
    grid = np.linspace(0, 1, 300)
    tpr = np.mean([np.interp(grid, *roc_curve(onehot[:, i], prob[:, i])[:2]) for i in range(len(SUBTYPES))], axis=0)
    return grid, tpr, auc(grid, tpr)


def read(path):
    d = pd.read_csv(path)
    return d.label.values, d[SUBTYPES].values


def draw(curves, title, name, out):
    plt = style_matplotlib()
    plt.rcParams.update({"font.size": 18, "axes.labelsize": 22, "xtick.labelsize": 18, "ytick.labelsize": 18,
                         "legend.fontsize": 12.5, "axes.linewidth": 1.2})
    fig, ax = plt.subplots(figsize=(9.8, 8.4))
    for clf, (y, prob) in curves.items():
        fpr, tpr, area = macro_roc(y, prob)
        ax.plot(fpr, tpr, lw=3, color=COLORS[clf], label=f"{LABELS.get(clf, clf)} (AUC={area:.3f})")
        ax.text(0.732, min(float(np.interp(0.72, fpr, tpr)), 0.98), f"{area:.2f}", color=COLORS[clf], fontsize=12.5,
                fontweight="bold", va="center")
    ax.plot([0, 1], [0, 1], "--", color="dimgrey", lw=1.5, label="Chance")
    ax.set_xlabel("1-Specificity", fontweight="bold")
    ax.set_ylabel("Sensitivity", fontweight="bold")
    ax.set_title(title, fontsize=26, fontweight="bold", pad=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, ls="--", color="lightgrey", lw=0.8, alpha=0.7)
    ax.legend(frameon=True, edgecolor="lightgrey", loc="lower right")
    plt.tight_layout()
    save_fig(fig, out / name)
    plt.close(fig)


def main():
    out = outdir("32_roc_panels")
    classical = ["RFC", "SVM", "NaiveBayes", "KNN", "XGBoost", "LR", "MLP"]
    cv = {c: read(OUT / "27_classifiers" / f"cv_probabilities_{c}.csv") for c in classical}
    test = {c: read(OUT / "27_classifiers" / f"test_probabilities_{c}.csv") for c in classical}
    cv["ResNet"] = read(OUT / "28_resnet" / "validation_probabilities.csv")       # single 12.5% validation split
    test["ResNet"] = read(OUT / "28_resnet" / "test_probabilities.csv")
    draw(cv, "CV AUROC", "A_ROC_cross_validation", out)
    draw(test, "Test AUROC", "B_ROC_held_out_test", out)


if __name__ == "__main__":
    main()
