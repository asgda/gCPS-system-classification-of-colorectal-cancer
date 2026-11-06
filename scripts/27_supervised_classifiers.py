"""Step 27. Recovery of the gCPS labels by seven classifiers, with SHAP importance (Figure S4, Supplementary Table S8)."""
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

from gcps_utils import COLORS, JOBS, SEED, SUBTYPES, load_labels, load_matrix, outdir, sample_id, save_fig, style_matplotlib

TEST_SIZE, CV_FOLDS, TOP_SHAP = 0.20, 10, 15

CLASSIFIERS = {
    "RFC": lambda s: RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", n_jobs=JOBS, random_state=s),
    "SVM": lambda s: SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=s),
    "NaiveBayes": lambda s: BernoulliNB(alpha=1.0),
    "KNN": lambda s: KNeighborsClassifier(n_neighbors=7, metric="hamming", n_jobs=JOBS),
    "XGBoost": lambda s: xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, eval_metric="mlogloss", nthread=JOBS,
                                           random_state=s, verbosity=0),
    "LR": lambda s: LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, class_weight="balanced", n_jobs=JOBS,
                                       random_state=s),
    "MLP": lambda s: MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", max_iter=500,
                                   early_stopping=True, validation_fraction=0.1, random_state=s),
}


def oversample(X, y, seed=SEED):
    return RandomOverSampler(random_state=seed).fit_resample(X, y)


def probabilities(clf, X):
    return clf.predict_proba(X)


def metrics(y, pred, prob):
    cm = multilabel_confusion_matrix(y, pred, labels=range(4))
    sens = [tp / (tp + fn) if tp + fn else 0 for tn, fp, fn, tp in (m.ravel() for m in cm)]
    spec = [tn / (tn + fp) if tn + fp else 0 for tn, fp, fn, tp in (m.ravel() for m in cm)]
    return dict(accuracy=accuracy_score(y, pred), sensitivity=np.mean(sens), specificity=np.mean(spec),
                macro_f1=f1_score(y, pred, average="macro", zero_division=0),
                auroc=roc_auc_score(y, prob, multi_class="ovr", average="macro"))


def main():
    out = outdir("27_classifiers")
    labels = load_labels()
    x = load_matrix()
    x.index = [sample_id(i) for i in x.index]
    x = x.loc[labels.index]
    X, y = x.values.astype(np.float32), labels.map({s: i for i, s in enumerate(SUBTYPES)}).values
    names = list(x.columns)
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    pd.DataFrame({"sample_id": labels.index, "split": np.where(np.isin(np.arange(len(y)), train_idx), "train", "test")}
                 ).to_csv(out / "train_test_split.csv", index=False)
    print(f"train {X_train.shape}, test {X_test.shape}")

    rows, test_prob = [], {}
    for name, make in CLASSIFIERS.items():
        folds, pooled_y, pooled_p = [], [], []
        for f, (tr, va) in enumerate(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED).split(X_train, y_train)):
            Xr, yr = oversample(X_train[tr], y_train[tr], SEED + f)
            clf = make(SEED + f).fit(Xr, yr)
            folds.append(metrics(y_train[va], clf.predict(X_train[va]), probabilities(clf, X_train[va])))
            pooled_y.append(y_train[va])
            pooled_p.append(probabilities(clf, X_train[va]))
        cv = pd.DataFrame(folds)
        pd.DataFrame(np.vstack(pooled_p), columns=SUBTYPES).assign(label=np.concatenate(pooled_y)).to_csv(
            out / f"cv_probabilities_{name}.csv", index=False)
        Xr, yr = oversample(X_train, y_train)
        clf = make(SEED).fit(Xr, yr)
        prob = probabilities(clf, X_test)
        test_prob[name] = prob
        test = metrics(y_test, clf.predict(X_test), prob)
        rows.append({"classifier": name, **{f"cv_mean_{k}": cv[k].mean() for k in cv}, **{f"cv_sd_{k}": cv[k].std() for k in cv},
                     **{f"test_{k}": v for k, v in test.items()}})
        print(f"{name}: test macro-F1 {test['macro_f1']:.3f}, AUROC {test['auroc']:.3f}")
        if name == "RFC":
            sv = np.asarray(shap.TreeExplainer(clf).shap_values(X_test))
            if sv.ndim == 3 and sv.shape[0] != 4:
                sv = sv.transpose(2, 0, 1)
            shap_rfc = np.abs(sv).mean(axis=(0, 1))
        if name == "XGBoost":
            contrib = clf.get_booster().predict(xgb.DMatrix(X_test), pred_contribs=True)
            shap_xgb = np.abs(contrib[:, :, :-1].transpose(1, 0, 2)).mean(axis=(0, 1))
    pd.DataFrame(rows).to_csv(out / "classifier_performance.csv", index=False)
    pd.DataFrame({"feature": names, "SHAP_RFC": shap_rfc, "SHAP_XGBoost": shap_xgb}).sort_values(
        "SHAP_RFC", ascending=False).to_csv(out / "shap_importance.csv", index=False)

    for name, prob in test_prob.items():
        pd.DataFrame(prob, columns=SUBTYPES).assign(label=y_test).to_csv(out / f"test_probabilities_{name}.csv", index=False)

    plt = style_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    y_bin = label_binarize(y_test, classes=range(4))
    for name, prob in test_prob.items():
        grid = np.linspace(0, 1, 200)
        tpr = np.mean([np.interp(grid, *roc_curve(y_bin[:, k], prob[:, k])[:2]) for k in range(4)], axis=0)
        ax.plot(grid, tpr, label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity (macro-average)")
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, out / "test_auroc_curves")
    plt.close(fig)

    shap_table = pd.read_csv(out / "shap_importance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col in zip(axes, ("SHAP_RFC", "SHAP_XGBoost")):
        top = shap_table.nlargest(TOP_SHAP, col).iloc[::-1]
        ax.barh(top.feature.str.replace("_", " ", n=1), top[col], color=[COLORS["gCPS1"]] * len(top))
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(col.replace("SHAP_", ""))
    save_fig(fig, out / "shap_top_features")
    plt.close(fig)


if __name__ == "__main__":
    main()
