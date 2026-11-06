"""Step 28. One-dimensional ResNet on the same train/test split as step 27 (Supplementary Table S8b).

Runs in the TensorFlow environment (versions/requirements_resnet.txt). Training uses a fixed seed and deterministic
TensorFlow operations, so repeated runs on one machine give identical results; results can still differ slightly on other
hardware or TensorFlow builds. Pass --evaluate <model.keras> to score a saved model on the held-out test tumors instead.
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from gcps_utils import SEED, SUBTYPES, load_labels, load_matrix, outdir, sample_id

EPOCHS, BATCH, LR = 200, 64, 1e-3
VALIDATION_FRACTION = 0.125


def residual_block(x, filters, stride=1):
    shortcut = x
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same", kernel_initializer="lecun_normal")(x)
    for i in range(3):
        x = layers.Conv1D(filters, 3, strides=stride if i == 0 else 1, padding="same",
                          kernel_initializer="lecun_normal")(x)
        x = layers.Activation("selu")(x)
    return layers.Activation("selu")(layers.Add()([x, shortcut]))


def build(n_features, n_classes):
    inp = keras.Input(shape=(n_features, 1))
    x = layers.Activation("selu")(layers.Conv1D(64, 7, padding="same", kernel_initializer="lecun_normal")(inp))
    x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
    for filters, blocks in [(64, 3), (128, 4), (256, 3), (512, 3)]:
        x = residual_block(x, filters, stride=2)
        for _ in range(blocks - 1):
            x = residual_block(x, filters)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(x)
    x = layers.Dropout(0.3, seed=SEED)(x)
    return keras.Model(inp, layers.Dense(n_classes, activation="softmax", kernel_initializer="lecun_normal")(x))


def schedule(epoch):
    """Five-epoch warm-up followed by cosine decay."""
    if epoch < 5:
        return LR * (epoch + 1) / 5
    progress = (epoch - 5) / max(1, EPOCHS - 5)
    return LR / 100 + 0.5 * (LR - LR / 100) * (1 + np.cos(np.pi * progress))


def cube_all(a):
    return a.reshape(-1, a.shape[1], 1)


def metrics(y, prob):
    pred = prob.argmax(1)
    cm = multilabel_confusion_matrix(y, pred, labels=range(4))
    return dict(accuracy=accuracy_score(y, pred),
                sensitivity=np.mean([m[1, 1] / (m[1, 1] + m[1, 0]) for m in cm]),
                specificity=np.mean([m[0, 0] / (m[0, 0] + m[0, 1]) for m in cm]),
                macro_f1=f1_score(y, pred, average="macro"), auroc=roc_auc_score(y, prob, multi_class="ovr"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", help="score this saved model instead of training a new one")
    args = parser.parse_args()
    out = outdir("28_resnet")
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    labels = load_labels()
    x = load_matrix()
    x.index = [sample_id(i) for i in x.index]
    X = x.loc[labels.index].values.astype(np.float32)
    y = labels.map({s: i for i, s in enumerate(SUBTYPES)}).values
    split = pd.read_csv(out.parent / "27_classifiers" / "train_test_split.csv")
    train, test = (split.split == "train").values, (split.split == "test").values
    X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]

    if args.evaluate:
        model = keras.models.load_model(args.evaluate)
        row = {"classifier": "ResNet1D", "model": args.evaluate,
               **{f"test_{k}": v for k, v in metrics(y_test, model.predict(cube_all(X_test), verbose=0)).items()}}
        pd.DataFrame([row]).to_csv(out / "resnet_evaluation.csv", index=False)
        print({k: round(v, 3) for k, v in row.items() if k.startswith("test_")})
        return

    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_FRACTION, stratify=y_train,
                                                  random_state=SEED)
    X_fit, y_fit = RandomOverSampler(random_state=SEED).fit_resample(X_fit, y_fit)
    cube = lambda a: a.reshape(-1, a.shape[1], 1)

    model = build(X.shape[1], 4)
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=LR), loss="categorical_crossentropy", metrics=["accuracy"])
    checkpoint = out / "best_model.keras"

    class KeepBestF1(keras.callbacks.Callback):
        best = 0.0

        def on_epoch_end(self, epoch, logs=None):
            f1 = f1_score(y_val, self.model.predict(cube(X_val), verbose=0).argmax(1), average="macro")
            if f1 > self.best:
                self.best = f1
                self.model.save(checkpoint)

    model.fit(cube(X_fit), keras.utils.to_categorical(y_fit, 4), epochs=EPOCHS, batch_size=BATCH, verbose=0,
              validation_data=(cube(X_val), keras.utils.to_categorical(y_val, 4)),
              callbacks=[keras.callbacks.LearningRateScheduler(schedule), KeepBestF1(),
                         keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)])
    model = keras.models.load_model(checkpoint)
    prob_val, prob_test = model.predict(cube(X_val), verbose=0), model.predict(cube(X_test), verbose=0)
    pd.DataFrame(prob_val, columns=SUBTYPES).assign(label=y_val).to_csv(out / "validation_probabilities.csv", index=False)
    pd.DataFrame(prob_test, columns=SUBTYPES).assign(label=y_test).to_csv(out / "test_probabilities.csv", index=False)
    row = {"classifier": "ResNet1D", "validation_protocol": "single stratified 12.5% split",
           **{f"validation_{k}": v for k, v in metrics(y_val, prob_val).items()},
           **{f"test_{k}": v for k, v in metrics(y_test, prob_test).items()}}
    pd.DataFrame([row]).to_csv(out / "resnet_performance.csv", index=False)
    print({k: round(v, 3) for k, v in row.items() if k.startswith("test_")})


if __name__ == "__main__":
    main()
