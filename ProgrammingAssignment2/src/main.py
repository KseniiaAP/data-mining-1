import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, transform, util
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# 2. Convert the images to edge histograms.
def edge_orientation_histogram(img_rgb, resize_to=(256, 256), bins=16):
    img = color.rgb2gray(img_rgb)
    if resize_to is not None:
        img = transform.resize(img, resize_to, anti_aliasing=True)
    gx = filters.sobel_h(img)
    gy = filters.sobel_v(img)
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    if mag.size == 0:
        return np.zeros(bins, dtype=np.float32)
    thresh = np.percentile(mag, 75)
    mask = mag >= thresh
    hist, _ = np.histogram(ang[mask], bins=bins, range=(0, 180), weights=mag[mask])
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


# 1. Use images from ALL classes in your dataset
def load_dataset(root_dir, extensions={".jpg", ".jpeg", ".png", ".bmp"}):
    root = Path(root_dir)
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]
    X, y = [], []
    for label, cdir in enumerate(class_dirs):
        for p in cdir.rglob("*"):
            if p.suffix.lower() in extensions:
                try:
                    img = io.imread(p)
                    if img.ndim == 2:
                        img = np.stack([img]*3, axis=-1)
                    elif img.shape[2] == 4:
                        img = util.img_as_float(img)[..., :3]
                    feat = edge_orientation_histogram(img)
                    X.append(feat)
                    y.append(label)
                except Exception as e:
                    print(f"Skip {p}: {e}")
    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, class_names


def get_models_dict():
    return {
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "LinearSVC": LinearSVC(dual="auto", random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=500, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
    }


def plot_confusion(ax, y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=str(Path(__file__).resolve().parents[1] / "Flowers Data Set" / "flowers"))
    parser.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--models", nargs="+", default=["GaussianNB","DecisionTree","RandomForest"])
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--svm_two_classes", nargs=2, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Use images from ALL classes in your dataset
    print("Extracting features from all classes...")
    X, y, class_names = load_dataset(args.data_root)
    print(f"Dataset: {X.shape}, classes: {class_names}")

    # 3. Split the dataset into a training set and a test set: For each class, perform a training/test split of 80/20.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.seed)

    # 4. Perform standardization on the training dataset.
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr)

    # 5. Perform standardization on the test dataset using the means and variances you obtained from the training dataset.
    X_te_std = scaler.transform(X_te)

    # 6. Performance Comparison
    # The following models and cross-validation setup are based on the official scikit-learn documentation:
    # https://scikit-learn.org/stable/user_guide.html
    models_all = get_models_dict()
    chosen = [m for m in args.models if m in models_all]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_results, test_acc, test_f1 = {}, {}, {}

    for name in chosen:
        clf = models_all[name]
        scores = cross_val_score(clf, X_tr_std, y_tr, cv=skf, scoring="accuracy")
        cv_results[name] = scores
        print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, name in zip(axes, chosen):
        clf = models_all[name]
        clf.fit(X_tr_std, y_tr)
        y_pred = clf.predict(X_te_std)
        acc = accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro")
        test_acc[name], test_f1[name] = acc, f1m
        plot_confusion(ax, y_te, y_pred, class_names, f"{name}\nAcc={acc:.3f}, F1={f1m:.3f}")

    plt.suptitle("Confusion Matrices for 3 Models")
    plt.tight_layout()
    plt.savefig(out_dir / "confusions_test.png", dpi=150)
    plt.close()
    print("Saved confusion matrices")


    with open(out_dir / "cv_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + [f"fold_{i+1}" for i in range(5)] + ["mean","std"])
        for name in chosen:
            s = cv_results[name]
            writer.writerow([name]+[f"{v:.4f}" for v in s]+[f"{s.mean():.4f}",f"{s.std():.4f}"])
    with open(out_dir / "test_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model","accuracy","f1_macro"])
        for name in chosen:
            writer.writerow([name,f"{test_acc[name]:.4f}",f"{test_f1[name]:.4f}"])

    # 7. Model Selection
    if args.svm_two_classes is None:
        two = class_names[:2]
    else:
        two = args.svm_two_classes
    print(f"Two classes for SVM: {two}")

    idx_two = [class_names.index(c) for c in two]
    mask_tr, mask_te = np.isin(y_tr, idx_two), np.isin(y_te, idx_two)
    Xtr2, ytr2 = X_tr_std[mask_tr], y_tr[mask_tr]
    Xte2, yte2 = X_te_std[mask_te], y_te[mask_te]
    remap = {idx_two[0]:0, idx_two[1]:1}
    ytr2, yte2 = np.vectorize(remap.get)(ytr2), np.vectorize(remap.get)(yte2)

    Cs = [0.1, 1, 10, 100]
    kf, skf2 = KFold(5, shuffle=True, random_state=args.seed), StratifiedKFold(5, shuffle=True, random_state=args.seed)
    train_err_kf, val_err_kf, train_err_skf, val_err_skf = [], [], [], []

    for C in Cs:
        tr_scores, val_scores = [], []
        for tr_idx, val_idx in kf.split(Xtr2):
            clf = LinearSVC(C=C, dual="auto", random_state=42)
            clf.fit(Xtr2[tr_idx], ytr2[tr_idx])
            tr_scores.append(clf.score(Xtr2[tr_idx], ytr2[tr_idx]))
            val_scores.append(clf.score(Xtr2[val_idx], ytr2[val_idx]))
        train_err_kf.append(1 - np.mean(tr_scores))
        val_err_kf.append(1 - np.mean(val_scores))

        tr_scores, val_scores = [], []
        for tr_idx, val_idx in skf2.split(Xtr2, ytr2):
            clf = LinearSVC(C=C, dual="auto", random_state=42)
            clf.fit(Xtr2[tr_idx], ytr2[tr_idx])
            tr_scores.append(clf.score(Xtr2[tr_idx], ytr2[tr_idx]))
            val_scores.append(clf.score(Xtr2[val_idx], ytr2[val_idx]))
        train_err_skf.append(1 - np.mean(tr_scores))
        val_err_skf.append(1 - np.mean(val_scores))

    plt.figure(figsize=(6,4))
    plt.plot(Cs,val_err_kf,"o-",label="KF val error")
    plt.plot(Cs,train_err_kf,"o-",label="KF train error")
    plt.plot(Cs,val_err_skf,"o-",label="StratKF val error")
    plt.plot(Cs,train_err_skf,"o-",label="StratKF train error")
    plt.xscale("log")
    plt.xlabel("C (log scale)")
    plt.ylabel("Mean error (1 - accuracy)")
    plt.title(f"SVM model selection ({two[0]} vs {two[1]})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "svm_two_classes_model_selection.png", dpi=150)
    plt.close()

    best_idx = int(np.argmin(val_err_skf))
    best_C = Cs[best_idx]
    best_svm = LinearSVC(C=best_C, dual="auto", random_state=42)
    best_svm.fit(Xtr2, ytr2)
    test_err_2 = 1 - best_svm.score(Xte2, yte2)
    print(f"Best C={best_C}; two class test error={test_err_2:.4f}")


if __name__ == "__main__":
    main()
