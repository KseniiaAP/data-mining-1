from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans,
    BisectingKMeans,
    SpectralClustering,
    DBSCAN,
    AgglomerativeClustering,
)
from sklearn.metrics import fowlkes_mallows_score, silhouette_score


def load_features_and_labels():
    src_dir = Path(__file__).resolve().parent
    base_dir = src_dir.parent
    features_dir = base_dir / "features"

    features_path = features_dir / "resnet18_lastconv_features.npy"
    labels_path = features_dir / "labels.npy"

    X = np.load(features_path)
    y = np.load(labels_path)

    return X, y


#2) Dimension Reduction to 2D
def reduce_to_2d_pca(X, n_components=2, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X_2d = pca.fit_transform(X)
    return X_2d, pca


#3) Clustering Algorithms
def run_all_clusterings(X_2d, random_state=42, target_k=4):

    results = {}

    # 3.(a) K-means clustering: (Use KMeans with init = ‘Random’)
    kmeans_random = KMeans(
        n_clusters=target_k,
        init="random",
        n_init=10,
        random_state=random_state,
    )
    results["kmeans_random"] = kmeans_random.fit_predict(X_2d)

    # 3.(b) KMeans with init=‘k-means++’
    kmeans_plus = KMeans(
        n_clusters=target_k,
        init="k-means++",
        n_init=10,
        random_state=random_state,
    )
    results["kmeans_kmeans++"] = kmeans_plus.fit_predict(X_2d)

    # 3.(c) Bisecting K-means (sklearn.cluster.BisectingKMeans with init = ‘Random’)
    bisecting = BisectingKMeans(
        n_clusters=target_k,
        init="random",
        random_state=random_state,
    )
    results["bisecting_kmeans"] = bisecting.fit_predict(X_2d)

    # 3.(d) spectral clustering (sklearn.cluster.SpectralClustering with default parameters)
    spectral = SpectralClustering(
        n_clusters=target_k,
        assign_labels="kmeans",
        random_state=random_state,
    )
    results["spectral_clustering"] = spectral.fit_predict(X_2d)

    # DBSCAN
    labels_dbscan, eps_used, min_samples_used = find_dbscan_for_k(
        X_2d, target_k=target_k
    )
    results["dbscan"] = labels_dbscan

    # Agglomerative clustering
    linkages = ["single", "complete", "average", "ward"]
    for link in linkages:
        agg = AgglomerativeClustering(
            n_clusters=target_k,
            linkage=link,
        )
        name = f"agglomerative_{link}"
        results[name] = agg.fit_predict(X_2d)

    return results


def find_dbscan_for_k(X_2d, target_k=4):
    eps_values = [round(e, 2) for e in np.linspace(0.05, 1.5, 30)]
    min_samples_values = list(range(2, 11))

    found_params = []
    best_labels = None

    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X_2d)

            n_clusters = len(set(labels) - {-1})

            if n_clusters == target_k:
                print(f"[DBSCAN] {target_k} clusters: eps={eps}, min_samples={ms}")
                return labels, eps, ms

            if n_clusters >= 2:
                found_params.append((n_clusters, eps, ms))
                best_labels = labels

    print("\n[DBSCAN] Exact match not found.")
    print("Closest cluster counts found:")
    for n, eps, ms in found_params[:10]:
        print(f"Clusters={n}, eps={eps}, min_samples={ms}")

    return best_labels, None, None


#4) Clustering Evaluations
def evaluate_clusterings(X_2d, y_true, clustering_results):
    rows = []

    for method_name, y_pred in clustering_results.items():

        labels_set = set(y_pred)
        n_clusters = len(labels_set - {-1})

        # (a) Fowlkes-Mallows index
        fm = fowlkes_mallows_score(y_true, y_pred)

        # (b) Silhouette Coefficient
        sil = np.nan
        try:
            if n_clusters >= 2:
                sil = silhouette_score(X_2d, y_pred)
        except Exception:
            sil = np.nan

        rows.append(
            {
                "method": method_name,
                "n_clusters": n_clusters,
                "fowlkes_mallows": fm,
                "silhouette": sil,
            }
        )

    df = pd.DataFrame(rows)
    return df


def print_rankings(df_results):
    print("\nResults table")
    print(df_results.to_string(index=False))

    print("\nRanking by Fowlkes–Mallows (best to worst)")
    print(
        df_results.sort_values("fowlkes_mallows", ascending=False)[
            ["method", "fowlkes_mallows"]
        ].to_string(index=False)
    )

    print("\nRanking by Silhouette Coefficient (best to worst)")
    df_sorted_sil = df_results.sort_values(
        "silhouette", ascending=False, na_position="last"
    )
    print(
        df_sorted_sil[["method", "silhouette"]].to_string(index=False)
    )


def main():
    # Data load
    X, y_true = load_features_and_labels()
    print(f"Loaded features: {X.shape}, labels: {y_true.shape}")

    #2) Dimension reduction to 2D
    X_2d, pca = reduce_to_2d_pca(X)
    print(f"Reduced to 2D: {X_2d.shape}")

    #3) Clustering methods
    clustering_results = run_all_clusterings(X_2d, target_k=4)

    #4) Evaluation
    df_results = evaluate_clusterings(X_2d, y_true, clustering_results)

    print_rankings(df_results)

    src_dir = Path(__file__).resolve().parent
    base_dir = src_dir.parent
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "clustering_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()