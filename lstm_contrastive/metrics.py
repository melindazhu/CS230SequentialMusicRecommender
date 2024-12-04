"""
Perform metrics gathering on the self-supervised LSTM model through
k-means clustering visualized by t-SNE. Also have the option of doing
DBSCAN instead of k-means.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt


def cluster_and_evaluate(csv_filename, df, embeddings, method="kmeans", n_clusters=5, eps=0.5, min_samples=5):
    if method == "kmeans":
        print(f"Running K-Means with {n_clusters} clusters...")
        model = KMeans(n_clusters=n_clusters, max_iter=1000, tol=1e-6, n_init=15, random_state=42)
        labels = model.fit_predict(embeddings)
        df['Cluster'] = labels
        df.to_csv(f'{csv_filename}_with_labels.csv', index=False)
        print("Clustering done")
    elif method == "dbscan":
        print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}...")
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(embeddings)

    # Evaluate clustering (ignoring noise points in DBSCAN labeled as -1)
    valid_indices = labels != -1
    if np.sum(valid_indices) > 1:  # score requires >= 2 clusters
        silhouette = silhouette_score(embeddings[valid_indices], labels[valid_indices], metric="cosine")
        db_index = davies_bouldin_score(embeddings[valid_indices], labels[valid_indices])
    else:
        silhouette = None
        db_index = None

    print(f"silhouette score: {silhouette:.4f}" if silhouette else "silhouette: none")
    print(f"davies-bouldin index: {db_index:.4f}" if db_index else "db index: none")

    return {
        "labels": labels,
        "silhouette_score": silhouette,
        "db_index": db_index,
    }


def visualize_embeddings(embeddings, labels, title="t-SNE Visualization"):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = reduced_embeddings[labels == label]
        plt.scatter(
            cluster_points[:,0],
            cluster_points[:,1],
            label=f"Cluster {label}",
            cmap="viridis",
            alpha=0.7
        )

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Clusters")
    plt.savefig("kmeans_len8.png", dpi=300)
    plt.show()


def get_variance(csv_file):
    df = pd.read_csv(csv_file)
    df = df.drop(columns=["track_names"])

    num_features = 12
    num_songs = df.shape[1] // num_features

    feature_variances = []

    for _, row in df.iterrows():
        sequence = row.values.reshape(num_songs, num_features)
        feature_var = np.mean(sequence, axis=0)
        feature_variances.append(feature_var)

    variance_df = pd.DataFrame(feature_variances, columns=[f"Feature_{i+1}" for i in range(num_features)])
    average_variance = variance_df.mean()

    print("Average Variance for Each Feature:")
    print(average_variance)



def main():
    csv_file = 'data/training_data_seqs.csv'
    get_variance(csv_file)


if __name__ == '__main__':
    main()