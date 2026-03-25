import time

import matplotlib.pyplot as plt
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType


def _ensure_vector_features(df):
    """Convert array<float> features column to DenseVector if needed."""
    feature_type = df.schema['features'].dataType
    if isinstance(feature_type, ArrayType):
        array_to_vector = udf(
            lambda arr: Vectors.dense(arr) if arr is not None else None,
            VectorUDT()
        )
        df = df.withColumn('features', array_to_vector('features'))
    return df


def cluster_users(user_factors, k=5, output_dir='outputs'):
    """Cluster users based on their ALS latent factors using K-Means."""
    print(f"\nClustering configuration:")
    print(f"Number of clusters (k): {k}")
    print(f"Algorithm: K-Means")
    print(f"Features: Latent factors from ALS")

    user_factors = _ensure_vector_features(user_factors)

    print("\nInitializing K-Means")
    kmeans = KMeans(
        k=k,
        seed=42,
        featuresCol='features',
        predictionCol='cluster'
    )

    print("Training K-Means model")
    start_time = time.time()

    kmeans_model = kmeans.fit(user_factors)

    elapsed_time = time.time() - start_time
    print(f"Training complete in {elapsed_time:.2f} seconds")

    clustered_users = kmeans_model.transform(user_factors)

    wssse = kmeans_model.summary.trainingCost

    print("CLUSTERING RESULTS")
    print(f"Within Set Sum of Squared Errors (WSSSE): {wssse:.2f}")

    print("CLUSTER DISTRIBUTION:")
    cluster_dist = clustered_users.groupBy('cluster').count().orderBy('cluster')
    cluster_dist.show()

    cluster_dist_pd = cluster_dist.toPandas()

    print("\nCreating cluster distribution visualization")
    plt.figure(figsize=(10, 6))
    plt.bar(cluster_dist_pd['cluster'], cluster_dist_pd['count'], color='skyblue', edgecolor='black')
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Users', fontsize=12)
    plt.title('User Cluster Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/user_cluster_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nUser clustering complete!")
    return kmeans_model, clustered_users


def find_optimal_k(factors_df, k_range=range(2, 11), output_dir='outputs'):
    """Use the elbow method to find the optimal number of clusters."""
    print(f"\nTesting k values from {min(k_range)} to {max(k_range)}...")

    factors_df = _ensure_vector_features(factors_df)

    wssse_values = []

    print("ELBOW METHOD PROGRESS")

    for i, k in enumerate(k_range, 1):
        print(f"\n[{i}/{len(k_range)}] Testing k={k}")

        kmeans = KMeans(k=k, seed=42, featuresCol='features')
        model = kmeans.fit(factors_df)
        wssse = model.summary.trainingCost
        wssse_values.append(wssse)

        print(f"WSSSE = {wssse:.2f}")

    print("ELBOW METHOD RESULTS")
    print(f"{'k':<10} {'WSSSE':>15}")
    for k, wssse in zip(k_range, wssse_values):
        print(f"{k:<10} {wssse:>15.2f}")

    changes = []
    if len(wssse_values) > 2:
        changes = [wssse_values[i - 1] - wssse_values[i] for i in range(1, len(wssse_values))]
        print(f"\nLargest improvement at k={list(k_range)[changes.index(max(changes)) + 1]}")

    print("\nCreating elbow curve visualization")
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), wssse_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Within-Set Sum of Squared Errors (WSSSE)', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    if len(wssse_values) > 2 and changes:
        elbow_k = list(k_range)[changes.index(max(changes)) + 1]
        elbow_wssse = wssse_values[list(k_range).index(elbow_k)]
        plt.plot(elbow_k, elbow_wssse, 'r*', markersize=20, label=f'Suggested k={elbow_k}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nOptimal k analysis complete!")
    return list(k_range), wssse_values


def cluster_movies(item_factors, k=10, output_dir='outputs'):
    """Cluster movies based on their ALS latent factors using K-Means."""
    print(f"\nClustering configuration:")
    print(f"Number of clusters (k): {k}")
    print(f"Algorithm: K-Means")
    print(f"Features: Latent factors from ALS")

    item_factors = _ensure_vector_features(item_factors)

    print("\nInitializing K-Means")
    kmeans = KMeans(
        k=k,
        seed=42,
        featuresCol='features',
        predictionCol='cluster'
    )

    print("Training K-Means model")
    start_time = time.time()

    kmeans_model = kmeans.fit(item_factors)

    elapsed_time = time.time() - start_time
    print(f"Training complete in {elapsed_time:.2f} seconds")

    clustered_movies = kmeans_model.transform(item_factors)

    wssse = kmeans_model.summary.trainingCost

    print("CLUSTERING RESULTS")
    print(f"Within Set Sum of Squared Errors (WSSSE): {wssse:.2f}")

    print("CLUSTER DISTRIBUTION:")
    cluster_dist = clustered_movies.groupBy('cluster').count().orderBy('cluster')
    cluster_dist.show()

    cluster_dist_pd = cluster_dist.toPandas()

    print("\nCreating cluster distribution visualization")
    plt.figure(figsize=(10, 6))
    plt.bar(cluster_dist_pd['cluster'], cluster_dist_pd['count'], color='lightcoral', edgecolor='black')
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Movies', fontsize=12)
    plt.title('Movie Cluster Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/movie_cluster_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nMovie clustering complete!")
    return kmeans_model, clustered_movies
