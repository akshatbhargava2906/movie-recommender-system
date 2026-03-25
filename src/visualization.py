import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import avg
from sklearn.decomposition import PCA


def visualize_dataset(ratings_df, movies_df, output_dir='outputs'):
    """Create exploratory visualizations for the dataset."""
    sample_size = 100000
    total_ratings = ratings_df.count()
    sample_fraction = min(sample_size / total_ratings, 1.0)

    print(f"Sampling {sample_fraction * 100:.1f}% of data ({min(sample_size, total_ratings):,} ratings)")
    ratings_sample = ratings_df.sample(False, sample_fraction, seed=42)

    ratings_pd = ratings_sample.toPandas()

    print("\nData prepared")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('MovieLens Dataset Exploration', fontsize=16, fontweight='bold')

    # 1. Distribution of ratings
    print("Plot 1/6: Rating distribution")
    axes[0, 0].hist(ratings_pd['rating'], bins=10, edgecolor='black', color='skyblue')
    axes[0, 0].set_xlabel('Rating', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Distribution of Ratings', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Ratings per user
    print("Plot 2/6: Ratings per user")
    ratings_per_user = ratings_df.groupBy('userId').count().toPandas()
    axes[0, 1].hist(ratings_per_user['count'], bins=50, edgecolor='black', color='lightcoral')
    axes[0, 1].set_xlabel('Number of Ratings', fontsize=10)
    axes[0, 1].set_ylabel('Number of Users (log scale)', fontsize=10)
    axes[0, 1].set_title('Ratings per User Distribution', fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Ratings per movie
    print("Plot 3/6: Ratings per movie")
    ratings_per_movie = ratings_df.groupBy('movieId').count().toPandas()
    axes[0, 2].hist(ratings_per_movie['count'], bins=50, edgecolor='black', color='lightgreen')
    axes[0, 2].set_xlabel('Number of Ratings', fontsize=10)
    axes[0, 2].set_ylabel('Number of Movies (log scale)', fontsize=10)
    axes[0, 2].set_title('Ratings per Movie Distribution', fontweight='bold')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Average rating distribution
    print("Plot 4/6: Average rating per movie")
    avg_ratings = ratings_df.groupBy('movieId').agg(avg('rating').alias('avg_rating')).toPandas()
    axes[1, 0].hist(avg_ratings['avg_rating'], bins=20, edgecolor='black', color='plum')
    axes[1, 0].set_xlabel('Average Rating', fontsize=10)
    axes[1, 0].set_ylabel('Number of Movies', fontsize=10)
    axes[1, 0].set_title('Average Rating per Movie', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Rating trends over time
    print("Plot 5/6: Rating trends over time")
    if 'timestamp' in ratings_pd.columns:
        ratings_pd['year'] = pd.to_datetime(ratings_pd['timestamp'], unit='s').dt.year
        yearly_avg = ratings_pd.groupby('year')['rating'].mean()
        axes[1, 1].plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=6, color='darkblue')
        axes[1, 1].set_xlabel('Year', fontsize=10)
        axes[1, 1].set_ylabel('Average Rating', fontsize=10)
        axes[1, 1].set_title('Average Rating Over Time', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Timestamp not available', ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Rating Trends (N/A)', fontweight='bold')

    # 6. Genre distribution
    print("Plot 6/6: Genre distribution")
    if movies_df is not None:
        movies_pd = movies_df.toPandas()
        if 'genres' in movies_pd.columns:
            genres = movies_pd['genres'].str.split('|').explode()
            genre_counts = genres.value_counts().head(10)
            axes[1, 2].barh(range(len(genre_counts)), genre_counts.values, color='coral', edgecolor='black')
            axes[1, 2].set_yticks(range(len(genre_counts)))
            axes[1, 2].set_yticklabels(genre_counts.index, fontsize=9)
            axes[1, 2].set_xlabel('Count', fontsize=10)
            axes[1, 2].set_title('Top 10 Movie Genres', fontweight='bold')
            axes[1, 2].grid(True, alpha=0.3, axis='x')
        else:
            axes[1, 2].text(0.5, 0.5, 'Genre data not available', ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_exploration.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved as: dataset_exploration.png")
    plt.show()

    print("All visualizations created successfully!")


def visualize_user_clusters(clustered_users, n_components=2, output_dir='outputs'):
    """Visualize user clusters using PCA dimensionality reduction."""
    print(f"\nPreparing data for visualization")

    sample_size = 10000
    total_users = clustered_users.count()

    if total_users > sample_size:
        sample_fraction = sample_size / total_users
        print(f"Sampling {sample_fraction * 100:.1f}% ({sample_size:,} users)")
        factors_sample = clustered_users.sample(False, sample_fraction, seed=42)
    else:
        print(f"Using all {total_users:,} users")
        factors_sample = clustered_users

    print("\nConverting to Pandas")
    factors_pd = factors_sample.toPandas()

    print("Extracting feature vectors")
    if hasattr(factors_pd['features'].iloc[0], 'toArray'):
        features_array = np.array([row.toArray() for row in factors_pd['features']])
    else:
        features_array = np.array(factors_pd['features'].tolist())

    print(f"Feature matrix shape: {features_array.shape}")

    pca = PCA(n_components=n_components)
    features_2d = pca.fit_transform(features_array)

    explained_var = pca.explained_variance_ratio_
    print(f"PCA complete")
    print(f"PC1 explains {explained_var[0] * 100:.1f}% variance")
    print(f"PC2 explains {explained_var[1] * 100:.1f}% variance")
    print(f"Total explained: {sum(explained_var) * 100:.1f}%")

    print("\nCreating visualization")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=factors_pd['cluster'],
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )

    cbar = plt.colorbar(scatter, label='Cluster')
    cbar.set_label('Cluster ID', fontsize=11)

    plt.xlabel(f'PC1 ({explained_var[0] * 100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_var[1] * 100:.1f}% variance)', fontsize=12)
    plt.title('User Clusters Visualization (PCA Projection)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    for cluster_id in factors_pd['cluster'].unique():
        cluster_points = features_2d[factors_pd['cluster'] == cluster_id]
        centroid = cluster_points.mean(axis=0)
        plt.plot(centroid[0], centroid[1], 'r*', markersize=20,
                 markeredgecolor='black', markeredgewidth=1.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/user_clusters_pca.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nCluster visualization complete!")
