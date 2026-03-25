import matplotlib.pyplot as plt
from pyspark.sql.functions import col, count, avg, expr, explode, split, row_number
from pyspark.sql.window import Window


def analyze_user_clusters(clustered_users, train_df, output_dir='outputs'):
    """Analyze rating behavior across user clusters."""
    print("\nJoining cluster assignments with training data")

    cluster_stats = train_df.join(
        clustered_users.select('id', 'cluster'),
        train_df.userId == clustered_users.id,
        'inner'
    )

    print("Data joined successfully")

    print("\nComputing cluster statistics")
    cluster_summary = cluster_stats.groupBy('cluster').agg(
        count('rating').alias('num_ratings'),
        avg('rating').alias('avg_rating'),
        expr('count(distinct userId)').alias('num_users')
    ).orderBy('cluster')

    print("USER CLUSTER SUMMARY")
    cluster_summary.show()

    cluster_summary_pd = cluster_summary.toPandas()

    print("DETAILED CLUSTER STATISTICS")
    print(f"{'Cluster':<10} {'Users':>12} {'Ratings':>15} {'Avg Rating':>15}")
    for _, row in cluster_summary_pd.iterrows():
        print(f"{int(row['cluster']):<10} {int(row['num_users']):>12,} "
              f"{int(row['num_ratings']):>15,} {row['avg_rating']:>15.2f}")

    print("\nCreating cluster analysis visualizations")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('User Cluster Analysis', fontsize=16, fontweight='bold')

    axes[0].bar(cluster_summary_pd['cluster'], cluster_summary_pd['num_users'],
                color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Cluster ID', fontsize=11)
    axes[0].set_ylabel('Number of Users', fontsize=11)
    axes[0].set_title('Users per Cluster', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(cluster_summary_pd['cluster'], cluster_summary_pd['avg_rating'],
                color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Cluster ID', fontsize=11)
    axes[1].set_ylabel('Average Rating', fontsize=11)
    axes[1].set_title('Average Rating per Cluster', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=cluster_summary_pd['avg_rating'].mean(), color='red',
                    linestyle='--', linewidth=2, label='Overall Avg')
    axes[1].legend()

    axes[2].bar(cluster_summary_pd['cluster'], cluster_summary_pd['num_ratings'],
                color='lightgreen', edgecolor='black')
    axes[2].set_xlabel('Cluster ID', fontsize=11)
    axes[2].set_ylabel('Total Ratings', fontsize=11)
    axes[2].set_title('Total Ratings per Cluster', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/user_cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nUser cluster analysis complete!")
    return cluster_summary


def analyze_movie_clusters(clustered_movies, movies_df, train_df, output_dir='outputs'):
    """Analyze genre composition and ratings across movie clusters."""
    cluster_movies_meta = clustered_movies.select('id', 'cluster').join(
        movies_df,
        clustered_movies.id == movies_df.movieId,
        'inner'
    )

    movie_stats = train_df.groupBy('movieId').agg(
        count('rating').alias('num_ratings'),
        avg('rating').alias('avg_rating')
    )

    print("\nJoining all data")
    cluster_movies_stats = cluster_movies_meta.join(
        movie_stats,
        cluster_movies_meta.movieId == movie_stats.movieId,
        'inner'
    )

    cluster_summary = cluster_movies_stats.groupBy('cluster').agg(
        count(cluster_movies_meta.movieId).alias('num_movies'),
        avg('avg_rating').alias('cluster_avg_rating'),
        avg('num_ratings').alias('avg_num_ratings')
    ).orderBy('cluster')

    print("MOVIE CLUSTER SUMMARY")
    cluster_summary.show()

    if 'genres' in movies_df.columns:
        print("DOMINANT GENRES PER CLUSTER (Top 3):")

        cluster_genres = cluster_movies_meta.select(
            'cluster', explode(split('genres', '\\|')).alias('genre')
        )
        genre_distribution = cluster_genres.groupBy('cluster', 'genre').count().orderBy(
            'cluster', col('count').desc()
        )

        windowSpec = Window.partitionBy('cluster').orderBy(col('count').desc())
        top_genres = genre_distribution.withColumn('rank', row_number().over(windowSpec)) \
                                      .filter(col('rank') <= 3)

        top_genres.show(30, truncate=False)

        print("\nCreating genre distribution visualization")
        top_genres_pd = top_genres.toPandas()

        n_clusters = len(top_genres_pd['cluster'].unique())
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Top Genres by Movie Cluster', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for cluster_id in range(min(n_clusters, 10)):
            cluster_data = top_genres_pd[top_genres_pd['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                axes[cluster_id].barh(cluster_data['genre'], cluster_data['count'], color='coral')
                axes[cluster_id].set_title(f'Cluster {cluster_id}', fontweight='bold')
                axes[cluster_id].set_xlabel('Count')
                axes[cluster_id].tick_params(axis='y', labelsize=8)

        for idx in range(n_clusters, 10):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/movie_cluster_genres.png', dpi=300, bbox_inches='tight')
        plt.show()

    print("\nMovie cluster analysis complete!")
    return cluster_summary
