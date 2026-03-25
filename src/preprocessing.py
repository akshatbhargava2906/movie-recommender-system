from pyspark.sql.functions import col


def preprocess_data(ratings_df, min_ratings_per_user=20, min_ratings_per_movie=20):
    """Filter ratings to keep only active users and popular movies."""
    print(f"\nFiltering criteria:")
    print(f"Minimum ratings per user: {min_ratings_per_user}")
    print(f"Minimum ratings per movie: {min_ratings_per_movie}")

    initial_count = ratings_df.count()
    initial_users = ratings_df.select('userId').distinct().count()
    initial_movies = ratings_df.select('movieId').distinct().count()

    print(f"Ratings: {initial_count:,}")
    print(f"Users: {initial_users:,}")
    print(f"Movies: {initial_movies:,}")

    user_counts = ratings_df.groupBy('userId').count()
    movie_counts = ratings_df.groupBy('movieId').count()

    active_users = user_counts.filter(col('count') >= min_ratings_per_user).select('userId')
    print(f"Found {active_users.count():,} active users")

    popular_movies = movie_counts.filter(col('count') >= min_ratings_per_movie).select('movieId')
    print(f"Found {popular_movies.count():,} popular movies")

    filtered_ratings = ratings_df \
        .join(active_users, on='userId', how='inner') \
        .join(popular_movies, on='movieId', how='inner')

    final_count = filtered_ratings.count()
    final_users = filtered_ratings.select('userId').distinct().count()
    final_movies = filtered_ratings.select('movieId').distinct().count()

    print("PREPROCESSING RESULTS")
    print(f"{'Metric':<20} {'Before':>15} {'After':>15} {'Change':>15}")
    print(f"{'Ratings':<20} {initial_count:>15,} {final_count:>15,} {final_count - initial_count:>15,}")
    print(f"{'Users':<20} {initial_users:>15,} {final_users:>15,} {final_users - initial_users:>15,}")
    print(f"{'Movies':<20} {initial_movies:>15,} {final_movies:>15,} {final_movies - initial_movies:>15,}")
    print(f"Retention rate: {final_count / initial_count * 100:.1f}%")

    return filtered_ratings


def split_data(ratings_df, train_ratio=0.7, seed=42):
    """Split ratings into train and test sets."""
    print(f"\nSplit configuration:")
    print(f"Train ratio: {train_ratio * 100:.0f}%")
    print(f"Test ratio: {(1 - train_ratio) * 100:.0f}%")
    print(f"Random seed: {seed}")

    print("\nPerforming split")
    train_df, test_df = ratings_df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)

    train_count = train_df.count()
    test_count = test_df.count()
    total = train_count + test_count

    print("TRAIN-TEST SPLIT RESULTS")
    print(f"{'Set':<15} {'Count':>15} {'Percentage':>15}")
    print(f"{'Train':<15} {train_count:>15,} {train_count / total * 100:>14.1f}%")
    print(f"{'Test':<15} {test_count:>15,} {test_count / total * 100:>14.1f}%")
    print(f"{'Total':<15} {total:>15,} {100.0:>14.1f}%")
    print("\nSplit complete!")
    return train_df, test_df
