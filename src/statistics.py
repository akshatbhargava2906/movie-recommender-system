from pyspark.sql.functions import avg, expr


def compute_statistics(ratings_df, movies_df):
    """Compute and display dataset statistics."""
    num_ratings = ratings_df.count()
    num_users = ratings_df.select('userId').distinct().count()
    num_movies = ratings_df.select('movieId').distinct().count()

    print("Counts computed")

    rating_stats = ratings_df.select(
        avg('rating').alias('avg_rating'),
        expr('min(rating)').alias('min_rating'),
        expr('max(rating)').alias('max_rating')
    ).collect()[0]

    sparsity = 1.0 - (num_ratings / (num_users * num_movies))

    print("DATASET OVERVIEW")
    print(f"{'Metric':<30} {'Value':>20}")
    print(f"{'Total Ratings':<30} {num_ratings:>20,}")
    print(f"{'Unique Users':<30} {num_users:>20,}")
    print(f"{'Unique Movies':<30} {num_movies:>20,}")
    print(f"{'Average Rating':<30} {rating_stats['avg_rating']:>20.2f}")
    print(f"{'Min Rating':<30} {rating_stats['min_rating']:>20.1f}")
    print(f"{'Max Rating':<30} {rating_stats['max_rating']:>20.1f}")
    print(f"{'Sparsity':<30} {sparsity * 100:>19.2f}%")

    print("\nComputing ratings per user")
    ratings_per_user = ratings_df.groupBy('userId').count().select('count')
    user_stats = ratings_per_user.summary('mean', 'min', 'max', '25%', '50%', '75%')

    print("RATINGS PER USER STATISTICS:")
    user_stats.show()

    print("Computing ratings per movie")
    ratings_per_movie = ratings_df.groupBy('movieId').count().select('count')
    movie_stats = ratings_per_movie.summary('mean', 'min', 'max', '25%', '50%', '75%')

    print("RATINGS PER MOVIE STATISTICS:")
    movie_stats.show()

    print("\nAll statistics computed successfully!")

    return {
        'num_ratings': num_ratings,
        'num_users': num_users,
        'num_movies': num_movies,
        'sparsity': sparsity,
        'avg_rating': rating_stats['avg_rating']
    }
