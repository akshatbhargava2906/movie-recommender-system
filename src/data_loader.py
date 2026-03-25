def load_data(spark, ratings_path='rating.csv', movies_path='movie.csv'):
    """Load ratings and movies CSV data into Spark DataFrames."""
    print(f"\nLoading ratings from: {ratings_path}")
    ratings_df = spark.read.csv(
        ratings_path,
        header=True,
        inferSchema=True
    )

    print(f"Loading movies from: {movies_path}")
    movies_df = spark.read.csv(
        movies_path,
        header=True,
        inferSchema=True
    )

    print("\nData loaded successfully!")

    print("SAMPLE RATINGS (5 rows):")
    ratings_df.show(5, truncate=False)

    print("SAMPLE MOVIES (5 rows):")
    movies_df.show(5, truncate=False)

    print("\nData preview complete!")
    return ratings_df, movies_df
