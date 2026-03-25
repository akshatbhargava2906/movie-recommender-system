from pyspark.sql.functions import col, explode


def get_user_recommendations(spark, model, user_id, n=10, movies_df=None):
    """Generate top-N movie recommendations for a specific user."""
    print(f"\nGenerating recommendations for User {user_id}")
    print(f"Number of recommendations: {n}")

    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    recommendations = model.recommendForUserSubset(user_df, n)

    print("Recommendations generated")

    recs_exploded = recommendations.select('userId', explode('recommendations').alias('rec'))
    recs_final = recs_exploded.select(
        'userId',
        col('rec.movieId').alias('movieId'),
        col('rec.rating').alias('predicted_rating')
    )

    print(f"TOP {n} RECOMMENDATIONS FOR USER {user_id}:")

    if movies_df is not None:
        recs_with_titles = recs_final.join(movies_df, on='movieId', how='left')
        recs_with_titles.select('movieId', 'title', 'genres', 'predicted_rating').show(n, truncate=False)
        return recs_with_titles
    else:
        recs_final.show(n, truncate=False)
        return recs_final
