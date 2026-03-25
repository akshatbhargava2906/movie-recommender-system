"""
MovieLens Recommender System - Main Pipeline

Runs the full pipeline: data loading, EDA, preprocessing, ALS training,
hyperparameter tuning, clustering, analysis, and recommendations.
"""

import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import create_spark_session
from src.data_loader import load_data
from src.statistics import compute_statistics
from src.visualization import visualize_dataset, visualize_user_clusters
from src.preprocessing import preprocess_data, split_data
from src.model import train_als_model, evaluate_model, tune_als_parameters, extract_factors
from src.clustering import cluster_users, find_optimal_k, cluster_movies
from src.analysis import analyze_user_clusters, analyze_movie_clusters
from src.recommend import get_user_recommendations

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # ── 1. Spark Session ──
    spark = create_spark_session()

    # ── 2. Load Data ──
    ratings_df, movies_df = load_data(spark, 'rating.csv', 'movie.csv')

    # ── 3. Exploratory Statistics ──
    stats = compute_statistics(ratings_df, movies_df)

    # ── 4. Dataset Visualizations ──
    visualize_dataset(ratings_df, movies_df, output_dir=OUTPUT_DIR)

    # ── 5. Preprocessing ──
    filtered_ratings = preprocess_data(ratings_df, min_ratings_per_user=20, min_ratings_per_movie=20)

    # ── 6. Train/Test Split ──
    train_df, test_df = split_data(filtered_ratings, train_ratio=0.7, seed=42)

    # ── 7. Train ALS Model ──
    model = train_als_model(train_df, rank=50, maxIter=10, regParam=0.1)

    # ── 8. Evaluate Model ──
    print("\nEvaluating with RMSE")
    rmse, predictions = evaluate_model(model, test_df, metric='rmse')

    print("\nEvaluating with MAE")
    mae, _ = evaluate_model(model, test_df, metric='mae')

    print("MODEL PERFORMANCE SUMMARY")
    print(f"{'Metric':<20} {'Score':>15}")
    print(f"{'RMSE':<20} {rmse:>15.4f}")
    print(f"{'MAE':<20} {mae:>15.4f}")

    # ── 9. Hyperparameter Tuning ──
    tuning_results = tune_als_parameters(train_df, test_df, output_dir=OUTPUT_DIR)

    # ── 10. Extract Factors ──
    user_factors, item_factors = extract_factors(model)

    # ── 11. User Clustering ──
    k_values, wssse_values = find_optimal_k(user_factors, k_range=range(2, 11), output_dir=OUTPUT_DIR)
    user_kmeans, clustered_users = cluster_users(user_factors, k=5, output_dir=OUTPUT_DIR)

    # ── 12. Visualize User Clusters ──
    visualize_user_clusters(clustered_users, output_dir=OUTPUT_DIR)

    # ── 13. Movie Clustering ──
    movie_kmeans, clustered_movies = cluster_movies(item_factors, k=10, output_dir=OUTPUT_DIR)

    # ── 14. Cluster Analysis ──
    user_cluster_summary = analyze_user_clusters(clustered_users, train_df, output_dir=OUTPUT_DIR)
    movie_cluster_summary = analyze_movie_clusters(clustered_movies, movies_df, train_df, output_dir=OUTPUT_DIR)

    # ── 15. Sample Recommendations ──
    user_recs = get_user_recommendations(spark, model, user_id=1, n=10, movies_df=movies_df)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"All plots saved to: {OUTPUT_DIR}/")
    print("=" * 60)

    spark.stop()


if __name__ == '__main__':
    main()
