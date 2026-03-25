import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


def train_als_model(train_df, rank=50, maxIter=10, regParam=0.1):
    """Train an ALS collaborative filtering model."""
    print(f"\nModel Configuration:")
    print(f"Rank (latent factors): {rank}")
    print(f"Max iterations: {maxIter}")
    print(f"Regularization parameter: {regParam}")
    print(f"Cold start strategy: drop")

    print("\nInitializing ALS model")
    als = ALS(
        rank=rank,
        maxIter=maxIter,
        regParam=regParam,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        seed=42
    )

    print("Training model")
    start_time = time.time()

    model = als.fit(train_df)

    elapsed_time = time.time() - start_time

    print(f"\nALS model trained successfully!")
    print(f"Training time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

    print(f"\nModel Details:")
    print(f"User factors: {model.userFactors.count():,}")
    print(f"Item factors: {model.itemFactors.count():,}")
    print(f"Rank: {model.rank}")

    return model


def evaluate_model(model, test_df, metric='rmse'):
    """Evaluate the ALS model on the test set."""
    print(f"\nEvaluating model using {metric.upper()}...")

    print("Making predictions on test set")
    predictions = model.transform(test_df)

    print(f"Predictions generated")
    print("\nSample predictions:")
    predictions.select('userId', 'movieId', 'rating', 'prediction').show(10, truncate=False)

    print("\nCalculating evaluation metric")
    evaluator = RegressionEvaluator(
        metricName=metric,
        labelCol="rating",
        predictionCol="prediction"
    )

    score = evaluator.evaluate(predictions)

    print(f"{metric.upper()} SCORE: {score:.4f}")

    return score, predictions


def tune_als_parameters(train_df, test_df, output_dir='outputs'):
    """Grid search over ALS hyperparameters."""
    print("\nHyperparameter grid:")
    ranks = [10, 30, 50]
    reg_params = [0.01, 0.1, 0.5]

    print(f"Ranks to test: {ranks}")
    print(f"Reg params to test: {reg_params}")
    print(f"Total combinations: {len(ranks) * len(reg_params)}")

    results = []
    total_combinations = len(ranks) * len(reg_params)
    current = 0

    print("TUNING PROGRESS")

    for rank in ranks:
        for reg_param in reg_params:
            current += 1
            print(f"\n[{current}/{total_combinations}] Testing rank={rank}, regParam={reg_param}")

            model = train_als_model(train_df, rank=rank, maxIter=10, regParam=reg_param)
            rmse, _ = evaluate_model(model, test_df, metric='rmse')

            results.append({
                'rank': rank,
                'regParam': reg_param,
                'rmse': rmse
            })

            print(f"Result: RMSE = {rmse:.4f}")

    results_df = pd.DataFrame(results)

    print("HYPERPARAMETER TUNING RESULTS")
    print(results_df.sort_values('rmse').to_string(index=False))

    best_result = results_df.loc[results_df['rmse'].idxmin()]
    print(f"\nBest configuration:")
    print(f"Rank: {int(best_result['rank'])}")
    print(f"Reg param: {best_result['regParam']}")
    print(f"RMSE: {best_result['rmse']:.4f}")

    print("\nCreating visualization")
    pivot_results = results_df.pivot(index='rank', columns='regParam', values='rmse')

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_results, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': 'RMSE'})
    plt.title('ALS Hyperparameter Tuning Results (RMSE)', fontsize=14, fontweight='bold')
    plt.xlabel('Regularization Parameter', fontsize=12)
    plt.ylabel('Rank (Number of Factors)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/als_hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
    print("Heatmap saved as: als_hyperparameter_tuning.png")
    plt.show()

    return results_df


def extract_factors(model):
    """Extract user and item latent factor matrices from the trained model."""
    print("\nExtracting factors from model...")

    user_factors = model.userFactors
    item_factors = model.itemFactors

    user_count = user_factors.count()
    item_count = item_factors.count()

    print(f"\nUser factors extracted: {user_count:,} users")
    print(f"Item factors extracted: {item_count:,} items")

    print("SAMPLE USER FACTORS (5 users):")
    user_factors.show(5, truncate=False)

    print("SAMPLE ITEM FACTORS (5 items):")
    item_factors.show(5, truncate=False)

    sample_features = user_factors.select('features').first()[0]
    factor_dim = len(sample_features)

    print(f"\nFactor dimensionality: {factor_dim}")
    print("Factors ready for clustering!")

    return user_factors, item_factors
