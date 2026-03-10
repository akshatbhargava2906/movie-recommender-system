# Movie Recommendation System using Apache Spark MLlib

A scalable, end-to-end movie recommendation system built on the **MovieLens dataset** using **Apache Spark MLlib** and deployed on **Google Cloud Platform**. The system leverages collaborative filtering via Alternating Least Squares (ALS) and unsupervised user/movie segmentation via K-Means clustering.

---

## Project Overview

| Detail | Value |
|---|---|
| **Dataset** | MovieLens (20M ratings) |
| **Users** | 138,493 |
| **Movies** | 26,744 |
| **Algorithm** | ALS (Collaborative Filtering) |
| **RMSE** | 0.8110 |
| **MAE** | 0.6349 |
| **Platform** | Google Cloud Dataproc (asia-southeast1) |

---

## Architecture

```
Google Cloud Storage (GCS)
        │
        ▼
  Dataproc Cluster (Spark)
        │
        ├── Data Preprocessing & EDA
        ├── ALS Model Training & Evaluation
        ├── Hyperparameter Tuning
        ├── K-Means Clustering (Users + Movies)
        └── Recommendation Generation
```

---

## Features

- **Collaborative Filtering** via ALS matrix factorization on 20M+ ratings
- **Hyperparameter Tuning** across rank × regularization grid (9 configurations)
- **User Segmentation** — 5 user clusters identified via K-Means on ALS latent factors
- **Movie Clustering** — 10 movie clusters analyzed by dominant genre composition
- **Personalized Top-N Recommendations** for any user in the dataset
- **PCA Visualization** of high-dimensional latent factor spaces
- **Distributed Computing** via Apache Spark on GCP Dataproc

---

## Methodology

### 1. Data Preprocessing
- Filtered users and movies with fewer than 20 ratings to reduce sparsity
- Retained **99.7% of ratings** after filtering (19,933,089 ratings)
- 70/30 train-test split with random seed 42

### 2. ALS Model
ALS decomposes the user-item rating matrix **R** into:
- **User factor matrix U** (users × k latent factors)
- **Item factor matrix V** (movies × k latent factors)

Such that: `R ≈ U × Vᵀ`

**Best Configuration:**
| Parameter | Value |
|---|---|
| Rank | 50 |
| Max Iterations | 10 |
| Regularization (λ) | 0.1 |
| Cold Start Strategy | drop |

### 3. Hyperparameter Tuning
Grid search over:
- **Ranks:** [10, 30, 50]
- **Regularization:** [0.01, 0.1, 0.5]

Best result: `rank=10, regParam=0.1, RMSE=0.8107`

### 4. Clustering
- **User Clustering (K=5):** Elbow method applied on user latent factors
- **Movie Clustering (K=10):** Elbow method applied on item latent factors
- PCA used to project 50-dimensional factors into 2D for visualization

---

## 📊 Results

### Model Performance
| Metric | Value |
|---|---|
| RMSE | 0.8110 |
| MAE | 0.6349 |

### User Clusters
| Cluster | Profile |
|---|---|
| 0 | Generous Raters — high avg ratings, mainstream preferences |
| 1 | Critical Viewers — lower avg ratings, discerning taste |
| 2 | Power Users — highly active, large number of ratings |
| 3 | Casual Users — low activity, fewer ratings |
| 4 | Niche Enthusiasts — specific genre preferences |

### Sample Recommendations (User 1)
| Movie | Genres | Predicted Rating |
|---|---|---|
| Lord of the Rings: Fellowship of the Ring (2001) | Adventure\|Fantasy | 4.21 |
| Shawshank Redemption, The (1994) | Crime\|Drama | 4.14 |
| Star Wars: Episode IV - A New Hope (1977) | Action\|Adventure\|Sci-Fi | 4.16 |
| Matrix, The (1999) | Action\|Sci-Fi\|Thriller | 4.13 |

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **Big Data** | Apache Spark (PySpark), Spark MLlib |
| **Cloud** | Google Cloud Platform — Dataproc, GCS |
| **ML** | ALS (Collaborative Filtering), K-Means Clustering |
| **Visualization** | Matplotlib, Seaborn |
| **Dimensionality Reduction** | PCA (sklearn) |
| **Data Processing** | PySpark SQL, Pandas, NumPy |
| **Environment** | Jupyter Notebook (via Dataproc Component Gateway) |

---

## GCP Setup

### Cloud Storage
- Bucket: `dsa5101-proj-bucket`
- Contains: `ratings.csv`, `movies.csv`, trained model checkpoints, parquet files

### Dataproc Cluster Config
| Setting | Value |
|---|---|
| Region | asia-southeast1 (Singapore) |
| Master Node | n4-standard-2 |
| Worker Nodes | 2 × n4-standard-2 |
| Disk | 50GB hyperdisk-balanced |
| Components | Jupyter |

---

## Repository Structure

```
├── notebooks/
│   └── recommendation_system.ipynb   # Main Jupyter notebook
├── data/
│   ├── ratings.csv                   # MovieLens ratings
│   └── movies.csv                    # Movie metadata
├── models/                           # Saved ALS model checkpoints
├── outputs/                          # Visualizations and results
└── README.md
```

---

## Limitations

- **Cold Start:** Cannot recommend for brand new users or unseen movies
- **Popularity Bias:** Tends to favour highly-rated popular movies
- **No Content Features:** Doesn't use movie metadata (cast, director, etc.)
- **Static Preferences:** No temporal modelling of changing user tastes

---

## Future Work

- Hybrid model incorporating content-based features (genres, actors, directors)
- Temporal dynamics to capture evolving user preferences
- Real-time recommendation API deployment
- Improved cold start handling via knowledge-based methods
- Diversity and serendipity enhancements

---

## References

1. Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context.* ACM TIIS, 5(4).
2. Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems.* Computer, 42(8).
3. [Apache Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
4. [GroupLens Research — MovieLens](https://grouplens.org/datasets/movielens/)

---

## Author

**Akshat Atul Bhargava**  
M.Sc. Data Science — National University of Singapore (NUS)
