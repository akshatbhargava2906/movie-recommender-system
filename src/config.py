from pyspark.sql import SparkSession


def create_spark_session(app_name="MovieLens Recommender System",
                         driver_memory="8g",
                         executor_memory="8g",
                         shuffle_partitions=200):
    """Create and configure a Spark session."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print(f"Spark Session created successfully!")
    print(f"Spark Version: {spark.version}")
    print(f"Driver Memory: {driver_memory}")
    print(f"Executor Memory: {executor_memory}")
    print(f"Application Name: {spark.sparkContext.appName}")

    return spark
