"""
Pytest configuration and shared fixtures for AirSpark tests.
"""
import os
import sys
from pathlib import Path
import pytest
from pyspark.sql import SparkSession

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows PySpark Python environment variables
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


@pytest.fixture(scope="session")
def spark_session():
    """Shared local SparkSession fixture for testing."""
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("AirSpark-Test-Suite")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()
