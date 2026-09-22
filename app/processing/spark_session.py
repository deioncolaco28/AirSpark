"""
SparkSession manager and factory for AirSpark.
Configured for local development on Windows / Linux / macOS with optimal defaults.
"""
import os
import sys
from typing import Optional
from pyspark.sql import SparkSession
from app.config.settings import get_settings
from app.utils.logging import get_logger
from app.utils.paths import ensure_dir

logger = get_logger("AirSpark.SparkSession")


class SparkSessionManager:
    """Manages the creation and lifecycle of the PySpark SparkSession."""
    _spark: Optional[SparkSession] = None

    @classmethod
    def get_session(cls, app_name: Optional[str] = None, master: Optional[str] = None) -> SparkSession:
        """
        Get existing SparkSession or initialize a new one with standard configurations.
        """
        if cls._spark is not None:
            try:
                if not cls._spark.sparkContext._jsc.sc().isStopped():
                    return cls._spark
            except Exception:
                pass

        # Configure environment for Windows Python worker compatibility
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        settings = get_settings()
        cfg = settings.spark_config
        
        name = app_name or cfg.get("app_name", "AirSpark-BDA-Engine")
        master_url = master or cfg.get("master", "local[*]")
        warehouse_dir = ensure_dir(settings.paths.get("staging_dir", "data/staging") + "/spark-warehouse")

        builder = (
            SparkSession.builder
            .appName(name)
            .master(master_url)
            .config("spark.driver.memory", cfg.get("driver_memory", "4g"))
            .config("spark.executor.memory", cfg.get("executor_memory", "2g"))
            .config("spark.sql.shuffle.partitions", str(cfg.get("shuffle_partitions", 4)))
            .config("spark.default.parallelism", str(cfg.get("default_parallelism", 4)))
            .config("spark.sql.adaptive.enabled", str(cfg.get("adaptive_execution_enabled", True)).lower())
            .config("spark.sql.warehouse.dir", str(warehouse_dir.as_uri()))
            .config("spark.hadoop.io.native.lib.available", "false")
            .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
        )

        logger.info(f"Initializing PySpark SparkSession [AppName: {name}, Master: {master_url}]")
        cls._spark = builder.getOrCreate()
        cls._spark.sparkContext.setLogLevel(cfg.get("log_level", "WARN"))
        
        return cls._spark

    @classmethod
    def stop_session(cls) -> None:
        """Gracefully stop the SparkSession."""
        if cls._spark is not None:
            try:
                logger.info("Stopping PySpark SparkSession")
                cls._spark.stop()
            except Exception as e:
                logger.warning(f"Error while stopping SparkSession: {e}")
            finally:
                cls._spark = None


def get_spark(app_name: Optional[str] = None) -> SparkSession:
    """Convenience function to get the current SparkSession."""
    return SparkSessionManager.get_session(app_name=app_name)
