"""
PySpark Schemas and definitions for AirSpark data sources.
"""
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
    IntegerType,
)

# Raw Air Quality Schema
AIR_QUALITY_SCHEMA = StructType([
    StructField("station_id", StringType(), False),
    StructField("timestamp", StringType(), False),
    StructField("pm2_5", DoubleType(), True),
    StructField("pm10", DoubleType(), True),
    StructField("no2", DoubleType(), True),
    StructField("so2", DoubleType(), True),
    StructField("co", DoubleType(), True),
    StructField("o3", DoubleType(), True),
])

# Raw Weather Schema
WEATHER_SCHEMA = StructType([
    StructField("station_id", StringType(), False),
    StructField("timestamp", StringType(), False),
    StructField("temperature", DoubleType(), True),
    StructField("humidity", DoubleType(), True),
    StructField("wind_speed", DoubleType(), True),
    StructField("wind_direction", DoubleType(), True),
    StructField("pressure", DoubleType(), True),
    StructField("rainfall", DoubleType(), True),
])

# Station Metadata Schema
STATION_METADATA_SCHEMA = StructType([
    StructField("station_id", StringType(), False),
    StructField("station_name", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("latitude", DoubleType(), False),
    StructField("longitude", DoubleType(), False),
    StructField("elevation", DoubleType(), True),
    StructField("station_type", StringType(), True),
])

# Required Columns for Validation
REQUIRED_AIR_QUALITY_COLS = ["station_id", "timestamp"]
POLLUTANT_COLS = ["pm2_5", "pm10", "no2", "so2", "co", "o3"]
WEATHER_COLS = ["temperature", "humidity", "wind_speed", "wind_direction", "pressure", "rainfall"]
REQUIRED_STATION_COLS = ["station_id", "latitude", "longitude", "city"]


class SparkSchemaRegistry:
    """Registry providing standard PySpark schemas across AirSpark."""

    @staticmethod
    def get_air_quality_schema() -> StructType:
        return AIR_QUALITY_SCHEMA

    @staticmethod
    def get_weather_schema() -> StructType:
        return WEATHER_SCHEMA

    @staticmethod
    def get_station_metadata_schema() -> StructType:
        return STATION_METADATA_SCHEMA
