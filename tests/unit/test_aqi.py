"""
Unit tests for the standard AQI calculation engine and breakpoint logic.
"""
from pyspark.sql import SparkSession
from app.aqi.breakpoints import calculate_sub_index
from app.aqi.categories import get_aqi_category_info, get_category_name
from app.aqi.calculator import AQICalculator


def test_epa_pm25_breakpoints():
    # Boundary 0.0 -> 0.0
    assert calculate_sub_index(0.0, "pm2_5", "US_EPA") == 0.0
    # Good upper breakpoint: 9.0 -> 50.0
    assert calculate_sub_index(9.0, "pm2_5", "US_EPA") == 50.0
    # Immediately above Good: 9.1 -> 51.0
    assert calculate_sub_index(9.1, "pm2_5", "US_EPA") == 51.0
    # Moderate upper breakpoint: 35.4 -> 100.0
    assert calculate_sub_index(35.4, "pm2_5", "US_EPA") == 100.0
    # Immediately above Moderate: 35.5 -> 101.0
    assert calculate_sub_index(35.5, "pm2_5", "US_EPA") == 101.0
    # USG upper breakpoint: 55.4 -> 150.0
    assert calculate_sub_index(55.4, "pm2_5", "US_EPA") == 150.0
    # Immediately above USG: 55.5 -> 151.0
    assert calculate_sub_index(55.5, "pm2_5", "US_EPA") == 151.0
    # Unhealthy upper breakpoint: 125.4 -> 200.0
    assert calculate_sub_index(125.4, "pm2_5", "US_EPA") == 200.0
    # Immediately above Unhealthy: 125.5 -> 201.0
    assert calculate_sub_index(125.5, "pm2_5", "US_EPA") == 201.0
    # Very Unhealthy upper breakpoint: 225.4 -> 300.0
    assert calculate_sub_index(225.4, "pm2_5", "US_EPA") == 300.0
    # Immediately above Very Unhealthy: 225.5 -> 301.0
    assert calculate_sub_index(225.5, "pm2_5", "US_EPA") == 301.0
    # Hazardous tier 1 upper breakpoint: 325.4 -> 400.0
    assert calculate_sub_index(325.4, "pm2_5", "US_EPA") == 400.0
    # Hazardous tier 2 lower breakpoint: 325.5 -> 401.0
    assert calculate_sub_index(325.5, "pm2_5", "US_EPA") == 401.0
    # Highest standard breakpoint: 500.4 -> 500.0
    assert calculate_sub_index(500.4, "pm2_5", "US_EPA") == 500.0


def test_aqi_categories():
    assert get_category_name(35.0, "US_EPA") == "Good"
    assert get_category_name(75.0, "US_EPA") == "Moderate"
    assert get_category_name(125.0, "US_EPA") == "Unhealthy for Sensitive Groups"
    assert get_category_name(175.0, "US_EPA") == "Unhealthy"
    assert get_category_name(250.0, "US_EPA") == "Very Unhealthy"
    assert get_category_name(350.0, "US_EPA") == "Hazardous"


def test_distributed_aqi_calculator(spark_session: SparkSession):
    data = [
        ("STN_001", "2026-01-01 00:00:00", 9.0, 54.0, 20.0, 10.0, 1.0, 20.0),   # AQI = 50 (Good)
        ("STN_001", "2026-01-01 01:00:00", 35.4, 20.0, 10.0, 5.0, 0.5, 10.0),   # PM2.5 dominates: AQI = 100 (Moderate)
        ("STN_001", "2026-01-01 02:00:00", 5.0, 254.0, 10.0, 5.0, 0.5, 10.0),   # PM10 dominates: AQI = 150 (USG)
    ]
    df = spark_session.createDataFrame(data, ["station_id", "timestamp", "pm2_5", "pm10", "no2", "so2", "co", "o3"])

    calc = AQICalculator(standard="US_EPA")
    result_df = calc.calculate_aqi_df(df)
    results = result_df.orderBy("timestamp").collect()

    # Row 0: AQI = 50, Good
    assert results[0]["aqi"] == 50.0
    assert results[0]["aqi_category"] == "Good"

    # Row 1: PM2.5 is dominant (AQI = 100)
    assert results[1]["aqi"] == 100.0
    assert results[1]["dominant_pollutant"] == "PM2.5"
    assert results[1]["aqi_category"] == "Moderate"

    # Row 2: PM10 is dominant (AQI = 150)
    assert results[2]["aqi"] == 150.0
    assert results[2]["dominant_pollutant"] == "PM10"
    assert results[2]["aqi_category"] == "Unhealthy for Sensitive Groups"


def test_cpcb_breakpoints():
    # India CPCB PM2.5: 0-30 -> 0-50, 31-60 -> 51-100
    assert calculate_sub_index(30.0, "pm2_5", "INDIA_CPCB") == 50.0
    assert calculate_sub_index(60.0, "pm2_5", "INDIA_CPCB") == 100.0
    assert get_category_name(45.0, "INDIA_CPCB") == "Good"
    assert get_category_name(75.0, "INDIA_CPCB") == "Satisfactory"


def test_out_of_range_aqi_handling(spark_session: SparkSession):
    # Extreme PM2.5 concentration beyond EPA breakpoint (e.g., 850 ug/m3)
    data = [
        ("STN_001", "2026-01-01 00:00:00", 850.0, 50.0, 10.0, 5.0, 0.5, 20.0),
    ]
    df = spark_session.createDataFrame(data, ["station_id", "timestamp", "pm2_5", "pm10", "no2", "so2", "co", "o3"])
    
    calc = AQICalculator(standard="US_EPA")
    result_df = calc.calculate_aqi_df(df)
    row = result_df.collect()[0]
    
    # Standard AQI must strictly not exceed 500
    assert row["aqi"] == 500.0
    assert row["is_aqi_out_of_range"] is True
    assert row["aqi_exceedance_flag"] is True
    assert row["extrapolated_aqi"] > 500.0
    assert row["aqi_category"] == "Hazardous"


