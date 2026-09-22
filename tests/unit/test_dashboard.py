"""
Unit tests for the AirSpark interactive analytics dashboard module.
Validates branding, data loaders, map builders, and fault tolerance.
"""
import pytest
import pandas as pd
import plotly.graph_objects as go
from dashboard.app import (
    PROJECT_NAME,
    PROJECT_TAGLINE,
    build_spatial_map,
    load_pipeline_summary,
    load_station_summary,
    load_daily_analytics,
    load_hourly_profile,
    load_hotspots,
    load_integrated_aqi,
    load_ml_metrics,
    load_predictions,
    load_benchmarks,
    load_partitioning_results,
    load_streaming_data,
)


def test_dashboard_branding_constants():
    """Verify official project name and tagline compliance."""
    assert PROJECT_NAME == "AirSpark"
    assert PROJECT_TAGLINE == (
        "A Scalable Big Data Analytics Framework for Multi-Source Air Quality Monitoring "
        "and Spatio-Temporal AQI Analysis Using Apache Spark"
    )


def test_build_spatial_map_with_valid_data():
    """Verify build_spatial_map constructs a valid Plotly Figure with modern MapLibre API."""
    sample_df = pd.DataFrame({
        "station_id": ["STN_001", "STN_002"],
        "station_name": ["Central Sensor #1", "Central Sensor #2"],
        "city": ["New Delhi", "Mumbai"],
        "latitude": [28.6139, 19.0760],
        "longitude": [77.2090, 72.8777],
        "mean_aqi": [185.5, 95.2],
        "max_aqi": [230.0, 120.0],
        "hotspot_score": [75.0, 30.0],
        "hotspot_level": ["Severe Hotspot", "Moderate Hotspot"],
        "unhealthy_pct": [65.0, 15.0]
    })

    fig = build_spatial_map(sample_df)
    assert fig is not None
    assert isinstance(fig, go.Figure)


def test_build_spatial_map_with_empty_or_invalid_data():
    """Verify build_spatial_map handles empty or invalid coordinate data gracefully."""
    # Empty DataFrame
    assert build_spatial_map(pd.DataFrame()) is None

    # Missing coordinates
    no_coord_df = pd.DataFrame({"station_id": ["STN_001"], "mean_aqi": [100.0]})
    assert build_spatial_map(no_coord_df) is None

    # Invalid / NaN coordinates
    nan_coord_df = pd.DataFrame({
        "station_id": ["STN_001"],
        "latitude": [None],
        "longitude": [None],
        "mean_aqi": [100.0]
    })
    assert build_spatial_map(nan_coord_df) is None

    # Out of range coordinates
    out_of_bounds_df = pd.DataFrame({
        "station_id": ["STN_001"],
        "latitude": [999.0],
        "longitude": [-999.0],
        "mean_aqi": [100.0]
    })
    assert build_spatial_map(out_of_bounds_df) is None


def test_dashboard_data_loaders():
    """Verify data loaders execute cleanly and return valid types."""
    summary = load_pipeline_summary()
    assert summary is None or isinstance(summary, dict)

    stations = load_station_summary()
    assert isinstance(stations, pd.DataFrame)

    daily = load_daily_analytics()
    assert isinstance(daily, pd.DataFrame)

    hourly = load_hourly_profile()
    assert isinstance(hourly, pd.DataFrame)

    hotspots = load_hotspots()
    assert isinstance(hotspots, pd.DataFrame)

    integrated = load_integrated_aqi()
    assert isinstance(integrated, pd.DataFrame)

    ml_metrics = load_ml_metrics()
    assert ml_metrics is None or isinstance(ml_metrics, dict)

    preds = load_predictions()
    assert isinstance(preds, pd.DataFrame)

    benchmarks = load_benchmarks()
    assert isinstance(benchmarks, list)

    partitioning = load_partitioning_results()
    assert isinstance(partitioning, list)

    alerts, win_stats, latest_stream = load_streaming_data()
    assert alerts is None or isinstance(alerts, dict)
    assert isinstance(win_stats, pd.DataFrame)
    assert isinstance(latest_stream, pd.DataFrame)
