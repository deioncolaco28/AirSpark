"""
Unit tests for the AirSpark interactive analytics dashboard module.
Validates branding, multi-tier data loaders, map builders, location filtering, and fault tolerance.
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
        "station_id": ["IND_DL_001", "IND_MH_001"],
        "station_name": ["Anand Vihar Station", "BKC Station"],
        "city": ["Delhi", "Mumbai"],
        "state": ["Delhi", "Maharashtra"],
        "latitude": [28.6469, 19.0657],
        "longitude": [77.3160, 72.8687],
        "mean_aqi": [245.5, 135.2],
        "max_aqi": [350.0, 180.0],
        "hotspot_score": [78.0, 42.0],
        "hotspot_level": ["Critical Hotspot", "Moderate Hotspot"],
        "unhealthy_pct": [72.0, 25.0]
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


def test_dashboard_data_loaders_for_both_tiers():
    """Verify data loaders execute cleanly for both India-scale and Dev dataset tiers."""
    for tier in ["india", "dev"]:
        summary = load_pipeline_summary(dataset_tier=tier)
        assert summary is None or isinstance(summary, dict)

        stations = load_station_summary(dataset_tier=tier)
        assert isinstance(stations, pd.DataFrame)

        daily = load_daily_analytics(dataset_tier=tier)
        assert isinstance(daily, pd.DataFrame)

        hourly = load_hourly_profile(dataset_tier=tier)
        assert isinstance(hourly, pd.DataFrame)

        hotspots = load_hotspots(dataset_tier=tier)
        assert isinstance(hotspots, pd.DataFrame)

        integrated = load_integrated_aqi(dataset_tier=tier)
        assert isinstance(integrated, pd.DataFrame)

        ml_metrics = load_ml_metrics(dataset_tier=tier)
        assert ml_metrics is None or isinstance(ml_metrics, dict)

        preds = load_predictions(dataset_tier=tier)
        assert isinstance(preds, pd.DataFrame)

    benchmarks = load_benchmarks()
    assert isinstance(benchmarks, list)

    partitioning = load_partitioning_results()
    assert isinstance(partitioning, list)

    alerts, win_stats, latest_stream = load_streaming_data()
    assert alerts is None or isinstance(alerts, dict)
    assert isinstance(win_stats, pd.DataFrame)
    assert isinstance(latest_stream, pd.DataFrame)
