"""
Unit tests for India-scale dataset ingestion and schema compliance.
"""
import pytest
import pandas as pd
from app.ingestion.india_dataset import IndiaScaleDataIngestor, INDIA_STATIONS_REGISTRY
from app.processing.schemas import REQUIRED_STATION_COLS, POLLUTANT_COLS, WEATHER_COLS


def test_india_stations_registry_validity():
    """Verify registry contains valid Indian stations, states, cities, and coordinates."""
    assert len(INDIA_STATIONS_REGISTRY) >= 30
    df = pd.DataFrame(INDIA_STATIONS_REGISTRY)

    # Check required columns
    for col in ["station_id", "station_name", "city", "state", "latitude", "longitude", "station_type"]:
        assert col in df.columns

    # Verify geographical bounds of India
    assert df["latitude"].between(6.0, 38.0).all()
    assert df["longitude"].between(68.0, 98.0).all()
    assert df["station_id"].is_unique


def test_india_scale_dataset_generation():
    """Verify IndiaScaleDataIngestor generates compliant multi-source DataFrames."""
    ingestor = IndiaScaleDataIngestor(seed=123)
    aq_df, w_df, st_df = ingestor.generate_datasets(days=2, frequency_minutes=60, inject_quality_issues=False)

    assert len(st_df) == len(INDIA_STATIONS_REGISTRY)
    assert len(aq_df) > 0
    assert len(w_df) > 0

    # Verify pollutant columns
    for pol in POLLUTANT_COLS:
        assert pol in aq_df.columns

    # Verify weather columns
    for w in WEATHER_COLS:
        assert w in w_df.columns
