"""
Unit tests for AirSpark Reusable Location Filtering Engine.
"""
import pytest
import pandas as pd
from app.analytics.location_filter import (
    LocationFilter,
    filter_by_country,
    filter_by_state,
    filter_by_city,
    filter_by_station,
    filter_by_scope,
)


@pytest.fixture
def sample_location_df():
    return pd.DataFrame([
        {"station_id": "IND_DL_001", "station_name": "Anand Vihar", "city": "Delhi", "state": "Delhi", "latitude": 28.6469, "longitude": 77.3160, "aqi": 250},
        {"station_id": "IND_DL_002", "station_name": "ITO", "city": "Delhi", "state": "Delhi", "latitude": 28.6312, "longitude": 77.2494, "aqi": 210},
        {"station_id": "IND_MH_001", "station_name": "BKC", "city": "Mumbai", "state": "Maharashtra", "latitude": 19.0657, "longitude": 72.8687, "aqi": 140},
        {"station_id": "IND_MH_002", "station_name": "Worli", "city": "Mumbai", "state": "Maharashtra", "latitude": 19.0166, "longitude": 72.8168, "aqi": 90},
        {"station_id": "IND_MH_004", "station_name": "Shivaji Nagar", "city": "Pune", "state": "Maharashtra", "latitude": 18.5314, "longitude": 73.8446, "aqi": 110},
        {"station_id": "IND_KA_001", "station_name": "Silk Board", "city": "Bengaluru", "state": "Karnataka", "latitude": 12.9172, "longitude": 77.6229, "aqi": 85},
    ])


def test_filter_by_country(sample_location_df):
    result = filter_by_country(sample_location_df)
    assert len(result) == 6


def test_filter_by_state(sample_location_df):
    mh = filter_by_state(sample_location_df, "Maharashtra")
    assert len(mh) == 3
    assert set(mh["city"].unique()) == {"Mumbai", "Pune"}

    dl = filter_by_state(sample_location_df, "Delhi")
    assert len(dl) == 2

    # All India / None check
    all_res = filter_by_state(sample_location_df, "All India")
    assert len(all_res) == 6


def test_filter_by_city(sample_location_df):
    mumbai = filter_by_city(sample_location_df, "Mumbai")
    assert len(mumbai) == 2
    assert set(mumbai["station_id"].unique()) == {"IND_MH_001", "IND_MH_002"}

    pune = filter_by_city(sample_location_df, "Pune", state="Maharashtra")
    assert len(pune) == 1
    assert pune.iloc[0]["station_id"] == "IND_MH_004"


def test_filter_by_station(sample_location_df):
    stn = filter_by_station(sample_location_df, "IND_KA_001")
    assert len(stn) == 1
    assert stn.iloc[0]["city"] == "Bengaluru"


def test_filter_by_scope_hierarchy(sample_location_df):
    # State only
    res1 = filter_by_scope(sample_location_df, state="Maharashtra")
    assert len(res1) == 3

    # State + City
    res2 = filter_by_scope(sample_location_df, state="Maharashtra", city="Mumbai")
    assert len(res2) == 2

    # State + City + Station
    res3 = filter_by_scope(sample_location_df, state="Maharashtra", city="Mumbai", station_id="IND_MH_001")
    assert len(res3) == 1
    assert res3.iloc[0]["station_name"] == "BKC"


def test_build_hierarchy_tree(sample_location_df):
    tree = LocationFilter.build_hierarchy_tree(sample_location_df)
    assert "Maharashtra" in tree
    assert "Delhi" in tree
    assert "Karnataka" in tree
    assert "Mumbai" in tree["Maharashtra"]
    assert "Pune" in tree["Maharashtra"]
    assert len(tree["Maharashtra"]["Mumbai"]) == 2


def test_format_scope_label(sample_location_df):
    label_india = LocationFilter.format_scope_label()
    assert "India" in label_india

    label_state = LocationFilter.format_scope_label(state="Maharashtra")
    assert "Maharashtra" in label_state

    label_stn = LocationFilter.format_scope_label(state="Maharashtra", city="Mumbai", station_id="IND_MH_001", stations_df=sample_location_df)
    assert "Mumbai" in label_stn
    assert "BKC" in label_stn or "IND_MH_001" in label_stn


def test_format_scope_title_and_ml_scope(sample_location_df):
    # National / Default
    assert LocationFilter.format_scope_title() == "All India"
    assert LocationFilter.format_ml_scope() == "India"

    # State Scope
    assert LocationFilter.format_scope_title(state="Maharashtra") == "Maharashtra"
    assert LocationFilter.format_ml_scope(state="Maharashtra") == "Maharashtra"

    # City Scope
    assert LocationFilter.format_scope_title(state="Maharashtra", city="Mumbai") == "Mumbai"
    assert LocationFilter.format_ml_scope(state="Maharashtra", city="Mumbai") == "Maharashtra → Mumbai"

    # Station Scope
    assert LocationFilter.format_scope_title(state="Maharashtra", city="Mumbai", station_id="IND_MH_001", stations_df=sample_location_df) == "BKC"
    assert LocationFilter.format_ml_scope(state="Maharashtra", city="Mumbai", station_id="IND_MH_001", stations_df=sample_location_df) == "Maharashtra → Mumbai → BKC"


def test_get_map_viewport(sample_location_df):
    vp = LocationFilter.get_map_viewport(sample_location_df)
    assert "lat" in vp
    assert "lon" in vp
    assert "zoom" in vp
    assert 10.0 <= vp["lat"] <= 35.0
    assert 68.0 <= vp["lon"] <= 90.0

