"""
Reusable Geographic Location Filtering Engine for AirSpark.
Provides consistent hierarchy filtering (Country -> State -> City -> Monitoring Station),
scope labeling, and dynamic geospatial map viewport calculations.
"""
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np


class LocationFilter:
    """Unified geographic hierarchy filtering and resolution engine."""

    @staticmethod
    def filter_by_country(df: pd.DataFrame) -> pd.DataFrame:
        """Return full nationwide dataset without geographical filtering."""
        if df is None or df.empty:
            return pd.DataFrame()
        return df.copy()

    @staticmethod
    def filter_by_state(df: pd.DataFrame, state: Optional[str]) -> pd.DataFrame:
        """Filter dataset by Indian State / Union Territory."""
        if df is None or df.empty or not state or state in ["All", "All India", "National"]:
            return df if df is not None else pd.DataFrame()
        
        if "state" in df.columns:
            return df[df["state"].astype(str).str.lower() == str(state).lower()].copy()
        return df.copy()

    @staticmethod
    def filter_by_city(df: pd.DataFrame, city: Optional[str], state: Optional[str] = None) -> pd.DataFrame:
        """Filter dataset by City, with optional state constraint."""
        if df is None or df.empty:
            return pd.DataFrame()

        filtered = df.copy()
        if state and state not in ["All", "All India", "National"] and "state" in filtered.columns:
            filtered = filtered[filtered["state"].astype(str).str.lower() == str(state).lower()]

        if city and city not in ["All", "All Cities"]:
            if "city" in filtered.columns:
                filtered = filtered[filtered["city"].astype(str).str.lower() == str(city).lower()]

        return filtered

    @staticmethod
    def filter_by_station(df: pd.DataFrame, station_id: Optional[str]) -> pd.DataFrame:
        """Filter dataset by specific Station ID."""
        if df is None or df.empty or not station_id or station_id in ["All", "All Stations"]:
            return df if df is not None else pd.DataFrame()

        if "station_id" in df.columns:
            return df[df["station_id"].astype(str) == str(station_id)].copy()
        return df.copy()

    @classmethod
    def filter_by_scope(
        cls,
        df: pd.DataFrame,
        state: Optional[str] = None,
        city: Optional[str] = None,
        station_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply hierarchical filter: Country -> State -> City -> Station.
        Works across all AirSpark tabular datasets (integrated, stations, daily, hourly, hotspots, predictions).
        """
        if df is None or df.empty:
            return pd.DataFrame()

        filtered = df.copy()

        # 1. State filter
        if state and state not in ["All", "All India", "National"]:
            if "state" in filtered.columns:
                filtered = filtered[filtered["state"].astype(str).str.lower() == str(state).lower()]

        # 2. City filter
        if city and city not in ["All", "All Cities"]:
            if "city" in filtered.columns:
                filtered = filtered[filtered["city"].astype(str).str.lower() == str(city).lower()]

        # 3. Station filter
        if station_id and station_id not in ["All", "All Stations"]:
            if "station_id" in filtered.columns:
                filtered = filtered[filtered["station_id"].astype(str) == str(station_id)]

        return filtered

    @staticmethod
    def build_hierarchy_tree(stations_df: pd.DataFrame) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        """
        Construct structured hierarchy tree: State -> City -> List of Stations.
        """
        if stations_df is None or stations_df.empty:
            return {}

        tree: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
        for _, row in stations_df.iterrows():
            stn_state = str(row.get("state", "Other")).strip()
            stn_city = str(row.get("city", "Other")).strip()
            stn_id = str(row.get("station_id", "")).strip()
            stn_name = str(row.get("station_name", stn_id)).strip()

            if stn_state not in tree:
                tree[stn_state] = {}
            if stn_city not in tree[stn_state]:
                tree[stn_state][stn_city] = []

            tree[stn_state][stn_city].append({
                "station_id": stn_id,
                "station_name": stn_name
            })

        return tree

    @staticmethod
    def format_scope_label(
        state: Optional[str] = None,
        city: Optional[str] = None,
        station_id: Optional[str] = None,
        stations_df: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate a clear, human-readable breadcrumb label for the active geographic scope.
        Example: 'India ➔ Maharashtra ➔ Mumbai ➔ Bandra Kurla Complex (BKC) Station'
        """
        parts = ["🇮🇳 India"]

        if state and state not in ["All", "All India", "National"]:
            parts.append(state)

        if city and city not in ["All", "All Cities"]:
            parts.append(city)

        if station_id and station_id not in ["All", "All Stations"]:
            station_name = station_id
            if stations_df is not None and not stations_df.empty and "station_id" in stations_df.columns:
                match = stations_df[stations_df["station_id"] == station_id]
                if not match.empty and "station_name" in match.columns:
                    station_name = f"{match.iloc[0]['station_name']} ({station_id})"
            parts.append(station_name)
        elif len(parts) > 1 and (not station_id or station_id == "All Stations"):
            parts.append("All Monitoring Stations")

        return " ➔ ".join(parts)

    @staticmethod
    def format_scope_title(
        state: Optional[str] = None,
        city: Optional[str] = None,
        station_id: Optional[str] = None,
        stations_df: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate a concise scope title for chart headings.
        Examples: 'All India', 'Maharashtra', 'Mumbai', 'Bandra Kurla Complex Station'.
        """
        if station_id and station_id not in ["All", "All Stations"]:
            if stations_df is not None and not stations_df.empty and "station_id" in stations_df.columns:
                match = stations_df[stations_df["station_id"] == station_id]
                if not match.empty and "station_name" in match.columns:
                    return f"{match.iloc[0]['station_name']}"
            return f"Station {station_id}"
        elif city and city not in ["All", "All Cities"]:
            return f"{city}"
        elif state and state not in ["All", "All India", "National"]:
            return f"{state}"
        return "All India"

    @staticmethod
    def format_ml_scope(
        state: Optional[str] = None,
        city: Optional[str] = None,
        station_id: Optional[str] = None,
        stations_df: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate ML prediction scope string (e.g. 'Maharashtra → Mumbai' or 'India').
        """
        parts = []
        if not state or state in ["All", "All India", "National"]:
            parts.append("India")
        else:
            parts.append(state)
            if city and city not in ["All", "All Cities"]:
                parts.append(city)
                if station_id and station_id not in ["All", "All Stations"]:
                    if stations_df is not None and not stations_df.empty and "station_id" in stations_df.columns:
                        match = stations_df[stations_df["station_id"] == station_id]
                        if not match.empty and "station_name" in match.columns:
                            parts.append(f"{match.iloc[0]['station_name']}")
                        else:
                            parts.append(str(station_id))
                    else:
                        parts.append(str(station_id))
        return " → ".join(parts)

    @staticmethod
    def get_map_viewport(filtered_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute optimal map centroid coordinates and zoom level based on filtered station dispersion.
        """
        default_viewport = {"lat": 22.5937, "lon": 78.9629, "zoom": 4.1}  # Center of India

        if filtered_df is None or filtered_df.empty or "latitude" not in filtered_df.columns or "longitude" not in filtered_df.columns:
            return default_viewport

        valid_coords = filtered_df.dropna(subset=["latitude", "longitude"])
        valid_coords = valid_coords[
            valid_coords["latitude"].between(-90, 90) & valid_coords["longitude"].between(-180, 180)
        ]

        if valid_coords.empty:
            return default_viewport

        min_lat, max_lat = valid_coords["latitude"].min(), valid_coords["latitude"].max()
        min_lon, max_lon = valid_coords["longitude"].min(), valid_coords["longitude"].max()

        center_lat = float((min_lat + max_lat) / 2.0)
        center_lon = float((min_lon + max_lon) / 2.0)

        lat_spread = max_lat - min_lat
        lon_spread = max_lon - min_lon
        max_spread = max(lat_spread, lon_spread)

        # Dynamic zoom calculation based on coordinate bounding box
        if len(valid_coords) == 1 or max_spread < 0.05:
            zoom = 12.5  # Single station / hyperlocal
        elif max_spread < 0.4:
            zoom = 10.5  # City level
        elif max_spread < 2.0:
            zoom = 7.5   # Metro region
        elif max_spread < 6.0:
            zoom = 5.8   # State level
        elif max_spread < 15.0:
            zoom = 4.8   # Regional zone
        else:
            zoom = 4.1   # National view

        return {"lat": center_lat, "lon": center_lon, "zoom": zoom}


# Module level convenience functions
filter_by_country = LocationFilter.filter_by_country
filter_by_state = LocationFilter.filter_by_state
filter_by_city = LocationFilter.filter_by_city
filter_by_station = LocationFilter.filter_by_station
filter_by_scope = LocationFilter.filter_by_scope
