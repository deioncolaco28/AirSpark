"""
AirSpark Interactive Analytics & Monitoring Dashboard.
Built with Streamlit and Plotly for high-performance Big Data visualization.
"""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Project Identity & Official Constants ---
PROJECT_NAME = "AirSpark"
PROJECT_TAGLINE = "A Scalable Big Data Analytics Framework for Multi-Source Air Quality Monitoring and Spatio-Temporal AQI Analysis Using Apache Spark"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Color Scheme for Standard AQI Categories ---
AQI_CATEGORY_COLORS = {
    "Good": "#00E400",
    "Moderate": "#FFFF00",
    "Unhealthy for Sensitive Groups": "#FF7E00",
    "Unhealthy": "#FF0000",
    "Very Unhealthy": "#8F3F97",
    "Hazardous": "#7E0023",
    "Satisfactory": "#00E400",
    "Poor": "#FF7E00",
    "Very Poor": "#8F3F97",
    "Severe": "#7E0023"
}

# Set Streamlit page configuration
st.set_page_config(
    page_title=f"{PROJECT_NAME} - Big Data AQI Analytics",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1a1c23;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #00E400;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #1a1c23;
        border-radius: 6px 6px 0px 0px;
        gap: 1px;
        padding-top: 8px;
        padding-bottom: 8px;
    }
    .project-header {
        margin-bottom: 0.5rem;
    }
    .project-tagline {
        color: #90caf9;
        font-style: italic;
        margin-bottom: 1.5rem;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# DATA LOADING HELPERS (Modular, Cached, Resilient)
# =====================================================================

@st.cache_data(ttl=60)
def load_pipeline_summary() -> Optional[Dict[str, Any]]:
    """Load batch pipeline summary metadata."""
    summary_file = PROJECT_ROOT / "data" / "processed" / "pipeline_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


@st.cache_data(ttl=60)
def load_ml_metrics() -> Optional[Dict[str, Any]]:
    """Load Spark ML evaluation metrics."""
    ml_file = PROJECT_ROOT / "data" / "processed" / "ml_metrics.json"
    if ml_file.exists():
        try:
            with open(ml_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    """Load Spark ML predictions dataset."""
    pred_file = PROJECT_ROOT / "data" / "processed" / "aqi_predictions.parquet"
    if pred_file.exists():
        try:
            return pd.read_parquet(pred_file)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_station_summary() -> pd.DataFrame:
    """Load station summary dataset from processed parquet or fallback raw CSV."""
    proc_file = PROJECT_ROOT / "data" / "processed" / "station_summary.parquet"
    raw_file = PROJECT_ROOT / "data" / "raw" / "stations.csv"
    if proc_file.exists():
        try:
            return pd.read_parquet(proc_file)
        except Exception:
            pass
    if raw_file.exists():
        try:
            return pd.read_csv(raw_file)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_daily_analytics() -> pd.DataFrame:
    """Load daily temporal analytics dataset."""
    daily_file = PROJECT_ROOT / "data" / "processed" / "daily_analytics.parquet"
    if daily_file.exists():
        try:
            return pd.read_parquet(daily_file)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_hourly_profile() -> pd.DataFrame:
    """Load diurnal hourly profile dataset."""
    hourly_file = PROJECT_ROOT / "data" / "processed" / "hourly_profile.parquet"
    if hourly_file.exists():
        try:
            return pd.read_parquet(hourly_file)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_hotspots() -> pd.DataFrame:
    """Load spatial hotspot ranking dataset."""
    hotspots_file = PROJECT_ROOT / "data" / "processed" / "hotspots.parquet"
    if hotspots_file.exists():
        try:
            return pd.read_parquet(hotspots_file)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_integrated_aqi() -> pd.DataFrame:
    """Load integrated AQI dataset (sample or partitioned parquet)."""
    integrated_file = PROJECT_ROOT / "data" / "processed" / "integrated_aqi.parquet"
    if integrated_file.exists():
        try:
            return pd.read_parquet(integrated_file)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_benchmarks() -> List[Dict[str, Any]]:
    """Load benchmark and scalability results."""
    bench_file = PROJECT_ROOT / "data" / "benchmarks" / "benchmark_results.json"
    if bench_file.exists():
        try:
            with open(bench_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


@st.cache_data(ttl=60)
def load_partitioning_results() -> List[Dict[str, Any]]:
    """Load partition scalability benchmark results."""
    part_file = PROJECT_ROOT / "data" / "benchmarks" / "partitioning_results.json"
    if part_file.exists():
        try:
            with open(part_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


@st.cache_data(ttl=5)
def load_streaming_data() -> Tuple[Optional[Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Load latest streaming alerts and windowed statistics with low TTL for live updates."""
    stream_dir = PROJECT_ROOT / "data" / "streaming" / "output"
    alerts_file = stream_dir / "streaming_alerts.json"
    window_stats_file = stream_dir / "windowed_stats.csv"
    latest_stream_file = stream_dir / "latest_stream.csv"

    alerts_data = None
    if alerts_file.exists():
        try:
            with open(alerts_file, "r", encoding="utf-8") as f:
                alerts_data = json.load(f)
        except Exception:
            alerts_data = None

    win_df = pd.DataFrame()
    if window_stats_file.exists():
        try:
            win_df = pd.read_csv(window_stats_file)
        except Exception:
            win_df = pd.DataFrame()

    latest_df = pd.DataFrame()
    if latest_stream_file.exists():
        try:
            latest_df = pd.read_csv(latest_stream_file)
        except Exception:
            latest_df = pd.DataFrame()

    return alerts_data, win_df, latest_df


# =====================================================================
# GEOSPATIAL MAP VISUALIZATION HELPER
# Compatible with modern Plotly 7.1+ MapLibre API (px.scatter_map)
# =====================================================================

def build_spatial_map(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Construct a responsive geo-spatial scatter map using Plotly Express scatter_map.
    Safely validates coordinates and handles missing/empty data.
    """
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return None

    # Ensure numeric coordinates
    map_df = df.copy()
    map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
    map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
    valid_df = map_df.dropna(subset=["latitude", "longitude"])
    valid_df = valid_df[
        valid_df["latitude"].between(-90, 90) & valid_df["longitude"].between(-180, 180)
    ]

    if valid_df.empty:
        return None

    # Determine size and color attributes
    size_col = "hotspot_score" if "hotspot_score" in valid_df.columns else None
    if size_col and (valid_df[size_col].isna().all() or (valid_df[size_col] <= 0).all()):
        size_col = None

    hover_cols = [c for c in ["city", "mean_aqi", "max_aqi", "hotspot_level", "unhealthy_pct"] if c in valid_df.columns]
    hover_name = "station_name" if "station_name" in valid_df.columns else "station_id" if "station_id" in valid_df.columns else None

    # Center map on centroid of stations
    center_lat = float(valid_df["latitude"].mean())
    center_lon = float(valid_df["longitude"].mean())

    try:
        # Modern Plotly 7.1+ px.scatter_map using MapLibre
        fig_map = px.scatter_map(
            valid_df,
            lat="latitude",
            lon="longitude",
            color="mean_aqi" if "mean_aqi" in valid_df.columns else None,
            size=size_col,
            hover_name=hover_name,
            hover_data=hover_cols,
            color_continuous_scale="Reds",
            size_max=24,
            zoom=4.2,
            center=dict(lat=center_lat, lon=center_lon),
            map_style="carto-darkmatter",
            title="Monitoring Station Distribution & Hotspot Severity"
        )
        fig_map.update_layout(
            template="plotly_dark",
            height=520,
            margin=dict(r=0, t=40, l=0, b=0)
        )
        return fig_map
    except Exception:
        # Fallback to scatter_geo if map tiles fail to load
        try:
            fig_geo = px.scatter_geo(
                valid_df,
                lat="latitude",
                lon="longitude",
                color="mean_aqi" if "mean_aqi" in valid_df.columns else None,
                hover_name=hover_name,
                hover_data=hover_cols,
                color_continuous_scale="Reds",
                title="Monitoring Station Geographic Distribution (Geo View)"
            )
            fig_geo.update_layout(template="plotly_dark", height=520)
            return fig_geo
        except Exception:
            return None


# =====================================================================
# TAB RENDERING FUNCTIONS
# =====================================================================

def render_overview_tab(summary: Dict[str, Any], stations_df: pd.DataFrame, daily_df: pd.DataFrame):
    """Render Tab 1: Executive AQI Overview & Monitoring KPIs."""
    st.header("Executive AQI Overview & Monitoring KPIs")

    aq_report = summary.get("quality_report_aq", {})
    avg_aqi = round(float(stations_df["mean_aqi"].mean()), 1) if not stations_df.empty and "mean_aqi" in stations_df.columns else round(summary.get("aqi_statistics", {}).get("mean_aqi", 0.0), 1)
    max_aqi = round(float(stations_df["max_aqi"].max()), 1) if not stations_df.empty and "max_aqi" in stations_df.columns else round(summary.get("aqi_statistics", {}).get("max_aqi", 0.0), 1)
    num_stations = len(stations_df) if not stations_df.empty else summary.get("join_stats", {}).get("station_count", 0)

    # Dominant pollutant
    dom_list = summary.get("dominance_distribution", [])
    top_dominant = dom_list[0]["dominant_pollutant"] if dom_list else "PM2.5"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Average AQI", value=f"{avg_aqi}", delta=f"Standard: {summary.get('aqi_standard', 'US_EPA')}")
    with col2:
        st.metric(label="Peak AQI Observed", value=f"{max_aqi}", delta_color="inverse")
    with col3:
        st.metric(label="Monitoring Stations", value=f"{num_stations}")
    with col4:
        st.metric(label="Dominant Pollutant", value=f"{top_dominant}")
    with col5:
        retention = aq_report.get("retention_percentage", 100)
        st.metric(label="Data Retention", value=f"{retention}%")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Daily Mean AQI Trend")
        if not daily_df.empty and "date" in daily_df.columns and "mean_aqi" in daily_df.columns:
            daily_sorted = daily_df.sort_values("date")
            fig_daily = px.line(
                daily_sorted, x="date", y="mean_aqi",
                title="Daily Average AQI Progression Across Station Network",
                markers=True,
                color_discrete_sequence=["#00E400"],
                labels={"date": "Date", "mean_aqi": "Mean AQI"}
            )
            fig_daily.add_hline(y=50, line_dash="dot", line_color="#00E400", annotation_text="Good (50)", annotation_position="top left")
            fig_daily.add_hline(y=100, line_dash="dot", line_color="#FFFF00", annotation_text="Moderate (100)", annotation_position="top left")
            fig_daily.add_hline(y=150, line_dash="dot", line_color="#FF7E00", annotation_text="Unhealthy for Sensitive (150)", annotation_position="top left")
            fig_daily.add_hline(y=200, line_dash="dot", line_color="#FF0000", annotation_text="Unhealthy (200)", annotation_position="top left")
            fig_daily.update_layout(template="plotly_dark", height=390)
            st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.info("Daily trend data unavailable. Run the batch pipeline to generate temporal daily aggregations.")

    with col_right:
        st.subheader("Pollutant Dominance Breakdown")
        if dom_list:
            df_dom = pd.DataFrame(dom_list)
            fig_dom = px.pie(
                df_dom, names="dominant_pollutant", values="count",
                title="Share of Driving Pollutants Determining Final AQI",
                hole=0.42,
                color_discrete_sequence=px.colors.sequential.Teal
            )
            fig_dom.update_layout(template="plotly_dark", height=390)
            st.plotly_chart(fig_dom, use_container_width=True)
        else:
            st.info("Dominant pollutant breakdown unavailable.")

    # Station Summary Metrics Table
    if not stations_df.empty:
        st.subheader("Station Summary & Overall Metrics")
        display_cols = [c for c in ["station_id", "station_name", "city", "mean_aqi", "min_aqi", "max_aqi", "total_readings"] if c in stations_df.columns]
        st.dataframe(stations_df[display_cols], use_container_width=True)


def render_aqi_analysis_tab(summary: Dict[str, Any], integrated_df: pd.DataFrame):
    """Render Tab 2: Comprehensive AQI Analysis, Sub-Indices, and Bounded Exceedances."""
    st.header("Comprehensive AQI Sub-Indices & Categorization")

    cat_counts = pd.DataFrame()
    if not integrated_df.empty and "aqi_category" in integrated_df.columns:
        cat_counts = integrated_df["aqi_category"].value_counts().reset_index()
        cat_counts.columns = ["aqi_category", "count"]
    elif "category_distribution" in summary:
        cat_dist = summary["category_distribution"]
        cat_counts = pd.DataFrame(list(cat_dist.items()), columns=["aqi_category", "count"])

    aq_c1, aq_c2 = st.columns(2)

    with aq_c1:
        st.subheader("AQI Category Distribution")
        if not cat_counts.empty:
            fig_cat = px.bar(
                cat_counts, x="aqi_category", y="count",
                color="aqi_category",
                color_discrete_map=AQI_CATEGORY_COLORS,
                title="Observation Frequency per Official AQI Category",
                labels={"aqi_category": "AQI Category", "count": "Record Count"}
            )
            fig_cat.update_layout(template="plotly_dark", height=380, showlegend=False)
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("AQI category distribution data unavailable.")

    with aq_c2:
        st.subheader("Sub-Index vs Final AQI Distribution")
        sub_cols = [c for c in ["sub_index_pm2_5", "sub_index_pm10", "sub_index_no2", "sub_index_so2", "sub_index_co", "sub_index_o3"] if not integrated_df.empty and c in integrated_df.columns]
        if sub_cols:
            sub_means = integrated_df[sub_cols].mean().reset_index()
            sub_means.columns = ["Pollutant Sub-Index", "Mean Sub-Index Value"]
            sub_means["Pollutant Sub-Index"] = sub_means["Pollutant Sub-Index"].str.replace("sub_index_", "").str.upper()
            fig_sub_bar = px.bar(
                sub_means, x="Pollutant Sub-Index", y="Mean Sub-Index Value",
                color="Mean Sub-Index Value",
                color_continuous_scale="OrRd",
                title="Average Sub-Index Contribution by Pollutant",
                labels={"Mean Sub-Index Value": "Mean Value", "Pollutant Sub-Index": "Pollutant"}
            )
            fig_sub_bar.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_sub_bar, use_container_width=True)
        else:
            st.info("Sub-index columns not present in the current view.")

    st.markdown("---")
    st.subheader("AQI Reporting Range & Extreme Event Verification")
    st.markdown("""
    *Note: Standard regulatory reporting constraints specify an official bounded range of **0–500**.
    Extreme events beyond standard breakpoints are preserved without data loss using scientific extrapolation flags.*
    """)

    aqi_stats = summary.get("aqi_statistics", {})
    max_standard_aqi = integrated_df["aqi"].max() if not integrated_df.empty and "aqi" in integrated_df.columns else aqi_stats.get("max_aqi", 500.0)
    oor_count = (integrated_df["is_aqi_out_of_range"] == True).sum() if not integrated_df.empty and "is_aqi_out_of_range" in integrated_df.columns else aqi_stats.get("out_of_range_count", 0)
    extrap_max = integrated_df["extrapolated_aqi"].max() if not integrated_df.empty and "extrapolated_aqi" in integrated_df.columns else max_standard_aqi

    e_c1, e_c2, e_c3 = st.columns(3)
    with e_c1:
        st.metric("Reported Standard AQI Maximum", f"{max_standard_aqi:.1f}", help="Capped at standard 500 ceiling per regulatory definition")
    with e_c2:
        st.metric("Out-of-Range (>500) Exceedance Events", f"{oor_count:,}", help="Count of acute events exceeding standard maximum breakpoint")
    with e_c3:
        st.metric("Research Extrapolated AQI Peak", f"{extrap_max:.1f}", help="Piecewise extrapolated scale for extreme event scientific analysis")


def render_temporal_trends_tab(summary: Dict[str, Any], hourly_df: pd.DataFrame):
    """Render Tab 3: Temporal Trends, Diurnal Profiles, and Seasonal Summaries."""
    st.header("Temporal Analytics & Diurnal Profiles")

    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.subheader("Diurnal Hourly Profile (0:00 - 23:00)")
        if not hourly_df.empty and "hour" in hourly_df.columns and "avg_aqi" in hourly_df.columns:
            hourly_sorted = hourly_df.sort_values("hour")
            fig_hour = px.bar(
                hourly_sorted, x="hour", y="avg_aqi",
                color="avg_aqi",
                color_continuous_scale="Viridis",
                title="Diurnal Hourly Average AQI (Rush-Hour & Inversion Dynamics)",
                labels={"hour": "Hour of Day (UTC)", "avg_aqi": "Average AQI"}
            )
            fig_hour.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_hour, use_container_width=True)
        else:
            st.info("Hourly profile data unavailable.")

    with t_col2:
        st.subheader("Diurnal Pollutant Concentrations")
        if not hourly_df.empty and "hour" in hourly_df.columns:
            hourly_sorted = hourly_df.sort_values("hour")
            fig_pol_hour = go.Figure()
            pol_configs = [
                ("avg_pm2_5", "PM2.5 (µg/m³)", "#FF7E00"),
                ("avg_pm10", "PM10 (µg/m³)", "#FFFF00"),
                ("avg_no2", "NO2 (ppb)", "#00E400"),
                ("avg_so2", "SO2 (ppb)", "#00BFFF"),
                ("avg_co", "CO (ppm)", "#FF69B4"),
                ("avg_o3", "O3 (ppb)", "#9370DB")
            ]
            for p_col, name, clr in pol_configs:
                if p_col in hourly_sorted.columns:
                    fig_pol_hour.add_trace(go.Scatter(
                        x=hourly_sorted["hour"], y=hourly_sorted[p_col],
                        mode="lines+markers", name=name, line=dict(color=clr, width=2)
                    ))
            fig_pol_hour.update_layout(
                title="Hourly Mean Pollutant Curves",
                template="plotly_dark",
                height=380,
                xaxis_title="Hour of Day",
                yaxis_title="Concentration Level",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_pol_hour, use_container_width=True)
        else:
            st.info("Pollutant diurnal curve data unavailable.")

    st.markdown("---")
    st.subheader("Seasonal Air Quality Distribution")
    seasonal = summary.get("seasonal_summary", [])
    if seasonal:
        seasonal_df = pd.DataFrame(seasonal)
        st.dataframe(seasonal_df, use_container_width=True)
    else:
        st.info("Seasonal summary data unavailable.")


def render_spatial_hotspots_tab(hotspots_df: pd.DataFrame):
    """Render Tab 4: Geographic Distribution & Pollution Hotspots."""
    st.header("Geographic Distribution & Pollution Hotspots")

    if hotspots_df.empty:
        st.info("Spatial hotspot data unavailable. Run the batch pipeline to compute geographic hotspot metrics.")
        return

    st_col1, st_col2 = st.columns([3, 2])

    with st_col1:
        st.subheader("Monitoring Station Map & Hotspot Severity")
        fig_map = build_spatial_map(hotspots_df)
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Could not render spatial map. Check that station coordinates (latitude/longitude) are valid.")

    with st_col2:
        st.subheader("Hotspot Ranking & Severity Audit")
        show_cols = [c for c in ["station_id", "station_name", "city", "mean_aqi", "max_aqi", "hotspot_score", "hotspot_level", "unhealthy_pct"] if c in hotspots_df.columns]
        sorted_hotspots = hotspots_df.sort_values(by="hotspot_score", ascending=False) if "hotspot_score" in hotspots_df.columns else hotspots_df
        st.dataframe(sorted_hotspots[show_cols], use_container_width=True, height=480)


def render_data_quality_tab(summary: Dict[str, Any]):
    """Render Tab 5: Data Quality Pipeline, Imputation Audit, and Outlier Analysis."""
    st.header("Data Quality Pipeline & Anomaly Audit")

    aq_report = summary.get("quality_report_aq", {})
    weather_report = summary.get("quality_report_weather", {})
    join_stats = summary.get("join_stats", {})

    st.subheader("Air Quality Cleaning Pipeline Metrics")
    q_c1, q_c2, q_c3, q_c4, q_c5 = st.columns(5)
    with q_c1:
        st.metric("Initial Ingested Rows", f"{aq_report.get('initial_records', 0):,}")
    with q_c2:
        st.metric("Duplicates Cleared", f"{aq_report.get('duplicate_records', 0):,}")
    with q_c3:
        domain_corr = aq_report.get("domain_outliers_corrected", {})
        total_dom = domain_corr.get("total_domain_outliers", 0) if isinstance(domain_corr, dict) else domain_corr
        st.metric("Domain Corrections", f"{total_dom:,}", help="Physical sensor domain limits enforced")
    with q_c4:
        st.metric("Statistical Outliers Flagged", f"{aq_report.get('statistical_outliers_flagged', 0):,}", help="IQR 3.0x extreme threshold flagged non-destructively")
    with q_c5:
        st.metric("Final Retained Records", f"{aq_report.get('final_records', 0):,}")

    st.markdown("---")

    col_miss, col_outliers = st.columns(2)

    with col_miss:
        st.subheader("Missing Values Audit (Before vs After Imputation)")
        before_miss = aq_report.get("missing_before_cleaning", {})
        after_miss = aq_report.get("missing_after_cleaning", {})
        if before_miss:
            miss_df = pd.DataFrame({
                "Pollutant": [k.upper() for k in before_miss.keys()],
                "Missing Before": list(before_miss.values()),
                "Missing After": [after_miss.get(k, 0) for k in before_miss.keys()],
            })
            fig_miss = px.bar(
                miss_df, x="Pollutant", y=["Missing Before", "Missing After"],
                barmode="group",
                title="Missing Value Resolution via Windowed Imputation",
                color_discrete_sequence=["#ff5252", "#4caf50"],
                labels={"value": "Missing Count", "variable": "Pipeline Stage"}
            )
            fig_miss.update_layout(template="plotly_dark", height=360)
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.info("Missing values audit data unavailable.")

    with col_outliers:
        st.subheader("Statistical Outlier Breakdown by Pollutant")
        stat_outliers = aq_report.get("statistical_outliers_breakdown", {})
        if not stat_outliers:
            stat_outliers = aq_report.get("statistical_outliers_by_column", {})
        if stat_outliers:
            clean_outliers = {k.upper(): v for k, v in stat_outliers.items() if k != "total_outlier_rows"}
            st_df = pd.DataFrame({
                "Pollutant": list(clean_outliers.keys()),
                "Outlier Rows Flagged": list(clean_outliers.values())
            })
            fig_st = px.bar(
                st_df, x="Pollutant", y="Outlier Rows Flagged",
                color="Outlier Rows Flagged",
                color_continuous_scale="Reds",
                title="Non-Destructive Statistical Outliers Flagged (IQR 3.0x)",
                labels={"Outlier Rows Flagged": "Flagged Records"}
            )
            fig_st.update_layout(template="plotly_dark", height=360)
            st.plotly_chart(fig_st, use_container_width=True)
        else:
            st.info("Statistical outlier breakdown data unavailable.")

    # Multi-source fusion summary
    if weather_report or join_stats:
        st.markdown("---")
        st.subheader("Multi-Source Ingestion & Fusion Statistics")
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric("Weather Ingested", f"{weather_report.get('initial_records', 0):,}")
        with m_c2:
            st.metric("Weather Retained", f"{weather_report.get('final_records', 0):,}")
        with m_c3:
            st.metric("Weather Retention %", f"{weather_report.get('retention_percentage', 0)}%")
        with m_c4:
            st.metric("Integrated Dataset", f"{join_stats.get('integrated_records', 0):,}", help="Exact temporal join on station_id and timestamp")


def render_streaming_alerts_tab():
    """Render Tab 6: Spark Structured Streaming Live Feed & Real-time Alerts."""
    st.header("Spark Structured Streaming Live Feed & Real-time Alerts")

    stream_data, win_stats, latest_df = load_streaming_data()

    if stream_data is None and win_stats.empty and latest_df.empty:
        st.warning("⚠️ Streaming results are not available yet.")
        st.info("Run the streaming pipeline to generate real-time streaming feeds:\n```powershell\npython run.py streaming --duration 10 --rate 5 --stations 5\n```")
        return

    stream_data = stream_data or {}
    total_streamed = stream_data.get("total_streamed_records", 0)
    total_alerts = stream_data.get("total_alerts", 0)
    alert_counts = stream_data.get("alert_type_counts", {})

    s_c1, s_c2, s_c3, s_c4 = st.columns(4)
    with s_c1:
        st.metric("Total Streamed Records", f"{total_streamed:,}")
    with s_c2:
        st.metric("Total Triggered Alerts", f"{total_alerts:,}")
    with s_c3:
        st.metric("Unhealthy AQI Alerts", f"{alert_counts.get('UNHEALTHY_AQI_ALERT', 0):,}")
    with s_c4:
        st.metric("Rapid Change Alerts", f"{alert_counts.get('RAPID_CHANGE_ALERT', 0):,}")

    st.markdown("---")

    # Recent Alerts
    alerts = stream_data.get("recent_alerts", [])
    st.subheader(f"Active Real-Time Threshold Alerts ({len(alerts)} Recorded)")

    if alerts:
        alert_cols = st.columns(min(len(alerts), 3) if len(alerts) > 0 else 1)
        for idx, alt in enumerate(alerts[:6]):
            col_idx = idx % 3
            severity = alt.get("severity", "WARNING")
            alert_type = alt.get("alert_type", "ALERT")
            station = alt.get("station_id", "N/A")
            aqi_val = alt.get("aqi", "N/A")
            ts = alt.get("timestamp", "N/A")
            msg = alt.get("message", "")

            with alert_cols[col_idx]:
                if severity == "CRITICAL":
                    st.error(f"🚨 **[{severity}] {alert_type}**\n\n**Station:** `{station}` | **AQI:** `{aqi_val}`\n\n*Time:* `{ts}`\n\n{msg}")
                else:
                    st.warning(f"⚠️ **[{severity}] {alert_type}**\n\n**Station:** `{station}` | **AQI:** `{aqi_val}`\n\n*Time:* `{ts}`\n\n{msg}")
    else:
        st.success("✅ No critical threshold breaches in the latest streaming window.")

    # Windowed Stats
    if not win_stats.empty:
        st.markdown("---")
        st.subheader("Sliding Window Aggregations (Window: 10m / Slide: 5m)")
        st.dataframe(win_stats, use_container_width=True)

    # Latest Stream Records
    latest_recs = stream_data.get("latest_records", [])
    if latest_recs:
        st.markdown("---")
        st.subheader("Latest Ingested Stream Events")
        st.dataframe(pd.DataFrame(latest_recs), use_container_width=True)
    elif not latest_df.empty:
        st.markdown("---")
        st.subheader("Latest Ingested Stream Events")
        st.dataframe(latest_df.tail(20), use_container_width=True)


def render_spark_ml_tab(ml_data: Optional[Dict[str, Any]], preds_df: pd.DataFrame):
    """Render Tab 7: Spark ML Predictive Modeling & Time-Aware Evaluation."""
    st.header("Spark ML AQI Forecasting & Time-Aware Model Evaluation")

    if not ml_data or "models" not in ml_data:
        st.warning("⚠️ No ML evaluation metrics found.")
        st.info("Run the batch pipeline to train and evaluate Spark ML models:\n```powershell\npython run.py batch\n```")
        return

    split_info = ml_data.get("split_strategy", {})
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1:
        st.metric("Split Strategy", f"{split_info.get('strategy', 'chronological').title()} (80/20)")
    with m_c2:
        st.metric("Training Records", f"{ml_data.get('train_records', 0):,}")
    with m_c3:
        st.metric("Holdout Test Records", f"{ml_data.get('test_records', 0):,}")
    with m_c4:
        st.metric("Engineered Features", f"{len(ml_data.get('features', []))}")

    st.markdown("---")
    st.subheader("Spark ML Regression Model Comparison")
    st.markdown("""
    *Note: Models are evaluated strictly on a forward-looking chronological holdout set to avoid data leakage.
    The selected model is chosen according to lowest Root Mean Squared Error (RMSE) on the empirical test dataset.*
    """)

    models_dict = ml_data.get("models", {})
    best_model_name = ml_data.get("best_model", "linear_regression")

    metrics_rows = []
    for m_name, m_stats in models_dict.items():
        is_selected = " ⭐ (Selected)" if m_name == best_model_name else ""
        metrics_rows.append({
            "Model Name": m_name.replace("_", " ").title() + is_selected,
            "RMSE": round(float(m_stats.get("rmse", 0)), 3),
            "MAE": round(float(m_stats.get("mae", 0)), 3),
            "R² Score": round(float(m_stats.get("r2", 0)), 3),
            "Model Persistence Path": m_stats.get("model_path", "N/A")
        })

    metrics_df = pd.DataFrame(metrics_rows)
    st.dataframe(metrics_df, use_container_width=True)

    if not preds_df.empty and "actual_aqi" in preds_df.columns:
        st.markdown("---")
        p_c1, p_c2 = st.columns(2)

        with p_c1:
            st.subheader("Actual vs Predicted AQI (Holdout Test Set)")
            sample_size = min(len(preds_df), 1000)
            sample_preds = preds_df.sample(sample_size, random_state=42) if len(preds_df) > sample_size else preds_df
            y_col = "best_predicted_aqi" if "best_predicted_aqi" in sample_preds.columns else "predicted_aqi" if "predicted_aqi" in sample_preds.columns else "predicted_aqi_lr"

            fig_pred = px.scatter(
                sample_preds,
                x="actual_aqi",
                y=y_col,
                color="model" if "model" in sample_preds.columns else None,
                title=f"Actual vs Predicted AQI (Sample: {sample_size:,} observations)",
                labels={"actual_aqi": "Actual Regulatory AQI", y_col: "Predicted AQI"}
            )
            min_val = min(float(sample_preds["actual_aqi"].min()), float(sample_preds[y_col].min()))
            max_val = max(float(sample_preds["actual_aqi"].max()), float(sample_preds[y_col].max()))
            fig_pred.add_shape(
                type="line", line=dict(dash="dash", color="white", width=1.5),
                x0=min_val, y0=min_val, x1=max_val, y1=max_val
            )
            fig_pred.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_pred, use_container_width=True)

        with p_c2:
            st.subheader("Prediction Residual Distribution")
            if "prediction_error" in preds_df.columns:
                fig_err = px.histogram(
                    preds_df,
                    x="prediction_error",
                    nbins=40,
                    title="Model Residual Error Distribution (Actual - Predicted)",
                    color_discrete_sequence=["#00E400"],
                    labels={"prediction_error": "Prediction Error (AQI Points)"}
                )
                fig_err.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_err, use_container_width=True)
            else:
                st.info("Prediction residual column not available.")


def render_scalability_tab(benchmarks: List[Dict[str, Any]], partitioning: List[Dict[str, Any]]):
    """Render Tab 8: Scalability, Distributed Throughput, and Partitioning Benchmarks."""
    st.header("Scalability & Performance Benchmarks")

    if not benchmarks:
        st.warning("⚠️ No benchmark results recorded yet.")
        st.info("Run the scalability benchmarking suite to measure empirical execution times:\n```powershell\npython run.py benchmark\n```")
        return

    bench_df = pd.DataFrame(benchmarks)

    st.subheader("Throughput & Processing Time by Workload Scale")
    p_c1, p_c2 = st.columns(2)

    with p_c1:
        time_cols = [c for c in ["spark_etl_time_sec", "spark_aqi_time_sec", "spark_analytics_time_sec"] if c in bench_df.columns]
        if time_cols:
            fig_time = px.bar(
                bench_df, x="dataset_name", y=time_cols,
                title="Spark Execution Time Breakdown by Stage (Seconds)",
                barmode="stack",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                labels={"dataset_name": "Dataset Scale", "value": "Execution Time (s)", "variable": "Pipeline Stage"}
            )
            fig_time.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("Time breakdown metrics not available.")

    with p_c2:
        if "spark_throughput_records_per_sec" in bench_df.columns:
            fig_tp = px.line(
                bench_df, x="dataset_name", y="spark_throughput_records_per_sec",
                title="Spark Processing Throughput Scaling (Records / Second)",
                markers=True,
                color_discrete_sequence=["#00E400"],
                labels={"dataset_name": "Dataset Scale", "spark_throughput_records_per_sec": "Throughput (rec/s)"}
            )
            fig_tp.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_tp, use_container_width=True)
        else:
            st.info("Throughput metrics not available.")

    st.markdown("---")
    st.subheader("Empirical Benchmark Execution Summary")
    st.dataframe(bench_df, use_container_width=True)

    # Partition Scalability Breakdown if available
    if partitioning:
        st.markdown("---")
        st.subheader("Spark Partition Scaling Analysis")
        part_df = pd.DataFrame(partitioning)
        part_c1, part_c2 = st.columns(2)

        with part_c1:
            if "partition_count" in part_df.columns and "duration_sec" in part_df.columns:
                fig_part = px.bar(
                    part_df, x="partition_count", y="duration_sec",
                    title="Execution Duration vs Partition Count",
                    color="duration_sec",
                    color_continuous_scale="Viridis",
                    labels={"partition_count": "Shuffle Partitions", "duration_sec": "Duration (s)"}
                )
                fig_part.update_layout(template="plotly_dark", height=340)
                st.plotly_chart(fig_part, use_container_width=True)

        with part_c2:
            st.write("Partition Benchmark Raw Measurements:")
            st.dataframe(part_df, use_container_width=True)


# =====================================================================
# MAIN APPLICATION ENTRY POINT
# =====================================================================

def main():
    # Sidebar Branding
    st.sidebar.image("https://img.icons8.com/fluency/96/wind.png", width=64)
    st.sidebar.title(PROJECT_NAME)
    st.sidebar.caption("Scalable Big Data AQI Analytics")

    # Primary App Header Branding
    st.markdown(f"<h1 class='project-header'>{PROJECT_NAME}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='project-tagline'>{PROJECT_TAGLINE}</div>", unsafe_allow_html=True)

    # Load All Datasets
    summary = load_pipeline_summary()
    stations_df = load_station_summary()
    daily_df = load_daily_analytics()
    hourly_df = load_hourly_profile()
    hotspots_df = load_hotspots()
    integrated_df = load_integrated_aqi()
    ml_data = load_ml_metrics()
    preds_df = load_predictions()
    benchmarks = load_benchmarks()
    partitioning = load_partitioning_results()

    # If no core batch data exists, display run instructions
    if not summary:
        st.warning("⚠️ No processed batch pipeline data detected.")
        st.info(
            "Please generate synthetic data and execute the batch pipeline first:\n\n"
            "```powershell\n"
            "python run.py generate --stations 10 --days 30 --frequency 60\n"
            "python run.py batch --aqi-standard US_EPA\n"
            "```"
        )
        return

    aq_report = summary.get("quality_report_aq", {})

    # Sidebar System Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ System Status")
    st.sidebar.write(f"**Last Processed:** {summary.get('pipeline_execution_time', 'N/A')}")
    st.sidebar.write(f"**Total Events:** {summary.get('total_records', 0):,}")
    st.sidebar.write(f"**Data Quality Score:** {aq_report.get('quality_score', 0)}%")
    st.sidebar.write(f"**AQI Standard:** {summary.get('aqi_standard', 'US_EPA')}")

    # Tabs for Multi-dimensional view
    tab_overview, tab_aqi, tab_temporal, tab_spatial, tab_quality, tab_streaming, tab_prediction, tab_perf = st.tabs([
        "📊 Overview",
        "🎯 AQI Analysis",
        "📈 Temporal Trends",
        "🗺️ Spatial & Hotspots",
        "🛡️ Data Quality",
        "⚡ Streaming & Alerts",
        "🤖 Spark ML Prediction",
        "⏱️ Scalability & Benchmarks"
    ])

    with tab_overview:
        render_overview_tab(summary, stations_df, daily_df)

    with tab_aqi:
        render_aqi_analysis_tab(summary, integrated_df)

    with tab_temporal:
        render_temporal_trends_tab(summary, hourly_df)

    with tab_spatial:
        render_spatial_hotspots_tab(hotspots_df)

    with tab_quality:
        render_data_quality_tab(summary)

    with tab_streaming:
        render_streaming_alerts_tab()

    with tab_prediction:
        render_spark_ml_tab(ml_data, preds_df)

    with tab_perf:
        render_scalability_tab(benchmarks, partitioning)


if __name__ == "__main__":
    main()
