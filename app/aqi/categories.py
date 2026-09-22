"""
AQI Category mapping and health advisory definitions.
"""
from typing import Dict, Any, Optional
from app.config.settings import get_settings


def get_aqi_category_info(aqi_value: Optional[float], standard: str = "US_EPA") -> Dict[str, Any]:
    """
    Get category name, color code, and health advisory for an AQI value.
    """
    if aqi_value is None or aqi_value < 0:
        return {
            "category": "Unknown",
            "color": "#808080",
            "description": "Insufficient data to compute AQI.",
        }

    settings = get_settings()
    std_config = settings.aqi_config.get(standard, {})
    categories = std_config.get("categories", [])

    for cat in categories:
        if cat["aqi_min"] <= aqi_value <= cat["aqi_max"]:
            return {
                "category": cat["name"],
                "color": cat.get("color", "#000000"),
                "description": cat.get("description", ""),
            }

    # Value above 500
    if aqi_value > 500:
        return {
            "category": "Hazardous (Extreme)",
            "color": "#7E0023",
            "description": "Emergency conditions: critical health risk to the entire population.",
        }

    return {
        "category": "Unknown",
        "color": "#808080",
        "description": "No matching AQI bracket.",
    }


def get_category_name(aqi_value: Optional[float], standard: str = "US_EPA") -> str:
    """Helper to return only category name."""
    return get_aqi_category_info(aqi_value, standard)["category"]
