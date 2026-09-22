"""
AQI Breakpoints definitions and mathematical sub-index calculator.
Implements standard US EPA and India CPCB piecewise linear interpolation.
"""
from typing import Dict, List, Tuple, Optional
from app.config.settings import get_settings


def calculate_sub_index(concentration: Optional[float], pollutant: str, standard: str = "US_EPA") -> Optional[float]:
    """
    Calculate sub-index for a single pollutant concentration using linear interpolation:
    I = [(I_hi - I_lo) / (BP_hi - BP_lo)] * (C - BP_lo) + I_lo
    """
    if concentration is None or concentration < 0:
        return None

    settings = get_settings()
    std_config = settings.aqi_config.get(standard, {})
    breakpoints_map = std_config.get("breakpoints", {})
    
    pollutant_key = pollutant.lower().replace(".", "_")
    breakpoints = breakpoints_map.get(pollutant_key)

    if not breakpoints:
        # Fallback default EPA breakpoints if config is missing
        return None

    for bp in breakpoints:
        c_low, c_high, i_low, i_high = bp[0], bp[1], bp[2], bp[3]
        if c_low <= concentration <= c_high:
            # Piecewise linear interpolation
            sub_index = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(sub_index, 1)

    # If concentration exceeds highest breakpoint, extrapolate from the highest bracket
    last_bp = breakpoints[-1]
    c_low, c_high, i_low, i_high = last_bp[0], last_bp[1], last_bp[2], last_bp[3]
    if concentration > c_high:
        sub_index = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
        return round(min(sub_index, 999.0), 1)

    return None
