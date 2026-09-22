"""
Streaming Alert Evaluation and Notification Engine for AirSpark.
Detects real-time threshold breaches, severe AQI spikes, and rapid pollutant surges.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from app.config.settings import get_settings
from app.utils.logging import get_logger

logger = get_logger("AirSpark.Streaming.Alerts")


class AlertEvaluator:
    """Evaluates real-time sensor records against threshold and rate-of-change rules."""

    def __init__(self):
        settings = get_settings()
        stream_cfg = settings.streaming_config
        self.thresholds = stream_cfg.get("alert_thresholds", {
            "aqi_unhealthy": 150.0,
            "aqi_hazardous": 300.0,
            "pm2_5_spike": 100.0,
            "rate_of_change_pct": 50.0,
        })
        self._previous_station_readings: Dict[str, Dict[str, float]] = {}

    def evaluate_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate single streaming record for threshold breaches and rapid rate-of-change surges.
        Returns structured alert payload dict if triggered, else None.
        """
        aqi = record.get("aqi")
        pm25 = record.get("pm2_5")
        station_id = record.get("station_id", "UNKNOWN")
        now_utc = datetime.now(timezone.utc)
        timestamp = record.get("timestamp", now_utc.isoformat())

        if aqi is None:
            return None

        alert_type = None
        severity = "INFO"
        message = ""

        # 1. Evaluate rate of change if previous reading exists for this station
        rate_of_change_threshold = float(self.thresholds.get("rate_of_change_pct", 50.0))
        prev_data = self._previous_station_readings.get(station_id)
        if prev_data and pm25 is not None:
            prev_pm25 = prev_data.get("pm2_5", 0.0)
            if prev_pm25 > 10.0:
                pct_increase = ((pm25 - prev_pm25) / prev_pm25) * 100.0
                if pct_increase >= rate_of_change_threshold:
                    alert_type = "RAPID_CHANGE_ALERT"
                    severity = "WARNING"
                    message = f"Rapid PM2.5 surge of +{pct_increase:.1f}% at {station_id} ({prev_pm25:.1f} -> {pm25:.1f} ug/m3)"

        # 2. Evaluate static threshold breaches
        if aqi >= self.thresholds.get("aqi_hazardous", 300.0):
            alert_type = "HAZARDOUS_AQI_ALERT"
            severity = "CRITICAL"
            message = f"Hazardous air quality detected at {station_id}: AQI = {aqi:.1f}"
        elif aqi >= self.thresholds.get("aqi_unhealthy", 150.0) and alert_type is None:
            alert_type = "UNHEALTHY_AQI_ALERT"
            severity = "WARNING"
            message = f"Unhealthy air quality detected at {station_id}: AQI = {aqi:.1f}"
        elif pm25 is not None and pm25 >= self.thresholds.get("pm2_5_spike", 100.0) and alert_type is None:
            alert_type = "PM25_SPIKE_ALERT"
            severity = "WARNING"
            message = f"Acute PM2.5 spike at {station_id}: PM2.5 = {pm25:.1f} ug/m3"

        # Update previous reading state
        if pm25 is not None:
            self._previous_station_readings[station_id] = {"pm2_5": float(pm25), "aqi": float(aqi)}

        if alert_type:
            alert = {
                "alert_id": f"ALT_{station_id}_{int(now_utc.timestamp())}_{abs(hash(message)) % 10000}",
                "station_id": station_id,
                "timestamp": str(timestamp),
                "alert_type": alert_type,
                "severity": severity,
                "aqi": round(float(aqi), 1),
                "pm2_5": round(float(pm25), 1) if pm25 is not None else None,
                "message": message,
                "generated_at": now_utc.isoformat(),
            }
            logger.warning(f"[STREAMING ALERT] [{severity}] {message}")
            return alert

        return None
