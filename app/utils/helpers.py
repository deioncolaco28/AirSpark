"""
General helper functions for serialization, time formatting, and formatting outputs.
"""
import json
from datetime import datetime
from typing import Any, Dict


def serialize_json_pretty(data: Dict[str, Any]) -> str:
    """Serialize dictionary to pretty formatted JSON string."""
    def default_serializer(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return str(obj)
    return json.dumps(data, indent=2, default=default_serializer)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    return f"{minutes}m {rem_seconds:.2f}s"
