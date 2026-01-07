"""
Configuration constants and default values for the Patient Vitals & Seizure Timeline app.
"""

# =============================================================================
# VITAL CONFIGURATION
# =============================================================================

VITALS_CONFIG = {
    "spo2": {
        "name": "SpO2",
        "unit": "%",
        "color": "#00D4AA",
        "normal_range": (95, 100),
        "critical_low": 90,
        "default_threshold_low": 92,
        "y_range": (85, 100),
    },
    "heart_rate": {
        "name": "Heart Rate",
        "unit": "bpm",
        "color": "#FF6B6B",
        "normal_range": (60, 100),
        "critical_high": 150,
        "default_threshold_high": 140,
        "y_range": (40, 180),
    },
    "temperature": {
        "name": "Temperature",
        "unit": "°C",
        "color": "#FFB347",
        "normal_range": (36.1, 37.2),
        "critical_high": 39.0,
        "default_threshold_high": 38.0,
        "y_range": (35, 41),
    },
    "blood_glucose": {
        "name": "Blood Glucose",
        "unit": "mg/dL",
        "color": "#9B59B6",
        "normal_range": (70, 140),
        "critical_low": 54,
        "critical_high": 250,
        "default_threshold_low": 70,
        "default_threshold_high": 180,
        "y_range": (40, 300),
    },
}

VITAL_KEYS = list(VITALS_CONFIG.keys())

# =============================================================================
# SAMPLING FREQUENCIES
# =============================================================================

SAMPLING_FREQUENCIES = {
    "per_second": {"label": "Per Second", "freq": "1s", "seconds": 1},
    "per_minute": {"label": "Per Minute", "freq": "1min", "seconds": 60},
    "per_5_minutes": {"label": "Per 5 Minutes", "freq": "5min", "seconds": 300},
    "per_hour": {"label": "Per Hour", "freq": "1h", "seconds": 3600},
    "per_6_hours": {"label": "Per 6 Hours", "freq": "6h", "seconds": 21600},
    "per_day": {"label": "Per Day", "freq": "1D", "seconds": 86400},
}

# =============================================================================
# RESAMPLING OPTIONS
# =============================================================================

RESAMPLE_OPTIONS = {
    "raw": {"label": "Raw (No Resample)", "freq": None},
    "auto": {"label": "Auto (Adaptive)", "freq": "auto"},
    "1s": {"label": "Per Second", "freq": "1s"},
    "1min": {"label": "Per Minute", "freq": "1min"},
    "5min": {"label": "5 Minutes", "freq": "5min"},
    "15min": {"label": "15 Minutes", "freq": "15min"},
    "1h": {"label": "Hourly", "freq": "1h"},
    "6h": {"label": "6 Hours", "freq": "6h"},
    "1D": {"label": "Daily", "freq": "1D"},
}

AGGREGATION_METHODS = ["mean", "median"]

# =============================================================================
# SEIZURE CONFIGURATION
# =============================================================================

SEIZURE_CONFIG = {
    "marker_color": "#E74C3C",
    "marker_size": 12,
    "marker_symbol": "circle",
    "density_color_scale": "Reds",
    "region_fill_color": "rgba(231, 76, 60, 0.3)",
    "region_line_color": "rgba(231, 76, 60, 0.8)",
}

SEIZURE_DENSITY_BINS = {
    "hour": {"label": "Hourly", "freq": "1h"},
    "day": {"label": "Daily", "freq": "1D"},
}

# =============================================================================
# X-AXIS TIME UNIT OPTIONS
# =============================================================================

XAXIS_TIME_UNITS = {
    "second": {"label": "Seconds", "dtick": 1000, "tickformat": "%H:%M:%S"},
    "minute": {"label": "Minutes", "dtick": 60000, "tickformat": "%H:%M"},
    "hour": {"label": "Hours", "dtick": 3600000, "tickformat": "%b %d %H:%M"},
    "day": {"label": "Days", "dtick": 86400000, "tickformat": "%b %d"},
    "month": {"label": "Months", "dtick": "M1", "tickformat": "%b %Y"},
}

# =============================================================================
# PERFORMANCE GUARDRAILS
# =============================================================================

MAX_RAW_POINTS = 200000  # Auto-downsample if exceeding this
DOWNSAMPLE_TARGET = 10000  # Target points after downsampling

# =============================================================================
# CSV SCHEMA
# =============================================================================

CSV_REQUIRED_COLUMNS = ["timestamp"]
CSV_OPTIONAL_COLUMNS = [
    "patient_id",
    "spo2",
    "heart_rate",
    "temperature",
    "blood_glucose",
    "seizure",
    "seizure_event",
    "seizure_start",
    "seizure_end",
]

# =============================================================================
# UI DEFAULTS
# =============================================================================

DEFAULT_DAYS = 7
DEFAULT_NUM_PATIENTS = 5

# Chart dimensions
CHART_HEIGHT_STACKED_ROW = 200
CHART_HEIGHT_COMBINED = 600
CHART_HEIGHT_DENSITY = 100

# Color palette for multi-axis
MULTI_AXIS_COLORS = ["#00D4AA", "#FF6B6B", "#FFB347", "#9B59B6"]

