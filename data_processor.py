"""
Data processing functions for Patient Vitals & Seizure Timeline app.
Includes filtering, resampling, frequency inference, and seizure density computation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict
from datetime import datetime, timedelta

from config import (
    VITALS_CONFIG, 
    VITAL_KEYS,
    MAX_RAW_POINTS, 
    DOWNSAMPLE_TARGET,
    RESAMPLE_OPTIONS,
)


def preprocess_and_filter(
    vitals_df: pd.DataFrame,
    seizures_df: pd.DataFrame,
    patient_id: str,
    time_range: Optional[Tuple[datetime, datetime]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter data for a specific patient and time range.
    
    Args:
        vitals_df: Full vitals DataFrame
        seizures_df: Full seizures DataFrame
        patient_id: Patient to filter for
        time_range: Optional (start, end) datetime tuple
        
    Returns:
        Filtered vitals_df, seizures_df
    """
    # Filter by patient
    vitals = vitals_df[vitals_df["patient_id"] == patient_id].copy()
    seizures = seizures_df[seizures_df["patient_id"] == patient_id].copy()
    
    # Apply time range filter if provided
    if time_range is not None:
        start, end = time_range
        vitals = vitals[
            (vitals["timestamp"] >= start) & 
            (vitals["timestamp"] <= end)
        ]
        
        if not seizures.empty:
            if "seizure_time" in seizures.columns:
                seizures = seizures[
                    (seizures["seizure_time"] >= start) & 
                    (seizures["seizure_time"] <= end)
                ]
    
    return vitals.reset_index(drop=True), seizures.reset_index(drop=True)


def infer_sampling_frequency_per_vital(
    vitals_df: pd.DataFrame,
) -> Dict[str, Dict]:
    """
    Infer the approximate sampling frequency for each vital.
    
    Returns:
        Dict mapping vital name to frequency info:
        {
            "heart_rate": {
                "median_interval_seconds": 60,
                "frequency_label": "~1 minute",
                "count": 10000,
                "coverage_pct": 95.5
            },
            ...
        }
    """
    result = {}
    
    for vital in VITAL_KEYS:
        if vital not in vitals_df.columns:
            continue
            
        # Get non-null values
        vital_data = vitals_df[vitals_df[vital].notna()][["timestamp", vital]].copy()
        
        if len(vital_data) < 2:
            result[vital] = {
                "median_interval_seconds": None,
                "frequency_label": "Insufficient data",
                "count": len(vital_data),
                "coverage_pct": 0,
            }
            continue
        
        # Calculate intervals between consecutive readings
        vital_data = vital_data.sort_values("timestamp")
        intervals = vital_data["timestamp"].diff().dt.total_seconds().dropna()
        
        if len(intervals) == 0:
            result[vital] = {
                "median_interval_seconds": None,
                "frequency_label": "Insufficient data",
                "count": len(vital_data),
                "coverage_pct": 0,
            }
            continue
        
        median_interval = intervals.median()
        
        # Convert to human-readable label
        freq_label = _interval_to_label(median_interval)
        
        # Calculate coverage percentage (actual vs expected samples)
        total_span = (vital_data["timestamp"].max() - vital_data["timestamp"].min()).total_seconds()
        expected_samples = total_span / median_interval if median_interval > 0 else 0
        coverage = (len(vital_data) / expected_samples * 100) if expected_samples > 0 else 0
        
        result[vital] = {
            "median_interval_seconds": median_interval,
            "frequency_label": freq_label,
            "count": len(vital_data),
            "coverage_pct": min(100, coverage),
        }
    
    return result


def _interval_to_label(seconds: float) -> str:
    """Convert interval in seconds to human-readable label."""
    if seconds < 2:
        return "~1 second"
    elif seconds < 120:
        return f"~{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"~{minutes} minute{'s' if minutes > 1 else ''}"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"~{hours} hour{'s' if hours > 1 else ''}"
    else:
        days = int(seconds / 86400)
        return f"~{days} day{'s' if days > 1 else ''}"


def resample_vitals(
    vitals_df: pd.DataFrame,
    granularity: str,
    aggregation: str = "mean",
    show_band: bool = False,
    ffill: bool = False,
    selected_vitals: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Resample vitals data to specified granularity.
    
    Args:
        vitals_df: Input vitals DataFrame
        granularity: One of "raw", "auto", "1s", "1min", "5min", "15min", "1h", "6h", "1D"
        aggregation: "mean" or "median"
        show_band: If True, also compute min/max (or p10/p90) bands
        ffill: If True, forward-fill missing values after resampling
        selected_vitals: List of vitals to process (defaults to all)
        
    Returns:
        resampled_df: Resampled vitals DataFrame
        band_min_df: Lower band DataFrame (if show_band=True, else None)
        band_max_df: Upper band DataFrame (if show_band=True, else None)
    """
    if vitals_df.empty:
        return vitals_df, None, None
    
    if selected_vitals is None:
        selected_vitals = [v for v in VITAL_KEYS if v in vitals_df.columns]
    
    # Handle raw mode
    if granularity == "raw":
        # Check if we need to auto-downsample for performance
        total_points = sum(vitals_df[v].notna().sum() for v in selected_vitals if v in vitals_df.columns)
        if total_points > MAX_RAW_POINTS:
            # Force auto-resample
            granularity = "auto"
        else:
            return vitals_df, None, None
    
    # Determine frequency for auto mode
    if granularity == "auto":
        freq = _determine_auto_frequency(vitals_df, selected_vitals)
    else:
        freq = RESAMPLE_OPTIONS.get(granularity, {}).get("freq", granularity)
    
    if freq is None:
        return vitals_df, None, None
    
    # Set timestamp as index for resampling
    df = vitals_df.copy()
    df = df.set_index("timestamp")
    
    # Prepare aggregation functions
    agg_func = aggregation
    
    # Resample each vital
    vitals_to_resample = [v for v in selected_vitals if v in df.columns]
    
    if not vitals_to_resample:
        return vitals_df, None, None
    
    # Main resampling
    resampled = df[vitals_to_resample].resample(freq).agg(agg_func)
    
    # Compute bands if requested
    band_min = None
    band_max = None
    
    if show_band:
        band_min = df[vitals_to_resample].resample(freq).quantile(0.1)
        band_max = df[vitals_to_resample].resample(freq).quantile(0.9)
    
    # Forward fill if requested
    if ffill:
        resampled = resampled.ffill()
        if band_min is not None:
            band_min = band_min.ffill()
            band_max = band_max.ffill()
    
    # Reset index
    resampled = resampled.reset_index()
    resampled["patient_id"] = vitals_df["patient_id"].iloc[0]
    
    if band_min is not None:
        band_min = band_min.reset_index()
        band_max = band_max.reset_index()
    
    return resampled, band_min, band_max


def _determine_auto_frequency(
    vitals_df: pd.DataFrame,
    selected_vitals: List[str],
) -> str:
    """
    Determine appropriate resampling frequency based on data density and time range.
    """
    if vitals_df.empty:
        return "1min"
    
    # Calculate total points
    total_points = sum(
        vitals_df[v].notna().sum() 
        for v in selected_vitals 
        if v in vitals_df.columns
    )
    
    # Calculate time span
    time_span = (vitals_df["timestamp"].max() - vitals_df["timestamp"].min()).total_seconds()
    
    if time_span == 0:
        return "1min"
    
    # Target density: DOWNSAMPLE_TARGET points across the view
    target_interval = time_span / DOWNSAMPLE_TARGET
    
    # Map to standard frequencies
    if target_interval < 1:
        return "1s"
    elif target_interval < 60:
        return "1s"
    elif target_interval < 300:
        return "1min"
    elif target_interval < 900:
        return "5min"
    elif target_interval < 3600:
        return "15min"
    elif target_interval < 21600:
        return "1h"
    elif target_interval < 86400:
        return "6h"
    else:
        return "1D"


def compute_seizure_density(
    seizures_df: pd.DataFrame,
    bin_size: str = "hour",
    time_range: Optional[Tuple[datetime, datetime]] = None,
) -> pd.DataFrame:
    """
    Compute seizure density (counts per time bin).
    
    Args:
        seizures_df: Seizures DataFrame
        bin_size: "hour" or "day"
        time_range: Optional (start, end) for the full time axis
        
    Returns:
        DataFrame with columns [time_bin, count, density_per_hour]
    """
    if seizures_df.empty:
        if time_range is not None:
            # Return empty density with time range
            freq = "1h" if bin_size == "hour" else "1D"
            time_index = pd.date_range(start=time_range[0], end=time_range[1], freq=freq)
            return pd.DataFrame({
                "time_bin": time_index,
                "count": 0,
                "density_per_hour": 0.0,
            })
        return pd.DataFrame(columns=["time_bin", "count", "density_per_hour"])
    
    # Get seizure times
    if "seizure_time" in seizures_df.columns:
        times = seizures_df["seizure_time"]
    elif "seizure_start" in seizures_df.columns:
        times = seizures_df["seizure_start"]
    else:
        return pd.DataFrame(columns=["time_bin", "count", "density_per_hour"])
    
    # Create time series
    freq = "1h" if bin_size == "hour" else "1D"
    bin_seconds = 3600 if bin_size == "hour" else 86400
    
    # Determine range
    if time_range is not None:
        start, end = time_range
    else:
        start, end = times.min(), times.max()
    
    # Create complete time index
    time_index = pd.date_range(start=start, end=end, freq=freq)
    
    # Count seizures per bin
    seizure_series = pd.Series(1, index=pd.to_datetime(times))
    counts = seizure_series.resample(freq).count()
    
    # Align with complete index
    density_df = pd.DataFrame({"time_bin": time_index})
    counts_df = counts.reset_index()
    counts_df.columns = ["time_bin", "count"]
    
    density_df = density_df.merge(counts_df, on="time_bin", how="left")
    density_df["count"] = density_df["count"].fillna(0).astype(int)
    
    # Calculate density per hour
    density_df["density_per_hour"] = density_df["count"] / (bin_seconds / 3600)
    
    return density_df


def compute_vital_stats(
    vitals_df: pd.DataFrame,
    vital: str,
) -> Dict:
    """
    Compute summary statistics for a vital.
    
    Returns:
        Dict with min, max, mean, median, std, count
    """
    if vital not in vitals_df.columns:
        return None
    
    values = vitals_df[vital].dropna()
    
    if len(values) == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "count": 0,
        }
    
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()) if len(values) > 1 else 0,
        "count": len(values),
    }


def compute_seizure_stats(
    seizures_df: pd.DataFrame,
    time_range: Optional[Tuple[datetime, datetime]] = None,
) -> Dict:
    """
    Compute seizure statistics.
    
    Returns:
        Dict with total_count, per_hour, per_day, types breakdown
    """
    if seizures_df.empty:
        return {
            "total_count": 0,
            "per_hour": 0,
            "per_day": 0,
            "types": {},
        }
    
    total_count = len(seizures_df)
    
    # Calculate time span
    if time_range is not None:
        time_span_hours = (time_range[1] - time_range[0]).total_seconds() / 3600
    else:
        if "seizure_time" in seizures_df.columns:
            times = seizures_df["seizure_time"]
        else:
            times = seizures_df.get("seizure_start", pd.Series())
        
        if len(times) > 0:
            time_span_hours = (times.max() - times.min()).total_seconds() / 3600
        else:
            time_span_hours = 1
    
    time_span_hours = max(1, time_span_hours)  # Avoid division by zero
    time_span_days = time_span_hours / 24
    
    # Type breakdown
    types = {}
    if "seizure_type" in seizures_df.columns:
        types = seizures_df["seizure_type"].value_counts().to_dict()
    
    return {
        "total_count": total_count,
        "per_hour": total_count / time_span_hours,
        "per_day": total_count / time_span_days,
        "types": types,
    }


def get_vital_context_around_seizure(
    vitals_df: pd.DataFrame,
    seizure_time: datetime,
    window_minutes: int = 5,
    vitals_list: Optional[List[str]] = None,
) -> Dict:
    """
    Get vital statistics in a window around a seizure event.
    
    Args:
        vitals_df: Vitals DataFrame
        seizure_time: Seizure timestamp
        window_minutes: Minutes before and after seizure
        vitals_list: List of vitals to include
        
    Returns:
        Dict mapping vital name to stats in the window
    """
    if vitals_list is None:
        vitals_list = [v for v in VITAL_KEYS if v in vitals_df.columns]
    
    window_start = seizure_time - timedelta(minutes=window_minutes)
    window_end = seizure_time + timedelta(minutes=window_minutes)
    
    window_data = vitals_df[
        (vitals_df["timestamp"] >= window_start) &
        (vitals_df["timestamp"] <= window_end)
    ]
    
    result = {}
    for vital in vitals_list:
        if vital in window_data.columns:
            values = window_data[vital].dropna()
            if len(values) > 0:
                result[vital] = {
                    "mean": float(values.mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "count": len(values),
                }
            else:
                result[vital] = None
        else:
            result[vital] = None
    
    return result


def build_seizure_events_table(
    seizures_df: pd.DataFrame,
    vitals_df: pd.DataFrame,
    window_minutes: int = 5,
) -> pd.DataFrame:
    """
    Build a table of seizure events with vital context.
    
    Returns:
        DataFrame with seizure info and vital context columns
    """
    if seizures_df.empty:
        return pd.DataFrame()
    
    rows = []
    
    for _, seizure in seizures_df.iterrows():
        row = {
            "Event Time": seizure.get("seizure_time", seizure.get("seizure_start")),
        }
        
        # Add interval end if present
        if "seizure_end" in seizure and pd.notna(seizure["seizure_end"]):
            row["End Time"] = seizure["seizure_end"]
            duration = (seizure["seizure_end"] - seizure["seizure_start"]).total_seconds()
            row["Duration (s)"] = int(duration)
        elif "duration_seconds" in seizure and pd.notna(seizure["duration_seconds"]):
            row["Duration (s)"] = int(seizure["duration_seconds"])
        
        # Add type and severity
        if "seizure_type" in seizure:
            row["Type"] = seizure["seizure_type"]
        if "severity" in seizure:
            row["Severity"] = seizure["severity"]
        
        # Get vital context
        seizure_time = row["Event Time"]
        context = get_vital_context_around_seizure(
            vitals_df, seizure_time, window_minutes
        )
        
        for vital, stats in context.items():
            if stats is not None:
                vital_name = VITALS_CONFIG[vital]["name"]
                unit = VITALS_CONFIG[vital]["unit"]
                row[f"{vital_name} (±{window_minutes}min)"] = f"{stats['mean']:.1f} {unit}"
        
        rows.append(row)
    
    return pd.DataFrame(rows)

