"""
CSV data loading and validation for Patient Vitals & Seizure Timeline app.
Handles file upload, schema validation, and data parsing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict
from datetime import datetime
import streamlit as st

from config import CSV_REQUIRED_COLUMNS, CSV_OPTIONAL_COLUMNS, VITAL_KEYS


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


def load_uploaded_data(
    uploaded_file,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load and validate uploaded CSV file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        vitals_df: Processed vitals DataFrame
        seizures_df: Extracted seizures DataFrame
        warnings: List of warning messages
    """
    warnings = []
    
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise DataValidationError(f"Failed to read CSV file: {str(e)}")
    
    # Validate required columns
    validation_result = validate_csv_schema(df)
    if not validation_result["valid"]:
        raise DataValidationError(validation_result["error"])
    
    warnings.extend(validation_result.get("warnings", []))
    
    # Parse timestamps
    df = parse_timestamps(df)
    
    # Handle patient_id
    if "patient_id" not in df.columns:
        df["patient_id"] = "uploaded_patient"
        warnings.append("No patient_id column found. Using 'uploaded_patient' as default.")
    
    # Extract vitals DataFrame
    vitals_df = extract_vitals(df)
    
    # Extract seizures DataFrame
    seizures_df, seizure_warnings = extract_seizures(df)
    warnings.extend(seizure_warnings)
    
    return vitals_df, seizures_df, warnings


def validate_csv_schema(df: pd.DataFrame) -> Dict:
    """
    Validate CSV schema against expected format.
    
    Returns:
        dict with keys: valid (bool), error (str or None), warnings (list)
    """
    result = {"valid": True, "error": None, "warnings": []}
    
    # Check for required columns
    missing_required = [col for col in CSV_REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        result["valid"] = False
        result["error"] = f"Missing required columns: {', '.join(missing_required)}"
        return result
    
    # Check if at least one vital or seizure column exists
    vital_columns = [col for col in VITAL_KEYS if col in df.columns]
    seizure_columns = [col for col in ["seizure", "seizure_event", "seizure_start", "seizure_end"] 
                       if col in df.columns]
    
    if not vital_columns and not seizure_columns:
        result["valid"] = False
        result["error"] = (
            "No vital or seizure columns found. "
            "Expected at least one of: spo2, heart_rate, temperature, blood_glucose, "
            "seizure, seizure_event, seizure_start, seizure_end"
        )
        return result
    
    # Warnings for missing optional data
    if not vital_columns:
        result["warnings"].append("No vital sign columns found. Only seizure data will be displayed.")
    
    if not seizure_columns:
        result["warnings"].append("No seizure columns found. Seizure overlay will be disabled.")
    
    # Check for unknown columns
    known_columns = set(CSV_REQUIRED_COLUMNS + CSV_OPTIONAL_COLUMNS)
    unknown_columns = [col for col in df.columns if col not in known_columns]
    if unknown_columns:
        result["warnings"].append(f"Unknown columns will be ignored: {', '.join(unknown_columns)}")
    
    return result


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamp column robustly (ISO8601 or Unix epoch).
    """
    df = df.copy()
    
    try:
        # Try parsing as datetime string first
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception:
        try:
            # Try parsing as Unix epoch (seconds)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        except Exception:
            try:
                # Try parsing as Unix epoch (milliseconds)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            except Exception as e:
                raise DataValidationError(
                    f"Failed to parse timestamp column. "
                    f"Expected ISO8601 string or Unix epoch. Error: {str(e)}"
                )
    
    # Sort and drop duplicates
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp", "patient_id"], keep="first")
    df = df.reset_index(drop=True)
    
    return df


def extract_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract vitals data from the DataFrame.
    """
    # Select relevant columns
    columns = ["timestamp", "patient_id"]
    
    for vital in VITAL_KEYS:
        if vital in df.columns:
            columns.append(vital)
    
    vitals_df = df[columns].copy()
    
    # Convert vital columns to numeric, coercing errors
    for vital in VITAL_KEYS:
        if vital in vitals_df.columns:
            vitals_df[vital] = pd.to_numeric(vitals_df[vital], errors="coerce")
    
    return vitals_df


def extract_seizures(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract seizure events from the DataFrame.
    
    Handles three formats:
    1. seizure or seizure_event column (0/1 or boolean)
    2. seizure_start + seizure_end columns (intervals)
    
    Returns:
        seizures_df: DataFrame with columns [patient_id, seizure_time, seizure_type, severity, duration_seconds]
                     OR for intervals: [patient_id, seizure_start, seizure_end, seizure_type, severity]
        warnings: List of warning messages
    """
    warnings = []
    
    # Check for interval format first
    if "seizure_start" in df.columns and "seizure_end" in df.columns:
        return _extract_seizures_interval(df), warnings
    
    # Check for point event format
    seizure_col = None
    if "seizure" in df.columns:
        seizure_col = "seizure"
    elif "seizure_event" in df.columns:
        seizure_col = "seizure_event"
    
    if seizure_col is None:
        # No seizure data
        return pd.DataFrame(columns=[
            "patient_id", "seizure_time", "seizure_type", "severity", "duration_seconds"
        ]), warnings
    
    return _extract_seizures_point(df, seizure_col), warnings


def _extract_seizures_point(df: pd.DataFrame, seizure_col: str) -> pd.DataFrame:
    """Extract point seizure events from a binary column."""
    # Convert to boolean
    df = df.copy()
    df[seizure_col] = pd.to_numeric(df[seizure_col], errors="coerce").fillna(0).astype(bool)
    
    # Filter to seizure events
    seizure_rows = df[df[seizure_col] == True].copy()
    
    if seizure_rows.empty:
        return pd.DataFrame(columns=[
            "patient_id", "seizure_time", "seizure_type", "severity", "duration_seconds"
        ])
    
    seizures_df = pd.DataFrame({
        "patient_id": seizure_rows["patient_id"],
        "seizure_time": seizure_rows["timestamp"],
        "seizure_type": "unknown",
        "severity": "unknown",
        "duration_seconds": None,
    })
    
    return seizures_df.reset_index(drop=True)


def _extract_seizures_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Extract interval seizure events."""
    # Filter rows where seizure_start is not null
    seizure_rows = df[df["seizure_start"].notna()].copy()
    
    if seizure_rows.empty:
        return pd.DataFrame(columns=[
            "patient_id", "seizure_start", "seizure_end", "seizure_type", "severity", "is_interval"
        ])
    
    # Parse seizure times
    seizure_rows["seizure_start"] = pd.to_datetime(seizure_rows["seizure_start"])
    seizure_rows["seizure_end"] = pd.to_datetime(seizure_rows["seizure_end"])
    
    seizures_df = pd.DataFrame({
        "patient_id": seizure_rows["patient_id"],
        "seizure_start": seizure_rows["seizure_start"],
        "seizure_end": seizure_rows["seizure_end"],
        "seizure_type": "unknown",
        "severity": "unknown",
        "is_interval": True,
    })
    
    # Also add seizure_time as start for compatibility
    seizures_df["seizure_time"] = seizures_df["seizure_start"]
    
    return seizures_df.reset_index(drop=True)


def get_csv_schema_help() -> str:
    """Return help text describing the expected CSV schema."""
    return """
### Expected CSV Schema

**Required columns:**
- `timestamp`: ISO8601 datetime string (e.g., "2024-01-15T10:30:00") or Unix epoch (seconds or milliseconds)

**Optional columns:**
- `patient_id`: Patient identifier (string). If missing, defaults to "uploaded_patient"
- `spo2`: Oxygen saturation percentage (numeric, 0-100)
- `heart_rate`: Heart rate in beats per minute (numeric)
- `temperature`: Body temperature in °C (numeric)
- `blood_glucose`: Blood glucose in mg/dL (numeric)

**Seizure representations (use one of these):**
- `seizure` or `seizure_event`: Binary column (0/1 or true/false) indicating seizure at that timestamp
- `seizure_start` + `seizure_end`: Datetime columns for seizure intervals

### Example CSV:
```csv
timestamp,patient_id,heart_rate,spo2,seizure
2024-01-15T10:00:00,patient_001,75,98,0
2024-01-15T10:01:00,patient_001,82,97,0
2024-01-15T10:02:00,patient_001,120,94,1
```

### Notes:
- Different vitals can have different sampling rates
- Missing values are allowed (empty cells or NaN)
- Duplicate timestamps per patient will be deduplicated
"""

