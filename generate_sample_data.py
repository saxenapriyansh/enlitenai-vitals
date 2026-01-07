"""
Script to generate sample CSV data files for 5 patients over 1 month.

Generates realistic vital signs data with:
- Different sampling frequencies per vital per patient
- Missing data periods for some signals
- Seizure events with varying patterns

Run: python generate_sample_data.py
Output: Creates 'sample_data/' folder with CSV files
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Create output directory
OUTPUT_DIR = "sample_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration for each patient
PATIENT_CONFIGS = {
    "patient_001": {
        "description": "ICU patient - High frequency monitoring",
        "heart_rate": {"freq_seconds": 1, "missing_periods": [(5, 6), (15, 15.5)]},  # per-second, missing day 5-6, 15-15.5
        "spo2": {"freq_seconds": 60, "missing_periods": [(10, 11)]},  # per-minute
        "temperature": {"freq_seconds": 3600, "missing_periods": []},  # per-hour
        "blood_glucose": {"freq_seconds": 21600, "missing_periods": [(20, 22)]},  # per-6-hours
        "seizure_rate": 0.8,  # per day
        "has_clusters": True,
    },
    "patient_002": {
        "description": "General ward - Standard monitoring",
        "heart_rate": {"freq_seconds": 60, "missing_periods": [(3, 4), (18, 19)]},  # per-minute
        "spo2": {"freq_seconds": 300, "missing_periods": [(7, 8)]},  # per-5-minutes
        "temperature": {"freq_seconds": 3600, "missing_periods": [(12, 13)]},  # per-hour
        "blood_glucose": {"freq_seconds": 86400, "missing_periods": []},  # per-day
        "seizure_rate": 0.3,
        "has_clusters": False,
    },
    "patient_003": {
        "description": "Stable outpatient - Sparse monitoring, NO seizures",
        "heart_rate": {"freq_seconds": 300, "missing_periods": [(1, 3), (14, 16)]},  # per-5-minutes
        "spo2": {"freq_seconds": 600, "missing_periods": [(5, 7), (22, 24)]},  # per-10-minutes
        "temperature": {"freq_seconds": 7200, "missing_periods": [(10, 12)]},  # per-2-hours
        "blood_glucose": {"freq_seconds": 86400, "missing_periods": [(8, 10)]},  # per-day
        "seizure_rate": 0.0,  # NO SEIZURES - test empty state
        "has_clusters": False,
    },
    "patient_004": {
        "description": "Epilepsy unit - Frequent seizures with clusters, IRREGULAR temp",
        "heart_rate": {"freq_seconds": 1, "missing_periods": [(9, 9.5)]},  # per-second
        "spo2": {"freq_seconds": 30, "missing_periods": []},  # per-30-seconds
        "temperature": {"irregular": True, "entries_per_hour": (2, 7), "missing_periods": [(25, 26)]},  # IRREGULAR: 2-7 per hour
        "blood_glucose": {"freq_seconds": 14400, "missing_periods": []},  # per-4-hours
        "seizure_rate": 3.0,  # frequent seizures
        "has_clusters": True,
    },
    "patient_005": {
        "description": "Diabetic patient - Focus on glucose monitoring",
        "heart_rate": {"freq_seconds": 120, "missing_periods": [(4, 5), (17, 18)]},  # per-2-minutes
        "spo2": {"freq_seconds": 300, "missing_periods": [(11, 13)]},  # per-5-minutes
        "temperature": {"freq_seconds": 3600, "missing_periods": []},  # per-hour
        "blood_glucose": {"freq_seconds": 900, "missing_periods": [(20, 21)]},  # per-15-minutes (CGM-like)
        "seizure_rate": 0.5,
        "has_clusters": False,
    },
}

# Baseline values for each patient (some variation)
PATIENT_BASELINES = {
    "patient_001": {"hr": 78, "spo2": 96, "temp": 37.1, "glucose": 115},
    "patient_002": {"hr": 72, "spo2": 97, "temp": 36.8, "glucose": 100},
    "patient_003": {"hr": 68, "spo2": 98, "temp": 36.6, "glucose": 95},
    "patient_004": {"hr": 85, "spo2": 95, "temp": 37.3, "glucose": 125},
    "patient_005": {"hr": 75, "spo2": 97, "temp": 36.9, "glucose": 145},  # higher glucose baseline
}


def generate_irregular_timestamps(start: datetime, end: datetime, entries_per_hour: tuple, missing_periods: list) -> list:
    """
    Generate irregular timestamps with variable entries per hour.
    
    Args:
        start: Start datetime
        end: End datetime
        entries_per_hour: Tuple (min, max) entries per hour
        missing_periods: List of (start_day, end_day) tuples for missing data
    
    Returns:
        List of irregular timestamps
    """
    timestamps = []
    current_hour = start.replace(minute=0, second=0, microsecond=0)
    total_days = (end - start).days
    min_entries, max_entries = entries_per_hour
    
    while current_hour < end:
        # Check if current hour falls in a missing period
        day_of_month = (current_hour - start).total_seconds() / 86400
        
        in_missing_period = False
        for missing_start, missing_end in missing_periods:
            if missing_start <= day_of_month < missing_end:
                in_missing_period = True
                break
        
        if not in_missing_period:
            # Generate random number of entries for this hour
            num_entries = random.randint(min_entries, max_entries)
            
            # Generate random timestamps within this hour
            for _ in range(num_entries):
                # Random offset within the hour (0-3599 seconds)
                offset_seconds = random.uniform(0, 3599)
                ts = current_hour + timedelta(seconds=offset_seconds)
                if start <= ts < end:
                    timestamps.append(ts)
        
        # Move to next hour
        current_hour += timedelta(hours=1)
    
    # Sort timestamps
    timestamps.sort()
    return timestamps


def generate_timestamps(start: datetime, end: datetime, freq_seconds: int, missing_periods: list) -> list:
    """Generate timestamps with specified frequency and missing periods."""
    timestamps = []
    current = start
    total_days = (end - start).days
    
    while current < end:
        # Check if current time falls in a missing period
        day_of_month = (current - start).total_seconds() / 86400
        
        in_missing_period = False
        for missing_start, missing_end in missing_periods:
            if missing_start <= day_of_month < missing_end:
                in_missing_period = True
                break
        
        if not in_missing_period:
            # Add some jitter for realism (±10% of interval)
            jitter = random.uniform(-freq_seconds * 0.05, freq_seconds * 0.05)
            ts = current + timedelta(seconds=jitter)
            timestamps.append(ts)
        
        current += timedelta(seconds=freq_seconds)
    
    return timestamps


def generate_heart_rate(timestamps: list, baseline: float, has_events: bool = False) -> list:
    """Generate realistic heart rate values."""
    n = len(timestamps)
    if n == 0:
        return []
    
    values = []
    for i, ts in enumerate(timestamps):
        # Circadian rhythm (lower at night)
        hour = ts.hour
        circadian = -5 * np.cos(2 * np.pi * (hour - 14) / 24)  # Peak around 2pm
        
        # Random walk component
        if i == 0:
            walk = 0
        else:
            walk = values[-1] - baseline + random.gauss(0, 0.5)
            walk = max(-10, min(10, walk))  # Clamp
        
        # Noise
        noise = random.gauss(0, 2)
        
        # Occasional spikes
        spike = 0
        if has_events and random.random() < 0.001:
            spike = random.uniform(20, 40)
        
        value = baseline + circadian + walk * 0.3 + noise + spike
        value = max(40, min(180, value))
        values.append(round(value, 1))
    
    return values


def generate_spo2(timestamps: list, baseline: float) -> list:
    """Generate realistic SpO2 values with occasional dips."""
    n = len(timestamps)
    if n == 0:
        return []
    
    values = []
    in_dip = False
    dip_remaining = 0
    dip_depth = 0
    
    for i in range(n):
        # Check for new dip event
        if not in_dip and random.random() < 0.0005:
            in_dip = True
            dip_remaining = random.randint(5, 30)
            dip_depth = random.uniform(3, 12)
        
        # Calculate value
        noise = random.gauss(0, 0.3)
        
        if in_dip:
            # Smooth dip shape
            progress = 1 - (dip_remaining / 30)
            dip_effect = dip_depth * np.sin(progress * np.pi)
            value = baseline - dip_effect + noise
            dip_remaining -= 1
            if dip_remaining <= 0:
                in_dip = False
        else:
            value = baseline + noise
        
        value = max(70, min(100, value))
        values.append(round(value, 1))
    
    return values


def generate_temperature(timestamps: list, baseline: float) -> list:
    """Generate realistic temperature values with circadian rhythm."""
    n = len(timestamps)
    if n == 0:
        return []
    
    values = []
    fever_active = False
    fever_remaining = 0
    
    for ts in timestamps:
        hour = ts.hour
        
        # Circadian rhythm (lower in morning, higher in evening)
        circadian = 0.3 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Random chance of fever episode
        if not fever_active and random.random() < 0.001:
            fever_active = True
            fever_remaining = random.randint(10, 30)
        
        # Noise
        noise = random.gauss(0, 0.1)
        
        # Fever effect
        fever_effect = 0
        if fever_active:
            fever_effect = random.uniform(0.5, 1.5)
            fever_remaining -= 1
            if fever_remaining <= 0:
                fever_active = False
        
        value = baseline + circadian + noise + fever_effect
        value = max(35.0, min(41.0, value))
        values.append(round(value, 2))
    
    return values


def generate_blood_glucose(timestamps: list, baseline: float) -> list:
    """Generate realistic blood glucose values with meal spikes."""
    n = len(timestamps)
    if n == 0:
        return []
    
    values = []
    
    for ts in timestamps:
        hour = ts.hour
        
        # Meal effects (breakfast ~8am, lunch ~12pm, dinner ~6pm)
        breakfast = 25 * np.exp(-((hour - 9) ** 2) / 4) if 6 <= hour <= 12 else 0
        lunch = 30 * np.exp(-((hour - 13) ** 2) / 4) if 10 <= hour <= 16 else 0
        dinner = 35 * np.exp(-((hour - 19) ** 2) / 4) if 16 <= hour <= 23 else 0
        
        meal_effect = breakfast + lunch + dinner
        
        # Random variation
        noise = random.gauss(0, 8)
        
        # Day-to-day variation
        daily_var = random.gauss(0, 5)
        
        value = baseline + meal_effect + noise + daily_var
        value = max(40, min(350, value))
        values.append(round(value, 0))
    
    return values


def generate_seizures(patient_id: str, config: dict, start: datetime, end: datetime) -> pd.DataFrame:
    """Generate seizure events for a patient."""
    seizure_rate = config["seizure_rate"]
    has_clusters = config["has_clusters"]
    
    if seizure_rate == 0:
        return pd.DataFrame(columns=["patient_id", "seizure_time", "seizure_type", "severity", "duration_seconds"])
    
    total_days = (end - start).days
    expected_seizures = int(seizure_rate * total_days)
    num_seizures = max(0, expected_seizures + random.randint(-3, 5))
    
    seizure_types = ["focal", "generalized", "absence", "tonic-clonic", "myoclonic"]
    severities = ["mild", "moderate", "severe"]
    
    seizures = []
    
    if has_clusters and num_seizures > 5:
        # Create 2-4 clusters
        num_clusters = random.randint(2, 4)
        seizures_per_cluster = num_seizures // num_clusters
        
        for _ in range(num_clusters):
            # Random cluster center
            cluster_day = random.uniform(1, total_days - 1)
            cluster_center = start + timedelta(days=cluster_day)
            
            for _ in range(seizures_per_cluster):
                # Seizures within ±8 hours of cluster center
                offset_hours = random.gauss(0, 3)
                seizure_time = cluster_center + timedelta(hours=offset_hours)
                
                if start <= seizure_time <= end:
                    seizures.append({
                        "patient_id": patient_id,
                        "seizure_time": seizure_time,
                        "seizure_type": random.choice(seizure_types),
                        "severity": random.choice(severities),
                        "duration_seconds": random.randint(15, 300),
                    })
    else:
        # Randomly distributed
        for _ in range(num_seizures):
            seizure_day = random.uniform(0, total_days)
            seizure_time = start + timedelta(days=seizure_day)
            
            seizures.append({
                "patient_id": patient_id,
                "seizure_time": seizure_time,
                "seizure_type": random.choice(seizure_types),
                "severity": random.choice(severities),
                "duration_seconds": random.randint(15, 300),
            })
    
    df = pd.DataFrame(seizures)
    if not df.empty:
        df = df.sort_values("seizure_time").reset_index(drop=True)
    
    return df


def generate_patient_data(patient_id: str, config: dict, baselines: dict, 
                          start: datetime, end: datetime) -> tuple:
    """Generate all vitals data for a single patient."""
    
    # Generate timestamps for each vital
    hr_timestamps = generate_timestamps(
        start, end, 
        config["heart_rate"]["freq_seconds"],
        config["heart_rate"]["missing_periods"]
    )
    
    spo2_timestamps = generate_timestamps(
        start, end,
        config["spo2"]["freq_seconds"],
        config["spo2"]["missing_periods"]
    )
    
    # Temperature: check if irregular sampling is configured
    temp_config = config["temperature"]
    if temp_config.get("irregular", False):
        # Use irregular timestamps (2-7 entries per hour for patient_004)
        temp_timestamps = generate_irregular_timestamps(
            start, end,
            temp_config["entries_per_hour"],
            temp_config["missing_periods"]
        )
    else:
        temp_timestamps = generate_timestamps(
            start, end,
            temp_config["freq_seconds"],
            temp_config["missing_periods"]
        )
    
    glucose_timestamps = generate_timestamps(
        start, end,
        config["blood_glucose"]["freq_seconds"],
        config["blood_glucose"]["missing_periods"]
    )
    
    # Generate values
    hr_values = generate_heart_rate(hr_timestamps, baselines["hr"], config.get("has_clusters", False))
    spo2_values = generate_spo2(spo2_timestamps, baselines["spo2"])
    temp_values = generate_temperature(temp_timestamps, baselines["temp"])
    glucose_values = generate_blood_glucose(glucose_timestamps, baselines["glucose"])
    
    # Create DataFrames
    hr_df = pd.DataFrame({
        "timestamp": hr_timestamps,
        "patient_id": patient_id,
        "heart_rate": hr_values,
    })
    
    spo2_df = pd.DataFrame({
        "timestamp": spo2_timestamps,
        "patient_id": patient_id,
        "spo2": spo2_values,
    })
    
    temp_df = pd.DataFrame({
        "timestamp": temp_timestamps,
        "patient_id": patient_id,
        "temperature": temp_values,
    })
    
    glucose_df = pd.DataFrame({
        "timestamp": glucose_timestamps,
        "patient_id": patient_id,
        "blood_glucose": glucose_values,
    })
    
    # Merge all vitals (outer join)
    vitals_df = hr_df.merge(spo2_df, on=["timestamp", "patient_id"], how="outer")
    vitals_df = vitals_df.merge(temp_df, on=["timestamp", "patient_id"], how="outer")
    vitals_df = vitals_df.merge(glucose_df, on=["timestamp", "patient_id"], how="outer")
    vitals_df = vitals_df.sort_values("timestamp").reset_index(drop=True)
    
    # Generate seizures
    seizures_df = generate_seizures(patient_id, config, start, end)
    
    return vitals_df, seizures_df


def main():
    """Generate all sample data files."""
    print("=" * 60)
    print("Generating Sample Patient Data (1 Month)")
    print("=" * 60)
    
    # Time range: 1 month
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    end_date = datetime(2024, 2, 1, 0, 0, 0)
    
    all_vitals = []
    all_seizures = []
    
    for patient_id, config in PATIENT_CONFIGS.items():
        print(f"\nGenerating data for {patient_id}...")
        print(f"  Description: {config['description']}")
        
        baselines = PATIENT_BASELINES[patient_id]
        
        vitals_df, seizures_df = generate_patient_data(
            patient_id, config, baselines, start_date, end_date
        )
        
        # Print statistics
        print(f"  Heart Rate samples: {vitals_df['heart_rate'].notna().sum():,} "
              f"(every {config['heart_rate']['freq_seconds']}s)")
        print(f"  SpO2 samples: {vitals_df['spo2'].notna().sum():,} "
              f"(every {config['spo2']['freq_seconds']}s)")
        
        # Temperature: handle irregular case
        temp_config = config['temperature']
        if temp_config.get('irregular', False):
            temp_info = f"IRREGULAR {temp_config['entries_per_hour'][0]}-{temp_config['entries_per_hour'][1]} per hour"
        else:
            temp_info = f"every {temp_config['freq_seconds']}s"
        print(f"  Temperature samples: {vitals_df['temperature'].notna().sum():,} ({temp_info})")
        
        print(f"  Blood Glucose samples: {vitals_df['blood_glucose'].notna().sum():,} "
              f"(every {config['blood_glucose']['freq_seconds']}s)")
        print(f"  Seizure events: {len(seizures_df)}")
        print(f"  Missing periods: HR={config['heart_rate']['missing_periods']}, "
              f"SpO2={config['spo2']['missing_periods']}, Temp={temp_config['missing_periods']}")
        
        # Save individual patient files
        patient_vitals_file = os.path.join(OUTPUT_DIR, f"{patient_id}_vitals.csv")
        vitals_df.to_csv(patient_vitals_file, index=False)
        print(f"  Saved: {patient_vitals_file}")
        
        if not seizures_df.empty:
            patient_seizures_file = os.path.join(OUTPUT_DIR, f"{patient_id}_seizures.csv")
            seizures_df.to_csv(patient_seizures_file, index=False)
            print(f"  Saved: {patient_seizures_file}")
        
        all_vitals.append(vitals_df)
        all_seizures.append(seizures_df)
    
    # Combine all patients into single files
    print("\n" + "=" * 60)
    print("Creating combined files...")
    
    combined_vitals = pd.concat(all_vitals, ignore_index=True)
    combined_vitals = combined_vitals.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    combined_vitals_file = os.path.join(OUTPUT_DIR, "all_patients_vitals.csv")
    combined_vitals.to_csv(combined_vitals_file, index=False)
    print(f"Saved: {combined_vitals_file} ({len(combined_vitals):,} rows)")
    
    combined_seizures = pd.concat(all_seizures, ignore_index=True)
    if not combined_seizures.empty:
        combined_seizures = combined_seizures.sort_values(["patient_id", "seizure_time"]).reset_index(drop=True)
    combined_seizures_file = os.path.join(OUTPUT_DIR, "all_patients_seizures.csv")
    combined_seizures.to_csv(combined_seizures_file, index=False)
    print(f"Saved: {combined_seizures_file} ({len(combined_seizures):,} rows)")
    
    # Create a combined file with seizure flag (for upload testing)
    print("\nCreating upload-ready combined file with seizure column...")
    
    # Add seizure flag to vitals
    combined_with_seizures = combined_vitals.copy()
    combined_with_seizures["seizure"] = 0
    
    for _, seizure in combined_seizures.iterrows():
        patient_mask = combined_with_seizures["patient_id"] == seizure["patient_id"]
        # Find closest timestamp
        if patient_mask.any():
            patient_data = combined_with_seizures[patient_mask]
            time_diffs = abs(patient_data["timestamp"] - seizure["seizure_time"])
            closest_idx = time_diffs.idxmin()
            if time_diffs[closest_idx].total_seconds() < 60:  # Within 1 minute
                combined_with_seizures.loc[closest_idx, "seizure"] = 1
    
    upload_ready_file = os.path.join(OUTPUT_DIR, "upload_ready_all_patients.csv")
    combined_with_seizures.to_csv(upload_ready_file, index=False)
    print(f"Saved: {upload_ready_file} ({len(combined_with_seizures):,} rows)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"Time range: {start_date.date()} to {end_date.date()} (1 month)")
    print(f"Total patients: {len(PATIENT_CONFIGS)}")
    print(f"Total vital records: {len(combined_vitals):,}")
    print(f"Total seizure events: {len(combined_seizures)}")
    print("\nFiles created:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        filepath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  - {f} ({size_kb:.1f} KB)")
    
    print("\n" + "=" * 60)
    print("DATA CHARACTERISTICS BY PATIENT")
    print("=" * 60)
    for patient_id, config in PATIENT_CONFIGS.items():
        print(f"\n{patient_id}: {config['description']}")
        print(f"  HR: every {config['heart_rate']['freq_seconds']}s, missing days: {config['heart_rate']['missing_periods']}")
        print(f"  SpO2: every {config['spo2']['freq_seconds']}s, missing days: {config['spo2']['missing_periods']}")
        
        # Temperature: handle irregular case
        temp_config = config['temperature']
        if temp_config.get('irregular', False):
            temp_freq = f"IRREGULAR {temp_config['entries_per_hour'][0]}-{temp_config['entries_per_hour'][1]} per hour"
        else:
            temp_freq = f"every {temp_config['freq_seconds']}s"
        print(f"  Temp: {temp_freq}, missing days: {temp_config['missing_periods']}")
        
        print(f"  Glucose: every {config['blood_glucose']['freq_seconds']}s, missing days: {config['blood_glucose']['missing_periods']}")
        print(f"  Seizures: ~{config['seizure_rate']}/day" + (" (clustered)" if config['has_clusters'] else ""))


if __name__ == "__main__":
    main()

