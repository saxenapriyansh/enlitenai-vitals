"""
Dummy data generation for Patient Vitals & Seizure Timeline app.
Generates realistic multi-frequency vitals and seizure events for 5 patients.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import random

from config import VITALS_CONFIG, SAMPLING_FREQUENCIES


def generate_dummy_data(
    num_patients: int = 5,
    days: int = 7,
    start_date: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate dummy vitals and seizure data for multiple patients.
    
    Returns:
        vitals_df: DataFrame with columns [timestamp, patient_id, spo2, heart_rate, temperature, blood_glucose]
        seizures_df: DataFrame with columns [patient_id, seizure_time, seizure_type, severity, duration_seconds]
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    end_date = start_date + timedelta(days=days)
    
    all_vitals = []
    all_seizures = []
    
    # Patient-specific configurations for variety
    patient_configs = _generate_patient_configs(num_patients)
    
    for i in range(num_patients):
        patient_id = f"patient_{i+1:03d}"
        config = patient_configs[i]
        
        # Generate vitals for this patient
        patient_vitals = _generate_patient_vitals(
            patient_id=patient_id,
            start_date=start_date,
            end_date=end_date,
            config=config,
        )
        all_vitals.append(patient_vitals)
        
        # Generate seizures for this patient
        patient_seizures = _generate_patient_seizures(
            patient_id=patient_id,
            start_date=start_date,
            end_date=end_date,
            config=config,
        )
        all_seizures.append(patient_seizures)
    
    vitals_df = pd.concat(all_vitals, ignore_index=True)
    seizures_df = pd.concat(all_seizures, ignore_index=True)
    
    return vitals_df, seizures_df


def _generate_patient_configs(num_patients: int) -> list:
    """Generate varied configurations for each patient."""
    configs = []
    
    # Predefined patient profiles for realistic variety
    profiles = [
        {
            "name": "High-frequency monitoring",
            "hr_freq": "per_second",
            "spo2_freq": "per_minute",
            "temp_freq": "per_hour",
            "glucose_freq": "per_6_hours",
            "seizure_rate": 0.8,  # seizures per day average
            "has_clusters": True,
            "baseline_hr": 75,
            "baseline_spo2": 97,
            "baseline_temp": 36.8,
            "baseline_glucose": 110,
        },
        {
            "name": "Standard ICU monitoring",
            "hr_freq": "per_minute",
            "spo2_freq": "per_minute",
            "temp_freq": "per_hour",
            "glucose_freq": "per_day",
            "seizure_rate": 0.3,
            "has_clusters": False,
            "baseline_hr": 82,
            "baseline_spo2": 96,
            "baseline_temp": 37.0,
            "baseline_glucose": 95,
        },
        {
            "name": "Stable patient - sparse monitoring",
            "hr_freq": "per_5_minutes",
            "spo2_freq": "per_5_minutes",
            "temp_freq": "per_hour",
            "glucose_freq": "per_day",
            "seizure_rate": 0.0,  # No seizures - test empty state
            "has_clusters": False,
            "baseline_hr": 68,
            "baseline_spo2": 98,
            "baseline_temp": 36.6,
            "baseline_glucose": 100,
        },
        {
            "name": "Critical patient - frequent seizures",
            "hr_freq": "per_second",
            "spo2_freq": "per_minute",
            "temp_freq": "per_hour",
            "glucose_freq": "per_6_hours",
            "seizure_rate": 2.5,
            "has_clusters": True,
            "baseline_hr": 95,
            "baseline_spo2": 94,
            "baseline_temp": 37.5,
            "baseline_glucose": 140,
        },
        {
            "name": "Diabetic patient focus",
            "hr_freq": "per_minute",
            "spo2_freq": "per_5_minutes",
            "temp_freq": "per_hour",
            "glucose_freq": "per_6_hours",
            "seizure_rate": 0.5,
            "has_clusters": False,
            "baseline_hr": 78,
            "baseline_spo2": 97,
            "baseline_temp": 36.9,
            "baseline_glucose": 160,
        },
    ]
    
    for i in range(num_patients):
        configs.append(profiles[i % len(profiles)])
    
    return configs


def _generate_patient_vitals(
    patient_id: str,
    start_date: datetime,
    end_date: datetime,
    config: dict,
) -> pd.DataFrame:
    """Generate vitals for a single patient with varied frequencies."""
    
    vitals_data = {}
    
    # Generate timestamps for each vital based on its frequency
    hr_timestamps = _generate_timestamps(
        start_date, end_date, config["hr_freq"], missing_rate=0.02
    )
    spo2_timestamps = _generate_timestamps(
        start_date, end_date, config["spo2_freq"], missing_rate=0.03
    )
    temp_timestamps = _generate_timestamps(
        start_date, end_date, config["temp_freq"], missing_rate=0.05
    )
    glucose_timestamps = _generate_timestamps(
        start_date, end_date, config["glucose_freq"], missing_rate=0.1
    )
    
    # Generate values for each vital
    hr_values = _generate_heart_rate(
        len(hr_timestamps), config["baseline_hr"], config.get("has_clusters", False)
    )
    spo2_values = _generate_spo2(
        len(spo2_timestamps), config["baseline_spo2"]
    )
    temp_values = _generate_temperature(
        len(temp_timestamps), config["baseline_temp"]
    )
    glucose_values = _generate_glucose(
        len(glucose_timestamps), config["baseline_glucose"]
    )
    
    # Create individual DataFrames for each vital
    hr_df = pd.DataFrame({
        "timestamp": hr_timestamps,
        "heart_rate": hr_values,
    })
    
    spo2_df = pd.DataFrame({
        "timestamp": spo2_timestamps,
        "spo2": spo2_values,
    })
    
    temp_df = pd.DataFrame({
        "timestamp": temp_timestamps,
        "temperature": temp_values,
    })
    
    glucose_df = pd.DataFrame({
        "timestamp": glucose_timestamps,
        "blood_glucose": glucose_values,
    })
    
    # Merge all vitals on timestamp (outer join to preserve all timestamps)
    merged = hr_df.merge(spo2_df, on="timestamp", how="outer")
    merged = merged.merge(temp_df, on="timestamp", how="outer")
    merged = merged.merge(glucose_df, on="timestamp", how="outer")
    
    merged["patient_id"] = patient_id
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    
    return merged


def _generate_timestamps(
    start_date: datetime,
    end_date: datetime,
    freq_key: str,
    missing_rate: float = 0.0,
) -> list:
    """Generate timestamps with optional missing data gaps."""
    freq_config = SAMPLING_FREQUENCIES[freq_key]
    interval_seconds = freq_config["seconds"]
    
    timestamps = []
    current = start_date
    
    while current < end_date:
        # Randomly skip some timestamps to simulate missing data
        if random.random() > missing_rate:
            # Add small jitter to make data more realistic
            jitter = random.uniform(-interval_seconds * 0.1, interval_seconds * 0.1)
            ts = current + timedelta(seconds=jitter)
            timestamps.append(ts)
        current += timedelta(seconds=interval_seconds)
    
    return timestamps


def _generate_heart_rate(n: int, baseline: float, has_events: bool = False) -> np.ndarray:
    """Generate realistic heart rate values."""
    # Base signal with slow variations (circadian rhythm simulation)
    t = np.linspace(0, 24 * (n / 3600), n)  # Approximate hours
    circadian = 5 * np.sin(2 * np.pi * t / 24 - np.pi / 2)  # Peak in afternoon
    
    # Random walk component for gradual changes
    walk = np.cumsum(np.random.normal(0, 0.5, n))
    walk = walk - np.linspace(walk[0], walk[-1], n)  # Remove trend
    
    # High-frequency noise
    noise = np.random.normal(0, 2, n)
    
    # Occasional spikes (e.g., movement, stress)
    spikes = np.zeros(n)
    if has_events:
        num_spikes = max(1, n // 5000)
        spike_indices = np.random.choice(n, num_spikes, replace=False)
        for idx in spike_indices:
            spike_len = min(60, n - idx)
            spike_shape = np.exp(-np.linspace(0, 3, spike_len))
            spikes[idx:idx+spike_len] += np.random.uniform(20, 40) * spike_shape
    
    hr = baseline + circadian + walk * 0.5 + noise + spikes
    hr = np.clip(hr, 40, 180)
    
    return np.round(hr, 1)


def _generate_spo2(n: int, baseline: float) -> np.ndarray:
    """Generate realistic SpO2 values with occasional dips."""
    # SpO2 is typically stable but can have sudden dips
    values = np.full(n, baseline, dtype=float)
    
    # Add small random variation
    values += np.random.normal(0, 0.5, n)
    
    # Add occasional desaturation events
    num_dips = max(1, n // 500)
    for _ in range(num_dips):
        dip_start = random.randint(0, max(0, n - 20))
        dip_len = random.randint(5, 20)
        dip_depth = random.uniform(3, 10)
        dip_end = min(dip_start + dip_len, n)
        
        # Create a smooth dip shape
        dip_shape = np.sin(np.linspace(0, np.pi, dip_end - dip_start))
        values[dip_start:dip_end] -= dip_depth * dip_shape
    
    values = np.clip(values, 70, 100)
    return np.round(values, 1)


def _generate_temperature(n: int, baseline: float) -> np.ndarray:
    """Generate realistic temperature values."""
    # Temperature has circadian rhythm (lower at night, higher in evening)
    t = np.linspace(0, 24 * (n / 24), n)
    circadian = 0.3 * np.sin(2 * np.pi * t / 24 - np.pi / 3)
    
    # Small random variations
    noise = np.random.normal(0, 0.1, n)
    
    # Occasional fever spikes
    fever_spikes = np.zeros(n)
    if random.random() < 0.3:  # 30% chance of fever episode
        fever_start = random.randint(0, max(0, n - 10))
        fever_len = random.randint(5, min(15, n - fever_start))
        fever_peak = random.uniform(0.5, 2.0)
        fever_shape = np.sin(np.linspace(0, np.pi, fever_len))
        fever_spikes[fever_start:fever_start+fever_len] = fever_peak * fever_shape
    
    temp = baseline + circadian + noise + fever_spikes
    temp = np.clip(temp, 35, 41)
    
    return np.round(temp, 2)


def _generate_glucose(n: int, baseline: float) -> np.ndarray:
    """Generate realistic blood glucose values."""
    # Glucose varies with meals (assume 3 meals per day)
    t = np.linspace(0, 24 * (n / 4), n)  # Approximate hours
    
    # Meal-related spikes at roughly 8am, 12pm, 6pm
    meal_effect = (
        20 * np.exp(-((t % 24 - 9) ** 2) / 4) +   # Breakfast spike
        25 * np.exp(-((t % 24 - 13) ** 2) / 4) +  # Lunch spike
        30 * np.exp(-((t % 24 - 19) ** 2) / 4)    # Dinner spike
    )
    
    # Random daily variation
    noise = np.random.normal(0, 10, n)
    
    glucose = baseline + meal_effect + noise
    glucose = np.clip(glucose, 40, 350)
    
    return np.round(glucose, 0)


def _generate_patient_seizures(
    patient_id: str,
    start_date: datetime,
    end_date: datetime,
    config: dict,
) -> pd.DataFrame:
    """Generate seizure events for a single patient."""
    seizure_rate = config["seizure_rate"]
    has_clusters = config.get("has_clusters", False)
    
    if seizure_rate == 0:
        return pd.DataFrame(columns=[
            "patient_id", "seizure_time", "seizure_type", "severity", "duration_seconds"
        ])
    
    total_days = (end_date - start_date).days
    expected_seizures = int(seizure_rate * total_days)
    
    # Add some randomness to count
    num_seizures = max(0, expected_seizures + random.randint(-2, 3))
    
    seizure_types = ["focal", "generalized", "absence", "tonic-clonic"]
    severities = ["mild", "moderate", "severe"]
    
    seizures = []
    
    if has_clusters and num_seizures > 3:
        # Create 1-2 clusters of seizures
        num_clusters = random.randint(1, 2)
        cluster_sizes = np.random.multinomial(num_seizures, [1/num_clusters] * num_clusters)
        
        for cluster_size in cluster_sizes:
            # Random cluster center time
            cluster_center = start_date + timedelta(
                seconds=random.uniform(0, total_days * 86400)
            )
            
            for _ in range(cluster_size):
                # Seizures within ±6 hours of cluster center
                offset = timedelta(hours=random.gauss(0, 2))
                seizure_time = cluster_center + offset
                
                if start_date <= seizure_time <= end_date:
                    seizures.append({
                        "patient_id": patient_id,
                        "seizure_time": seizure_time,
                        "seizure_type": random.choice(seizure_types),
                        "severity": random.choice(severities),
                        "duration_seconds": random.randint(30, 300),
                    })
    else:
        # Randomly distributed seizures
        for _ in range(num_seizures):
            seizure_time = start_date + timedelta(
                seconds=random.uniform(0, total_days * 86400)
            )
            seizures.append({
                "patient_id": patient_id,
                "seizure_time": seizure_time,
                "seizure_type": random.choice(seizure_types),
                "severity": random.choice(severities),
                "duration_seconds": random.randint(30, 300),
            })
    
    df = pd.DataFrame(seizures)
    if not df.empty:
        df = df.sort_values("seizure_time").reset_index(drop=True)
    
    return df


def get_patient_info(vitals_df: pd.DataFrame, seizures_df: pd.DataFrame) -> Dict:
    """Get summary information about each patient's data."""
    info = {}
    
    for patient_id in vitals_df["patient_id"].unique():
        patient_vitals = vitals_df[vitals_df["patient_id"] == patient_id]
        patient_seizures = seizures_df[seizures_df["patient_id"] == patient_id]
        
        # Calculate data availability for each vital
        vital_info = {}
        for vital in ["spo2", "heart_rate", "temperature", "blood_glucose"]:
            if vital in patient_vitals.columns:
                valid_count = patient_vitals[vital].notna().sum()
                if valid_count > 0:
                    vital_info[vital] = {
                        "count": valid_count,
                        "start": patient_vitals.loc[patient_vitals[vital].notna(), "timestamp"].min(),
                        "end": patient_vitals.loc[patient_vitals[vital].notna(), "timestamp"].max(),
                    }
        
        info[patient_id] = {
            "vitals": vital_info,
            "seizure_count": len(patient_seizures),
            "total_records": len(patient_vitals),
            "date_range": (
                patient_vitals["timestamp"].min(),
                patient_vitals["timestamp"].max(),
            ),
        }
    
    return info

