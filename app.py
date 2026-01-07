"""
Patient Vitals & Seizure Timeline - Streamlit Application

A production-quality visualization app for patient time-series vitals and seizure events.

=== USAGE ===
Run with: streamlit run app.py

=== MODES ===
1. DUMMY DATA MODE (default):
   - 5 pre-generated patients (patient_001 to patient_005)
   - Realistic multi-frequency vitals: HR (per-second/minute), SpO2 (per-minute),
     Temperature (per-hour), Blood Glucose (per-6-hours/day)
   - Seizure events with clustering patterns
   
2. FILE UPLOAD MODE:
   - Upload your own CSV file
   - Required column: timestamp (ISO8601 or Unix epoch)
   - Optional columns: patient_id, spo2, heart_rate, temperature, blood_glucose
   - Seizure columns: seizure (0/1), OR seizure_event (0/1), OR seizure_start + seizure_end

=== RESAMPLING ===
- Raw: Show all data points (auto-downsamples if >200k points)
- Auto: Adaptive resampling based on time window and data density
- Manual: Select specific granularity (second, minute, hour, day)
- Aggregation: mean or median
- Variability band: Shows p10-p90 range when resampling

=== SEIZURE DISPLAY ===
- Markers: Circle markers on vital charts at seizure times
- Density: Bar chart showing seizure counts per hour/day
- Intervals: If seizure_start/end provided, shows shaded regions

=== VISUALIZATION MODES ===
A) Stacked Small Multiples: Each vital in its own row with shared x-axis
B) Single Combined Chart: All vitals on one chart with multiple y-axes (best for 2-3 vitals)

Author: EnlitenAI
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# Local imports
from config import (
    VITALS_CONFIG,
    VITAL_KEYS,
    RESAMPLE_OPTIONS,
    AGGREGATION_METHODS,
    SEIZURE_DENSITY_BINS,
    XAXIS_TIME_UNITS,
    DEFAULT_DAYS,
    DEFAULT_NUM_PATIENTS,
)
from data_generator import generate_dummy_data, get_patient_info
from data_loader import load_uploaded_data, get_csv_schema_help, DataValidationError
from data_processor import (
    preprocess_and_filter,
    infer_sampling_frequency_per_vital,
    resample_vitals,
    compute_seizure_density,
    compute_vital_stats,
    compute_seizure_stats,
    build_seizure_events_table,
)
from charts import build_plot_stacked, build_plot_combined


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Patient Vitals & Seizure Timeline",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6C7D;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stat-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
    }
    .freq-badge {
        background-color: #E8F4FD;
        color: #1E88E5;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 4px 4px 0;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 4px 4px 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.3rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "data_source": "dummy",
        "vitals_df": None,
        "seizures_df": None,
        "patient_list": [],
        "selected_patient": None,
        "data_loaded": False,
        "upload_warnings": [],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_dummy_data(num_patients: int = DEFAULT_NUM_PATIENTS, days: int = DEFAULT_DAYS):
    """Load or generate dummy data (cached)."""
    return generate_dummy_data(num_patients=num_patients, days=days)


def load_data():
    """Load data based on selected source."""
    if st.session_state.data_source == "dummy":
        with st.spinner("Generating dummy data for 5 patients..."):
            vitals_df, seizures_df = load_dummy_data()
            st.session_state.vitals_df = vitals_df
            st.session_state.seizures_df = seizures_df
            st.session_state.patient_list = sorted(vitals_df["patient_id"].unique().tolist())
            st.session_state.data_loaded = True
            st.session_state.upload_warnings = []
            
            if st.session_state.selected_patient not in st.session_state.patient_list:
                st.session_state.selected_patient = st.session_state.patient_list[0]


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar with all controls."""
    st.sidebar.title("⚙️ Controls")
    
    # Data Source Section
    st.sidebar.header("📊 Data Source")
    
    data_source = st.sidebar.radio(
        "Select data source",
        options=["dummy", "upload"],
        format_func=lambda x: "Dummy Data (5 Patients)" if x == "dummy" else "Upload CSV",
        key="data_source_radio",
        horizontal=True,
    )
    
    if data_source != st.session_state.data_source:
        st.session_state.data_source = data_source
        st.session_state.data_loaded = False
    
    # File upload
    if st.session_state.data_source == "upload":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Upload a CSV file with patient vitals data",
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Processing uploaded file..."):
                    vitals_df, seizures_df, warnings = load_uploaded_data(uploaded_file)
                    st.session_state.vitals_df = vitals_df
                    st.session_state.seizures_df = seizures_df
                    st.session_state.patient_list = sorted(vitals_df["patient_id"].unique().tolist())
                    st.session_state.data_loaded = True
                    st.session_state.upload_warnings = warnings
                    
                    if st.session_state.selected_patient not in st.session_state.patient_list:
                        st.session_state.selected_patient = st.session_state.patient_list[0]
                    
                    st.sidebar.success("✅ File loaded successfully!")
                    
            except DataValidationError as e:
                st.sidebar.error(f"❌ Validation Error: {str(e)}")
                st.session_state.data_loaded = False
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
                st.session_state.data_loaded = False
        
        # Show schema help
        with st.sidebar.expander("📋 CSV Schema Help"):
            st.markdown(get_csv_schema_help())
    
    else:
        # Auto-load dummy data
        if not st.session_state.data_loaded:
            load_data()
    
    # Show warnings if any
    if st.session_state.upload_warnings:
        for warning in st.session_state.upload_warnings:
            st.sidebar.warning(warning)
    
    st.sidebar.divider()
    
    # Patient Selection
    st.sidebar.header("👤 Patient Selection")
    
    if st.session_state.data_loaded and st.session_state.patient_list:
        selected_patient = st.sidebar.selectbox(
            "Select Patient",
            options=st.session_state.patient_list,
            index=st.session_state.patient_list.index(st.session_state.selected_patient) 
                  if st.session_state.selected_patient in st.session_state.patient_list else 0,
            key="patient_selector",
        )
        st.session_state.selected_patient = selected_patient
    else:
        st.sidebar.info("Load data to select a patient")
        return None
    
    st.sidebar.divider()
    
    # Time Range Selection
    st.sidebar.header("📅 Time Range")
    
    patient_vitals = st.session_state.vitals_df[
        st.session_state.vitals_df["patient_id"] == st.session_state.selected_patient
    ]
    
    min_date = patient_vitals["timestamp"].min().date()
    max_date = patient_vitals["timestamp"].max().date()
    
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_range",
    )
    
    # Handle single date selection
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range if not isinstance(date_range, tuple) else date_range[0]
    
    # Convert to datetime
    time_range = (
        datetime.combine(start_date, datetime.min.time()),
        datetime.combine(end_date, datetime.max.time()),
    )
    
    st.sidebar.divider()
    
    # Vital Selection
    st.sidebar.header("💓 Vital Signs")
    
    available_vitals = [
        v for v in VITAL_KEYS 
        if v in patient_vitals.columns and patient_vitals[v].notna().any()
    ]
    
    selected_vitals = st.sidebar.multiselect(
        "Select vitals to display",
        options=available_vitals,
        default=available_vitals,
        format_func=lambda x: VITALS_CONFIG[x]["name"],
        key="vital_selector",
    )
    
    st.sidebar.divider()
    
    # Visualization Mode
    st.sidebar.header("📈 Visualization")
    
    viz_mode = st.sidebar.radio(
        "Display mode",
        options=["stacked", "combined"],
        format_func=lambda x: "Stacked Small Multiples" if x == "stacked" else "Single Combined Chart",
        key="viz_mode",
        horizontal=True,
    )
    
    st.sidebar.divider()
    
    # Resampling Controls
    st.sidebar.header("🔄 Resolution & Resampling")
    
    granularity = st.sidebar.selectbox(
        "Granularity",
        options=list(RESAMPLE_OPTIONS.keys()),
        format_func=lambda x: RESAMPLE_OPTIONS[x]["label"],
        index=1,  # Default to "auto"
        key="granularity",
    )
    
    aggregation = st.sidebar.selectbox(
        "Aggregation method",
        options=AGGREGATION_METHODS,
        format_func=str.capitalize,
        key="aggregation",
    )
    
    show_band = st.sidebar.checkbox(
        "Show variability band (P10-P90)",
        value=False,
        key="show_band",
    )
    
    st.sidebar.divider()
    
    # X-Axis Time Unit
    st.sidebar.header("⏱️ X-Axis Time Scale")
    
    xaxis_time_unit = st.sidebar.radio(
        "X-axis tick interval",
        options=["auto", "second", "minute", "hour", "day", "month"],
        format_func=lambda x: "Auto" if x == "auto" else XAXIS_TIME_UNITS[x]["label"],
        key="xaxis_time_unit",
        horizontal=True,
    )
    
    st.sidebar.divider()
    
    # Seizure Overlay Controls
    st.sidebar.header("⚡ Seizure Overlays")
    
    show_seizure_markers = st.sidebar.checkbox(
        "Show seizure markers",
        value=True,
        key="show_seizure_markers",
    )
    
    show_seizure_density = st.sidebar.checkbox(
        "Show seizure density track",
        value=True,
        key="show_seizure_density",
    )
    
    density_bin = st.sidebar.selectbox(
        "Density bin size",
        options=list(SEIZURE_DENSITY_BINS.keys()),
        format_func=lambda x: SEIZURE_DENSITY_BINS[x]["label"],
        key="density_bin",
    )
    
    st.sidebar.divider()
    
    # Threshold Controls
    st.sidebar.header("🎯 Thresholds")
    
    show_thresholds = st.sidebar.checkbox(
        "Show threshold lines",
        value=False,
        key="show_thresholds",
    )
    
    thresholds = {}
    
    if show_thresholds:
        with st.sidebar.expander("Edit Thresholds", expanded=True):
            if "spo2" in selected_vitals:
                thresholds["spo2_low"] = st.number_input(
                    "SpO2 Low (%)",
                    value=VITALS_CONFIG["spo2"]["default_threshold_low"],
                    min_value=70,
                    max_value=100,
                    key="spo2_low_thresh",
                )
            
            if "heart_rate" in selected_vitals:
                thresholds["heart_rate_high"] = st.number_input(
                    "HR High (bpm)",
                    value=VITALS_CONFIG["heart_rate"]["default_threshold_high"],
                    min_value=60,
                    max_value=200,
                    key="hr_high_thresh",
                )
            
            if "temperature" in selected_vitals:
                thresholds["temperature_high"] = st.number_input(
                    "Temp High (°C)",
                    value=VITALS_CONFIG["temperature"]["default_threshold_high"],
                    min_value=36.0,
                    max_value=42.0,
                    step=0.1,
                    key="temp_high_thresh",
                )
            
            if "blood_glucose" in selected_vitals:
                thresholds["blood_glucose_low"] = st.number_input(
                    "Glucose Low (mg/dL)",
                    value=VITALS_CONFIG["blood_glucose"]["default_threshold_low"],
                    min_value=30,
                    max_value=150,
                    key="glucose_low_thresh",
                )
                thresholds["blood_glucose_high"] = st.number_input(
                    "Glucose High (mg/dL)",
                    value=VITALS_CONFIG["blood_glucose"]["default_threshold_high"],
                    min_value=100,
                    max_value=400,
                    key="glucose_high_thresh",
                )
    
    return {
        "time_range": time_range,
        "selected_vitals": selected_vitals,
        "viz_mode": viz_mode,
        "granularity": granularity,
        "aggregation": aggregation,
        "show_band": show_band,
        "xaxis_time_unit": xaxis_time_unit,
        "show_seizure_markers": show_seizure_markers,
        "show_seizure_density": show_seizure_density,
        "density_bin": density_bin,
        "show_thresholds": show_thresholds,
        "thresholds": thresholds,
    }


# =============================================================================
# MAIN PAGE
# =============================================================================

def render_header(patient_id: str, time_range: Tuple[datetime, datetime]):
    """Render the main page header."""
    st.markdown('<h1 class="main-header">🏥 Patient Vitals & Seizure Timeline</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown(f"**Patient:** `{patient_id}`")
    
    with col2:
        start_str = time_range[0].strftime("%Y-%m-%d %H:%M")
        end_str = time_range[1].strftime("%Y-%m-%d %H:%M")
        st.markdown(f"**Time Window:** {start_str} → {end_str}")


def render_sampling_info(freq_info: Dict, selected_vitals: list):
    """Render sampling frequency information for each vital."""
    with st.expander("📊 Data Sampling Information", expanded=False):
        cols = st.columns(len(selected_vitals) if selected_vitals else 1)
        
        for i, vital in enumerate(selected_vitals):
            if vital in freq_info:
                info = freq_info[vital]
                config = VITALS_CONFIG[vital]
                
                with cols[i]:
                    st.markdown(f"**{config['name']}**")
                    st.markdown(f"- Frequency: `{info['frequency_label']}`")
                    st.markdown(f"- Data points: `{info['count']:,}`")
                    st.markdown(f"- Coverage: `{info['coverage_pct']:.1f}%`")


def render_summary_stats(
    vitals_df: pd.DataFrame,
    seizures_df: pd.DataFrame,
    selected_vitals: list,
    time_range: Tuple[datetime, datetime],
):
    """Render summary statistics cards."""
    st.subheader("📋 Summary Statistics")
    
    # Calculate number of columns needed
    n_vital_cols = len(selected_vitals)
    n_cols = n_vital_cols + 1  # +1 for seizures
    
    cols = st.columns(n_cols)
    
    # Vital stats
    for i, vital in enumerate(selected_vitals):
        stats = compute_vital_stats(vitals_df, vital)
        config = VITALS_CONFIG[vital]
        
        with cols[i]:
            if stats and stats["count"] > 0:
                st.metric(
                    label=f"{config['name']} ({config['unit']})",
                    value=f"{stats['mean']:.1f}",
                    delta=f"Range: {stats['min']:.1f} - {stats['max']:.1f}",
                    delta_color="off",
                )
            else:
                st.metric(
                    label=f"{config['name']}",
                    value="No data",
                )
    
    # Seizure stats
    seizure_stats = compute_seizure_stats(seizures_df, time_range)
    
    with cols[-1]:
        st.metric(
            label="⚡ Seizures",
            value=f"{seizure_stats['total_count']}",
            delta=f"{seizure_stats['per_day']:.2f}/day",
            delta_color="off" if seizure_stats['total_count'] == 0 else "inverse",
        )


def render_chart(
    vitals_df: pd.DataFrame,
    seizures_df: pd.DataFrame,
    settings: Dict,
):
    """Render the main chart based on visualization mode."""
    
    # Resample data (no forward fill - missing data stays missing)
    resampled_df, band_min, band_max = resample_vitals(
        vitals_df=vitals_df,
        granularity=settings["granularity"],
        aggregation=settings["aggregation"],
        show_band=settings["show_band"],
        ffill=False,  # Never fill gaps - show missing data as gaps
        selected_vitals=settings["selected_vitals"],
    )
    
    # Compute seizure density
    density_df = compute_seizure_density(
        seizures_df=seizures_df,
        bin_size=settings["density_bin"],
        time_range=settings["time_range"],
    )
    
    # Build chart based on mode
    if settings["viz_mode"] == "stacked":
        fig = build_plot_stacked(
            vitals_df=resampled_df,
            seizures_df=seizures_df,
            density_df=density_df,
            selected_vitals=settings["selected_vitals"],
            band_min_df=band_min,
            band_max_df=band_max,
            show_seizure_markers=settings["show_seizure_markers"],
            show_seizure_density=settings["show_seizure_density"],
            thresholds=settings["thresholds"],
            show_thresholds=settings["show_thresholds"],
            xaxis_time_unit=settings["xaxis_time_unit"],
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # combined
        fig, warning = build_plot_combined(
            vitals_df=resampled_df,
            seizures_df=seizures_df,
            density_df=density_df,
            selected_vitals=settings["selected_vitals"],
            band_min_df=band_min,
            band_max_df=band_max,
            show_seizure_markers=settings["show_seizure_markers"],
            show_seizure_density=settings["show_seizure_density"],
            thresholds=settings["thresholds"],
            show_thresholds=settings["show_thresholds"],
            xaxis_time_unit=settings["xaxis_time_unit"],
        )
        
        if warning:
            st.warning(warning)
        
        st.plotly_chart(fig, use_container_width=True)


def render_seizure_table(seizures_df: pd.DataFrame, vitals_df: pd.DataFrame):
    """Render the seizure events table."""
    if seizures_df.empty:
        st.info("No seizure events in the selected time window.")
        return
    
    with st.expander(f"📋 Seizure Events Table ({len(seizures_df)} events)", expanded=False):
        table_df = build_seizure_events_table(seizures_df, vitals_df, window_minutes=5)
        
        if not table_df.empty:
            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No seizure events to display.")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Check if data is loaded
    if not st.session_state.data_loaded:
        st.info("👈 Select a data source in the sidebar to get started.")
        
        # Show demo info
        with st.expander("ℹ️ About this app"):
            st.markdown("""
            ### Patient Vitals & Seizure Timeline
            
            This application visualizes patient time-series vital signs and seizure events on a unified timeline.
            
            **Features:**
            - Support for multiple vitals: SpO2, Heart Rate, Temperature, Blood Glucose
            - Different sampling rates per vital (per-second to per-day)
            - Seizure event markers and density visualization
            - Two visualization modes: Stacked and Combined
            - Interactive zoom, pan, and tooltips
            - Resampling controls for handling dense data
            - Threshold overlays for clinical monitoring
            
            **Getting Started:**
            1. Select "Dummy Data" to explore with pre-generated patient data
            2. Or upload your own CSV file with patient vitals
            """)
        return
    
    if settings is None:
        return
    
    # Render header
    render_header(st.session_state.selected_patient, settings["time_range"])
    
    # Filter data for selected patient and time range
    vitals_df, seizures_df = preprocess_and_filter(
        vitals_df=st.session_state.vitals_df,
        seizures_df=st.session_state.seizures_df,
        patient_id=st.session_state.selected_patient,
        time_range=settings["time_range"],
    )
    
    # Check for empty data
    if vitals_df.empty:
        st.warning("No data available for the selected patient and time range.")
        return
    
    # Infer sampling frequencies
    freq_info = infer_sampling_frequency_per_vital(vitals_df)
    
    # Render sampling info
    render_sampling_info(freq_info, settings["selected_vitals"])
    
    # Render summary stats
    render_summary_stats(vitals_df, seizures_df, settings["selected_vitals"], settings["time_range"])
    
    st.divider()
    
    # Check for empty vital selection
    if not settings["selected_vitals"]:
        st.warning("⚠️ No vitals selected. Please select at least one vital sign to display.")
        return
    
    # Render main chart
    st.subheader("📈 Vitals Timeline")
    render_chart(vitals_df, seizures_df, settings)
    
    st.divider()
    
    # Render seizure events table
    render_seizure_table(seizures_df, vitals_df)


if __name__ == "__main__":
    main()

