"""
Plotly chart building functions for Patient Vitals & Seizure Timeline app.
Implements stacked small multiples and single combined multi-axis views.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    VITALS_CONFIG,
    VITAL_KEYS,
    SEIZURE_CONFIG,
    CHART_HEIGHT_STACKED_ROW,
    CHART_HEIGHT_COMBINED,
    CHART_HEIGHT_DENSITY,
    MULTI_AXIS_COLORS,
    XAXIS_TIME_UNITS,
)


def build_plot_stacked(
    vitals_df: pd.DataFrame,
    seizures_df: pd.DataFrame,
    density_df: pd.DataFrame,
    selected_vitals: List[str],
    band_min_df: Optional[pd.DataFrame] = None,
    band_max_df: Optional[pd.DataFrame] = None,
    show_seizure_markers: bool = True,
    show_seizure_density: bool = True,
    thresholds: Optional[Dict] = None,
    show_thresholds: bool = False,
    xaxis_time_unit: str = "auto",
) -> go.Figure:
    """
    Build stacked small multiples chart with shared x-axis.
    
    Each vital gets its own subplot row, with seizure density track at bottom.
    Missing data is shown as gaps (no interpolation/extrapolation).
    
    Args:
        vitals_df: Vitals DataFrame (filtered/resampled)
        seizures_df: Seizures DataFrame
        density_df: Seizure density DataFrame
        selected_vitals: List of vital keys to display
        band_min_df: Lower band DataFrame (optional)
        band_max_df: Upper band DataFrame (optional)
        show_seizure_markers: Show seizure markers on vital plots
        show_seizure_density: Show seizure density track
        thresholds: Dict of threshold values
        show_thresholds: Whether to show threshold lines
        xaxis_time_unit: X-axis tick unit ("auto", "second", "minute", "hour", "day", "month")
        
    Returns:
        Plotly Figure
    """
    if not selected_vitals:
        return _create_empty_figure("No vitals selected")
    
    # Calculate number of rows
    n_vital_rows = len(selected_vitals)
    n_density_rows = 1 if show_seizure_density else 0
    n_rows = n_vital_rows + n_density_rows
    
    # Row heights
    row_heights = [CHART_HEIGHT_STACKED_ROW] * n_vital_rows
    if show_seizure_density:
        row_heights.append(CHART_HEIGHT_DENSITY)
    
    # Normalize heights
    total_height = sum(row_heights)
    row_heights_ratio = [h / total_height for h in row_heights]
    
    # Create subplot titles
    subplot_titles = [VITALS_CONFIG[v]["name"] for v in selected_vitals]
    if show_seizure_density:
        subplot_titles.append("Seizure Density")
    
    # Create figure with subplots
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights_ratio,
        subplot_titles=subplot_titles,
    )
    
    # Get time range for consistent x-axis
    if not vitals_df.empty:
        x_min = vitals_df["timestamp"].min()
        x_max = vitals_df["timestamp"].max()
    else:
        x_min = x_max = datetime.now()
    
    # Plot each vital
    for i, vital in enumerate(selected_vitals):
        row = i + 1
        
        if vital in vitals_df.columns:
            vital_data = vitals_df[["timestamp", vital]].dropna()
            config = VITALS_CONFIG[vital]
            
            # Add variability band if provided
            if band_min_df is not None and band_max_df is not None:
                if vital in band_min_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=band_max_df["timestamp"],
                            y=band_max_df[vital],
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=band_min_df["timestamp"],
                            y=band_min_df[vital],
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor=_hex_to_rgba(config["color"], 0.2),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=1,
                    )
            
            # Main vital line - connectgaps=False to show gaps in data
            fig.add_trace(
                go.Scatter(
                    x=vital_data["timestamp"],
                    y=vital_data[vital],
                    mode="lines+markers",
                    name=config["name"],
                    line=dict(color=config["color"], width=1.5),
                    marker=dict(size=3, color=config["color"]),
                    connectgaps=False,  # Don't connect gaps - show missing data as gaps
                    hovertemplate=f"{config['name']}: %{{y:.1f}} {config['unit']}<br>%{{x}}<extra></extra>",
                ),
                row=row,
                col=1,
            )
            
            # Add threshold lines
            if show_thresholds and thresholds:
                _add_threshold_lines(fig, vital, thresholds, row, x_min, x_max)
            
            # Add seizure markers on this vital
            if show_seizure_markers and not seizures_df.empty:
                _add_seizure_markers_to_row(
                    fig, seizures_df, vital_data, vital, row, config
                )
            
            # Set y-axis range
            fig.update_yaxes(
                title_text=config["unit"],
                range=config.get("y_range"),
                row=row,
                col=1,
            )
    
    # Add seizure density track
    if show_seizure_density and not density_df.empty:
        density_row = n_vital_rows + 1
        
        fig.add_trace(
            go.Bar(
                x=density_df["time_bin"],
                y=density_df["count"],
                name="Seizure Count",
                marker_color=SEIZURE_CONFIG["marker_color"],
                opacity=0.7,
                hovertemplate="Seizures: %{y}<br>%{x}<extra></extra>",
            ),
            row=density_row,
            col=1,
        )
        
        fig.update_yaxes(
            title_text="Count",
            row=density_row,
            col=1,
        )
    
    # Update layout
    total_height = sum(row_heights) + 100  # Add margin
    fig.update_layout(
        height=total_height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=80, b=60),
    )
    
    # Apply x-axis time unit formatting
    xaxis_config = {}
    if xaxis_time_unit != "auto" and xaxis_time_unit in XAXIS_TIME_UNITS:
        unit_config = XAXIS_TIME_UNITS[xaxis_time_unit]
        xaxis_config = {
            "dtick": unit_config["dtick"],
            "tickformat": unit_config["tickformat"],
        }
    
    # Add range slider on bottom x-axis
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.05),
        row=n_rows,
        col=1,
        **xaxis_config,
    )
    
    # Apply time formatting to all x-axes
    if xaxis_config:
        for i in range(1, n_rows + 1):
            fig.update_xaxes(
                dtick=xaxis_config.get("dtick"),
                tickformat=xaxis_config.get("tickformat"),
                row=i,
                col=1,
            )
    
    return fig


def build_plot_combined(
    vitals_df: pd.DataFrame,
    seizures_df: pd.DataFrame,
    density_df: pd.DataFrame,
    selected_vitals: List[str],
    band_min_df: Optional[pd.DataFrame] = None,
    band_max_df: Optional[pd.DataFrame] = None,
    show_seizure_markers: bool = True,
    show_seizure_density: bool = True,
    thresholds: Optional[Dict] = None,
    show_thresholds: bool = False,
    xaxis_time_unit: str = "auto",
) -> Tuple[go.Figure, Optional[str]]:
    """
    Build single combined chart with multiple y-axes.
    Missing data is shown as gaps (no interpolation/extrapolation).
    
    Args:
        (same as build_plot_stacked)
        xaxis_time_unit: X-axis tick unit ("auto", "second", "minute", "hour", "day", "month")
        
    Returns:
        Plotly Figure, warning message (if any)
    """
    warning = None
    
    if not selected_vitals:
        return _create_empty_figure("No vitals selected"), None
    
    # Warn if too many vitals selected
    if len(selected_vitals) > 3:
        warning = (
            f"⚠️ {len(selected_vitals)} vitals selected. "
            "Combined view works best with 2-3 vitals. Consider using Stacked mode."
        )
    
    # Determine if we need a density subplot
    n_rows = 2 if show_seizure_density else 1
    row_heights = [0.85, 0.15] if show_seizure_density else [1.0]
    
    # Create figure
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        specs=[[{"secondary_y": True}]] + ([[{}]] if show_seizure_density else []),
    )
    
    # Get time range
    if not vitals_df.empty:
        x_min = vitals_df["timestamp"].min()
        x_max = vitals_df["timestamp"].max()
    else:
        x_min = x_max = datetime.now()
    
    # Plot vitals with multiple y-axes
    # Primary y-axis: first vital
    # Secondary y-axis: second vital (if exists)
    # Additional vitals: overlay with primary axis scaling
    
    for i, vital in enumerate(selected_vitals):
        if vital not in vitals_df.columns:
            continue
            
        vital_data = vitals_df[["timestamp", vital]].dropna()
        config = VITALS_CONFIG[vital]
        color = MULTI_AXIS_COLORS[i % len(MULTI_AXIS_COLORS)]
        
        # Use secondary_y for second vital
        secondary_y = (i == 1) and len(selected_vitals) >= 2
        
        # Add variability band
        if band_min_df is not None and band_max_df is not None:
            if vital in band_min_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=band_max_df["timestamp"],
                        y=band_max_df[vital],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                    secondary_y=secondary_y,
                )
                fig.add_trace(
                    go.Scatter(
                        x=band_min_df["timestamp"],
                        y=band_min_df[vital],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=_hex_to_rgba(color, 0.15),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                    secondary_y=secondary_y,
                )
        
        # Main line - connectgaps=False to show gaps in data
        fig.add_trace(
            go.Scatter(
                x=vital_data["timestamp"],
                y=vital_data[vital],
                mode="lines+markers",
                name=f"{config['name']} ({config['unit']})",
                line=dict(color=color, width=2),
                marker=dict(size=3, color=color),
                connectgaps=False,  # Don't connect gaps - show missing data as gaps
                hovertemplate=f"{config['name']}: %{{y:.1f}} {config['unit']}<br>%{{x}}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=secondary_y,
        )
    
    # Add seizure markers
    if show_seizure_markers and not seizures_df.empty:
        seizure_times = _get_seizure_times(seizures_df)
        
        # Place markers at top of chart
        fig.add_trace(
            go.Scatter(
                x=seizure_times,
                y=[None] * len(seizure_times),  # Will be positioned via yref
                mode="markers",
                name="Seizure Events",
                marker=dict(
                    color=SEIZURE_CONFIG["marker_color"],
                    size=SEIZURE_CONFIG["marker_size"],
                    symbol=SEIZURE_CONFIG["marker_symbol"],
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="Seizure<br>%{x}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        
        # Add vertical lines for seizures
        for t in seizure_times:
            fig.add_vline(
                x=t,
                line_dash="dash",
                line_color=SEIZURE_CONFIG["marker_color"],
                line_width=1,
                opacity=0.5,
                row=1,
                col=1,
            )
    
    # Add seizure intervals if present
    if show_seizure_markers and not seizures_df.empty and "is_interval" in seizures_df.columns:
        _add_seizure_intervals(fig, seizures_df)
    
    # Set y-axis labels
    if len(selected_vitals) >= 1:
        config = VITALS_CONFIG[selected_vitals[0]]
        fig.update_yaxes(
            title_text=f"{config['name']} ({config['unit']})",
            range=config.get("y_range"),
            row=1,
            col=1,
            secondary_y=False,
        )
    
    if len(selected_vitals) >= 2:
        config = VITALS_CONFIG[selected_vitals[1]]
        fig.update_yaxes(
            title_text=f"{config['name']} ({config['unit']})",
            range=config.get("y_range"),
            row=1,
            col=1,
            secondary_y=True,
        )
    
    # Add threshold lines
    if show_thresholds and thresholds:
        for i, vital in enumerate(selected_vitals[:2]):  # Only first two axes
            secondary_y = (i == 1)
            _add_threshold_lines_combined(fig, vital, thresholds, secondary_y, x_min, x_max)
    
    # Add seizure density track
    if show_seizure_density and not density_df.empty:
        fig.add_trace(
            go.Bar(
                x=density_df["time_bin"],
                y=density_df["count"],
                name="Seizure Count",
                marker_color=SEIZURE_CONFIG["marker_color"],
                opacity=0.7,
                hovertemplate="Seizures: %{y}<br>%{x}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        
        fig.update_yaxes(
            title_text="Seizures",
            row=2,
            col=1,
        )
    
    # Update layout
    height = CHART_HEIGHT_COMBINED + (CHART_HEIGHT_DENSITY if show_seizure_density else 0)
    fig.update_layout(
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
        margin=dict(l=60, r=60, t=80, b=60),
    )
    
    # Apply x-axis time unit formatting
    xaxis_config = {}
    if xaxis_time_unit != "auto" and xaxis_time_unit in XAXIS_TIME_UNITS:
        unit_config = XAXIS_TIME_UNITS[xaxis_time_unit]
        xaxis_config = {
            "dtick": unit_config["dtick"],
            "tickformat": unit_config["tickformat"],
        }
    
    # Add range slider
    bottom_row = 2 if show_seizure_density else 1
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.05),
        row=bottom_row,
        col=1,
        **xaxis_config,
    )
    
    # Apply time formatting to all x-axes
    if xaxis_config:
        for i in range(1, bottom_row + 1):
            fig.update_xaxes(
                dtick=xaxis_config.get("dtick"),
                tickformat=xaxis_config.get("tickformat"),
                row=i,
                col=1,
            )
    
    return fig, warning


def _create_empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _get_seizure_times(seizures_df: pd.DataFrame) -> List:
    """Extract seizure times from DataFrame."""
    if "seizure_time" in seizures_df.columns:
        return seizures_df["seizure_time"].tolist()
    elif "seizure_start" in seizures_df.columns:
        return seizures_df["seizure_start"].tolist()
    return []


def _add_seizure_markers_to_row(
    fig: go.Figure,
    seizures_df: pd.DataFrame,
    vital_data: pd.DataFrame,
    vital: str,
    row: int,
    config: Dict,
) -> None:
    """Add seizure markers to a specific subplot row."""
    seizure_times = _get_seizure_times(seizures_df)
    
    if not seizure_times:
        return
    
    # Find y-values at seizure times (interpolate if needed)
    y_values = []
    for t in seizure_times:
        # Find closest vital reading
        if not vital_data.empty:
            idx = (vital_data["timestamp"] - t).abs().idxmin()
            y_val = vital_data.loc[idx, vital]
            y_values.append(y_val if pd.notna(y_val) else None)
        else:
            y_values.append(None)
    
    fig.add_trace(
        go.Scatter(
            x=seizure_times,
            y=y_values,
            mode="markers",
            name="Seizure",
            marker=dict(
                color=SEIZURE_CONFIG["marker_color"],
                size=SEIZURE_CONFIG["marker_size"],
                symbol=SEIZURE_CONFIG["marker_symbol"],
                line=dict(width=2, color="white"),
            ),
            hovertemplate="Seizure Event<br>%{x}<extra></extra>",
            showlegend=(row == 1),  # Only show in legend once
        ),
        row=row,
        col=1,
    )


def _add_seizure_intervals(fig: go.Figure, seizures_df: pd.DataFrame) -> None:
    """Add shaded regions for seizure intervals."""
    interval_seizures = seizures_df[seizures_df.get("is_interval", False) == True]
    
    for _, seizure in interval_seizures.iterrows():
        if pd.notna(seizure.get("seizure_start")) and pd.notna(seizure.get("seizure_end")):
            fig.add_vrect(
                x0=seizure["seizure_start"],
                x1=seizure["seizure_end"],
                fillcolor=SEIZURE_CONFIG["region_fill_color"],
                line=dict(
                    color=SEIZURE_CONFIG["region_line_color"],
                    width=1,
                ),
                layer="below",
                row=1,
                col=1,
            )


def _add_threshold_lines(
    fig: go.Figure,
    vital: str,
    thresholds: Dict,
    row: int,
    x_min: datetime,
    x_max: datetime,
) -> None:
    """Add threshold lines to a subplot."""
    config = VITALS_CONFIG[vital]
    
    # Low threshold
    low_key = f"{vital}_low"
    if low_key in thresholds and thresholds[low_key] is not None:
        fig.add_hline(
            y=thresholds[low_key],
            line_dash="dash",
            line_color="orange",
            line_width=1,
            annotation_text="Low",
            annotation_position="right",
            row=row,
            col=1,
        )
    
    # High threshold
    high_key = f"{vital}_high"
    if high_key in thresholds and thresholds[high_key] is not None:
        fig.add_hline(
            y=thresholds[high_key],
            line_dash="dash",
            line_color="red",
            line_width=1,
            annotation_text="High",
            annotation_position="right",
            row=row,
            col=1,
        )


def _add_threshold_lines_combined(
    fig: go.Figure,
    vital: str,
    thresholds: Dict,
    secondary_y: bool,
    x_min: datetime,
    x_max: datetime,
) -> None:
    """Add threshold lines to combined chart."""
    config = VITALS_CONFIG[vital]
    
    # Low threshold
    low_key = f"{vital}_low"
    if low_key in thresholds and thresholds[low_key] is not None:
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[thresholds[low_key], thresholds[low_key]],
                mode="lines",
                line=dict(dash="dash", color="orange", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
            secondary_y=secondary_y,
        )
    
    # High threshold
    high_key = f"{vital}_high"
    if high_key in thresholds and thresholds[high_key] is not None:
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[thresholds[high_key], thresholds[high_key]],
                mode="lines",
                line=dict(dash="dash", color="red", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
            secondary_y=secondary_y,
        )

