"""
SentinelDC — Predictive Maintenance Platform
=============================================
Phase 2: Live Telemetry Table + Real-Time Plotly Probability Timeline

Run with:
    streamlit run app.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

try:
    from groq import Groq as GroqClient
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Ensure project root is on the path so 'pipeline' package is importable
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.router import DriveTelemetryRouter, ModelNotFoundError
from pipeline.pipeline import DriveModelPipeline

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelDC",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — Clean, professional dark-accented theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Top header bar */
    .sentinel-header {
        padding: 1rem 0 0.5rem 0;
        border-bottom: 2px solid #334155;
        margin-bottom: 1.5rem;
    }
    .sentinel-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #f8fafc;
        margin: 0;
    }
    .sentinel-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    /* Status badge */
    .badge-online {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
    }
    .badge-offline {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }

    /* Risk pills */
    .risk-healthy {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 4px;
        background-color: #d1fae5;
        color: #065f46;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .risk-warning {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 4px;
        background-color: #fef3c7;
        color: #92400e;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .risk-critical {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 4px;
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .risk-buffering {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 4px;
        background-color: #e2e8f0;
        color: #475569;
        font-weight: 600;
        font-size: 0.78rem;
    }

    /* Sidebar section divider */
    .sidebar-section {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin: 1.2rem 0 0.4rem 0;
    }

    /* Tab placeholder content */
    .tab-placeholder {
        padding: 3rem;
        text-align: center;
        color: #9ca3af;
        border: 2px dashed #334155;
        border-radius: 8px;
        margin-top: 1rem;
    }

    /* Chart section header */
    .chart-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #cbd5e1;
        margin: 1.25rem 0 0.4rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")

# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state():
    """Bootstrap all session state keys if not already set."""

    if "router" not in st.session_state:
        with st.spinner("Loading model registry..."):
            st.session_state.router = DriveTelemetryRouter(models_dir=str(MODELS_DIR))

    if "stream_data" not in st.session_state:
        st.session_state.stream_data = _load_all_demo_streams()

    if "all_days" not in st.session_state:
        st.session_state.all_days = _compute_day_index(st.session_state.stream_data)

    if "current_day" not in st.session_state:
        st.session_state.current_day = 0

    if "telemetry_history" not in st.session_state:
        st.session_state.telemetry_history = pd.DataFrame()

    if "groq_api_key" not in st.session_state:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            try:
                key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                pass
        st.session_state.groq_api_key = key

    if "selected_serial" not in st.session_state:
        st.session_state.selected_serial = "DEMO_DRIVE_001"

    if "real_time_streaming" not in st.session_state:
        st.session_state.real_time_streaming = False

    if "rt_stream_sub_idx" not in st.session_state:
        st.session_state.rt_stream_sub_idx = 0  # index into the drives list within current day
    if "selected_model" not in st.session_state:
        st.session_state.selected_model  = "ST12000NM0007"

    # Per-serial probability history for the timeline chart
    # dict[serial_number -> list of {"day": int, "prob": float, "risk": str, "drive_model": str}]
    if "prob_history" not in st.session_state:
        st.session_state.prob_history = {}

    # Groq LLM RCA results cache: dict[serial_number -> str (generated RCA text)]
    if "groq_rca_cache" not in st.session_state:
        st.session_state.groq_rca_cache = {}

    # Deterministic SHAP RCA cache: dict[serial_number -> rca dict from router.generate_rca()]
    if "shap_rca_cache" not in st.session_state:
        st.session_state.shap_rca_cache = {}

    # Persistent critical incident registry: once a drive crosses CRITICAL it stays
    # in this dict even if later days drop it back to WARNING or HEALTHY.
    # dict[serial_number -> {"drive_model", "first_critical_day", "peak_prob"}]
    if "critical_incident_registry" not in st.session_state:
        st.session_state.critical_incident_registry = {}



def _load_all_demo_streams() -> dict:
    """
    Load every *_demo_stream.csv in the models directory into a dictionary.
    """
    streams = {}
    for csv_path in sorted(MODELS_DIR.glob("*_demo_stream.csv")):
        drive_name = csv_path.stem.replace("_demo_stream", "")
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
        streams[drive_name] = df
    return streams


def _compute_day_index(stream_data: dict) -> list:
    """
    Build a globally sorted list of step dicts across all demo streams.
    Each step corresponds to one calendar day per drive.
    """
    per_drive = {}
    for drive, df in stream_data.items():
        dates = sorted(df["date"].unique())
        per_drive[drive] = dates

    max_steps = max((len(v) for v in per_drive.values()), default=0)

    steps = []
    for i in range(max_steps):
        cutoffs = {}
        for drive, dates in per_drive.items():
            if i < len(dates):
                cutoffs[drive] = dates[i]
        steps.append({
            "step":    i + 1,
            "label":   f"Day {i + 1}",
            "cutoffs": cutoffs,
        })

    return steps


# ─────────────────────────────────────────────────────────────────────────────
# Stream Engine
# ─────────────────────────────────────────────────────────────────────────────

def advance_stream():
    """
    Inject the next batch of telemetry rows into session state.
    Handles both 'accumulating' (buffering days 1-9) and 'predicted' statuses.
    """
    router      = st.session_state.router
    stream_data = st.session_state.stream_data
    all_days    = st.session_state.all_days
    current_day = st.session_state.current_day

    if current_day >= len(all_days):
        st.toast("Stream complete — all 20 days have been replayed.")
        return

    step_info = all_days[current_day]
    new_rows  = []

    for drive_model, cutoff_date in step_info["cutoffs"].items():
        if drive_model not in stream_data:
            continue

        df_drive = stream_data[drive_model]
        batch    = df_drive[df_drive["date"] == cutoff_date].copy()

        for _, row in batch.iterrows():
            log_dict      = row.to_dict()
            serial_number = str(row["serial_number"])

            try:
                result = router.predict_single_log(log_dict)
            except ModelNotFoundError:
                result = {
                    "status":          "error",
                    "probability_pct": 0.0,
                    "risk_level":      "unknown",
                    "risk_score":      0,
                    "prediction":      False,
                    "days_buffered":   0,
                }

            # ── Build display row ──
            if result.get("status") == "error":
                prob_display = "No Model Found"
                risk_display = "UNREGISTERED"
                prob_pct     = 0.0
                risk_score   = -1
                pred_display = "UNREGISTERED"
            elif result["status"] == "accumulating":
                days_buf     = result.get("days_buffered", current_day + 1)
                prob_display = f"Buffering ({days_buf}/10)"
                risk_display = "BUFFERING"
                prob_pct     = 0.0
                risk_score   = 0
                pred_display = "BUFFERING"
            else:
                prob_pct     = result.get("probability_pct", 0.0)
                risk_display = result.get("risk_level", "unknown").upper()
                risk_score   = result.get("risk_score", 0)
                pred_display = "FAILING" if result.get("prediction") else "HEALTHY"
                prob_display = f"{prob_pct:.2f}%"

                # Track probability history per serial for the Plotly chart
                if serial_number not in st.session_state.prob_history:
                    st.session_state.prob_history[serial_number] = []
                st.session_state.prob_history[serial_number].append({
                    "day":        current_day + 1,
                    "day_label":  f"Day {current_day + 1}",
                    "prob":       prob_pct,
                    "risk":       risk_display,
                    "drive_model": drive_model,
                })

                # Update the persistent critical incident registry
                if risk_display == "CRITICAL":
                    registry = st.session_state.critical_incident_registry
                    if serial_number not in registry:
                        registry[serial_number] = {
                            "drive_model":        drive_model,
                            "first_critical_day": current_day + 1,
                            "peak_prob":          prob_pct,
                        }
                    else:
                        # Update peak probability seen so far
                        registry[serial_number]["peak_prob"] = max(
                            registry[serial_number]["peak_prob"], prob_pct
                        )

            new_rows.append({
                "sim_day":         current_day + 1,
                "timestamp":       pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "drive_model":     drive_model,
                "serial_number":   serial_number,
                "failure_label":   int(row.get("failure", 0)),
                "probability_pct": prob_pct,
                "prob_display":    prob_display,
                "risk_level":      risk_display,
                "risk_score":      risk_score,
                "prediction":      pred_display,
                **{k: v for k, v in log_dict.items() if str(k).startswith("smart_")},
            })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if st.session_state.telemetry_history.empty:
            st.session_state.telemetry_history = new_df
        else:
            st.session_state.telemetry_history = pd.concat(
                [st.session_state.telemetry_history, new_df],
                ignore_index=True,
            )

    st.session_state.current_day += 1


def advance_single_log():
    """
    Emit exactly ONE drive's telemetry row for the real-time stream ticker.

    State managed:
      - rt_stream_sub_idx : which drive in the current day's drive list to emit next.
      - current_day       : incremented only when ALL drives for a day are done.

    Each call emits one row and returns. The main() ticker sleeps 0.5 s between calls.
    """
    router      = st.session_state.router
    stream_data = st.session_state.stream_data
    all_days    = st.session_state.all_days
    current_day = st.session_state.current_day
    sub_idx     = st.session_state.rt_stream_sub_idx

    if current_day >= len(all_days):
        return

    step_info   = all_days[current_day]
    drives      = sorted(step_info["cutoffs"].keys())   # consistent ordering

    if sub_idx >= len(drives):
        # All drives for this day done — advance day, reset sub_idx
        st.session_state.current_day    += 1
        st.session_state.rt_stream_sub_idx = 0
        return

    drive_model  = drives[sub_idx]
    cutoff_date  = step_info["cutoffs"][drive_model]

    if drive_model not in stream_data:
        st.session_state.rt_stream_sub_idx += 1
        return

    df_drive = stream_data[drive_model]
    batch    = df_drive[df_drive["date"] == cutoff_date].copy()

    new_rows = []
    for _, row in batch.iterrows():
        log_dict      = row.to_dict()
        serial_number = str(row["serial_number"])

        try:
            result = router.predict_single_log(log_dict)
        except ModelNotFoundError:
            result = {
                "status":          "error",
                "probability_pct": 0.0,
                "risk_level":      "unknown",
                "risk_score":      0,
                "prediction":      False,
                "days_buffered":   0,
            }

        if result.get("status") == "error":
            prob_display = "No Model Found"
            risk_display = "UNREGISTERED"
            prob_pct     = 0.0
            risk_score   = -1
            pred_display = "UNREGISTERED"
        elif result["status"] == "accumulating":
            days_buf     = result.get("days_buffered", current_day + 1)
            prob_display = f"Buffering ({days_buf}/10)"
            risk_display = "BUFFERING"
            prob_pct     = 0.0
            risk_score   = 0
            pred_display = "BUFFERING"
        else:
            prob_pct     = result.get("probability_pct", 0.0)
            risk_display = result.get("risk_level", "unknown").upper()
            risk_score   = result.get("risk_score", 0)
            pred_display = "FAILING" if result.get("prediction") else "HEALTHY"
            prob_display = f"{prob_pct:.2f}%"

            if serial_number not in st.session_state.prob_history:
                st.session_state.prob_history[serial_number] = []
            st.session_state.prob_history[serial_number].append({
                "day":        current_day + 1,
                "day_label":  f"Day {current_day + 1}",
                "prob":       prob_pct,
                "risk":       risk_display,
                "drive_model": drive_model,
            })

            if risk_display == "CRITICAL":
                registry = st.session_state.critical_incident_registry
                if serial_number not in registry:
                    registry[serial_number] = {
                        "drive_model":        drive_model,
                        "first_critical_day": current_day + 1,
                        "peak_prob":          prob_pct,
                    }
                else:
                    registry[serial_number]["peak_prob"] = max(
                        registry[serial_number]["peak_prob"], prob_pct
                    )

        new_rows.append({
            "sim_day":         current_day + 1,
            "timestamp":       pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "drive_model":     drive_model,
            "serial_number":   serial_number,
            "failure_label":   int(row.get("failure", 0)),
            "probability_pct": prob_pct,
            "prob_display":    prob_display,
            "risk_level":      risk_display,
            "risk_score":      risk_score,
            "prediction":      pred_display,
            **{k: v for k, v in log_dict.items() if str(k).startswith("smart_")},
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if st.session_state.telemetry_history.empty:
            st.session_state.telemetry_history = new_df
        else:
            st.session_state.telemetry_history = pd.concat(
                [st.session_state.telemetry_history, new_df],
                ignore_index=True,
            )

    # Move to next drive within this day
    st.session_state.rt_stream_sub_idx += 1

    # If that was the last drive, advance the day counter
    if st.session_state.rt_stream_sub_idx >= len(drives):
        st.session_state.current_day       += 1
        st.session_state.rt_stream_sub_idx  = 0


def reset_stream():
    """Reset stream back to Day 0 (no data) and clear the rolling window cache."""
    st.session_state.current_day                = 0
    st.session_state.telemetry_history          = pd.DataFrame()
    st.session_state.prob_history               = {}
    st.session_state.groq_rca_cache             = {}
    st.session_state.shap_rca_cache             = {}
    st.session_state.critical_incident_registry = {}
    st.session_state.selected_serial            = "DEMO_DRIVE_001"
    st.session_state.selected_model             = "ST12000NM0007"
    st.session_state.real_time_streaming        = False
    st.session_state.rt_stream_sub_idx          = 0
    st.session_state.router.clear_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### SentinelDC")
        st.markdown("Predictive Maintenance Platform")
        st.divider()

        # System status
        n_models = len(st.session_state.router.list_models())
        if n_models > 0:
            st.markdown(
                f'<span class="badge-online">System Online</span>&nbsp; '
                f'{n_models} model{"s" if n_models != 1 else ""} active',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="badge-offline">No Models Loaded</span>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Stream Controls ──
        st.markdown('<div class="sidebar-section">Stream Controls</div>', unsafe_allow_html=True)

        all_days    = st.session_state.all_days
        current_day = st.session_state.current_day
        total_steps = len(all_days)
        is_streaming = st.session_state.real_time_streaming

        if total_steps == 0:
            st.caption("No demo data loaded. Train a model first.")
        else:
            if current_day > 0:
                st.caption(f"Timeline: Day {current_day} of {total_steps}")
                st.progress(current_day / total_steps)
            else:
                st.caption("Stream not started. Press Advance Day to begin.")

            # ── Real-Time Stream toggle ──
            stream_finished = current_day >= total_steps
            if not stream_finished:
                toggle_label = "[ LIVE ] Stop Real-Time Stream" if is_streaming else "[ > ] Start Real-Time Stream"
                toggle_type  = "primary" if is_streaming else "secondary"
                if st.button(toggle_label, use_container_width=True, type=toggle_type, key="rt_stream_toggle"):
                    st.session_state.real_time_streaming = not is_streaming
                    st.rerun()
            else:
                st.caption("Stream complete.")

            col1, col2 = st.columns(2)
            with col1:
                advance_disabled = current_day >= total_steps or is_streaming
                if st.button(
                    "Advance Day" if current_day < total_steps else "Complete",
                    disabled=advance_disabled,
                    use_container_width=True,
                    type="primary",
                ):
                    advance_stream()
                    st.rerun()

            with col2:
                if st.button("Reset", use_container_width=True, type="secondary"):
                    reset_stream()
                    st.rerun()

            if current_day < total_steps and not is_streaming:
                if st.button("Stream All 20 Days", use_container_width=True):
                    while st.session_state.current_day < total_steps:
                        advance_stream()
                    st.rerun()

        st.divider()

        # ── Active Models ──
        st.markdown('<div class="sidebar-section">Active Models</div>', unsafe_allow_html=True)
        for model_name in st.session_state.router.list_models():
            info = st.session_state.router.get_model_info(model_name)
            st.caption(
                f"**{model_name}**  \n"
                f"Threshold: `{info['threshold']}`  |  "
                f"F1: `{info['metrics'].get('f1_failure', 'N/A')}`"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="sentinel-header">
        <div class="sentinel-title">SentinelDC</div>
        <div class="sentinel-subtitle">AI-Powered Hard Drive Predictive Maintenance &amp; Incident Response</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Live Telemetry (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

def _risk_color(risk: str) -> str:
    """Return a hex color for a given risk level string."""
    return {
        "CRITICAL":  "#ef4444",
        "WARNING":   "#f59e0b",
        "HEALTHY":   "#10b981",
        "BUFFERING": "#64748b",
    }.get(risk, "#94a3b8")


def render_probability_chart(serial_number: str, drive_model: str):
    """
    Render a Plotly timeline chart for a single drive's failure probability
    across the days that have been streamed so far.
    """
    prob_hist = st.session_state.prob_history.get(serial_number, [])

    if not prob_hist:
        st.info(
            f"**{serial_number} ({drive_model})** is currently in the initial **Buffering (Days 1–9)** window.  \n"
            "Real-time probability predictions will begin rendering on **Day 10** when the 10-day sliding window is primed."
        )
        return

    df_hist = pd.DataFrame(prob_hist)

    # Get the threshold for this drive model
    threshold_pct = 50.0
    try:
        info          = st.session_state.router.get_model_info(drive_model)
        threshold_pct = info["threshold"] * 100
    except Exception:
        pass

    # ── Build Plotly figure ──
    fig = go.Figure()

    # Background zone fills for risk bands
    fig.add_hrect(
        y0=0,  y1=threshold_pct * 0.40,
        fillcolor="rgba(16,185,129,0.06)", line_width=0,
        annotation_text="Healthy Zone", annotation_position="left",
        annotation_font_size=10, annotation_font_color="#6ee7b7",
    )
    fig.add_hrect(
        y0=threshold_pct * 0.40, y1=threshold_pct * 0.80,
        fillcolor="rgba(245,158,11,0.06)", line_width=0,
        annotation_text="Watch Zone", annotation_position="left",
        annotation_font_size=10, annotation_font_color="#fbbf24",
    )
    fig.add_hrect(
        y0=threshold_pct * 0.80, y1=100,
        fillcolor="rgba(239,68,68,0.06)", line_width=0,
        annotation_text="Danger Zone", annotation_position="left",
        annotation_font_size=10, annotation_font_color="#f87171",
    )

    # Decision threshold reference line
    fig.add_hline(
        y=threshold_pct,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=1.5,
        annotation_text=f"Decision Threshold ({threshold_pct:.1f}%)",
        annotation_position="top right",
        annotation_font_size=11,
        annotation_font_color="#f59e0b",
    )

    # Probability line
    fig.add_trace(go.Scatter(
        x=df_hist["day"],
        y=df_hist["prob"],
        mode="lines+markers",
        name="Failure Probability",
        line=dict(color="#3b82f6", width=2.5, shape="spline", smoothing=0.5),
        marker=dict(
            size=9,
            color=[_risk_color(r) for r in df_hist["risk"]],
            line=dict(color="#1e293b", width=1.5),
        ),
        hovertemplate=(
            "<b>Day %{x}</b><br>"
            "Failure Probability: <b>%{y:.2f}%</b><br>"
            "<extra></extra>"
        ),
    ))

    # Mark the first day prediction went CRITICAL
    critical_days = df_hist[df_hist["risk"] == "CRITICAL"]
    if not critical_days.empty:
        first_crit = critical_days.iloc[0]
        fig.add_vline(
            x=first_crit["day"],
            line_dash="dot",
            line_color="#ef4444",
            line_width=1.5,
            annotation_text="First Alert",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color="#ef4444",
        )

    fig.update_layout(
        title=dict(
            text=f"Failure Risk Timeline — {serial_number} ({drive_model})",
            font=dict(size=14, color="#f8fafc"),
        ),
        xaxis=dict(
            title="Simulation Day",
            tickmode="linear",
            tick0=10,
            dtick=1,
            range=[9.5, 20.5],
            color="#94a3b8",
            gridcolor="#1e293b",
            showgrid=True,
        ),
        yaxis=dict(
            title="Failure Probability (%)",
            range=[0, 105],
            ticksuffix="%",
            color="#94a3b8",
            gridcolor="#1e293b",
        ),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#94a3b8"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1"),
        ),
        hovermode="x unified",
        margin=dict(l=50, r=30, t=60, b=50),
        height=380,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_tab1(tab):
    with tab:
        history = st.session_state.telemetry_history

        if history.empty:
            st.markdown("""
            <div class="tab-placeholder">
                <strong>No telemetry data streamed yet.</strong><br>
                Use the <em>Advance Day</em> or <em>Stream All 20 Days</em> button in the sidebar to start live simulation.
            </div>
            """, unsafe_allow_html=True)
            return

        # ── Summary KPI row ──
        n_logs     = len(history)
        n_serials  = history["serial_number"].nunique()
        n_critical = int((history["risk_level"] == "CRITICAL").sum())
        n_warning  = int((history["risk_level"] == "WARNING").sum())
        n_healthy  = int((history["risk_level"] == "HEALTHY").sum())
        n_buffering= int((history["risk_level"] == "BUFFERING").sum())

        st.markdown(
            f"**{n_logs} log entries** across **{n_serials} active drive streams** &nbsp;|&nbsp; "
            f"Day **{st.session_state.current_day}** of **{len(st.session_state.all_days)}**"
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Logs",  n_logs)
        c2.metric("Critical",    n_critical)
        c3.metric("Warning",     n_warning)
        c4.metric("Healthy",     n_healthy)
        c5.metric("Buffering",   n_buffering)

        st.divider()

        # ── 1. Live Telemetry Stream (Full Width) ──
        st.markdown("#### Live Telemetry Stream")

        display_cols = [
            "sim_day", "timestamp", "drive_model", "serial_number",
            "prob_display", "risk_level", "prediction",
        ]
        display_df = history[display_cols].copy()
        display_df.columns = [
            "Day", "Date", "Drive Model", "Serial Number",
            "Failure Probability", "Risk Level", "Prediction",
        ]

        st.dataframe(
            display_df.sort_values(["Day", "Drive Model"], ascending=[False, True]),
            use_container_width=True,
            hide_index=True,
            height=300,
            column_config={
                "Day": st.column_config.NumberColumn("Day", width="small"),
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Drive Model": st.column_config.TextColumn("Drive Model", width="medium"),
                "Serial Number": st.column_config.TextColumn("Serial Number", width="medium"),
                "Failure Probability": st.column_config.TextColumn("Failure Probability", width="medium"),
                "Risk Level": st.column_config.TextColumn("Risk Level", width="small"),
                "Prediction": st.column_config.TextColumn("Prediction", width="small"),
            },
        )

        st.divider()

        # ── 2. Drive Inspector (Full Width below Stream) ──
        st.markdown("#### Drive Inspector")

        all_serials = (
            history.sort_values("sim_day", ascending=False)
            .drop_duplicates("serial_number")[["serial_number", "drive_model", "risk_level", "probability_pct", "prob_display"]]
            .reset_index(drop=True)
        )

        options = all_serials["serial_number"].tolist()

        def _format_option(sn):
            match = all_serials[all_serials["serial_number"] == sn]
            if match.empty:
                return sn
            row = match.iloc[0]
            risk = row["risk_level"]
            model = row["drive_model"]
            if risk == "BUFFERING":
                return f"[...] {sn} ({model}) — Buffering"
            if risk == "UNREGISTERED":
                return f"[?] {sn} ({model}) — No Model"
            icon = {"CRITICAL": "[!]", "WARNING": "[~]", "HEALTHY": "[OK]"}.get(risk, "[ ]")
            return f"{icon} {sn} ({model}) — {row['probability_pct']:.1f}%"

        current_idx = 0
        if st.session_state.selected_serial in options:
            current_idx = options.index(st.session_state.selected_serial)

        chosen = st.selectbox(
            "Select Drive to Inspect",
            options=options,
            index=current_idx,
            format_func=_format_option,
            key="drive_inspector_select",
        )

        if chosen != st.session_state.selected_serial:
            st.session_state.selected_serial = chosen
            match = all_serials[all_serials["serial_number"] == chosen]
            if not match.empty:
                st.session_state.selected_model = match.iloc[0]["drive_model"]
            st.rerun()

        # Display Selected Drive Metrics & Status in a clean 5-column card row
        if st.session_state.selected_serial:
            sel_sn   = st.session_state.selected_serial
            sel_rows = history[history["serial_number"] == sel_sn]

            if not sel_rows.empty:
                latest   = sel_rows.sort_values("sim_day").iloc[-1]
                risk     = latest["risk_level"]
                prob_str = latest["prob_display"]
                model_nm = latest["drive_model"]

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Selected Serial", sel_sn)
                m2.metric("Drive Model", model_nm)
                m3.metric("Current Status", risk)
                m4.metric("Failure Risk", prob_str)
                m5.metric("Days Observed", f"{len(sel_rows)} / 20")

        st.divider()

        # ── 3. Failure Risk Timeline Chart ──
        st.markdown('<div class="chart-header">Failure Risk Timeline</div>', unsafe_allow_html=True)

        sel_sn    = st.session_state.selected_serial
        sel_model = st.session_state.selected_model

        if sel_sn and sel_model:
            render_probability_chart(sel_sn, sel_model)
        else:
            st.info("Select a drive in the Drive Inspector above to view its failure risk timeline.")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs 2–5 — Placeholders (Phase 3 & 4)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_shap_rca(serial_number: str, drive_model: str) -> dict:
    """
    Retrieve or compute the deterministic SHAP-based RCA for a drive.
    Result is cached in session_state.shap_rca_cache.
    """
    cache = st.session_state.shap_rca_cache
    if serial_number in cache:
        return cache[serial_number]

    try:
        rca = st.session_state.router.generate_rca(serial_number, drive_model, top_n=5)
        cache[serial_number] = rca
        return rca
    except Exception as e:
        return {"error": str(e)}


def _call_groq_llm(serial_number: str, drive_model: str, rca: dict, api_key: str) -> str:
    """
    Call the Groq API to generate a rich LLM-powered Root Cause Analysis narrative.
    Returns the generated text string. Caches result in session_state.groq_rca_cache.
    """
    if serial_number in st.session_state.groq_rca_cache:
        return st.session_state.groq_rca_cache[serial_number]

    pred    = rca.get("prediction_summary", {})
    factors = rca.get("top_risk_factors", [])
    action  = rca.get("action_required", "")

    factor_lines = "\n".join([
        f"  - {f['smart_id']} ({f['name']}): last_value={f['last_value']:,.0f}, "
        f"SHAP contribution={f['total_shap']:+.4f}, trend={f['trend']}"
        for f in factors
    ])

    prompt = f"""You are a Principal Storage Hardware Engineer diagnosing an imminent drive failure in a hyperscale data center.

CONTEXT: Our predictive maintenance ML pipeline has flagged a drive.

DRIVE TELEMETRY DATA:
- Drive Model: {drive_model}
- Serial Number: {serial_number}
- Imminent Failure Probability: {pred.get('probability_pct', 0):.1f}%
- Risk Classification: {pred.get('risk_level', 'UNKNOWN').upper()}
- Observation Window: {pred.get('days_in_window', 10)} days

SHAP ATTRIBUTION (Top Risk Factors):
{factor_lines}

SYSTEM RUNBOOK RECOMMENDATION:
{action}

TASK: Write a highly technical, low-level Root Cause Analysis (RCA) narrative. 
Assume the audience consists of L3 Storage Engineers and Hardware Reliability SREs.

CRITICAL INSTRUCTIONS:
- DO NOT generate markdown tables or repeat raw SMART/SHAP values (the user already sees those in the dashboard).
- Use **bold markdown** to highlight: the primary failure mode diagnosis, specific SMART attribute names, critical risk terms (e.g., **Magnetic Platter Degradation**, **SMART 187**, **URE Risk**, **P1 Evacuation Required**).
- Use section headers with bold text for each of the 4 sections below.
- Provide ONLY the written engineering analysis in paragraph form.

**1. Low-Level Mechanical/Firmware RCA**
What is physically failing inside the chassis (e.g., servo tracking degradation, thermal asperities, read/write head micro-crashes, ECC exhaustion, platter defect propagation, or bus interface CRC timeouts) based strictly on the provided attributes.

**2. Degradation Kinetics**
Explain the mechanical or electrical mechanism of how these specific SMART trends correlate and cascade towards catastrophic failure.

**3. Blast Radius & Risk**
The immediate threat to the RAID array/storage pool, rebuild IOPS implications, and URE (Unrecoverable Read Error) risks.

**4. Hardware Remediation**
Specific low-level operational steps (e.g., NVMe/SAS lane isolation, LBA mapping analysis, node drain, physical spindown procedures).

Tone: Extremely technical, precise, objective, and analytical. No introductory fluff, no emojis, no generic advice. Use strict storage engineering terminology."""

    try:
        client   = GroqClient(api_key=api_key)
        
        # Dynamically select an available model to prevent 404 errors
        available_models = [m.id for m in client.models.list().data]
        if "llama-3.3-70b-versatile" in available_models:
            model_id = "llama-3.3-70b-versatile"
        elif "openai/gpt-oss-20b" in available_models:
            model_id = "openai/gpt-oss-20b"
        elif "groq/compound" in available_models:
            model_id = "groq/compound"
        elif "qwen/qwen3.8-27b" in available_models:
            model_id = "qwen/qwen3.8-27b"
        elif available_models:
            model_id = available_models[0]
        else:
            model_id = "llama-3.3-70b-versatile" # Fallback

        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
        )
        result = response.choices[0].message.content.strip()
        st.session_state.groq_rca_cache[serial_number] = result
        return result
    except Exception as e:
        err_msg = f"**[ERROR] Groq API call failed:**\n\n`{e}`\n\n*(Note: Check if the model 'llama-3.3-70b-versatile' is available for your API key)*"
        st.session_state.groq_rca_cache[serial_number] = err_msg
        return err_msg


def _render_shap_factors_table(factors: list):
    """Render the top SHAP risk factors as a clean styled dataframe."""
    if not factors:
        st.caption("No SHAP factors available.")
        return

    rows = []
    for f in factors:
        rows.append({
            "Rank":        f["rank"],
            "SMART ID":    f["smart_id"],
            "Attribute":   f["name"],
            "Last Value":  f"{f['last_value']:,.0f}",
            "Trend":       f["trend"].capitalize(),
            "SHAP Impact": f"{f['total_shap']:+.4f}",
            "Effect":      f["impact"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={
                     "Rank":       st.column_config.NumberColumn("Rank",    width="small"),
                     "SMART ID":   st.column_config.TextColumn("SMART ID",  width="small"),
                     "Attribute":  st.column_config.TextColumn("Attribute", width="large"),
                     "Last Value": st.column_config.TextColumn("Last Value",width="small"),
                     "Trend":      st.column_config.TextColumn("Trend",     width="small"),
                     "SHAP Impact":st.column_config.TextColumn("SHAP Impact",width="small"),
                     "Effect":     st.column_config.TextColumn("Effect",    width="medium"),
                 })


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Alerts & Diagnostics (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────
def _render_shap_trend_charts(serial_number: str, top_factors: list):
    """
    Renders a 2-column grid of Plotly line charts showing the raw value trends
    from Day 1 to the current day for the specific top risk factors identified by SHAP.
    """
    if not top_factors:
        return

    history = st.session_state.telemetry_history
    if history.empty:
        return

    # Filter history for this specific drive
    df_drive = history[history["serial_number"] == serial_number].sort_values("sim_day")
    if df_drive.empty:
        return

    # Compute global medians for healthy drives to use as baselines
    df_healthy = history[history["failure_label"] == 0]

    st.markdown("**Top Risk Factor Trends (Day 1 to Current)**")

    # Map SHAP 'smart_id' (e.g., "SMART 197") to telemetry column (e.g., "smart_197_raw")
    factors_to_plot = []
    for f in top_factors:
        col_name = f"smart_{f['smart_id'].replace('SMART ', '')}_raw"
        if col_name in df_drive.columns:
            factors_to_plot.append({
                "title":    f"{f['smart_id']} — {f['name']}",
                "col_name": col_name
            })

    if not factors_to_plot:
        return

    # Render in 2 columns
    cols = st.columns(2)
    for i, factor in enumerate(factors_to_plot):
        col_idx  = i % 2
        col_name = factor["col_name"]
        title    = factor["title"]

        fig = go.Figure()
        if col_name in df_drive.columns:
            fig.add_trace(go.Scatter(
                x=df_drive["sim_day"],
                y=df_drive[col_name],
                mode="lines+markers",
                line=dict(color="#ef4444", width=2),
                marker=dict(size=4),
                name=title,
            ))

        # Add healthy baseline reference line if available
        if col_name in df_healthy.columns and not df_healthy.empty:
            baseline = df_healthy[col_name].median()
            if pd.notna(baseline):
                fig.add_hline(
                    y=baseline,
                    line_dash="dot",
                    line_color="green",
                    annotation_text="Healthy Median",
                    annotation_position="bottom right",
                    opacity=0.6,
                )

        # Minimalist sparkline styling
        fig.update_layout(
            title=title,
            margin=dict(l=20, r=20, t=30, b=20),
            height=200,
            xaxis_title=None,
            yaxis_title=None,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            title_font_size=13,
            xaxis=dict(showgrid=True, gridcolor="#334155", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#334155", zeroline=False),
        )
        fig.update_traces(line_color="#ef4444", marker=dict(size=4))

        with cols[col_idx]:
            st.plotly_chart(fig, use_container_width=True, key=f"trend_{serial_number}_{col_name}")


def render_tab2(tab):
    with tab:
        history = st.session_state.telemetry_history

        if history.empty:
            st.markdown("""
            <div class="tab-placeholder">
                <strong>No telemetry data yet.</strong><br>
                Advance the stream from the sidebar to generate alerts.
            </div>
            """, unsafe_allow_html=True)
            return

        # Filter to WARNING and CRITICAL only
        alerts = history[history["risk_level"].isin(["WARNING", "CRITICAL"])].copy()

        if alerts.empty:
            st.info(
                "No warning or critical alerts at this time.  \n"
                "All monitored drives are currently within healthy parameters."
            )
            return

        # Latest status per serial
        latest_per_serial = (
            alerts.sort_values("sim_day", ascending=False)
            .drop_duplicates("serial_number")
            .reset_index(drop=True)
        )

        n_critical = int((latest_per_serial["risk_level"] == "CRITICAL").sum())
        n_warning  = int((latest_per_serial["risk_level"] == "WARNING").sum())

        # Summary bar
        st.markdown(
            f"**{len(latest_per_serial)} drives** require attention:  "
            f"**{n_critical} Critical** &nbsp;|&nbsp; **{n_warning} Warning**"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Drives with Active Alerts", len(latest_per_serial))
        c2.metric("Critical Severity", n_critical)
        c3.metric("Warning Severity",  n_warning)

        st.divider()
        st.markdown("#### Drive Alert Details")
        st.caption(
            "Expand a drive below to view SHAP risk factor analysis and generate an AI-powered RCA report."
        )

        # Sort: Critical first, then Warning
        latest_per_serial["sort_order"] = latest_per_serial["risk_level"].map(
            {"CRITICAL": 0, "WARNING": 1}
        )
        latest_per_serial = latest_per_serial.sort_values(
            ["sort_order", "probability_pct"], ascending=[True, False]
        )

        groq_key = st.session_state.groq_api_key

        for _, row in latest_per_serial.iterrows():
            sn         = row["serial_number"]
            model_name = row["drive_model"]
            risk       = row["risk_level"]
            prob       = row["probability_pct"]
            day        = int(row["sim_day"])

            risk_icon  = "[CRITICAL]" if risk == "CRITICAL" else "[WARNING]"
            risk_cls   = "risk-critical" if risk == "CRITICAL" else "risk-warning"

            expander_label = (
                f"{risk_icon}  {sn}  ({model_name})  —  "
                f"{prob:.1f}% failure probability  |  Day {day}"
            )

            with st.expander(expander_label, expanded=(risk == "CRITICAL")):

                # ── Alert header row ──
                col_meta, col_badge = st.columns([3, 1])
                with col_meta:
                    st.markdown(f"**Drive Model:** `{model_name}`")
                    st.markdown(f"**Serial Number:** `{sn}`")
                    st.markdown(f"**Failure Probability:** `{prob:.2f}%`")
                    st.markdown(f"**Last observation:** Day `{day}`")

                with col_badge:
                    st.markdown(
                        f"<br><span class='{risk_cls}' style='font-size:1.1rem;padding:0.4rem 1rem;'>"
                        f"{risk}</span>",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                # ── SHAP Risk Factor Analysis ──
                st.markdown("**Top Risk Factors (SHAP Attribution)**")

                with st.spinner("Computing SHAP attribution..."):
                    rca = _get_shap_rca(sn, model_name)

                if "error" in rca:
                    st.warning(f"SHAP analysis unavailable: {rca['error']}")
                else:
                    # Deterministic RCA text from router
                    st.caption(rca.get("rca_text", ""))
                    st.markdown(" ")
                    _render_shap_factors_table(rca.get("top_risk_factors", []))

                    st.markdown("---")
                    
                    # ── Dynamic Trend Charts ──
                    _render_shap_trend_charts(sn, rca.get("top_risk_factors", []))
                    
                    st.markdown("---")

                    # ── Groq LLM RCA Section ──
                    st.markdown("**AI-Powered Root Cause Analysis**")

                    already_generated = sn in st.session_state.groq_rca_cache

                    if already_generated:
                        st.markdown(st.session_state.groq_rca_cache[sn])
                        if st.button(
                            "Regenerate AI Analysis",
                            key=f"regen_{sn}",
                            type="secondary",
                        ):
                            del st.session_state.groq_rca_cache[sn]
                            with st.spinner("Regenerating AI RCA narrative..."):
                                _call_groq_llm(sn, model_name, rca, groq_key)
                            st.rerun()
                    else:
                        if not GROQ_AVAILABLE:
                            st.warning("Groq Python package not installed. Run: `pip install groq`")
                        elif not groq_key:
                            st.info("AI-powered RCA is not configured. Contact the administrator.")
                        else:
                            if st.button(
                                "Generate AI Root Cause Analysis",
                                key=f"groq_{sn}",
                                type="primary",
                            ):
                                with st.spinner("Calling Groq LLM — generating RCA narrative..."):
                                    _call_groq_llm(sn, model_name, rca, groq_key)
                                st.rerun()



# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — SRE Archetype Engine & TTF/SLA Calculator
# ─────────────────────────────────────────────────────────────────────────────

# SMART attribute → failure archetype mapping
_ARCHETYPE_MAP = {
    # A: Physical Platter / Bad Sectors
    "A": {"smart_5", "smart_187", "smart_196", "smart_197", "smart_198"},
    # B: Thermal / Overheating
    "B": {"smart_190", "smart_194", "smart_231"},
    # C: Mechanical / Vibration / Servo
    "C": {"smart_1", "smart_3", "smart_7", "smart_10", "smart_11", "smart_16"},
    # D: Interface / CRC / Bus Timeouts
    "D": {"smart_184", "smart_188", "smart_199"},
    # E: Endurance / Aging / Workload Wear
    "E": {"smart_4", "smart_9", "smart_12", "smart_192", "smart_193",
          "smart_240", "smart_241", "smart_242"},
}

_ARCHETYPE_META = {
    "A": {
        "title": "Archetype A — Physical Platter Damage & Bad Sectors",
        "summary": (
            "Physical platter head scratch or localized magnetic media degradation detected. "
            "Reallocated and pending bad sectors indicate areas of the platter that can no longer "
            "reliably store data. Bad sectors typically propagate exponentially under sustained read "
            "load. Immediate data evacuation is critical before further platter surface deterioration."
        ),
    },
    "B": {
        "title": "Archetype B — Thermal Runaway & Environmental Overheating",
        "summary": (
            "Drive internal temperature exceeds the safe operating thermal envelope. The platter "
            "mechanics may be intact, but sustained overheating causes lubricant breakdown, thermal "
            "expansion head collisions, and accelerated bearing wear. Thermal remediation must precede "
            "any decision on physical drive replacement."
        ),
    },
    "C": {
        "title": "Archetype C — Mechanical Wear & Spindle/Actuator Stress",
        "summary": (
            "Actuator arm mechanical fatigue, seek servo tracking instability, or excessive "
            "rotational vibration (RV) detected. This can be caused by adjacent drive resonance, "
            "chassis fan harmonic frequencies, or head flying height instability. The drive may "
            "still be recoverable if vibration source is isolated before a full mechanical lockup."
        ),
    },
    "D": {
        "title": "Archetype D — Interface / Cabling & Controller CRC Errors",
        "summary": (
            "SATA/SAS data link layer CRC corruption or command timeout bursts detected. The platter "
            "magnetic surface is likely intact, but data packets are corrupting in transit over the "
            "backplane, PCIe HBA controller, or through a degraded/oxidized data cable. Check "
            "physical interconnects before ordering a replacement drive."
        ),
    },
    "E": {
        "title": "Archetype E — End-of-Life, Endurance & Workload Wear",
        "summary": (
            "Drive is approaching or has exceeded its designed Mean Time Between Failures (MTBF) "
            "or rated Terabytes Written (TBW) endurance limit. Power-on hours, load/unload cycle "
            "counts, or emergency power retract counts indicate a drive nearing natural end-of-life. "
            "Graceful fleet rotation and proactive replacement is recommended over emergency action."
        ),
    },
    "F": {
        "title": "Archetype F — Composite / Multi-Factor Anomaly",
        "summary": (
            "The AI model detected complex cross-feature SMART correlation shifts across multiple "
            "channels simultaneously that do not map cleanly to a single failure mode. This may "
            "indicate early-stage multi-system degradation or an unusual drive firmware behavior. "
            "Full extended SMART diagnostics and escalation to the hardware lab is recommended."
        ),
    },
}


def _classify_archetype(top_smart_id: str) -> str:
    """Return archetype letter (A-E) or 'F' (fallback) for a given SMART attribute column name."""
    key = top_smart_id.replace("_raw", "").replace("_normalized", "").lower()
    for arch, smart_set in _ARCHETYPE_MAP.items():
        if key in smart_set:
            return arch
    return "F"


def _get_ttf_sla(prob: float) -> dict:
    """Return a TTF/SLA dict based on current failure probability."""
    if prob >= 85.0:
        return {
            "tier":       "P1 Critical",
            "sla":        "4 Hours",
            "ttf":        "< 24 Hours",
            "color":      "#ef4444",
            "action":     "Emergency hot-swap required. Page on-call storage lead immediately.",
        }
    elif prob >= 70.0:
        return {
            "tier":       "P2 High",
            "sla":        "24 Hours",
            "ttf":        "2 – 5 Days",
            "color":      "#f59e0b",
            "action":     "Evacuate data during current shift. Order replacement unit.",
        }
    else:
        return {
            "tier":       "P3 Warning",
            "sla":        "72 Hours",
            "ttf":        "5 – 10 Days",
            "color":      "#3b82f6",
            "action":     "Monitor hourly. Run extended SMART test. Verify hot-spare availability.",
        }


def _build_runbook(archetype: str, sn: str, model: str, prob: float,
                   factors: list, first_day: int, peak_prob: float) -> list:
    """
    Build a list of (title, body, commands) tuples for the SRE runbook.
    Commands is a multi-line string of bash commands with injected drive identifiers.
    """
    did = f"ata-{model}_{sn}"   # Disk identifier for udev /dev/disk/by-id paths
    top = factors[0] if factors else {}
    top_name  = top.get("name", "Unknown Attribute")
    top_val   = top.get("last_value", 0)
    top_trend = top.get("trend", "rising")
    top_id    = top.get("smart_id", "SMART ?")
    ttf       = _get_ttf_sla(prob)

    steps = []

    # ── Step 1: Alert Escalation (common to all archetypes) ──
    steps.append((
        "Step 1 — Immediate Alert Escalation",
        (
            f"Assign this incident **{ttf['tier']}** priority. SLA response window: **{ttf['sla']}**. "
            f"Estimated Time-To-Failure: **{ttf['ttf']}**.\n\n"
            f"Page the on-call storage SRE team and open an incident ticket with the following details:\n"
            f"- **Drive:** `{sn}` (Model: `{model}`)\n"
            f"- **Failure Probability:** `{prob:.1f}%` (Peak: `{peak_prob:.1f}%`)\n"
            f"- **First detected:** Day `{first_day}`\n"
            f"- **Primary SMART signal:** {top_id} — {top_name} = `{top_val:,.0f}` (trend: {top_trend})\n\n"
            f"**Required action:** {ttf['action']}"
        ),
        None,
    ))

    # ── Step 2+: Archetype-specific steps ──
    if archetype == "A":
        steps.append((
            "Step 2 — Locate & Isolate Drive in Chassis",
            (
                "Turn on the physical bay locator LED so data center technicians identify the "
                "exact drive slot. Then immediately mark the drive as failed in the storage pool "
                "to prevent further read/write thrashing on the damaged platter sectors."
            ),
            f"""\
# Turn on chassis bay locator LED (blinks the physical LED on the disk enclosure)
ledctl locate=/dev/disk/by-id/{did}

# MDADM Software RAID — mark drive as failed and remove from array
mdadm --manage /dev/md0 --fail /dev/disk/by-id/{did}
mdadm --manage /dev/md0 --remove /dev/disk/by-id/{did}""",
        ))
        steps.append((
            "Step 3 — Run Extended SMART Self-Test & Map Bad Blocks",
            (
                f"Execute an extended SMART self-test to enumerate all unreadable sectors. "
                f"Monitor **{top_id} ({top_name})** closely. "
                f"Current observed value: **{top_val:,.0f}** (trend: {top_trend}). "
                "Do NOT power cycle the drive before the test completes."
            ),
            f"""\
# Run extended SMART self-test (non-destructive, ~60-90 mins)
smartctl -t long /dev/disk/by-id/{did}

# Poll test progress every 60 seconds
watch -n 60 "smartctl -a /dev/disk/by-id/{did} | grep -E 'test remaining|Reallocated|Pending'"

# Extract full SMART bad sector report
smartctl -A /dev/disk/by-id/{did} | grep -E "Reallocated|Pending|Uncorrectable" """,
        ))
        steps.append((
            "Step 4 — Emergency Data Evacuation",
            (
                "Evacuate all data from the degraded drive before physical platter damage propagates. "
                "Use `ddrescue` for filesystem-level recovery or ZFS/MDADM for RAID-aware migration. "
                "Do NOT issue a power cycle at any point during evacuation."
            ),
            f"""\
# ZFS: Replace with hot-spare immediately (triggers automatic resilvering)
zpool replace tank /dev/disk/by-id/{did} /dev/disk/by-id/ata-SPARE_DRIVE

# Ceph: Drain OSD weight and mark out before removal
ceph osd reweight osd.<id> 0 && ceph osd out osd.<id>
ceph osd safe-to-destroy osd.<id>

# Low-level rescue (if drive not in RAID): byte-level copy ignoring bad sectors
ddrescue --force --no-split /dev/disk/by-id/{did} /dev/disk/by-id/ata-TARGET_DRIVE rescue.log""",
        ))
        steps.append((
            "Step 5 — Physical Drive Replacement",
            (
                f"After evacuation is confirmed complete, perform physical hot-swap. "
                f"Order a replacement unit: same form factor and capacity as **`{model}`**. "
                "Update CMDB and asset tracking. Verify RAID rebuild completion before closing the ticket."
            ),
            f"""\
# After hot-swap: add new drive to MDADM RAID array
mdadm --manage /dev/md0 --add /dev/disk/by-id/ata-NEW_DRIVE

# Monitor RAID resilvering progress
watch cat /proc/mdstat

# Verify ZFS resilver completion
zpool status -v tank""",
        ))

    elif archetype == "B":
        steps.append((
            "Step 2 — Query Drive Thermal Sensors & Chassis Environment",
            (
                f"The primary trigger is **{top_id} ({top_name})** at **{top_val:,.0f}°C** "
                f"(trend: {top_trend}). Check BMC/IPMI thermal sensor matrix to distinguish between "
                "a drive-only issue vs. a rack-level thermal event (e.g., failed fans, clogged air filters)."
            ),
            f"""\
# Query drive internal temperature log and historical thermal trend
smartctl -l scttemp /dev/disk/by-id/{did}

# Check all chassis fan tachometers and ambient inlet/exhaust temperatures via IPMI
ipmitool sdr type "Fan"
ipmitool sensor get "Inlet Temp" "Exhaust Temp" """,
        ))
        steps.append((
            "Step 3 — Emergency Chassis Cooling Override",
            (
                "Force chassis fans to 100% full-speed PWM duty cycle immediately. "
                "Drive thermal envelope: Seagate enterprise drives typically safe-operate at 0°C–60°C. "
                f"Current reading of {top_val:,.0f}°C indicates the drive is near or beyond the safe ceiling."
            ),
            """\
# Force all chassis fans to maximum RPM (Dell iDRAC example — raw IPMI command)
ipmitool raw 0x30 0x30 0x01 0x00
ipmitool raw 0x30 0x30 0x02 0xff 0x64

# HP iLO: Set fan to 100% via BIOS or iLO Fan Control policy
# (Use iLO REST API or iLO web interface → Fan Control → Manual override)""",
        ))
        steps.append((
            "Step 4 — Throttle I/O & Redistribute Workload",
            (
                "Reduce I/O load on the overheating drive to lower internal actuator heat generation. "
                "Migrate high-IOPS workloads (databases, logging) to drives on cooler chassis shelves."
            ),
            f"""\
# Throttle I/O bandwidth to the hot drive (reduce actuator friction heat)
systemctl set-property storage-daemon.service \\
  BlockIOReadBandwidth="/dev/disk/by-id/{did} 20M" \\
  BlockIOWriteBandwidth="/dev/disk/by-id/{did} 10M"

# Monitor real-time drive temperature every 30 seconds until it drops below 45°C
watch -n 30 "smartctl -A /dev/disk/by-id/{did} | grep -i temp" """,
        ))
        steps.append((
            "Step 5 — Physical Rack Inspection & Filter Cleaning",
            (
                "If temperature does not drop below 45°C within 30 minutes of fan override, "
                "dispatch a data center technician to inspect: (1) air filter blockages, "
                "(2) missing rack blanking panels creating hot-air recirculation, "
                "(3) drive bay positioning and airflow baffling."
            ),
            f"""\
# If temperature persists above safe range, migrate data to adjacent chassis
zpool replace tank /dev/disk/by-id/{did} /dev/disk/by-id/ata-COOLER_RACK_DRIVE""",
        ))

    elif archetype == "C":
        steps.append((
            "Step 2 — Run Non-Destructive Short Servo Calibration Test",
            (
                f"The primary trigger is **{top_id} ({top_name})** at **{top_val:,.0f}** "
                f"(trend: {top_trend}), indicating actuator servo tracking errors or rotational vibration. "
                "Run a short non-destructive test to assess whether errors are consistent or intermittent. "
                "Intermittent patterns strongly suggest chassis vibration rather than a dying drive head."
            ),
            f"""\
# Run non-destructive short servo calibration test (~2 minutes)
smartctl -t short /dev/disk/by-id/{did}

# Inspect seek error rate, spin retry count, and raw read error rate
smartctl -A /dev/disk/by-id/{did} | grep -E "Seek_Error_Rate|Spin_Retry_Count|Raw_Read_Error_Rate|Load_Cycle"

# Check system I/O latency and dropped operations
iostat -xz 1 10""",
        ))
        steps.append((
            "Step 3 — Inspect Chassis Vibration & Drive Mounting",
            (
                "Check for rotational vibration (RV) resonance — a common cause of seek errors in "
                "high-density drive shelves. Verify physical drive tray mounting screws are finger-tight "
                "and rubber vibration-isolation grommets are present and intact."
            ),
            """\
# Check kernel vibration event logs
dmesg -T | grep -iE "vibration|resonance|mechanical"

# Check SAS/SATA backplane link health (look for PHY resets)
dmesg -T | grep -iE "ata[0-9]+|link|reset|frozen" | tail -50""",
        ))
        steps.append((
            "Step 4 — Redistribute Adjacent Drive I/O",
            (
                "High concurrent random I/O on adjacent drives in the same shelf can cause sympathetic "
                "vibration that destabilizes the actuator head. Spread heavy workloads across alternate "
                "shelves or chassis to isolate the degraded drive from vibration excitation sources."
            ),
            f"""\
# Check real-time I/O utilization per drive
iostat -xz 1 5

# Limit concurrent I/O on adjacent drives to reduce vibration coupling
ionice -c 3 -p $(pgrep -d',' storage_workload)""",
        ))
        steps.append((
            "Step 5 — Schedule Proactive Replacement at Next Maintenance Window",
            (
                f"If seek error rate continues to rise, schedule proactive drive replacement in the "
                "next off-peak maintenance window. This archetype rarely requires emergency hot-swap "
                "unless `SMART 5` (Reallocated Sectors) also begins incrementing, which would reclassify "
                "the incident to Archetype A (Physical Platter Damage)."
            ),
            f"""\
# Monitor SMART 5 (Reallocated Sectors) for escalation to Archetype A
watch -n 300 "smartctl -A /dev/disk/by-id/{did} | grep Reallocated"

# If mechanical failure is confirmed, initiate ZFS proactive replacement
zpool replace tank /dev/disk/by-id/{did} /dev/disk/by-id/ata-REPLACEMENT""",
        ))

    elif archetype == "D":
        steps.append((
            "Step 2 — Inspect Kernel Bus Reset & CRC Error Logs",
            (
                f"The primary trigger is **{top_id} ({top_name})** at **{top_val:,.0f}** "
                f"(trend: {top_trend}). CRC errors and command timeouts indicate data packet corruption "
                "in transit over the SATA/SAS link — not necessarily physical platter damage. "
                "Check `dmesg` for AHCI link resets and SATA error bursts before pulling the drive."
            ),
            f"""\
# Search kernel journal for SATA link resets, frozen commands, and bus timeouts
dmesg -T | grep -E "ata[0-9]+|hard resetting link|frozen|timeout|NCQ" | tail -100

# Check SATA PHY layer CRC error statistics
smartctl -l sataphy /dev/disk/by-id/{did}

# Check drive CRC error counter (SMART 199)
smartctl -A /dev/disk/by-id/{did} | grep -E "CRC|Command_Timeout|End-to-End" """,
        ))
        steps.append((
            "Step 3 — Reseat or Replace SATA/SAS Data Cable",
            (
                "A majority of SATA CRC errors are caused by a degraded, oxidized, or improperly "
                "seated data cable rather than a failing drive. Dispatch a data center technician to "
                "physically reseat or swap the data cable connecting the drive to the backplane."
            ),
            f"""\
# After cable reseat: rescan SATA host bus to re-establish clean link synchronization
echo 1 > /sys/block/sdX/device/rescan

# Check if CRC error counter has stopped incrementing (compare with baseline above)
smartctl -A /dev/disk/by-id/{did} | grep CRC""",
        ))
        steps.append((
            "Step 4 — Verify HBA Controller Firmware & PCIe Link",
            (
                "Check the Host Bus Adapter (HBA) controller firmware version. Outdated HBA firmware "
                "is a known cause of spurious SATA command timeouts. Verify the PCIe slot link speed "
                "and width are negotiated correctly."
            ),
            """\
# Check PCIe HBA storage controller status and link speed
lspci -vvv -s $(lspci | grep -i storage | awk '{print $1}')

# Check HBA firmware version (LSI/Broadcom example)
sas3flash -listall

# Update HBA firmware if below vendor recommended version
# (Vendor-specific — refer to HBA manufacturer documentation)""",
        ))
        steps.append((
            "Step 5 — Monitor for CRC Error Cessation",
            (
                "After cable reseat and HBA firmware update, monitor CRC error counters for 24 hours. "
                "If `SMART 199` (UDMA CRC Error Count) ceases incrementing, the root cause was the "
                "physical interconnect. If CRC errors continue after cable replacement, escalate to "
                "Archetype A (physical platter test) and consider drive replacement."
            ),
            f"""\
# Monitor SMART 199 CRC error count hourly for 24 hours
watch -n 3600 "smartctl -A /dev/disk/by-id/{did} | grep -E 'CRC|Timeout'"

# If errors persist after cable fix, run extended SMART to check platter health
smartctl -t long /dev/disk/by-id/{did}""",
        ))

    elif archetype == "E":
        steps.append((
            "Step 2 — Audit Full Operational Lifetime Metrics",
            (
                f"The primary trigger is **{top_id} ({top_name})** at **{top_val:,.0f}** "
                f"(trend: {top_trend}). This drive is approaching its designed MTBF or TBW endurance "
                "limit. Extract the complete operational lifetime metrics to quantify remaining lifespan."
            ),
            f"""\
# Extract complete lifetime metrics: power-on hours, load cycles, cumulative TBW
smartctl -a /dev/disk/by-id/{did} | grep -E "Power_On_Hours|Load_Cycle_Count|Total_LBAs|Spin_Up_Time"

# Check storage pool redundancy health before initiating rotation
zpool status -v tank""",
        ))
        steps.append((
            "Step 3 — Graceful Data Migration (Off-Peak Window)",
            (
                "Unlike Archetypes A–D, this failure mode allows graceful planned migration. "
                "Schedule the drive rotation during the next off-peak maintenance window. "
                "Do not emergency-evacuate unless `SMART 5` or `SMART 187` are also incrementing."
            ),
            f"""\
# ZFS: Graceful online replacement (drive stays serving data during resilvering)
zpool replace tank /dev/disk/by-id/{did} /dev/disk/by-id/ata-NEW_DRIVE

# Monitor resilvering progress (typically 4-12 hours for large drives)
watch zpool status tank

# MDADM: Add replacement drive before removing aged drive
mdadm --manage /dev/md0 --add /dev/disk/by-id/ata-NEW_DRIVE
mdadm --manage /dev/md0 --fail /dev/disk/by-id/{did}""",
        ))
        steps.append((
            "Step 4 — Fleet Cohort Review",
            (
                f"Check if adjacent drives in the same manufacturing batch (same model `{model}`) "
                "share a similar deployment age. Drives manufactured in the same quarter often show "
                "correlated failure patterns. Issue a fleet-wide proactive replacement advisory if "
                "3 or more drives in the same cohort are now in this end-of-life bracket."
            ),
            f"""\
# List all drives of the same model and their power-on hours
for dev in /dev/disk/by-id/ata-{model}_*; do
    echo "$dev:"
    smartctl -A $dev | grep Power_On_Hours
done""",
        ))
        steps.append((
            "Step 5 — Cryptographic Secure Erase Before Decommission",
            (
                "After data has been fully migrated and verified on the replacement drive, "
                "execute a cryptographic secure erasure on the decommissioned unit per NIST SP 800-88 "
                "before physical removal for recycling or RMA return."
            ),
            f"""\
# NIST 800-88 compliant cryptographic secure erase (SSD/SATA ATA Secure Erase)
hdparm --security-erase NULL /dev/disk/by-id/{did}

# Alternative: NVMe secure format
nvme format /dev/nvme0 --ses=1

# For spinning HDD: use blkdiscard or ATA enhanced erase
blkdiscard --secure /dev/disk/by-id/{did}""",
        ))

    else:  # Archetype F — Composite Fallback
        steps.append((
            "Step 2 — Extract Full Extended SMART Diagnostic Log",
            (
                f"The AI model detected an unusual multi-feature correlation pattern with "
                f"**{top_id} ({top_name})** as the primary signal ({top_val:,.0f}, trend: {top_trend}). "
                "Extract a comprehensive SMART report including vendor-specific log pages before "
                "making any remediation decisions."
            ),
            f"""\
# Dump complete extended SMART attribute log including vendor-specific pages
smartctl -x /dev/disk/by-id/{did}

# Export full SMART data to file for hardware lab analysis
smartctl -x /dev/disk/by-id/{did} > /tmp/smart_report_{sn}.txt""",
        ))
        steps.append((
            "Step 3 — Run Non-Destructive Short Diagnostic Test",
            (
                "Run the drive's built-in firmware self-test. This is completely non-destructive "
                "and takes approximately 2 minutes. The result will indicate whether the drive "
                "firmware itself considers the drive in a healthy or degraded state."
            ),
            f"""\
# Run 2-minute firmware self-test
smartctl -t short /dev/disk/by-id/{did}

# Poll result after 3 minutes
smartctl -l selftest /dev/disk/by-id/{did}""",
        ))
        steps.append((
            "Step 4 — Audit RAID Array Redundancy",
            (
                "Before any further action, verify that the storage array still has sufficient "
                "parity redundancy. If the array is in a degraded state (missing a parity disk), "
                "do NOT remove this drive until the array is fully rebuilt."
            ),
            """\
# Check MDADM array state
cat /proc/mdstat

# Check ZFS pool health and fault tolerance margin
zpool status -v

# Check Ceph cluster health and PG distribution
ceph health detail""",
        ))
        steps.append((
            "Step 5 — Escalate to Storage Hardware Lab for RMA Testing",
            (
                "If the firmware self-test passes but the AI model continues flagging this drive "
                "at high probability, escalate to the storage hardware lab for Return Merchandise "
                "Authorization (RMA) testing. The AI may be detecting subtle pre-failure indicators "
                "not yet visible in standard SMART attribute thresholds."
            ),
            f"""\
# Search kernel hardware journal for unhandled I/O exceptions
dmesg -T --level=err,warn | grep -iE "storage|scsi|sata|nvme|io|disk"

# Generate full system hardware health report for RMA submission
sosreport --batch --name=storage_rma_{sn}""",
        ))

    # ── Common Final Step: Post-Incident Review ──
    steps.append((
        f"Step {len(steps) + 1} — Post-Incident Review & Model Feedback",
        (
            f"After the incident is fully resolved, file a Post-Incident Review (PIR).\n\n"
            f"**Document:**\n"
            f"- First detection: Day `{first_day}` at `{prob:.1f}%` failure probability\n"
            f"- Peak probability reached: `{peak_prob:.1f}%`\n"
            f"- Root cause archetype: **{archetype}** — {_ARCHETYPE_META[archetype]['title']}\n"
            f"- Time-to-replace vs SLA target: _(fill in actual replacement time)_\n"
            f"- Data loss assessment: _(fill in after verification)_\n\n"
            "Update predictive model decision thresholds in `Train Model` tab if the detection "
            "was too early (excessive false positive) or too late."
        ),
        None,
    ))

    return steps


def render_tab3(tab):
    with tab:
        history  = st.session_state.telemetry_history
        registry = st.session_state.critical_incident_registry

        if history.empty:
            st.markdown("""
            <div class="tab-placeholder">
                <strong>No telemetry data yet.</strong><br>
                Advance the stream from the sidebar to generate audit records.
            </div>
            """, unsafe_allow_html=True)
            return

        # ── Use the persistent incident registry, not just "currently critical" rows
        if not registry:
            st.success(
                "No critical incidents recorded yet.  \n"
                "Critical incident runbooks will appear the first time any drive crosses the "
                "CRITICAL threshold and will persist here even if the drive later drops to Warning."
            )
            return

        # For each drive in the registry, pull its LATEST prediction row from history
        incident_rows = []
        for sn, meta in registry.items():
            drive_history = history[history["serial_number"] == sn]
            if drive_history.empty:
                continue
            latest = drive_history.sort_values("sim_day").iloc[-1]
            incident_rows.append({
                "serial_number":    sn,
                "drive_model":      meta["drive_model"],
                "current_prob":     latest["probability_pct"],
                "current_risk":     latest["risk_level"],
                "peak_prob":        meta["peak_prob"],
                "first_critical_day": meta["first_critical_day"],
                "latest_day":       int(latest["sim_day"]),
            })

        # Sort: highest current probability first
        incident_rows.sort(key=lambda x: x["current_prob"], reverse=True)

        st.markdown(
            f"**{len(incident_rows)} open critical incident(s).**  \n"
            "Incidents remain open until manually resolved. Drive status reflects the latest prediction."
        )
        st.divider()

        for idx, inc in enumerate(incident_rows):
            sn         = inc["serial_number"]
            model_name = inc["drive_model"]
            prob       = inc["current_prob"]
            peak       = inc["peak_prob"]
            risk       = inc["current_risk"]
            first_day  = inc["first_critical_day"]
            latest_day = inc["latest_day"]

            ttf       = _get_ttf_sla(peak)
            risk_cls  = "risk-critical" if risk == "CRITICAL" else "risk-warning" if risk == "WARNING" else "risk-healthy"

            st.markdown(f"### Incident #{idx + 1} — `{sn}`")

            col_a, col_b, col_c, col_d, col_e = st.columns(5)
            col_a.metric("Drive Model",       model_name)
            col_b.metric("Current Risk",      f"{prob:.1f}%")
            col_c.metric("Peak Risk",         f"{peak:.1f}%")
            col_d.metric("First Detected",    f"Day {first_day}")
            col_e.metric("Latest Prediction", f"Day {latest_day}")

            # TTF/SLA badge
            st.markdown(
                f"<div style='background:{ttf['color']}22;border-left:4px solid {ttf['color']};"
                f"border-radius:4px;padding:0.6rem 1rem;margin:0.5rem 0;'>"
                f"<strong style='color:{ttf['color']};'>{ttf['tier']}</strong> &nbsp;|&nbsp; "
                f"SLA Response: <strong>{ttf['sla']}</strong> &nbsp;|&nbsp; "
                f"Est. TTF: <strong>{ttf['ttf']}</strong></div>",
                unsafe_allow_html=True,
            )

            # Current status badge
            st.markdown(
                f"<span class='{risk_cls}'>Current Status: {risk}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(" ")

            # ── Failure Assessment (SHAP RCA) ──
            with st.spinner(f"Loading SHAP analysis for {sn}..."):
                rca = _get_shap_rca(sn, model_name)

            if "error" in rca:
                st.error(f"SHAP RCA failed: {rca['error']}")
                st.divider()
                continue

            factors = rca.get("top_risk_factors", [])

            # Classify archetype from the top SHAP factor
            top_smart_col = ""
            if factors:
                smart_id = factors[0]["smart_id"]   # e.g. "SMART 5"
                num      = smart_id.split()[-1]
                top_smart_col = f"smart_{num}"

            archetype = _classify_archetype(top_smart_col)
            arch_meta = _ARCHETYPE_META[archetype]

            st.markdown("#### Failure Assessment")
            st.error(
                f"**{arch_meta['title']}**\n\n{arch_meta['summary']}"
            )

            st.markdown("#### Root Cause Attribution (SHAP)")
            _render_shap_factors_table(factors)

            # ── Dynamic SRE Runbook ──
            st.markdown("#### SRE Remediation Runbook")
            runbook = _build_runbook(archetype, sn, model_name, prob, factors, first_day, peak)

            for step_title, step_body, step_commands in runbook:
                with st.expander(step_title, expanded=True):
                    st.markdown(step_body)
                    if step_commands:
                        st.code(step_commands, language="bash")

            if idx < len(incident_rows) - 1:
                st.divider()







# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Model Registry (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def _metric_badge(label: str, value: str, color: str = "#3b82f6") -> str:
    return (
        f"<div style='background:{color}18;border:1px solid {color}44;"
        f"border-radius:6px;padding:0.5rem 0.75rem;text-align:center;margin:0.2rem;'>"
        f"<div style='font-size:0.72rem;color:#94a3b8;letter-spacing:0.05em;'>{label}</div>"
        f"<div style='font-size:1.1rem;font-weight:700;color:{color};margin-top:0.1rem;'>{value}</div>"
        f"</div>"
    )


def render_tab4(tab):
    with tab:
        router = st.session_state.router
        models = router.list_models()

        if not models:
            st.markdown("""
            <div class="tab-placeholder">
                <strong>No models in registry.</strong><br>
                Use the Train Model tab to train your first drive model.
            </div>
            """, unsafe_allow_html=True)
            return

        st.markdown(
            f"**{len(models)} model(s) registered** in the active registry.  "
            "Models are automatically loaded from the `models/` directory on startup."
        )
        st.divider()

        for model_name in models:
            try:
                info    = router.get_model_info(model_name)
                metrics = info.get("metrics", {})
            except Exception as e:
                st.error(f"Could not load info for `{model_name}`: {e}")
                continue

            threshold  = info.get("threshold", 0.5)
            smart_cols = info.get("smart_feature_cols", [])
            flat_cols  = info.get("flat_feature_cols", [])
            seq_len    = info.get("sequence_len", 10)
            demo_csv   = info.get("demo_csv")

            roc_auc   = metrics.get("roc_auc", 0.0)
            f1        = metrics.get("f1_failure", 0.0)
            precision = metrics.get("precision_failure", 0.0)
            recall    = metrics.get("recall_failure", 0.0)

            if roc_auc >= 0.95:
                health_color, health_label = "#22c55e", "Excellent"
            elif roc_auc >= 0.85:
                health_color, health_label = "#3b82f6", "Good"
            elif roc_auc >= 0.70:
                health_color, health_label = "#f59e0b", "Fair"
            else:
                health_color, health_label = "#ef4444", "Poor"

            with st.expander(
                f"{model_name}  —  ROC-AUC {roc_auc:.1%}  |  {len(smart_cols)} SMART features  |  {health_label}",
                expanded=True,
            ):
                # ── Performance metrics ──
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("ROC-AUC",   f"{roc_auc:.1%}")
                c2.metric("F1 Score",  f"{f1:.1%}")
                c3.metric("Precision", f"{precision:.1%}")
                c4.metric("Recall",    f"{recall:.1%}")
                c5.metric("Threshold", f"{threshold:.4f}")

                st.markdown("---")

                # ── Architecture badges ──
                bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.markdown(_metric_badge("Window Length", f"{seq_len} days", "#8b5cf6"), unsafe_allow_html=True)
                bc2.markdown(_metric_badge("SMART Features", str(len(smart_cols)), "#0ea5e9"), unsafe_allow_html=True)
                bc3.markdown(_metric_badge("Flat Features",  str(len(flat_cols)),  "#0ea5e9"), unsafe_allow_html=True)
                bc4.markdown(
                    _metric_badge("Demo Stream", "Available" if demo_csv else "None",
                                  "#22c55e" if demo_csv else "#6b7280"),
                    unsafe_allow_html=True,
                )

                st.markdown("---")

                # ── SMART feature list ──
                with st.expander("View SMART feature columns used in training", expanded=False):
                    if smart_cols:
                        col_a, col_b = st.columns(2)
                        half = len(smart_cols) // 2
                        with col_a:
                            for c in smart_cols[:half]:
                                st.markdown(f"- `{c}`")
                        with col_b:
                            for c in smart_cols[half:]:
                                st.markdown(f"- `{c}`")
                    else:
                        st.caption("No SMART feature info available.")

                # ── Artifact file status ──
                st.markdown("---")
                fc1, fc2, fc3 = st.columns(3)
                joblib_p = MODELS_DIR / f"{model_name}.joblib"
                meta_p   = MODELS_DIR / f"{model_name}_metadata.json"
                demo_p   = MODELS_DIR / f"{model_name}_demo_stream.csv"
                fc1.markdown(f"**Model:** `{joblib_p.name}` {'(exists)' if joblib_p.exists() else '(missing)'}")
                fc2.markdown(f"**Metadata:** `{meta_p.name}` {'(exists)' if meta_p.exists() else '(missing)'}")
                fc3.markdown(f"**Demo stream:** `{demo_p.name}` {'(exists)' if demo_p.exists() else '(missing)'}")

                # ── Delete ──
                st.markdown("---")
                confirm_key = f"confirm_delete_{model_name}"
                if confirm_key not in st.session_state:
                    st.session_state[confirm_key] = False

                if not st.session_state[confirm_key]:
                    if st.button(
                        f"Delete `{model_name}` from registry",
                        key=f"delete_btn_{model_name}",
                        type="secondary",
                    ):
                        st.session_state[confirm_key] = True
                        st.rerun()
                else:
                    st.warning(
                        f"This will permanently delete `{model_name}.joblib`, "
                        f"`{model_name}_metadata.json`, and `{model_name}_demo_stream.csv`. "
                        "This cannot be undone."
                    )
                    cy, cn, _ = st.columns([1, 1, 4])
                    with cy:
                        if st.button("Confirm Delete", key=f"confirm_yes_{model_name}", type="primary"):
                            deleted = []
                            # Only delete model + metadata. The demo_stream.csv is intentionally
                            # kept so the drive stays visible in Live Telemetry showing UNREGISTERED.
                            for fname in [
                                f"{model_name}.joblib",
                                f"{model_name}_metadata.json",
                            ]:
                                p = MODELS_DIR / fname
                                if p.exists():
                                    p.unlink()
                                    deleted.append(fname)
                            # Clear RAM immediately so no ghost predictions
                            router.reload_registry()
                            # Also reload the stream data so Tab 1 sees the updated state
                            st.session_state.stream_data = _load_all_demo_streams()
                            st.session_state.all_days    = _compute_day_index(st.session_state.stream_data)
                            st.session_state[confirm_key] = False
                            st.success(f"Deleted: {', '.join(deleted)}. Drive will now show UNREGISTERED in Live Telemetry.")
                            st.rerun()
                    with cn:
                        if st.button("Cancel", key=f"confirm_no_{model_name}"):
                            st.session_state[confirm_key] = False
                            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Train New Drive Model (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def _validate_uploaded_csv(df: pd.DataFrame) -> tuple:
    """Validate uploaded CSV structure. Returns (is_valid, list_of_errors)."""
    errors = []
    required_cols = {"date", "serial_number", "model", "failure"}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {', '.join(sorted(missing))}")

    smart_cols = [c for c in df.columns if c.startswith("smart_") and c.endswith("_raw")]
    if len(smart_cols) < 5:
        errors.append(f"Only {len(smart_cols)} SMART raw columns found — need at least 5.")

    if "failure" in df.columns:
        if df["failure"].nunique() < 2:
            errors.append("Dataset has only one class in 'failure' — need both 0 and 1.")
        fail_pct = df["failure"].mean() * 100
        if fail_pct < 0.5:
            errors.append(f"Failure rate is only {fail_pct:.2f}% — model may not learn effectively.")

    if "serial_number" in df.columns and df["serial_number"].nunique() < 20:
        errors.append(f"Only {df['serial_number'].nunique()} unique drives — need at least 20.")

    if "model" in df.columns and df["model"].nunique() > 1:
        errors.append(f"CSV has {df['model'].nunique()} drive models — train each model separately.")

    return (len(errors) == 0), errors


def render_tab5(tab):
    with tab:
        st.markdown(
            "Upload a cleaned hard drive telemetry CSV to train a new drive-model-specific "
            "XGBoost predictor and register it automatically in the live model registry."
        )
        st.divider()

        # ── Format guide ──
        with st.expander("CSV Format Requirements", expanded=False):
            st.markdown("""
**Required columns:**
- `date` — Daily observation date (`YYYY-MM-DD`)
- `serial_number` — Unique drive identifier
- `model` — Drive model name (single model per file)
- `failure` — Binary: `1` on the day of failure, `0` otherwise

**SMART columns (minimum 5):**
- Named `smart_<N>_raw` (e.g. `smart_5_raw`, `smart_187_raw`)

**Minimum data scale:**
- At least 20 unique serial numbers
- At least 50 unique failing drives
- 10 consecutive days of readings per drive before label
            """)

        # ── Uploader ──
        uploaded_file = st.file_uploader(
            "Upload drive telemetry CSV",
            type=["csv"],
            key="train_csv_uploader",
        )

        if uploaded_file is None:
            st.info("No file selected. Upload a CSV above to begin.")
            return

        try:
            df_upload = pd.read_csv(uploaded_file, parse_dates=["date"])
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            return

        # ── Dataset summary ──
        st.markdown("#### Dataset Preview")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Total Rows",    f"{len(df_upload):,}")
        p2.metric("Unique Drives", f"{df_upload['serial_number'].nunique():,}" if "serial_number" in df_upload.columns else "N/A")
        p3.metric("Failure Rate",  f"{df_upload['failure'].mean()*100:.2f}%" if "failure" in df_upload.columns else "N/A")
        p4.metric("SMART Columns", str(len([c for c in df_upload.columns if c.startswith("smart_") and c.endswith("_raw")])))

        if "model" in df_upload.columns and df_upload["model"].nunique() == 1:
            drive_model_detected = str(df_upload["model"].iloc[0]).strip()
            st.success(f"Drive model detected: **`{drive_model_detected}`**")

        st.dataframe(df_upload.head(8), hide_index=True, use_container_width=True)

        # ── Validation results ──
        st.markdown("#### Dataset Validation")
        is_valid, errors = _validate_uploaded_csv(df_upload)
        for err in errors:
            st.error(f"[!] {err}")
        if is_valid:
            st.success("[OK] All validation checks passed. Ready to train.")

        # Hard-stop on structural errors (not just warnings)
        structural_errors = [e for e in errors if "Missing required" in e or "SMART raw" in e or "drive models" in e]
        if structural_errors:
            return

        # ── Training config ──
        st.divider()
        st.markdown("#### Training Configuration")
        out_dir = st.text_input("Output directory", value="models")

        train_key  = "train_in_progress"
        result_key = "train_last_result"
        log_key    = "train_log_lines"
        for k, v in [(train_key, False), (result_key, None), (log_key, [])]:
            if k not in st.session_state:
                st.session_state[k] = v

        # ── Show results if training already completed ──
        if st.session_state[result_key] is not None:
            result = st.session_state[result_key]
            st.divider()
            st.markdown("#### Training Complete")
            st.success(f"Model `{result['drive_model']}` trained and registered.")

            m = result["metrics"]
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("ROC-AUC",   f"{m.get('roc_auc', 0):.1%}")
            r2.metric("F1 Score",  f"{m.get('f1_failure', 0):.1%}")
            r3.metric("Precision", f"{m.get('precision_failure', 0):.1%}")
            r4.metric("Recall",    f"{m.get('recall_failure', 0):.1%}")
            r5.metric("Threshold", f"{m.get('best_threshold', 0.5):.4f}")

            st.markdown("**Saved artifacts:**")
            for k, v in result.get("artifacts", {}).items():
                st.markdown(f"- `{v}`")

            st.markdown("**Training log:**")
            for line in st.session_state[log_key]:
                st.markdown(f"`{line}`")

            if st.button("Train Another Model", type="secondary"):
                st.session_state[result_key] = None
                st.session_state[log_key]    = []
                st.rerun()
            return

        # ── Train button ──
        if st.button(
            "Start Training",
            type="primary",
            disabled=st.session_state[train_key],
        ):
            st.session_state[train_key]  = True
            st.session_state[log_key]    = []
            st.session_state[result_key] = None

            import tempfile
            tmp_dir = Path(tempfile.mkdtemp())
            tmp_csv = tmp_dir / (uploaded_file.name or "uploaded_drive_data.csv")
            tmp_csv.write_bytes(uploaded_file.getvalue())

            st.markdown("#### Training Progress")
            progress_bar = st.progress(0, text="Initialising pipeline...")
            log_area     = st.empty()
            log_lines    = []

            STEP_LABELS = {
                1: "Step 1/5 — Loading and validating dataset",
                2: "Step 2/5 — Flattening 10-day sequences and splitting",
                3: "Step 3/5 — Training XGBoost model",
                4: "Step 4/5 — Tuning decision threshold (PR curve)",
                5: "Step 5/5 — Saving model artifacts to registry",
            }

            def _on_progress(step: int, total: int, message: str):
                progress_bar.progress(int(step / total * 100), text=STEP_LABELS.get(step, message))
                log_lines.append(f"[Step {step}/{total}] {message}")
                log_area.markdown("\n".join(f"`{l}`" for l in log_lines[-10:]))

            try:
                pipeline = DriveModelPipeline(
                    csv_path=str(tmp_csv),
                    output_dir=out_dir,
                    progress_callback=_on_progress,
                )
                result = pipeline.run_full_pipeline()
                progress_bar.progress(100, text="Training complete.")

                # Hot-reload: register the new model immediately without restart
                st.session_state.router.reload_registry()
                st.session_state.stream_data = _load_all_demo_streams()
                st.session_state.all_days    = _compute_day_index(st.session_state.stream_data)

                st.session_state[result_key] = result
                st.session_state[log_key]    = log_lines

            except Exception as exc:
                progress_bar.progress(0, text="Training failed.")
                st.error(f"Training error: {exc}")
            finally:
                st.session_state[train_key] = False
                try:
                    tmp_csv.unlink()
                    tmp_dir.rmdir()
                except Exception:
                    pass

            st.rerun()





# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 — Research & Methodology
# ─────────────────────────────────────────────────────────────────────────────

RESEARCH_METRICS_PATH = MODELS_DIR / "research_metrics.json"

REJECTION_RATIONALE = {
    "LightGBM": {
        "verdict": "Competitive — Narrowly Rejected",
        "verdict_color": "#f59e0b",
        "summary": (
            "LightGBM achieved the highest average F1 score (0.7743) and trained 2.5x faster "
            "than XGBoost. However, XGBoost was retained for production for two precise reasons."
        ),
        "reasons": [
            ("AUC Deficit",
             "XGBoost achieved AUC=0.9595 vs LightGBM's 0.9480 — a delta of 0.0115. In the context "
             "of highly imbalanced failure datasets (failure rates of 0.39%–14.8%), AUC is the "
             "dominant metric because it measures ranking quality across the full probability "
             "distribution. A higher AUC means XGBoost more reliably separates healthy from failing "
             "drives at every possible threshold, not just the tuned optimum."),
            ("Precision Advantage",
             "XGBoost Precision of 0.8114 vs LightGBM's 0.7939 means fewer false alerts per drive "
             "family. In an SRE dashboard, Precision directly controls alert fatigue — every false "
             "positive is a wasted escalation. This 1.75% Precision gap compounds at scale across "
             "thousands of monitored drives."),
            ("SHAP Compatibility",
             "Both models support SHAP TreeExplainer. However, XGBoost's depth-wise tree growth "
             "produces more interpretable SHAP waterfall plots compared to LightGBM's leaf-wise "
             "splits, which generate sparser attribution chains for multi-day flattened features."),
        ],
    },
    "Random Forest": {
        "verdict": "Rejected — Precision Too Low",
        "verdict_color": "#ef4444",
        "summary": (
            "Random Forest demonstrated strong Recall (0.7433) and AUC (0.9520) but was rejected "
            "due to its significantly lower Precision, which is critical in a real-time SRE alerting system."
        ),
        "reasons": [
            ("Precision Gap",
             "Random Forest Precision of 0.7549 vs XGBoost's 0.8114 is a 7.65% deficit. Because "
             "all trees are built independently via bootstrap aggregation, the ensemble cannot "
             "iteratively correct for false positives the way gradient boosting does. This produces "
             "more noise-driven splits in the 190-feature space, leading to higher false alarm rates."),
            ("Lowest F1 Score",
             "Average F1 of 0.7251 is the lowest among the three supervised models — a 5.6% deficit "
             "from XGBoost (0.7680). RF was only competitive on ST8000DM002 (F1=0.8525) due to that "
             "dataset's particular SMART distribution, but underperformed on every other family."),
            ("Inference Latency",
             "Inference latency of 0.0375ms per prediction is 16x slower than XGBoost's 0.0023ms. "
             "Storing 300 independent full-depth trees also produces a significantly larger .joblib "
             "artifact. Both factors are problematic for real-time streaming at 0.5s intervals."),
        ],
    },
    "Elliptic Envelope": {
        "verdict": "Rejected — Unsupervised Limitation",
        "verdict_color": "#ef4444",
        "summary": (
            "Elliptic Envelope was the strongest unsupervised method overall (F1=0.3511, AUC=0.8932) "
            "but falls 2.2x below XGBoost's F1. It was studied to validate the necessity of labelled "
            "failure data."
        ),
        "reasons": [
            ("Gaussian Assumption",
             "The Elliptic Envelope assumes the healthy SMART feature distribution follows a "
             "multivariate Gaussian. This holds reasonably after PCA(30) reduction, explaining "
             "its competitive AUC, but fails for non-linear, multi-modal degradation patterns in "
             "the original 190-feature space. SMART data is right-skewed and zero-inflated for "
             "many attributes, violating this assumption."),
            ("No Failure Labels",
             "As a purely unsupervised method, it trains exclusively on healthy drives and has no "
             "mechanism to learn the specific feature signatures that precede failure. It only learns "
             "'unusual' — not 'failing'. This caps its F1 at approximately 0.35 across all datasets, "
             "regardless of tuning."),
            ("PCA Information Loss",
             "Dimensionality reduction to 30 principal components was required to make MCD estimation "
             "tractable on 190 features. This discards fine-grained SMART interaction effects that "
             "XGBoost's tree splits explicitly capture — for example, the joint signal of rising "
             "smart_197_raw alongside elevated smart_194_raw across 10 consecutive days."),
        ],
    },
    "LOF": {
        "verdict": "Rejected — Unsupervised Limitation",
        "verdict_color": "#ef4444",
        "summary": (
            "Local Outlier Factor was the most consistent unsupervised method across datasets "
            "(AUC=0.8734) but was rejected for production due to fundamental memory and "
            "scalability constraints."
        ),
        "reasons": [
            ("Memory at Inference",
             "LOF with novelty=True must retain the entire training set in memory at inference time "
             "to compute k-nearest-neighbor distances for new observations. On production infrastructure "
             "with 37,000+ drives, this means storing an 8,000x190 float32 matrix (~5.8MB per model) "
             "that must be scanned for every incoming log row — incompatible with the real-time "
             "0.5-second streaming architecture."),
            ("No Probability Output",
             "LOF produces a signed outlier score but no calibrated probability. The SentinelDC "
             "dashboard requires a continuous failure probability (0–100%) for trend charts, "
             "risk level classification (WARNING/CRITICAL), and PR-curve threshold tuning. "
             "LOF's binary anomaly flag cannot support this without additional calibration layers."),
            ("Subsampling Bias",
             "On the largest dataset (ST12000NM0007, 37K drives), LOF required capping training "
             "at 8,000 drives for tractability. This subsampling degrades neighborhood density "
             "estimates — drives in sparsely sampled regions appear more anomalous not because "
             "they are failing, but because their neighbors were excluded from the reference set."),
        ],
    },
}


def render_tab6(tab):
    import json
    import numpy as np
    import pandas as pd
    
    with tab:
        if not RESEARCH_METRICS_PATH.exists():
            st.info(
                "Research metrics not yet generated. "
                "Run `python research/run_all_datasets.py` and "
                "`python research/run_extra_anomaly.py` from the project root."
            )
            return

        with open(RESEARCH_METRICS_PATH, encoding="utf-8") as f:
            data = json.load(f)

        agg     = data.get("aggregate_averages", {})
        lof_agg = data.get("local_outlier_factor", {}).get("aggregate_averages", {})
        ee_agg  = data.get("elliptic_envelope",    {}).get("aggregate_averages", {})

        def _accuracy_avg(model_key, section=None):
            if section is None:
                vals = [
                    d["models"].get(model_key, {}).get("accuracy")
                    for d in data.get("per_dataset", {}).values()
                    if d["models"].get(model_key, {}).get("accuracy") is not None
                ]
            else:
                vals = [
                    d.get("accuracy")
                    for d in data.get(section, {}).get("per_dataset", {}).values()
                    if d.get("accuracy") is not None
                ]
            return round(float(np.mean(vals)), 4) if vals else None

        MODELS = {
            "XGBoost":           {**agg.get("XGBoost", {}),     "accuracy": _accuracy_avg("XGBoost"),     "category": "Supervised",   "is_production": True},
            "LightGBM":          {**agg.get("LightGBM", {}),    "accuracy": _accuracy_avg("LightGBM"),    "category": "Supervised",   "is_production": False},
            "Random Forest":     {**agg.get("RandomForest", {}),"accuracy": _accuracy_avg("RandomForest"),"category": "Supervised",   "is_production": False},
            "LOF":               {**lof_agg,                     "accuracy": _accuracy_avg(None, "local_outlier_factor"), "category": "Unsupervised", "is_production": False},
            "Elliptic Envelope": {**ee_agg,                      "accuracy": _accuracy_avg(None, "elliptic_envelope"),    "category": "Unsupervised", "is_production": False},
        }

        # ── Header ──
        st.markdown(
            "This tab documents the complete model selection research for SentinelDC. "
            "All five candidate architectures were benchmarked against the same **6 real-world "
            "Backblaze hard drive datasets** using identical 70/30 serial-level stratified splits, "
            "10-day flattened feature windows (190 features), and PR-curve threshold tuning. "
            "Metrics below are averaged across all 6 drive families."
        )
        st.caption(
            f"Datasets evaluated: {len(data.get('datasets_evaluated', []))} drive families  |  "
            "Methodology: 10-Day Rolling Window, Serial-Level 70/30 Split, PR-Curve Threshold Tuning  |  "
            f"Generated: {data.get('generated_at', 'N/A')[:10]}"
        )
        st.divider()

        # ── Section 1: Metrics Table ──
        st.markdown("#### Model Performance Comparison — Averaged Across 6 Drive Families")

        METRIC_COLS = ["accuracy", "precision", "recall", "f1_score", "roc_auc",
                       "train_time_sec", "infer_latency_ms"]
        METRIC_LABELS = {
            "accuracy":         "Accuracy",
            "precision":        "Precision",
            "recall":           "Recall",
            "f1_score":         "F1 Score",
            "roc_auc":          "AUC-ROC",
            "train_time_sec":   "Train Time (s)",
            "infer_latency_ms": "Latency (ms/pred)",
        }

        table_rows = []
        for model_name, m in MODELS.items():
            row = {
                "Model":    ("[PRODUCTION] " if m["is_production"] else "") + model_name,
                "Category": m["category"],
            }
            for col in METRIC_COLS:
                val = m.get(col)
                row[METRIC_LABELS[col]] = f"{val:.4f}" if isinstance(val, float) else "N/A"
            table_rows.append(row)

        df_table = pd.DataFrame(table_rows).set_index("Model")
        st.dataframe(df_table, use_container_width=True)
        st.caption("[PRODUCTION] = Selected for deployment in SentinelDC")

        st.divider()

        # ── Section 2: Bar Charts ──
        st.markdown("#### Metric-by-Metric Visual Comparison")

        model_names = list(MODELS.keys())
        colors = {
            "XGBoost":           "#10b981",
            "LightGBM":          "#3b82f6",
            "Random Forest":     "#8b5cf6",
            "LOF":               "#f59e0b",
            "Elliptic Envelope": "#f97316",
        }

        chart_metrics = [
            ("F1 Score",  "f1_score"),
            ("AUC-ROC",   "roc_auc"),
            ("Precision", "precision"),
            ("Recall",    "recall"),
            ("Accuracy",  "accuracy"),
            ("Train Time (s)", "train_time_sec"),
        ]
        pairs = [chart_metrics[i:i+2] for i in range(0, len(chart_metrics), 2)]
        for pair in pairs:
            cols_ui = st.columns(2)
            for ci, (label, key) in enumerate(pair):
                vals = [MODELS[m].get(key) or 0 for m in model_names]
                fig = go.Figure(go.Bar(
                    x=model_names, y=vals,
                    marker_color=[colors.get(m, "#94a3b8") for m in model_names],
                    text=[f"{v:.4f}" for v in vals],
                    textposition="outside", textfont_size=11,
                ))
                y_max = max(vals) * 1.25 if max(vals) > 0 else 1
                fig.update_layout(
                    title=label, title_font_size=14,
                    margin=dict(l=10, r=10, t=40, b=10), height=280,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(showgrid=True, gridcolor="#334155", range=[0, y_max]),
                    xaxis=dict(showgrid=False), showlegend=False,
                )
                with cols_ui[ci]:
                    st.plotly_chart(fig, use_container_width=True, key=f"research_{key}")

        # Inference latency chart (very small values — separate scale)
        lat_vals = [MODELS[m].get("infer_latency_ms") or 0 for m in model_names]
        fig_lat = go.Figure(go.Bar(
            x=model_names, y=lat_vals,
            marker_color=[colors.get(m, "#94a3b8") for m in model_names],
            text=[f"{v:.4f}" for v in lat_vals],
            textposition="outside", textfont_size=11,
        ))
        fig_lat.update_layout(
            title="Inference Latency (ms per prediction)", title_font_size=14,
            margin=dict(l=10, r=10, t=40, b=10), height=280,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=True, gridcolor="#334155"),
            xaxis=dict(showgrid=False), showlegend=False,
        )
        st.plotly_chart(fig_lat, use_container_width=True, key="research_latency")

        st.divider()

        # ── Section 3: Rejection Rationale ──
        st.markdown("#### Model Selection Rationale — Why XGBoost")
        st.markdown(
            "XGBoost was selected based on its superior **AUC-ROC (0.9595)** and "
            "**Precision (0.8114)** — the two metrics most critical for a production SRE "
            "alerting system where false positives cause alert fatigue. "
            "The following cards detail why each alternative was eliminated."
        )

        for model_name, r in REJECTION_RATIONALE.items():
            with st.expander(f"{model_name}  —  {r['verdict']}", expanded=False):
                st.markdown(
                    f"<span style='background:{r['verdict_color']};color:#fff;"
                    f"padding:2px 12px;border-radius:4px;font-size:0.8rem;font-weight:600;'>"
                    f"{r['verdict']}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(" ")
                st.markdown(r["summary"])

                # Delta metrics vs XGBoost for supervised models
                prod = MODELS["XGBoost"]
                cand = MODELS.get(model_name)
                if cand and cand.get("category") == "Supervised":
                    c1, c2, c3 = st.columns(3)
                    d_f1   = round((cand.get("f1_score",   0) or 0) - (prod.get("f1_score",   0) or 0), 4)
                    d_auc  = round((cand.get("roc_auc",    0) or 0) - (prod.get("roc_auc",    0) or 0), 4)
                    d_prec = round((cand.get("precision",  0) or 0) - (prod.get("precision",  0) or 0), 4)
                    c1.metric("F1 vs XGBoost",        f"{d_f1:+.4f}")
                    c2.metric("AUC-ROC vs XGBoost",   f"{d_auc:+.4f}")
                    c3.metric("Precision vs XGBoost",  f"{d_prec:+.4f}")
                    st.markdown(" ")

                for i, (title, body) in enumerate(r["reasons"], 1):
                    st.markdown(f"**{i}. {title}**")
                    st.markdown(f"> {body}")
                    st.markdown(" ")

        # ── Section 4: Architectural & Theoretical Foundations (Research Paper Alignment) ──
        st.divider()
        st.markdown("#### Architectural Paradigm: Two-Tier Deterministic-to-Generative (D2G) RCA")
        st.markdown(
            "SentinelDC implements the **Two-Tier Deterministic-to-Generative (D2G)** triage architecture "
            "designed to eliminate hallucination risks and latency penalties in hyperscale infrastructure monitoring."
        )

        col_tier1, col_tier2 = st.columns(2)
        with col_tier1:
            st.markdown("""
##### Tier 1: Deterministic Expert Rule Engine
* **Execution:** Synchronous, fully offline, sub-millisecond ($O(1)$ constant response).
* **Mechanism:** Exact game-theoretic **TreeSHAP** attribution values:
  $$\\sum_{m=1}^{M} \\phi_m(x) = f(x) - \\mathbb{E}[f(x)]$$
* **Functions:**
  1. Identifies physical degradation modes (e.g., `MAGNETIC_PLATTER_DEGRADATION` via $\\phi_{\\text{SMART5}} > \\varepsilon \\land \\text{Raw}(\\text{SMART5}) > 0$, `HEAD_PREAMP_INSTABILITY` via $\\phi_{\\text{SMART187}} > \\varepsilon$).
  2. Computes storage blast radius (affected ZFS pools, Ceph OSDs, RAID arrays, VM mountpoints).
  3. Attaches pre-verified SRE remediation runbooks.
* **Output:** Strictly validated JSON schema payload (Zero hallucination risk).
            """)

        with col_tier2:
            st.markdown("""
##### Tier 2: Grounded LLM Synthesizer
* **Execution:** Asynchronous background execution (LLaMA-3.3-70B via Groq).
* **Mechanism:** Instruction-tuned LLM bounded strictly by Tier 1 JSON contracts.
* **Functions:**
  1. Translates quantitative attribution vectors into an L3 SRE incident brief.
  2. Synthesizes mechanical/firmware degradation kinetics and timeline.
  3. Formulates structured hardware evacuation instructions.
* **Fail-Safe SLA:** In the event of an API timeout ($> 400\\text{ ms}$) or network disconnect, Tier 1 diagnostic output is dispatched directly to on-call engineers.
            """)

        st.markdown("#### Multi-Day Temporal Feature Aggregation vs. Cross-Fleet Distortion")
        st.markdown(
            "* **10-Day Rolling Lookback ($x \\in \\mathbb{R}^{M \\times 10}$):** Captures temporal degradation slopes (e.g., rapid acceleration in pending sectors vs. stable static count) without quadratic attention overhead.\n"
            "* **Vendor-Isolated Model Routing:** Drive manufacturers employ disparate SMART register semantics (e.g., vendor-specific bitmask encodings in Seagate vs. raw integer event counters in Toshiba/HGST). Segregated model routing prevents cross-fleet feature distortion while maintaining specialized decision boundaries."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import time
    init_session_state()
    render_header()
    render_sidebar()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Live Telemetry",
        "Alerts & Diagnostics",
        "Audits",
        "Model Registry",
        "Train Model",
        "Model Comparisons and Others",
    ])

    render_tab1(tab1)
    render_tab2(tab2)
    render_tab3(tab3)
    render_tab4(tab4)
    render_tab5(tab5)
    render_tab6(tab6)

    # ── Real-Time Stream auto-advance loop ──
    # If the toggle is ON and there are still days left, advance ONE log row,
    # wait 0.5 s, then rerun. Each call to advance_single_log() emits exactly
    # one drive's row. When all drives for a day are done, current_day increments.
    if (
        st.session_state.real_time_streaming
        and st.session_state.current_day < len(st.session_state.all_days)
    ):
        advance_single_log()
        time.sleep(0.5)
        st.rerun()
    elif st.session_state.real_time_streaming and st.session_state.current_day >= len(st.session_state.all_days):
        # Stream finished — auto-stop the toggle
        st.session_state.real_time_streaming = False


if __name__ == "__main__":
    main()
