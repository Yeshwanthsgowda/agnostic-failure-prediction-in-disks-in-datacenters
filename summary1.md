# SentinelDC: Hard Drive Failure Prediction & Telemetry System
## Comprehensive Project Checkpoint & Handover Document (`summary1.md`)

**Generated at**: 2026-08-28  
**Project Workspace**: `e:\projects\Prediction-of-Hard-Drive-Failure`  
**Purpose**: Complete handover context for picking up development seamlessly across sessions/accounts.

---

## 1. Project Overview & Architecture

**Goal**: Build an enterprise-grade, real-time predictive maintenance system (**SentinelDC**) using Backblaze hard drive SMART telemetry data.

### Core Components
1. **Machine Learning Pipeline (`pipeline/pipeline.py`)**: Automated per-drive-model training pipeline using a **10-day flattened rolling window** and cost-sensitive XGBoost with Precision-Recall threshold optimization.
2. **Stateful Telemetry Router (`pipeline/router.py`)**: Multi-model dynamic registry with an in-memory per-serial sliding window cache (`maxlen=10`) and SHAP-based Root Cause Analysis (RCA).
3. **Streamlit Frontend Dashboard (`app.py`)**: Professional 5-tab SRE dashboard simulating a live data center telemetry firehose with real-time inference, alert triage, Groq LLM RCA, deterministic audit runbooks, and model lifecycle management.

---

## 2. Key Research Breakthroughs & Model Decisions

### A. Deep Learning (LSTM) vs. Tree Ensemble (XGBoost)
We conducted a rigorous head-to-head benchmark on 382,000 rows (`ST12000NM0007` dataset) comparing a 3D sequential Keras LSTM vs. a 2D flattened XGBoost on 10-day windows:

| Metric | Baseline Keras LSTM (3D Sequential) | XGBoost (2D Flattened Window) |
| :--- | :---: | :---: |
| **Architecture** | 10 timesteps, hidden LSTM layers | 1 row = 10 days flattened (190 features) |
| **Precision** | N/A (Poor convergence) | **76.9%** (un-tuned) / **96.5%** (tuned) |
| **Recall** | 53.1% | **86.6%** (un-tuned) / **83.1%** (tuned) |
| **PR-AUC** | 8.3% | **91.1%** |
| **ROC-AUC** | ~72.0% | **99.15%** |

*Findings recorded in `research.md`: SMART telemetry is characterized by sudden threshold spikes rather than wave-like temporal patterns. XGBoost handles tabular splits and extreme class imbalance (~33:1) vastly better than LSTM.*

### B. Evolution from v1 (Stateless) to v2 (Stateful 10-Day Flattened)
* **v1 Stateless (`1 row = 1 day`, 19 features)**: Model evaluated each daily log in isolation. Low recall (~47%), unable to capture degradation trends over time.
* **v2 Stateful (`1 row = 10 days`, 190-210 features)**: Model looks at the entire 10-day degradation curve (`smart_5_raw_day_1` through `smart_5_raw_day_10`). ROC-AUC jumped to **99.1%**, F1 jumped to **89.3%**.
* **Stateful Router Ingestion**:
  * Days 1–9 for any drive serial return `status: "accumulating"` (Buffering in rolling deque).
  * Day 10+ emits `status: "predicted"` with probability, risk level (`healthy`, `warning`, `critical`), and normalized `risk_score` (0–100).

---

## 3. Current Trained Models in Registry (`models/`)

Both drive models have been fully trained with v2 architecture and saved to `models/`:

### 1. `ST12000NM0007` (Seagate Enterprise 12TB)
* **Features**: 190 (19 non-zero-variance SMART features × 10 days)
* **Optimal Threshold**: `0.7739` (tuned on PR curve for maximum F1)
* **Performance**:
  * **Accuracy**: 99.42%
  * **Precision**: 96.55%
  * **Recall**: 83.09%
  * **F1-Score**: 89.31%
  * **ROC-AUC**: 99.15%
* **Artifacts**: `models/ST12000NM0007.joblib`, `models/ST12000NM0007_metadata.json`, `models/ST12000NM0007_demo_stream.csv`

### 2. `ST4000DM000` (Seagate Desktop 4TB)
* **Features**: 210 (21 non-zero-variance SMART features × 10 days)
* **Optimal Threshold**: `0.6551`
* **Performance**:
  * **Accuracy**: 99.43%
  * **Precision**: 89.11%
  * **Recall**: 75.63%
  * **F1-Score**: 81.82%
  * **ROC-AUC**: 96.90%
* **Artifacts**: `models/ST4000DM000.joblib`, `models/ST4000DM000_metadata.json`, `models/ST4000DM000_demo_stream.csv`

---

## 4. Curated 20-Day Demo Streams (Gradual Degradation)

To showcase a realistic gradual degradation curve during live presentations (without jumping instantly from 0% to 99%), we stitched 10 days of real healthy data with 10 days of real failing data for each drive model:

### Drive 1 (`ST12000NM0007`) -> `models/perfect_demo_stream.csv`
* **Serial Number**: `DEMO_DRIVE_001`
* **Progression**:
  * **Days 1–9**: Buffering (`status: "accumulating"`)
  * **Day 10**: `7.03%` (HEALTHY) — baseline healthy window
  * **Day 11**: `10.95%` (HEALTHY) — early seek/read error fluctuations
  * **Day 12**: `70.56%` (CRITICAL) — early mechanical strain detected
  * **Day 13**: `56.88%` (WARNING) — transient stabilization
  * **Day 14**: `67.45%` (CRITICAL) — thermal stress (+18°C rise)
  * **Day 15**: `86.61%` (CRITICAL) — compounding thermal/read errors
  * **Day 16–17**: `~88%` (CRITICAL) — severe degradation plateau
  * **Day 18–20**: `99.89%` -> `99.99%` (CRITICAL) — fatal platter scratch (`SMART 5` spikes to 384 then 12,352)

### Drive 2 (`ST4000DM000`) -> `models/ST4000DM000_stitched_demo.csv`
* **Serial Number**: `DEMO_DRIVE_002`
* **Progression**:
  * **Days 1–9**: Buffering
  * **Day 10**: `0.04%` (HEALTHY)
  * **Day 14**: `2.39%` (HEALTHY)
  * **Day 15**: `20.99%` (HEALTHY)
  * **Day 16**: `55.25%` (CRITICAL) — `SMART 5` = 48
  * **Day 17**: `76.08%` (CRITICAL) — `SMART 5` = 56
  * **Day 18**: `83.57%` (CRITICAL) — `SMART 5` = 144
  * **Day 19**: `94.54%` (CRITICAL) — `SMART 5` = 800
  * **Day 20**: `99.90%` (CRITICAL) — `SMART 5` = 928

---

## 5. Frontend Architecture & Build Status (`app.py`)

### What is Completed (Phase 1):
* **App Shell & Theme**: Professional Dark-Mode styling with high-contrast typography (clean `#0f172a` slate palette, no emojis).
* **Header & Top Navigation**: Clean "SentinelDC" brand with 5 tabs:
  1. `Live Telemetry`
  2. `Alerts & Root Cause`
  3. `Critical Audits & SRE Runbooks`
  4. `Model Registry`
  5. `Train New Drive`
* **Sidebar Telemetry Engine**: Controls for simulated stream speed, "Advance 1 Day" button, "Reset Stream" button, and active drive selector.
* **State Management**: `st.session_state` initialized for day pointer, accumulated logs, prediction cache, and alert storage.

---

## 6. Detailed Next Steps (Phases 2–5)

When resuming this project, proceed strictly in this order:

### Phase 2: Tab 1 — Live Telemetry & Real-Time Plotly Chart (NEXT STEP)
1. **Update Tab 1 Data Table**:
   * Feed incoming stream logs from the curated demo CSVs (`perfect_demo_stream.csv` / `ST4000DM000_stitched_demo.csv`).
   * Render table using `st.dataframe(..., on_select="rerun", selection_mode="single-row")`.
   * Display `Status` column as `Buffering (X/10)` for days 1–9, and formatted probability/risk badge for day 10+.
2. **Interactive Plotly Failure Risk Timeline**:
   * When a row (serial number) is clicked in the table, query that serial's historical predictions from `st.session_state`.
   * Render an interactive Plotly line chart below the table tracking `Failure Probability (%)` across the 20-day timeline with a reference line for the model's decision threshold.

### Phase 3: Tabs 2 & 3 — Alert Triage, LLM RCA & SRE Runbooks
1. **Tab 2 (Alerts & Root Cause)**:
   * Filter logs for `risk_level in ["warning", "critical"]`.
   * Implement hierarchical accordion: `Drive Model -> Serial Number -> Alert Card`.
   * Display grouped SHAP top risk factors.
   * Add a prominent button: **"Generate AI Root Cause Analysis"** (calls Groq API using `llama-3.3-70b-versatile` or `mixtral-8x7b-32768` on-demand to conserve tokens).
2. **Tab 3 (Critical Audits & SRE Runbooks)**:
   * Filter strictly for `critical` risk events.
   * Generate instantaneous, deterministic, system-level RCA (zero external LLM calls) with exact SRE mitigation runbooks (e.g., RAID re-silvering, hot-spare migration, power-cycle lockouts).

### Phase 4: Tabs 4 & 5 — Model Registry & Self-Serve Training
1. **Tab 4 (Model Registry)**:
   * Display card/table of all loaded models from `router.list_models()`.
   * Show PR-AUC, Recall, Optimal Threshold, and feature counts.
   * Include a **"Delete Model"** button with confirmation dialog that removes artifacts from `models/` and reloads the registry.
2. **Tab 5 (Train New Drive)**:
   * File uploader for new drive CSVs (Backblaze format).
   * "Train Model" button that triggers `DriveModelPipeline` with a live Streamlit progress bar callback.
   * Auto-refreshes the registry and makes the new drive immediately selectable in the telemetry stream.

---

## 7. Key File Index

* `pipeline/pipeline.py`: 10-day flattened XGBoost training pipeline.
* `pipeline/router.py`: Stateful sliding window router and SHAP engine.
* `pipeline/test_router.py`: Verification smoke test for stateful streaming & RCA.
* `app.py`: Streamlit frontend application.
* `research/train_models.py`: XGBoost vs LSTM research benchmark script.
* `research.md`: Full research experiment documentation.
* `models/perfect_demo_stream.csv`: Curated 20-day stream for `ST12000NM0007`.
* `models/ST4000DM000_stitched_demo.csv`: Curated 20-day stream for `ST4000DM000`.
