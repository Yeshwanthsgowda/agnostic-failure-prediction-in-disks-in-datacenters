# SentinelDC: Hard Drive Failure Prediction System
## Project Handover & Architecture Context

**Goal**: A complete, end-to-end Streamlit dashboard designed for SREs to monitor live hard drive telemetry, receive predictive alerts, view AI-generated root cause analyses, and manage ML models. 

This document serves as the absolute source of truth for the project's architecture, workflows, and edge cases. Future AI agents should read this file to understand the entire system before making any modifications.

---

### 1. Architecture Overview

The system consists of a Python Streamlit frontend (`app.py`), a custom ML pipeline (`pipeline.py`), and a stateful inference router (`router.py`). 

* **Machine Learning**: XGBoost Classifier.
* **Temporal Strategy (Crucial)**: The model does *not* look at a single day in isolation. It uses a **10-day flattened rolling window**. 19 raw SMART attributes × 10 days = 190 features per row. This gives the model access to degradation *trends*, significantly boosting Precision and Recall.
* **Inference Engine**: Because the model requires 10 days of data, `router.py` maintains an internal `deque(maxlen=10)` cache per drive serial. Days 1-9 return `status: "accumulating"`. Day 10+ returns `status: "predicted"`.

### 2. File Structure

* `app.py`: The main Streamlit dashboard containing the UI shell, state management, and 5 tabs.
* `pipeline/pipeline.py`: The `DriveModelPipeline` class. Handles loading CSVs, flattening 10-day sequences, training XGBoost, tuning the PR-Curve threshold, and saving artifacts.
* `pipeline/router.py`: The `DriveTelemetryRouter` class. Scans the registry, maintains the rolling deque cache, runs live predictions, and computes SHAP feature importance.
* `models/`: The model registry directory. Contains:
  * `<model>.joblib`: The trained XGBoost model.
  * `<model>_metadata.json`: Feature definitions, metrics, and threshold.
  * `<model>_demo_stream.csv`: **Hand-crafted** demo telemetry sequences (20 days).

---

### 3. Core Workflows (The 5 Tabs)

#### Tab 1: Live Telemetry
* **Behavior**: Simulates live streaming by injecting telemetry data day-by-day from `_demo_stream.csv` files.
* **Real-Time Streaming**: Features a live auto-advance ticker. When `[ > ] Start Real-Time Stream` is toggled, `app.py` emits exactly **one drive's log row** every 0.5 seconds, automatically advancing days when all drives are processed. The stream can be paused and resumed without losing state.
* **Missing Models**: If a drive's model is deleted from the registry but its stream data remains, the router throws `ModelNotFoundError`. The UI catches this and gracefully displays `[?] UNREGISTERED`.

#### Tab 2: Alerts & Diagnostics
* **Behavior**: Filters telemetry for drives currently in `WARNING` (10-80%) or `CRITICAL` (>80%) states.
* **Explainability**: Uses `shap.TreeExplainer` in the router to return the top 5 SMART factors contributing to the risk score.
* **Dynamic Sparklines**: Below the SHAP table, a 2-column grid dynamically renders Plotly sparkline charts tracking the exact 5 SHAP risk factors for that specific drive from Day 1 to the current day, overlaid against the global healthy median for reference.
* **AI RCA**: Integrates with the **Groq API**.
  * **Dynamic Model Fallback**: Automatically queries `client.models.list()` and falls back through a hierarchy of available models (`openai/gpt-oss-20b`, `groq/compound`, etc.) to prevent 404 errors. Exceptions are heavily cached and printed directly to the UI to avoid silent failures.
  * **L3 SRE Prompting**: The LLM prompt is strictly engineered to output dense, low-level hardware diagnostics (e.g., servo tracking, ECC exhaustion, LBA mapping) and never repeats the tabular SHAP data.

#### Tab 3: Audits & SRE Runbooks
* **Behavior**: A persistent registry of any drive that has *ever* crossed the CRITICAL threshold (`st.session_state.critical_incident_registry`). Even if the drive drops back to Warning later, the incident card stays open here.
* **Dynamic Archetypes**: The system dynamically maps the top SHAP factor (e.g., `SMART 5`) to a failure archetype (A: Platter, B: Thermal, C: Mechanical, etc.).
* **Runbooks**: Generates 5-step SRE runbooks with real, injected bash commands (e.g., `ledctl locate=/dev/disk/by-id/ata-ST12000NM0007_ZCH068EH`). 

#### Tab 4: Model Registry
* **Behavior**: Lists all active models loaded from the `models/` directory. Shows ROC-AUC, Precision/Recall, and architecture badges.
* **Delete Flow**: Deletes `.joblib` and `_metadata.json` from disk, then instantly clears the router's RAM (`self.registry.clear()`) and hot-reloads. **Crucial Note**: It intentionally *leaves* `_demo_stream.csv` on disk so Tab 1 can demonstrate the `UNREGISTERED` fallback behavior.

#### Tab 5: Train New Model
* **Behavior**: Upload a CSV from the `dataset/` directory. Validates the data, shows a 5-step progress bar, trains the model, and hot-reloads the registry without restarting the app.

---

### 4. Critical "Gotchas" & Agent Rules

1. **Demo Streams Must Not Be Overwritten**:
   * *The Problem*: Real hard drive telemetry usually has `SMART 5` or `SMART 197` jump instantly. XGBoost catches this and immediately spikes the failure probability to 99%, ruining the visual curve.
   * *The Solution*: `_demo_stream.csv` files (e.g. `ST12000NM0007`, `TOSHIBA MQ01ABF050`) are **hand-crafted** using a "slow burn" approach: temperature and G-sense shocks rise slowly over days 10-15 to trigger a WARNING (10-50%), and pending sectors appear late to trigger a CRITICAL (80%+).
   * *The Rule*: `pipeline.py` was specifically modified to **never auto-generate or overwrite** `_demo_stream.csv` files during training. Do not re-add demo generation to the pipeline.

2. **Router RAM Leaks**:
   * Whenever manipulating the registry, `router.reload_registry()` must execute `self.registry.clear()` first, otherwise deleted models remain alive in memory.

3. **No Database Used**:
   * The entire application relies purely on the filesystem (`models/` dir) and Streamlit's `st.session_state`. Do not attempt to add SQLite, Redis, or PostgreSQL. 

4. **UI Aesthetics**:
   * The user explicitly requested **no emojis** anywhere in the UI (professional SRE aesthetic only). Use ASCII brackets like `[!]` or `[OK]` instead. All LLM prompts must strictly forbid generic advice or introductory fluff.
