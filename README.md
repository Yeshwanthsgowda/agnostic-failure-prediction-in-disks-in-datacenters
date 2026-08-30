# SentinelDC: Datacenter Telemetry Monitoring & Two-Tier Root Cause Attribution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EB6424.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-TreeSHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Groq LLaMA-3.3](https://img.shields.io/badge/LLM-LLaMA--3.3--70B-orange.svg)](https://groq.com/)
[![Dataset](https://img.shields.io/badge/Dataset-Backblaze_SMART-black.svg)](https://www.backblaze.com/b2/hard-drive-test-data.html)

**SentinelDC** is an end-to-end predictive maintenance and automated Site Reliability Engineering (SRE) root cause analysis framework for hyperscale datacenter storage fleets. Evaluated across **348,810 Backblaze drive-days**, SentinelDC couples temporal gradient boosting with game-theoretic feature attribution and a **Two-Tier Deterministic-to-Generative (D2G)** triage pipeline to deliver sub-millisecond failure forecasting and grounded, hallucination-free incident briefs.

---

## Key Capabilities

* **Multi-Day Temporal Degradation Modeling:** Rolling 10-day sliding window feature engineering ( \in \mathbb{R}^{M \times 10}$) captures degradation rates of change (acceleration in uncorrectable/pending sectors) without the quadratic memory overhead $\mathcal{O}(N^2)$ of self-attention transformers.
* **Vendor-Isolated Model Routing:** Drive manufacturers implement disparate SMART register semantics (composite bitmask encodings vs. monotonic event counters). Dynamic vendor-segregated model routing prevents cross-fleet feature distortion while maintaining specialized decision boundaries.
* **Two-Tier Deterministic-to-Generative (D2G) RCA:**
  * **Tier 1 (Deterministic Expert Engine):** Runs synchronously offline with $\mathcal{O}(1)$ latency. Computes exact game-theoretic **TreeSHAP** attributions, identifies physical failure modes (MAGNETIC_PLATTER_DEGRADATION, HEAD_PREAMP_INSTABILITY), maps storage blast radius (affected ZFS pools, Ceph OSDs, RAID arrays), and attaches verified runbooks.
  * **Tier 2 (Grounded LLM Synthesizer):** Dispatches asynchronous calls to instruction-tuned **LLaMA-3.3-70B** via Groq. Strict JSON prompt contracts prohibit the introduction of ungrounded entities or fabricated metric values. Guaranteed sub-400ms SLA fallback to Tier 1 upon network failure.
* **Real-Time Telemetry Streaming:** Live simulation mode with step-by-step telemetry emission (0.5s cadence per drive log) and multi-day health tracking.
* **Dynamic SHAP Sparklines:** Interactive multi-day metric trend charts comparing degraded attributes against fleet-wide healthy medians.
* **Autonomous Model Registry & Training:** Train and hot-swap new vendor-specific XGBoost models with automated PR-curve threshold tuning and filesystem persistence.

---

## System Architecture

`
+-----------------------------------------------------------------------------------+
|              Backblaze Daily SMART Telemetry (ST12000, ST4000, TOSHIBA, HGST)     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|               Feature Pipeline: 10-Day Flattening + Vendor Routing                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|       Inference Core: Gradient Boosted Ensembles (XGBoost / PR-Curve Tuned)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|           Explainability Layer: Exact Game-Theoretic TreeSHAP Attribution          |
|                   \sum_{m=1}^M \phi_m(x) = f(x) - E[f(x)]                         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|    TIER 1 (Deterministic Rule Engine — Offline, O(1) Critical-Path SLA)           |
|    - Physical Failure Classification (Platter wear, Head instability, Thermal)    |
|    - Storage Blast Radius Calculation (ZFS pools, Ceph OSDs, VM mountpoints)      |
|    - Standard Operating Procedure (SOP) Runbook Attachment                        |
|    - Validated Immutable JSON Schema Payload Generation                           |
+-----------------------------------------------------------------------------------+
                         |                               |
          [Direct Alert / API Timeout]                   | [Grounded JSON Payload]
                         |                               v
                         |       +--------------------------------------------------+
                         |       |  TIER 2 (Grounded LLM Synthesizer — Asynchronous)|
                         |       |  - LLaMA-3.3-70B (via Groq API)                  |
                         |       |  - Strict Zero-Hallucination JSON Bound          |
                         |       |  - Executive SRE Narrative Brief                 |
                         |       +--------------------------------------------------+
                         |                               |
                         v                               v
+-----------------------------------------------------------------------------------+
|               SentinelDC SRE Operations Dashboard (Streamlit UI)                  |
+-----------------------------------------------------------------------------------+
`

---

## Empirical Benchmark & Model Selection

Benchmarked across 6 diverse Backblaze enterprise drive families (including 12TB, 8TB, 4TB Seagate, Toshiba 2.5\" high-failure rate drives, and HGST arrays):

| Model | Category | Avg F1 | Avg AUC-ROC | Avg Precision | Avg Recall | Train Time | Latency | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | **Supervised** | **0.7680** | **0.9595** | **0.8114** | **0.7359** | 1.48s | **0.0023ms** | **[PRODUCTION] Optimal AUC & Precision** |
| **LightGBM** | Supervised | 0.7743 | 0.9480 | 0.7939 | 0.7582 | 0.58s | 0.0039ms | Narrowly rejected (AUC deficit & leaf-wise SHAP sparsity) |
| **Random Forest** | Supervised | 0.7251 | 0.9520 | 0.7549 | 0.7433 | 1.53s | 0.0375ms | Rejected (Low precision, high false alarms in SRE ops) |
| **Elliptic Envelope** | Unsupervised | 0.3511 | 0.8932 | 0.2662 | 0.5347 | 3.54s | 0.0009ms | Rejected (Gaussian assumption fails on non-linear SMART) |
| **LOF** | Unsupervised | 0.3374 | 0.8734 | 0.2561 | 0.5147 | 0.51s | 0.0371ms | Rejected (Inference memory overhead & no probability output) |

*Why XGBoost was selected for production:* In datacenter SRE monitoring, **AUC-ROC (0.9595)** and **Precision (0.8114)** are the dominant metrics. High precision prevents alert fatigue by avoiding false drive replacement tickets, while depth-wise tree splits generate dense, actionable SHAP feature attributions.

---

## Dashboard Walkthrough

The SentinelDC operations console consists of 6 dedicated operational tabs:

1. **Live Telemetry:** Real-time stream monitor emitting telemetry one drive at a time (0.5s intervals) with interactive risk filtering, global failure probability progress bars, and fleet health counters.
2. **Alerts & Diagnostics:** Grounded L3 SRE incident investigation interface. Features live Groq LLaMA-3.3-70B synthesis, TreeSHAP impact tables, and dynamic 10-day sparkline charts plotted against healthy fleet baselines.
3. **Audits:** Full hardware triage history, tracking automated drive isolations, Ceph OSD evacuations, and SRE audit logs.
4. **Model Registry:** Live registry inspecting all active .joblib model artifacts, metadata, optimal PR thresholds, and ROC-AUC curves.
5. **Train Model:** Dynamic runtime training interface. Upload a 10-day raw Backblaze CSV to automatically validate, engineer features, train an XGBoost predictor, calibrate thresholds, and register the new drive family into the live registry.
6. **Model Comparisons and Others:** Complete empirical research tab displaying the cross-dataset multi-model bake-off table, 7 interactive metric comparison charts, detailed architectural rejection rationale cards, and theoretical foundations.

---

## Project Structure

`
Prediction-of-Hard-Drive-Failure/
├── app.py                      # Main SentinelDC Streamlit SRE Dashboard
├── pipeline/
│   ├── __init__.py
│   ├── pipeline.py             # Feature flattening, XGBoost training, PR threshold tuning
│   ├── router.py               # TreeSHAP attribution, model routing, severity triage
│   └── test_router.py          # Unit tests for router and scoring logic
├── models/
│   ├── ST12000NM0007.joblib     # Pre-trained Seagate 12TB enterprise model
│   ├── ST12000NM0007_metadata.json
│   ├── ST12000NM0007_demo_stream.csv
│   ├── ST4000DM000.joblib       # Pre-trained Seagate 4TB desktop model
│   ├── ST4000DM000_metadata.json
│   ├── ST4000DM000_demo_stream.csv
│   ├── TOSHIBA_MQ01ABF050.joblib # Pre-trained Toshiba 2.5\" model
│   ├── TOSHIBA_MQ01ABF050_metadata.json
│   ├── TOSHIBA MQ01ABF050_demo_stream.csv
│   └── research_metrics.json    # Complete multi-dataset benchmark results
├── research/
│   ├── run_all_datasets.py     # Cross-dataset 5-model bake-off benchmark script
│   ├── run_extra_anomaly.py    # LOF, Elliptic Envelope, Autoencoder evaluations
│   └── run_ocsvm.py            # One-Class SVM anomaly detection benchmark
├── dataset/                    # Backblaze raw & normalized 10-day sequence CSVs
├── .env.example                # Template for Groq API key configuration
├── .gitignore
├── requirements.txt
└── README.md
`

---

## Quickstart & Installation

### 1. Clone the Repository
`ash
git clone https://github.com/Yeshwanthsgowda/agnostic-failure-prediction-in-disks-in-datacenters.git
cd agnostic-failure-prediction-in-disks-in-datacenters
`

### 2. Set Up Environment
`ash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
`

### 3. Configure Environment Variables
Create a .env file in the root directory (or copy from .env.example):
`ash
cp .env.example .env
`
Add your Groq API key for LLM Root Cause Analysis:
`ini
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
`
*(Note: If no key is provided, the dashboard automatically operates in deterministic Tier 1 offline fallback mode without crashing).*

### 4. Launch the Dashboard
`ash
streamlit run app.py
`
Open your browser and navigate to http://localhost:8501.

---

## Research & Academic Attribution

This repository accompanies the research paper:
> **Coupled Selective State-Space Reconstruction and Deterministic-to-Generative Root Cause Attribution for Hyperscale Datacenter Telemetry Monitoring**  
> *Sreedevi N, Yeshwanth S Gowda, Veeraj D, Sanjay K C*  
> Dept. of Computer Science and Engineering, CMR Institute of Technology, Bengaluru, India.

---

## License

Distributed under the MIT License. See LICENSE for details.
