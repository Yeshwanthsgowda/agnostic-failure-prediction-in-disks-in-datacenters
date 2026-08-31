# SentinelDC: Comprehensive Technical System Summary & Architecture Specification

---

## 1. Project Overview & Problem Statement
**SentinelDC** is an end-to-end predictive maintenance, explainability, and automated Site Reliability Engineering (SRE) root cause analysis framework for hyperscale datacenter storage infrastructure.

* **The Problem:** Storage hard disk drive (HDD) failures represent the largest physical maintenance ticket volume in cloud infrastructure. In traditional Network Operations Centers (NOCs), static threshold alerts (e.g., alert if SMART 5 > 0) generate >90% false alarms on transient spikes. Unscheduled crashes force emergency RAID/Ceph rebuilds, causing massive network replication storms and data durability risks.
* **The Solution:** SentinelDC couples **multi-day temporal gradient boosting** with **game-theoretic TreeSHAP feature attribution** and a novel **Two-Tier Deterministic-to-Generative (D2G) Root Cause Analysis Engine**, delivering sub-millisecond failure forecasting (24–72 hour lead time) and grounded, hallucination-free SRE incident remediation briefs.

---

## 2. Dataset, Scale & Class Imbalance Handling
* **Data Source:** Production telemetry from **Backblaze Hard Drive Stats** (open datacenter storage repository).
* **Evaluated Scale:** **348,810 drive-days** across **6 distinct enterprise drive models**:
  1. `ST12000NM0007` (Seagate Enterprise 12TB, 38,202 drives, 2.94% AFR)
  2. `ST4000DM000` (Seagate Desktop 4TB, 23,121 drives, 1.72% AFR)
  3. `ST8000NM0055` (Seagate Enterprise 8TB, 14,544 drives, 1.50% AFR)
  4. `ST8000DM002` (Seagate Desktop 8TB, 9,864 drives, 1.20% AFR)
  5. `HGST HMS5C4040BLE640` (HGST Enterprise 4TB, 13,944 drives, 0.39% AFR)
  6. `TOSHIBA MQ01ABF050` (Toshiba 2.5" High-Stress, 507 drives, 14.79% AFR)
* **Extreme Class Imbalance Mitigation:**
  * Real-world Annualized Failure Rates are typically < 3% (100:1 healthy-to-failed ratio).
  * **Cost-Sensitive Loss:** Loss functions are weighted with positive-class scaling: `scale_pos_weight = N_healthy / N_failing`.
  * **Serial-Level Stratified Splitting:** 70/30 train/test split is enforced strictly at the **drive serial number level** (not row level), preventing temporal data leakage between days of the same drive.
  * **Precision-Recall (PR) Curve Optimization:** Optimal decision thresholds (tau*) are calibrated dynamically at training time by maximizing F1 along the PR curve rather than defaulting to a naive 0.5 cutoff.

---

## 3. Temporal Feature Engineering (10-Day Sliding Lookback)
* **Lookback Window Formulation (W = 10):** Telemetry is grouped per drive serial into 10 consecutive daily snapshots.
* **Temporal Flattening:** Pivots the 2D temporal sequence into a 1D vector x in R^(M x 10):
  `x = [ s_{1,day1}, ..., s_{M,day1}, s_{1,day2}, ..., s_{M,day10} ]`
* **Engineering Advantage:**
  * Captures physical degradation **velocity (Delta)** and **acceleration (d^2/dt^2)** (e.g., pending sectors jumping from 0 to 8 in 48 hours vs. a stable count of 8).
  * Operates in **linear time O(N)** with sub-0.003ms inference latency, completely bypassing the quadratic compute and memory complexity O(N^2) of sequence Transformers.
* **Vendor-Isolated Dynamic Model Routing:** Different manufacturers use disparate SMART register conventions (e.g., Seagate composite raw bitmasks in SMART 1/7 vs. Toshiba/HGST monotonic event counters). Segregated model routing prevents cross-fleet feature distortion.

---

## 4. Explainability & Local Attribution Layer (TreeSHAP)
* **Exact Game-Theoretic Attribution:** Calculates exact Shapley additive explanations synchronously alongside inference in O(T * L * D^2):
  `sum_{m=1}^{M x 10} phi_m(x) = f(x) - E[f(x)]`
* **Attribution Output:** Extracts the top-5 SMART attributes driving the failure probability, calculating:
  * Current raw sensor value
  * Fleet-wide healthy baseline median
  * Velocity / trend arrow (▲)
  * Percentage impact on total failure attribution
* **Dynamic Sparklines:** Plotted in real-time on the dashboard against fleet-wide healthy medians to visualize physical degradation kinetics over the 10-day lookback window.

---

## 5. Novelty: Two-Tier Deterministic-to-Generative (D2G) Root Cause Analysis
Directly prompting raw telemetry into generative LLMs causes severe hallucinations (invented metrics, fake hostnames) and 2–4s latency penalties. SentinelDC decouples diagnosis into two specialized tiers:

```
[ Ingested SMART Telemetry ] ---> [ TreeSHAP Attribution Vector ]
                                                |
                                                v
+-----------------------------------------------------------------------------------------------+
| TIER 1: Deterministic Expert Rule Engine (Synchronous, 100% Offline, Sub-Millisecond O(1))     |
| • Evaluates TreeSHAP conditions against verified physical failure modes:                      |
|   - MAGNETIC_PLATTER_DEGRADATION: phi_{SMART5} > eps AND Raw(SMART5) > 0                      |
|   - HEAD_PREAMP_INSTABILITY: phi_{SMART187} > eps (Uncorrectable read errors)                 |
|   - THERMAL_RUNAWAY_STRESS: phi_{SMART194} > eps AND Raw(SMART194) > 55°C                    |
| • Computes Storage Blast Radius: Affected ZFS Pools, Ceph OSD IDs, VM mount points            |
| • Binds SOP Runbooks: Pre-approved commands (`ceph osd out`, `zfs set offline`)               |
| • Emits strictly typed, immutable JSON Schema Payload                                         |
+-----------------------------------------------------------------------------------------------+
                         |                                               |
           [Direct Alert / API Timeout]                                  | [Grounded JSON Payload]
                         |                                               v
                         |       +--------------------------------------------------------------+
                         |       | TIER 2: Grounded LLM Synthesizer (Asynchronous)              |
                         |       | • LLaMA-3.3-70B via Groq API                                 |
                         |       | • Strictly bounded by Tier 1 JSON schema (Zero Hallucination)|
                         |       | • Generates 4-Section L3 SRE Executive Incident Brief        |
                         |       | • Fail-Safe SLA: Falls back to Tier 1 if API timeout > 400ms |
                         |       +--------------------------------------------------------------+
                         v                                               v
+-----------------------------------------------------------------------------------------------+
|                             On-Call SRE Dashboard & Alerting System                           |
+-----------------------------------------------------------------------------------------------+
```

---

## 6. Empirical Benchmarking & Model Selection Bake-Off
We evaluated **5 algorithmic paradigms** across all 6 Backblaze datasets (348,810 drive-days):

| Candidate Model | Category | Avg F1 | Avg AUC-ROC | Avg Precision | Avg Recall | Train Time | Latency | Selection Verdict & Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | **Supervised Ensemble** | **0.7680** | **0.9595** | **0.8114** | **0.7359** | 1.48s | **0.0023ms** | **[PRODUCTION] Highest AUC & Precision. Dense, actionable TreeSHAP attributions.** |
| **LightGBM** | Supervised Ensemble | 0.7743 | 0.9480 | 0.7939 | 0.7582 | 0.58s | 0.0039ms | *Rejected.* AUC deficit (-0.0115) and leaf-wise SHAP sparsity on flattened sequences. |
| **Random Forest** | Supervised Ensemble | 0.7251 | 0.9520 | 0.7549 | 0.7433 | 1.53s | 0.0375ms | *Rejected.* Precision deficit (7.65% lower than XGBoost), causing unacceptable false alarms. |
| **Elliptic Envelope** | Unsupervised (Mahalanobis on PCA-30) | 0.3511 | 0.8932 | 0.2662 | 0.5347 | 3.54s | 0.0009ms | *Rejected.* 2.2x lower F1. Multivariate Gaussian assumption fails on zero-inflated SMART data. |
| **Local Outlier Factor (LOF)** | Unsupervised (Density k-NN) | 0.3374 | 0.8734 | 0.2561 | 0.5147 | 0.51s | 0.0371ms | *Rejected.* High RAM overhead at inference, non-probabilistic binary output, low precision. |

---

## 7. Interactive SRE Operations Dashboard (`app.py`)
Built in **Streamlit** (zero-database dependency, filesystem-backed persistence):
1. **Live Telemetry:** Stateful real-time stream simulation (0.5s playback cadence per drive) with sliding window deques, global failure probability progress bars, and fleet health KPIs.
2. **Alerts & Diagnostics:** Drive Inspector with dynamic SHAP sparklines vs. healthy medians, top risk factors table, Tier 1 rule diagnostics, and live Groq LLaMA-3.3-70B incident briefs.
3. **Audits:** Hardware quarantine log tracking automated drive isolations and Ceph OSD evacuations.
4. **Model Registry:** Live artifact inspector displaying model metadata, calibrated PR thresholds, and ROC-AUC curves.
5. **Train Model:** Dynamic runtime training interface—upload any 10-day raw Backblaze CSV to automatically engineer features, train an XGBoost model, optimize threshold, and hot-load it into the live registry.
6. **Model Comparisons and Others:** Complete research tab displaying the multi-model bake-off table, 7 interactive metric charts, rejection rationale cards, and theoretical foundations.

---

## 8. Lifecycle Stage Artifacts (`stages/`)
The `stages/` folder contains clean, self-contained artifacts demonstrating the transformation of a real failing drive (`S300TS9Z` — Seagate 4TB `ST4000DM000`) at every lifecycle stage:
* `01_raw_telemetry_input.csv`: Raw 10-day sequence table (10 rows x 26 SMART columns).
* `02_flattened_features_vector.csv`: 1D temporal feature vector (1 row x 210 flattened columns).
* `03_model_metadata_and_threshold.json`: Model hyperparameters, metrics (ROC-AUC: 0.9690, F1: 0.8182), and calibrated threshold (tau* = 0.6551).
* `04_inference_prediction_output.json`: Supervised inference output (Probability: 100.0%, Risk: CRITICAL).
* `05_treeshap_attribution_table.csv` & `.json`: Exact TreeSHAP factor attributions and impact percentages.
* `06_tier1_deterministic_rca.json`: Tier 1 failure mode (`HEAD_PREAMP_INSTABILITY`), Ceph OSD `osd.572` blast radius, and SOP runbook.
* `06_tier2_grounded_llm_brief.md`: Tier 2 natural language incident brief synthesized by LLaMA-3.3-70B via Groq.
* `generate_stage_outputs.py`: Reproducible script to re-generate lifecycle outputs for any drive serial.

---

## 9. Academic & Technical Summary
* **Codebase Language:** Python 3.10+
* **Primary Frameworks:** XGBoost, Scikit-Learn, SHAP, Streamlit, Groq Python SDK, Pandas, NumPy, Plotly.
* **Key Strengths:** High enterprise domain relevance (AIOps), rigorous temporal feature engineering, strict class imbalance mitigation, exact game-theoretic explainability, and a novel hallucination-free Two-Tier D2G architecture with sub-400ms fail-safe SLA.
