"""
DriveTelemetryRouter  (v2 - Stateful 10-Day Rolling Window Cache)
=================================================================
Dynamic model registry and stateful inference engine for the
Hard Drive Failure Prediction system.

Key architectural change from v1:
  The v1 router was stateless (1 log -> 1 prediction).
  This v2 router maintains a per-serial-number rolling window cache.
  When a new daily log arrives:
    - Days 1-9 : cached, status = "accumulating"
    - Day 10+  : 10-day window flattened and passed to XGBoost,
                 then the window slides forward (drop oldest, keep latest 10)

  This matches the training data format: the model was trained on
  wide rows (smart_5_raw_day_1 ... smart_5_raw_day_10), so inference
  must supply exactly the same shape.

Responsibilities:
  1. Scan models/ directory and load all v2 model artifacts.
  2. Maintain a per-serial rolling deque cache of daily logs.
  3. On Day 10+ emit a calibrated failure probability and risk level.
  4. Generate SHAP-based explainability grouped by SMART attribute.
  5. Produce a deterministic RCA report for critical logs.

Usage:
    from pipeline.router import DriveTelemetryRouter

    router = DriveTelemetryRouter(models_dir="models")

    # Feed logs one day at a time
    result = router.ingest_log({
        "serial_number": "ZCH01CPY",
        "model":         "ST12000NM0007",
        "smart_5_raw":   4809,
        ...
    })

    # Returns {"status": "accumulating", "days_buffered": 3}  for days 1-9
    # Returns full prediction dict on day 10+
"""

import json
import logging
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import shap

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SMART Attribute Domain Descriptions
# ─────────────────────────────────────────────────────────────────────────────
SMART_DESCRIPTIONS = {
    "smart_1_raw":   ("Raw Read Error Rate",           "Read head / platter surface errors"),
    "smart_3_raw":   ("Spin-Up Time",                  "Motor spin-up latency increase"),
    "smart_4_raw":   ("Start/Stop Count",              "Power cycle wear indicator"),
    "smart_5_raw":   ("Reallocated Sectors Count",     "Bad sectors remapped — primary failure signal"),
    "smart_7_raw":   ("Seek Error Rate",               "Read/write head positioning errors"),
    "smart_9_raw":   ("Power-On Hours",                "Total operational lifetime"),
    "smart_12_raw":  ("Power Cycle Count",             "Total power cycles — wear indicator"),
    "smart_183_raw": ("SATA Downshifts",               "Interface speed downgrade events"),
    "smart_184_raw": ("End-to-End Error",              "Data path integrity errors"),
    "smart_187_raw": ("Reported Uncorrectable Errors", "Errors ECC could not correct — critical signal"),
    "smart_188_raw": ("Command Timeout",               "Commands that timed out — firmware stress"),
    "smart_189_raw": ("High Fly Writes",               "Off-track write events"),
    "smart_190_raw": ("Airflow Temperature",           "Internal drive temperature deviation"),
    "smart_191_raw": ("G-Sense Error Rate",            "Shock/vibration impact events"),
    "smart_192_raw": ("Power-Off Retract Count",       "Emergency head retract events"),
    "smart_193_raw": ("Load/Unload Cycle Count",       "Head load/unload wear"),
    "smart_194_raw": ("Drive Temperature",             "Current operating temperature"),
    "smart_195_raw": ("Hardware ECC Recovered",        "Corrected read errors — ECC activity"),
    "smart_196_raw": ("Reallocation Event Count",      "Total reallocation attempts"),
    "smart_197_raw": ("Current Pending Sector Count",  "Unstable sectors awaiting remap — critical"),
    "smart_198_raw": ("Offline Uncorrectable Sectors", "Sectors unreadable in offline scan — critical"),
    "smart_199_raw": ("UDMA CRC Error Count",          "Interface / cable data corruption events"),
    "smart_222_raw": ("Loaded Hours",                  "Head loaded operational hours"),
    "smart_226_raw": ("Load-In Time",                  "Head load time measurement"),
    "smart_240_raw": ("Head Flying Hours",             "Head in-flight operational hours"),
    "smart_241_raw": ("Total LBAs Written",            "Total data written lifetime"),
    "smart_242_raw": ("Total LBAs Read",               "Total data read lifetime"),
}


class ModelNotFoundError(Exception):
    """Raised when no registered model matches the incoming drive model name."""


class DriveTelemetryRouter:
    """
    Stateful multi-model registry and inference engine.

    Maintains a per-serial rolling deque cache of daily logs.
    Predictions are only emitted once 10 days are accumulated.
    After that, the window slides forward with each new log.

    Parameters
    ----------
    models_dir : str
        Path to the directory containing trained model artifacts.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)

        # registry[drive_model] = {model, smart_feature_cols, flat_feature_cols,
        #                           threshold, sequence_len, metrics, demo_csv}
        self.registry:   dict = {}

        # explainers[drive_model] = shap.TreeExplainer (lazily created)
        self.explainers: dict = {}

        # cache[serial_number] = deque(maxlen=sequence_len) of daily log dicts
        self.cache: dict = {}

        self.reload_registry()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Registry Management
    # ─────────────────────────────────────────────────────────────────────────

    def reload_registry(self) -> None:
        """
        Scan models_dir for *_metadata.json + *.joblib pairs and load
        them into the in-memory registry.  Safe to call at any time.

        Always clears the in-memory registry first so that any model
        whose .joblib / _metadata.json was deleted is immediately evicted
        from RAM and will not continue producing predictions as a ghost.
        """
        if not self.models_dir.exists():
            logger.warning("models_dir '%s' does not exist yet.", self.models_dir)
            return

        # Evict all current entries before re-scanning
        self.registry.clear()
        self.explainers.clear()

        found = 0
        for meta_path in sorted(self.models_dir.glob("*_metadata.json")):
            drive_name = meta_path.stem.replace("_metadata", "")
            model_path = self.models_dir / f"{drive_name}.joblib"

            if not model_path.exists():
                logger.warning("Metadata found but no model file for '%s' — skipping.", drive_name)
                continue

            try:
                with open(meta_path) as f:
                    meta = json.load(f)

                model               = joblib.load(model_path)
                smart_feature_cols  = meta["smart_feature_cols"]   # e.g. ["smart_5_raw", ...]
                flat_feature_cols   = meta["flat_feature_cols"]     # e.g. ["smart_5_raw_day_1", ...]
                sequence_len        = meta.get("sequence_len", 10)
                threshold           = meta.get("best_threshold", 0.5)
                metrics             = meta.get("metrics", {})
                demo_csv            = self.models_dir / f"{drive_name}_demo_stream.csv"

                self.registry[drive_name] = {
                    "model":              model,
                    "smart_feature_cols": smart_feature_cols,
                    "flat_feature_cols":  flat_feature_cols,
                    "sequence_len":       sequence_len,
                    "threshold":          threshold,
                    "metrics":            metrics,
                    "model_path":         str(model_path),
                    "demo_csv":           str(demo_csv) if demo_csv.exists() else None,
                }

                # Invalidate cached SHAP explainer
                self.explainers.pop(drive_name, None)

                logger.info(
                    "Registered model: %-30s | features=%d (%d SMART x %d days) | threshold=%.4f",
                    drive_name, len(flat_feature_cols),
                    len(smart_feature_cols), sequence_len, threshold,
                )
                found += 1

            except Exception as exc:
                logger.error("Failed to load model '%s': %s", drive_name, exc)

        logger.info("Registry loaded: %d model(s) available.", found)

    def list_models(self) -> list:
        return sorted(self.registry.keys())

    def is_registered(self, drive_model: str) -> bool:
        return self._normalise_name(drive_model) in self.registry

    def get_model_info(self, drive_model: str) -> dict:
        entry = self._get_entry(drive_model)
        return {
            "drive_model":       drive_model,
            "threshold":         entry["threshold"],
            "metrics":           entry["metrics"],
            "smart_feature_cols": entry["smart_feature_cols"],
            "flat_feature_cols": entry["flat_feature_cols"],
            "sequence_len":      entry["sequence_len"],
            "demo_csv":          entry["demo_csv"],
        }

    def clear_cache(self, serial_number: str = None) -> None:
        """Clear rolling window cache for a specific serial or all serials."""
        if serial_number:
            self.cache.pop(serial_number, None)
        else:
            self.cache.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 + 3 — Stateful Ingestion & Inference
    # ─────────────────────────────────────────────────────────────────────────

    def ingest_log(self, log_data: dict) -> dict:
        """
        Ingest a single daily telemetry log entry for a specific drive.

        Adds the log to the serial's rolling window cache.
        Returns a "accumulating" status for days 1-9.
        Returns a full prediction on day 10+ (sliding window).

        Parameters
        ----------
        log_data : dict
            Must contain 'model' and 'serial_number' keys plus any
            available SMART attributes.

        Returns
        -------
        dict
            On accumulation:
                {"status": "accumulating", "days_buffered": N, "serial_number": ...}
            On prediction:
                Full prediction dict (see _predict_from_window).
        """
        drive_model   = self._extract_model_name(log_data)
        serial_number = str(log_data.get("serial_number", "unknown"))
        entry         = self._get_entry(drive_model)
        seq_len       = entry["sequence_len"]

        # Initialise deque for this serial if not seen before
        if serial_number not in self.cache:
            self.cache[serial_number] = deque(maxlen=seq_len)

        # Add today's log to the cache
        self.cache[serial_number].append(log_data)
        days_buffered = len(self.cache[serial_number])

        if days_buffered < seq_len:
            return {
                "status":        "accumulating",
                "days_buffered": days_buffered,
                "days_needed":   seq_len - days_buffered,
                "serial_number": serial_number,
                "drive_model":   drive_model,
            }

        # Full window available — predict
        return self._predict_from_window(serial_number, drive_model, entry)

    def predict_single_log(self, log_data: dict) -> dict:
        """
        Convenience alias for ingest_log.
        Maintains API compatibility with the Streamlit app.
        """
        return self.ingest_log(log_data)

    def predict_batch(self, logs: list) -> list:
        """Feed a list of logs through ingest_log in order."""
        return [self.ingest_log(log) for log in logs]

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3b — Core Inference (from window cache)
    # ─────────────────────────────────────────────────────────────────────────

    def _predict_from_window(self, serial_number: str, drive_model: str, entry: dict) -> dict:
        """
        Flatten the 10-day window from cache and run XGBoost inference.
        The window slides automatically (deque maxlen handles eviction).
        """
        window       = list(self.cache[serial_number])          # 10 daily dicts
        flat_cols    = entry["flat_feature_cols"]
        smart_cols   = entry["smart_feature_cols"]
        threshold    = entry["threshold"]

        # Build flat feature row from window
        X, missing = self._build_flat_vector(window, smart_cols, flat_cols)

        prob              = float(entry["model"].predict_proba(X)[0, 1])
        predicted_failure = prob >= threshold
        risk_level        = self._classify_risk(prob, threshold)
        risk_score        = self._compute_risk_score(prob, threshold)

        return {
            "status":          "predicted",
            "drive_model":     drive_model,
            "serial_number":   serial_number,
            "probability":     round(prob, 4),
            "probability_pct": round(prob * 100, 2),
            "threshold":       round(threshold, 4),
            "prediction":      bool(predicted_failure),
            "risk_level":      risk_level,
            "risk_score":      risk_score,
            "days_in_window":  len(window),
            "missing_feats":   missing,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 + 5 — SHAP Explainability & RCA
    # ─────────────────────────────────────────────────────────────────────────

    def generate_rca(self, serial_number: str, drive_model: str, top_n: int = 5) -> dict:
        """
        Generate a Root Cause Analysis report for a drive using its cached window.

        Groups SHAP contributions by SMART attribute (summing across all 10 days)
        so the report shows "SMART 5 overall impact = +X" rather than one entry
        per day.  The day-level breakdown is also available in the raw output.

        Parameters
        ----------
        serial_number : str
        drive_model   : str
        top_n         : int   Number of top SMART attributes to highlight.

        Returns
        -------
        dict with prediction_summary, grouped_shap, top_risk_factors,
             rca_text, action_required.
        """
        if serial_number not in self.cache or len(self.cache[serial_number]) < 1:
            raise ValueError(
                f"No cached window for serial '{serial_number}'. "
                "At least 1 log must be ingested before generating RCA."
            )

        entry      = self._get_entry(drive_model)
        window     = list(self.cache[serial_number])
        flat_cols  = entry["flat_feature_cols"]
        smart_cols = entry["smart_feature_cols"]

        # Pad window to full length if < 10 days (fill missing with 0)
        padded_window = ([{s: 0.0 for s in smart_cols}] * (entry["sequence_len"] - len(window))
                         + window)
        X, _ = self._build_flat_vector(padded_window, smart_cols, flat_cols)
        X_df = pd.DataFrame(X, columns=flat_cols)

        # Build or retrieve SHAP explainer
        if drive_model not in self.explainers:
            logger.info("Building SHAP TreeExplainer for '%s' ...", drive_model)
            self.explainers[drive_model] = shap.TreeExplainer(entry["model"])

        explainer  = self.explainers[drive_model]
        shap_vals  = explainer.shap_values(X_df)   # shape (1, n_flat_features)
        shap_row   = shap_vals[0]

        # ── Day-level contributions ──
        day_contribs = []
        for feat, sv in zip(flat_cols, shap_row):
            # feat = "smart_5_raw_day_7"
            parts = feat.rsplit("_day_", 1)
            smart_col = parts[0]           # "smart_5_raw"
            day_num   = int(parts[1]) if len(parts) == 2 else 0
            val       = float(X_df[feat].iloc[0])
            name, detail = SMART_DESCRIPTIONS.get(smart_col, (smart_col, "Custom SMART attribute"))
            day_contribs.append({
                "feature":   feat,
                "smart_col": smart_col,
                "day":       day_num,
                "value":     val,
                "shap":      round(float(sv), 6),
                "name":      name,
                "detail":    detail,
            })

        # ── Group by SMART attribute (sum SHAP across all 10 days) ──
        grouped: dict = {}
        for c in day_contribs:
            sc = c["smart_col"]
            if sc not in grouped:
                grouped[sc] = {
                    "smart_col":   sc,
                    "name":        c["name"],
                    "detail":      c["detail"],
                    "total_shap":  0.0,
                    "day_values":  {},   # day_num -> raw value
                    "day_shaps":   {},   # day_num -> shap
                }
            grouped[sc]["total_shap"]        += c["shap"]
            grouped[sc]["day_values"][c["day"]] = c["value"]
            grouped[sc]["day_shaps"][c["day"]]  = c["shap"]

        # Sort by absolute grouped SHAP
        sorted_groups = sorted(
            grouped.values(), key=lambda x: abs(x["total_shap"]), reverse=True
        )

        # Top risk factors
        top_factors = []
        for rank, g in enumerate(sorted_groups[:top_n], start=1):
            last_val  = g["day_values"].get(max(g["day_values"].keys(), default=0), 0)
            trend     = "rising" if (
                len(g["day_values"]) > 1
                and list(g["day_values"].values())[-1] > list(g["day_values"].values())[0]
            ) else "stable"
            top_factors.append({
                "rank":      rank,
                "smart_id":  g["smart_col"].replace("_raw", "").replace("smart_", "SMART "),
                "name":      g["name"],
                "detail":    g["detail"],
                "total_shap": round(g["total_shap"], 4),
                "last_value": last_val,
                "trend":      trend,
                "impact":     "Increases Risk" if g["total_shap"] > 0 else "Decreases Risk",
            })

        # Prediction summary using current window
        pred_summary = self._predict_from_window(serial_number, drive_model, entry)

        rca_text, action = self._compose_rca_text(pred_summary, top_factors)

        return {
            "prediction_summary": pred_summary,
            "day_level_shap":     day_contribs,
            "grouped_shap":       list(grouped.values()),
            "top_risk_factors":   top_factors,
            "rca_text":           rca_text,
            "action_required":    action,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _normalise_name(self, name: str) -> str:
        return str(name).strip().replace(" ", "_")

    def _extract_model_name(self, log_data: dict) -> str:
        raw = log_data.get("model", log_data.get("drive_model", ""))
        if not raw:
            raise ModelNotFoundError(
                "Log entry has no 'model' field. Cannot route to a registered model."
            )
        return self._normalise_name(raw)

    def _get_entry(self, drive_model: str) -> dict:
        name = self._normalise_name(drive_model)
        if name not in self.registry:
            available = ", ".join(self.registry.keys()) or "none"
            raise ModelNotFoundError(
                f"No trained model for drive '{drive_model}'. "
                f"Available: [{available}]. Train a model first."
            )
        return self.registry[name]

    def _build_flat_vector(
        self,
        window: list,
        smart_cols: list,
        flat_cols: list,
    ):
        """
        Flatten a list of daily log dicts into the wide feature row
        expected by the v2 XGBoost model.

        window[0] = oldest day, window[-1] = most recent day.
        Column naming: smart_5_raw_day_1 corresponds to window[0].
        """
        row     = {}
        missing = []

        for day_idx, daily_log in enumerate(window, start=1):
            for feat in smart_cols:
                col_name = f"{feat}_day_{day_idx}"
                val      = daily_log.get(feat, None)
                if val is None:
                    missing.append(col_name)
                    row[col_name] = 0.0
                else:
                    row[col_name] = float(val)

        # Ensure all expected flat columns are present in the correct order
        X = np.array([[row.get(c, 0.0) for c in flat_cols]], dtype=np.float32)
        return X, missing

    def _classify_risk(self, prob: float, threshold: float) -> str:
        """
        Risk band relative to the model's optimal threshold.
        """
        ratio = prob / max(threshold, 1e-9)
        if ratio >= 0.80:
            return "critical"
        elif ratio >= 0.40:
            return "warning"
        return "healthy"

    def _compute_risk_score(self, prob: float, threshold: float) -> float:
        """
        0-100 risk score where 50 = exactly at threshold.
        Below threshold: 0-50 linearly.
        Above threshold: 50-100 linearly.
        """
        if prob < threshold:
            return round(prob / max(threshold, 1e-9) * 50, 1)
        else:
            return round(50 + (prob - threshold) / max(1 - threshold, 1e-9) * 50, 1)

    def _compose_rca_text(self, pred: dict, risk_factors: list) -> tuple:
        drive  = pred["drive_model"]
        prob   = pred["probability_pct"]
        level  = pred["risk_level"]
        days   = pred.get("days_in_window", 10)

        pos_factors = [f for f in risk_factors if f["total_shap"] > 0]

        if not pos_factors:
            rca_text = (
                f"Drive model {drive} shows a failure probability of {prob:.1f}% "
                f"based on the last {days} days. No single SMART attribute is "
                "the dominant contributor. The aggregate pattern is within normal range."
            )
            action = "Continue standard monitoring. No immediate action required."
            return rca_text, action

        primary   = pos_factors[0]
        secondary = pos_factors[1] if len(pos_factors) > 1 else None

        trend_note = (
            f" This attribute shows a {primary['trend']} trend over the observation window."
        )

        primary_line = (
            f"The primary driver is {primary['name']} ({primary['smart_id']}) "
            f"with a most-recent reading of {primary['last_value']:,.0f} and "
            f"a cumulative SHAP contribution of +{primary['total_shap']:.4f}.{trend_note}"
        )

        secondary_line = ""
        if secondary:
            secondary_line = (
                f" A secondary driver is {secondary['name']} ({secondary['smart_id']}, "
                f"contribution +{secondary['total_shap']:.4f}, trend: {secondary['trend']})."
            )

        if level == "critical":
            rca_text = (
                f"[CRITICAL] FAILURE RISK DETECTED on drive model {drive} "
                f"(failure probability: {prob:.1f}% over last {days} days). "
                + primary_line + secondary_line +
                " Immediate intervention is required."
            )
            action = (
                "IMMEDIATE ACTION: Initiate emergency data evacuation to a hot-spare "
                "drive. Schedule physical replacement within 24-48 hours. Do NOT power "
                "cycle this drive before backup is confirmed complete."
            )
        elif level == "warning":
            rca_text = (
                f"[WARNING] DEGRADATION DETECTED on drive model {drive} "
                f"(failure probability: {prob:.1f}% over last {days} days). "
                + primary_line + secondary_line +
                " The drive is exhibiting early wear patterns."
            )
            action = (
                "RECOMMENDED ACTION: Schedule a full SMART diagnostic scan within "
                "48 hours. Prepare a replacement drive and initiate incremental backup. "
                "Monitor daily for further escalation."
            )
        else:
            rca_text = (
                f"[OK] Drive model {drive} is operating within healthy parameters "
                f"(failure probability: {prob:.1f}% over last {days} days). "
                + primary_line
            )
            action = "No immediate action required. Continue standard monitoring interval."

        return rca_text, action
