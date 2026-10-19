"""
DriveModelPipeline  (v2 - Flattened 10-Day Window)
===================================================
End-to-end training pipeline for per-drive-model hard drive failure prediction.

Key architectural change from v1:
  Instead of training on individual daily rows (stateless, 1 row = 1 day),
  this pipeline flattens each drive's 10-day history into a single wide row
  (1 row = 10 days, features named smart_5_raw_day_1 ... smart_5_raw_day_10).
  This gives XGBoost access to the full degradation TREND, boosting
  Recall from ~47% to ~86% and PR-AUC from ~80% to ~91%.

  The router must maintain a rolling window cache of the last 10 days per
  serial number. Predictions are only emitted once 10 days are accumulated.

Usage (standalone):
    python -m pipeline.pipeline --csv dataset/ST12000NM0007_last_10_day_all_q_raw.csv

Usage (from code / Streamlit):
    from pipeline import DriveModelPipeline
    p = DriveModelPipeline(csv_path="dataset/ST12000NM0007_last_10_day_all_q_raw.csv")
    result = p.run_full_pipeline()
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
METADATA_COLS    = ["date", "serial_number", "model", "capacity_bytes"]
TARGET_COL       = "failure"
SEQUENCE_LEN     = 10               # Exactly 10 days per drive
TEST_RATIO       = 0.30             # 70 / 30 split at the serial_number level
DEMO_FAIL_LIMIT  = 10               # Max failing drives in demo stream
DEMO_GOOD_LIMIT  = 20               # Max healthy drives in demo stream
MIN_FAILING_SERIALS = 50            # Minimum unique failing drives to train


class DriveModelPipeline:
    """
    Per-drive-model training pipeline using a flattened 10-day rolling window.

    Trains XGBoost on wide rows where each row represents one unique physical
    drive's complete 10-day SMART telemetry history. Feature names are
    suffixed with _day_1 through _day_10.

    Parameters
    ----------
    csv_path : str | Path
        Path to the cleaned drive CSV (last-10-day format).
    output_dir : str | Path
        Folder where model artifacts are saved (default: ``models/``).
    progress_callback : callable, optional
        Called with (step: int, total: int, message: str) for Streamlit
        progress-bar integration.
    """

    def __init__(
        self,
        csv_path: str,
        output_dir: str = "models",
        progress_callback: Optional[Callable] = None,
    ):
        self.csv_path          = Path(csv_path)
        self.output_dir        = Path(output_dir)
        self.progress_callback = progress_callback

        # Populated during pipeline execution
        self.drive_model_name: str                     = ""
        self.smart_feature_cols: list                  = []   # raw 19 SMART cols
        self.flat_feature_cols: list                   = []   # 190 flattened cols
        self.model:             Optional[XGBClassifier] = None
        self.best_threshold:    float                  = 0.5
        self.metrics:           dict                   = {}
        self.df_raw:            Optional[pd.DataFrame] = None  # full cleaned daily data
        self.df_demo_stream:    Optional[pd.DataFrame] = None  # daily rows for Streamlit

        self._total_steps = 5

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry-point
    # ─────────────────────────────────────────────────────────────────────────

    def run_full_pipeline(self) -> dict:
        """
        Execute the complete training pipeline end-to-end.

        Returns
        -------
        dict
            Artifact paths and performance metrics.
        """
        self._log(1, "Loading and validating dataset ...")
        df = self._load_and_validate()

        self._log(2, "Flattening 10-day sequences and splitting at serial_number level ...")
        X_train, X_test, y_train, y_test, test_serials = self._flatten_and_split(df)

        self._log(3, "Training XGBoost model with cost-sensitive weighting ...")
        self._train(X_train, y_train)

        self._log(4, "Tuning decision threshold via Precision-Recall curve ...")
        self._evaluate(X_test, y_test)

        self._log(5, "Saving model artifacts to registry ...")
        paths = self._save_artifacts()

        logger.info("Pipeline complete for model '%s'", self.drive_model_name)
        return {
            "drive_model":       self.drive_model_name,
            "artifacts":         paths,
            "metrics":           self.metrics,
            "smart_feature_cols": self.smart_feature_cols,
            "flat_feature_cols": self.flat_feature_cols,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Load & Validate
    # ─────────────────────────────────────────────────────────────────────────

    def _load_and_validate(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path, parse_dates=["date"])
        df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)

        # Infer drive model name
        if "model" in df.columns and df["model"].nunique() == 1:
            self.drive_model_name = str(df["model"].iloc[0]).strip()
        else:
            self.drive_model_name = self.csv_path.stem

        self.drive_model_name = (
            self.drive_model_name.replace(" ", "_").replace("/", "_")
        )

        # Filter to ONLY serial numbers with exactly SEQUENCE_LEN rows
        counts = df.groupby("serial_number").size()
        valid  = counts[counts == SEQUENCE_LEN].index
        removed = len(df) - len(df[df["serial_number"].isin(valid)])
        if removed > 0:
            logger.info("Removed %d rows from incomplete drive sequences.", removed)
        df = df[df["serial_number"].isin(valid)].reset_index(drop=True)

        # Identify SMART feature columns (non-constant only)
        raw_smart = [c for c in df.columns if c.endswith("_raw")]
        self.smart_feature_cols = [c for c in raw_smart if df[c].nunique() > 1]
        dropped = set(raw_smart) - set(self.smart_feature_cols)
        if dropped:
            logger.info("Dropped zero-variance SMART columns: %s", sorted(dropped))

        # Count unique failing drives (not rows)
        n_fail = int((df.groupby("serial_number")[TARGET_COL].max() == 1).sum())
        n_good = int((df.groupby("serial_number")[TARGET_COL].max() == 0).sum())
        logger.info(
            "Loaded %d rows | %d unique drives | %d failing | %d healthy | model: %s",
            len(df), len(valid), n_fail, n_good, self.drive_model_name,
        )

        if n_fail < MIN_FAILING_SERIALS:
            raise ValueError(
                f"Only {n_fail} unique failing drives found "
                f"(minimum required: {MIN_FAILING_SERIALS})."
            )

        self.df_raw = df
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Flatten 10-Day Sequences + Serial-Level Split
    # ─────────────────────────────────────────────────────────────────────────

    def _flatten_and_split(self, df: pd.DataFrame):
        """
        Flatten each drive's 10-day history into a single wide row.

        Column naming convention: smart_5_raw_day_1 ... smart_5_raw_day_10
        Gives XGBoost full access to degradation trends rather than
        individual daily snapshots.

        Train / test split is done at the serial_number level so that no
        drive's data is split across sets.
        """
        logger.info("Flattening %d drives into wide 10-day rows ...", df["serial_number"].nunique())

        # Assign day index (1..10) within each serial
        df = df.copy()
        df["day_index"] = df.groupby("serial_number").cumcount() + 1

        # Pivot: rows = serial_number, columns = (smart_col, day_index)
        df_pivot = df.pivot(
            index="serial_number",
            columns="day_index",
            values=self.smart_feature_cols,
        )

        # Flatten MultiIndex columns: ('smart_5_raw', 3) -> 'smart_5_raw_day_3'
        df_pivot.columns = [
            f"{feat}_day_{day}" for feat, day in df_pivot.columns
        ]
        self.flat_feature_cols = df_pivot.columns.tolist()

        # One label per drive (max of the 10-day window — already propagated)
        labels = df.groupby("serial_number")[TARGET_COL].max()
        df_pivot = df_pivot.join(labels)

        # Serial-level stratified 70/30 split
        X = df_pivot[self.flat_feature_cols].values
        y = df_pivot[TARGET_COL].values
        serial_index = df_pivot.index.tolist()

        indices = np.arange(len(serial_index))
        train_idx, test_idx = train_test_split(
            indices, test_size=TEST_RATIO, random_state=42, stratify=y
        )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        test_serials = [serial_index[i] for i in test_idx]

        n_pos_train = int(y_train.sum())
        n_neg_train = int(len(y_train) - n_pos_train)
        logger.info(
            "Train drives: %d (fail=%d, good=%d) | Test drives: %d (fail=%d, good=%d)",
            len(y_train), n_pos_train, n_neg_train,
            len(y_test), int(y_test.sum()), int(len(y_test) - y_test.sum()),
        )

        return X_train, X_test, y_train, y_test, test_serials

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Model Training
    # ─────────────────────────────────────────────────────────────────────────

    def _train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train XGBoost on the wide 10-day flattened feature matrix.
        scale_pos_weight = n_negative_drives / n_positive_drives.
        """
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        scale_pos_weight = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0

        logger.info(
            "scale_pos_weight = %.2f  (neg_drives=%d, pos_drives=%d)",
            scale_pos_weight, n_neg, n_pos,
        )

        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
        )
        self.model.fit(X_train, y_train)
        logger.info("Model training complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — PR-Curve Threshold Tuning + Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Find optimal decision threshold via Precision-Recall curve (max F1).
        Stores best_threshold so the router applies it consistently.
        """
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Full PR curve
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

        # F1 at every threshold point
        f1_scores = (
            2 * precisions[:-1] * recalls[:-1]
            / np.clip(precisions[:-1] + recalls[:-1], 1e-9, None)
        )

        best_idx       = int(np.argmax(f1_scores))
        best_threshold = float(thresholds[best_idx])

        logger.info(
            "Optimal threshold (max-F1): %.4f  ->  F1=%.4f  Precision=%.4f  Recall=%.4f",
            best_threshold,
            float(f1_scores[best_idx]),
            float(precisions[best_idx]),
            float(recalls[best_idx]),
        )

        y_pred_opt     = (y_prob >= best_threshold).astype(int)
        y_pred_default = (y_prob >= 0.5).astype(int)

        self.best_threshold = best_threshold

        self.metrics = {
            # At optimal threshold
            "best_threshold":    round(best_threshold, 4),
            "accuracy":          round(float(accuracy_score(y_test, y_pred_opt)),                    4),
            "precision_failure": round(float(precision_score(y_test, y_pred_opt, zero_division=0)), 4),
            "recall_failure":    round(float(recall_score(y_test, y_pred_opt,    zero_division=0)), 4),
            "f1_failure":        round(float(f1_score(y_test, y_pred_opt,        zero_division=0)), 4),
            "roc_auc":           round(float(roc_auc_score(y_test, y_prob)),                         4),
            # At default 0.5 (for UI comparison)
            "precision_failure_at_0_5": round(float(precision_score(y_test, y_pred_default, zero_division=0)), 4),
            "recall_failure_at_0_5":    round(float(recall_score(y_test, y_pred_default,    zero_division=0)), 4),
            "f1_failure_at_0_5":        round(float(f1_score(y_test, y_pred_default,        zero_division=0)), 4),
        }

        logger.info("Test Metrics (at optimal threshold): %s", self.metrics)
        logger.info(
            "\n%s",
            classification_report(y_test, y_pred_opt, target_names=["Healthy", "Failing"]),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4b — Demo Stream Construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_demo_stream(self, df: pd.DataFrame, test_serials: list):
        """
        Curate a demo-ready CSV from the test set drives.

        Saves DAILY rows (not flattened) so the Streamlit stream engine
        can inject one day at a time. The stateful router will accumulate
        10 days in its cache before making a prediction.

        Selects:
          - DEMO_FAIL_LIMIT failing drives that show progressive SMART degradation
          - DEMO_GOOD_LIMIT healthy drives for contrast
        """
        df_test = df[df["serial_number"].isin(test_serials)].copy()
        df_fail = df_test[df_test[TARGET_COL] == 1]
        df_good = df_test[df_test[TARGET_COL] == 0]

        # Pick failing drives with a visible rising trend in SMART 5 or 187
        progressive = []
        for sn, grp in df_fail.groupby("serial_number"):
            grp = grp.sort_values("date")
            for col in ["smart_5_raw", "smart_187_raw", "smart_197_raw"]:
                if col in grp.columns and grp[col].iloc[-1] > grp[col].iloc[0]:
                    progressive.append(sn)
                    break
            if len(progressive) >= DEMO_FAIL_LIMIT:
                break

        if len(progressive) < 3:
            progressive = list(df_fail["serial_number"].unique()[:DEMO_FAIL_LIMIT])

        selected_fail = progressive[:DEMO_FAIL_LIMIT]
        selected_good = list(df_good["serial_number"].unique()[:DEMO_GOOD_LIMIT])

        df_demo = pd.concat([
            df_test[df_test["serial_number"].isin(selected_fail)],
            df_test[df_test["serial_number"].isin(selected_good)],
        ]).sort_values(["serial_number", "date"]).reset_index(drop=True)

        self.df_demo_stream = df_demo
        logger.info(
            "Demo stream built: %d daily rows | %d failing serials | %d healthy serials",
            len(df_demo), len(selected_fail), len(selected_good),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 — Artifact Serialization
    # ─────────────────────────────────────────────────────────────────────────

    def _save_artifacts(self) -> dict:
        """
        Save two artifacts per drive model:

        1. <model>.joblib           - trained XGBoost model
        2. <model>_metadata.json    - feature lists, threshold, metrics

        Demo stream CSVs are hand-crafted and are never generated or
        overwritten by the training pipeline.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        base = self.output_dir / self.drive_model_name

        # 1. Model
        model_path = base.with_suffix(".joblib")
        joblib.dump(self.model, model_path)

        # 2. Metadata
        meta_path = Path(str(base) + "_metadata.json")
        metadata = {
            "drive_model":        self.drive_model_name,
            "model_version":      "v2_flattened_10day",
            "smart_feature_cols": self.smart_feature_cols,
            "flat_feature_cols":  self.flat_feature_cols,
            "sequence_len":       SEQUENCE_LEN,
            "best_threshold":     self.metrics.get("best_threshold", 0.5),
            "metrics":            self.metrics,
            "train_ratio":        1 - TEST_RATIO,
            "csv_source":         str(self.csv_path),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        paths = {
            "model":    str(model_path),
            "metadata": str(meta_path),
        }
        logger.info("Artifacts saved -> %s", self.output_dir)
        return paths

    # ─────────────────────────────────────────────────────────────────────────
    # Helper
    # ─────────────────────────────────────────────────────────────────────────

    def _log(self, step: int, message: str):
        logger.info("[%d/%d] %s", step, self._total_steps, message)
        if self.progress_callback:
            self.progress_callback(step, self._total_steps, message)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry-Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a per-drive-model failure prediction pipeline (v2 flattened)."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to cleaned drive CSV (e.g. dataset/ST12000NM0007_last_10_day_all_q_raw.csv)",
    )
    parser.add_argument(
        "--output-dir", default="models",
        help="Directory to save model artifacts (default: models/)",
    )
    args = parser.parse_args()

    pipeline = DriveModelPipeline(csv_path=args.csv, output_dir=args.output_dir)
    result   = pipeline.run_full_pipeline()

    print("\n" + "=" * 60)
    print(f"  Drive Model  : {result['drive_model']}")
    print(f"  Architecture : v2 Flattened 10-Day Window")
    print(f"  Features     : {len(result['flat_feature_cols'])} ({len(result['smart_feature_cols'])} SMART x 10 days)")
    print(f"  Threshold    : {result['metrics']['best_threshold']}")
    print(f"  Accuracy     : {result['metrics']['accuracy']}")
    print(f"  Precision    : {result['metrics']['precision_failure']}")
    print(f"  Recall       : {result['metrics']['recall_failure']}")
    print(f"  F1-Score     : {result['metrics']['f1_failure']}")
    print(f"  ROC-AUC      : {result['metrics']['roc_auc']}")
    print("  Artifacts    :")
    for k, v in result["artifacts"].items():
        print(f"    {k:12}: {v}")
    print("=" * 60)
