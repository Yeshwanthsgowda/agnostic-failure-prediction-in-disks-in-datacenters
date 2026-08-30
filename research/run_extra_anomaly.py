import json, logging, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, precision_recall_curve)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEQUENCE_LEN = 10
TEST_RATIO   = 0.30
LOF_SUBSAMPLE = 8000
AE_EPOCHS     = 15
DATASETS = [
    "dataset/ST12000NM0007_last_10_day_all_q_raw.csv",
    "dataset/ST4000DM000_last_10_day_all_q_raw.csv",
    "dataset/ST8000DM002_last_10_day_all_q_raw.csv",
    "dataset/ST8000NM0055_last_10_day_all_q_raw.csv",
    "dataset/HGST HMS5C4040BLE640_last_10_day_all_q_raw.csv",
    "dataset/TOSHIBA MQ01ABF050_last_10_day_all_q_raw.csv",
]

def load_and_flatten(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values(["serial_number","date"]).reset_index(drop=True)
    counts = df.groupby("serial_number").size()
    valid  = counts[counts == SEQUENCE_LEN].index
    df     = df[df["serial_number"].isin(valid)].reset_index(drop=True)
    raw_smart  = [c for c in df.columns if c.endswith("_raw")]
    smart_cols = [c for c in raw_smart if df[c].nunique() > 1]
    df["day_index"] = df.groupby("serial_number").cumcount() + 1
    df_pivot = df.pivot(index="serial_number", columns="day_index", values=smart_cols)
    df_pivot.columns = [f"{feat}_day_{day}" for feat, day in df_pivot.columns]
    flat_cols = df_pivot.columns.tolist()
    labels   = df.groupby("serial_number")["failure"].max()
    df_pivot = df_pivot.join(labels)
    X = df_pivot[flat_cols].values.astype(np.float32)
    y = df_pivot["failure"].values.astype(int)
    n_fail = int(y.sum()); n_good = len(y)-n_fail
    return X, y, len(smart_cols), n_fail, n_good

def metrics_from_scores(y_te, anomaly_scores, y_pred_binary):
    s_min, s_max = anomaly_scores.min(), anomaly_scores.max()
    y_prob = (anomaly_scores - s_min) / max(s_max - s_min, 1e-9)
    m = {
        "accuracy":  round(float(accuracy_score(y_te, y_pred_binary)), 4),
        "precision": round(float(precision_score(y_te, y_pred_binary, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_te, y_pred_binary, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_te, y_pred_binary, zero_division=0)), 4),
    }
    try: m["roc_auc"] = round(float(roc_auc_score(y_te, y_prob)), 4)
    except: m["roc_auc"] = None
    try:
        p_arr, r_arr, _ = precision_recall_curve(y_te, y_prob)
        step = max(1, len(p_arr)//100)
        m["pr_curve_prec"] = [round(float(v),4) for v in p_arr[::step]]
        m["pr_curve_rec"]  = [round(float(v),4) for v in r_arr[::step]]
    except: pass
    tp = int(((y_pred_binary==1)&(y_te==1)).sum())
    fp = int(((y_pred_binary==1)&(y_te==0)).sum())
    fn = int(((y_pred_binary==0)&(y_te==1)).sum())
    tn = int(((y_pred_binary==0)&(y_te==0)).sum())
    n_fail = int(y_te.sum())
    m["confusion"] = {"TP":tp,"FP":fp,"FN":fn,"TN":tn}
    m["detection_rate"]   = round(tp/max(n_fail,1)*100, 2)
    m["false_alarm_rate"] = round(fp/max(tn+fp,1)*100, 2)
    return m, y_prob

# ── Load existing metrics ──
metrics_path = Path("models/research_metrics.json")
existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

# ── Results stores ──
lof_results = {}
ee_results  = {}
ae_results  = {}
agg = {k: {"f1_score":[],"roc_auc":[],"precision":[],"recall":[],
            "train_time_sec":[],"infer_latency_ms":[]}
       for k in ["LOF","EllipticEnvelope","Autoencoder"]}

for csv_path in DATASETS:
    p = Path(csv_path)
    if not p.exists(): logger.warning("MISSING: %s", csv_path); continue
    drive = p.stem.replace("_last_10_day_all_q_raw","").replace("_"," ").strip()
    logger.info("==============================")
    logger.info("Dataset: %s", drive)

    try:
        X, y, n_smart, n_fail, n_good = load_and_flatten(csv_path)
    except Exception as e:
        logger.error("Load failed: %s", e); continue
    if n_fail < 10:
        logger.warning("Skipping — too few failures"); continue

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_RATIO, random_state=42, stratify=y)
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    X_tr_healthy = X_tr_sc[y_tr == 0]
    failure_rate  = max(0.01, float(y_tr.mean()))

    # ──────────────────────────────────────────────────────────────
    # 1. LOCAL OUTLIER FACTOR
    # ──────────────────────────────────────────────────────────────
    logger.info("[LOF] Training ...")
    try:
        # Subsample healthy for tractability on large datasets
        n_lof = min(LOF_SUBSAMPLE, len(X_tr_healthy))
        if len(X_tr_healthy) > LOF_SUBSAMPLE:
            rng = np.random.default_rng(42)
            idx_sub = rng.choice(len(X_tr_healthy), LOF_SUBSAMPLE, replace=False)
            X_lof_fit = X_tr_healthy[idx_sub]
            logger.info("  Subsampling %d -> %d for LOF", len(X_tr_healthy), LOF_SUBSAMPLE)
        else:
            X_lof_fit = X_tr_healthy

        contamination_lof = min(0.49, failure_rate * 1.5)
        t0 = time.perf_counter()
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True,
                                  contamination=contamination_lof, n_jobs=-1)
        lof.fit(X_lof_fit)
        tt = round(time.perf_counter()-t0, 2)

        t0 = time.perf_counter()
        # negative_outlier_factor: more negative = more anomalous
        raw_scores = -lof.decision_function(X_te_sc)   # invert: higher = more anomalous
        lat = round((time.perf_counter()-t0)/len(X_te_sc)*1000, 4)

        y_pred_lof = (lof.predict(X_te_sc) == -1).astype(int)
        m, _ = metrics_from_scores(y_te, raw_scores, y_pred_lof)
        m.update({"train_time_sec": tt, "infer_latency_ms": lat,
                  "n_neighbors": 20, "contamination": round(contamination_lof,4),
                  "subsample_used": n_lof,
                  "approach": "Unsupervised (Local Outlier Factor, density-based)",
                  "window_type": "Flat 10-Day Window (z-score normalised)",
                  "description": (
                      "Density-based novelty detection. Measures how isolated each drive is "
                      "relative to its k=20 nearest healthy neighbors in SMART feature space. "
                      "A drive in a sparse region (far from similar drives) receives a high LOF score. "
                      "Unlike global methods (OC-SVM, IF), LOF captures local density variations — "
                      "effective when different drive failure modes cluster in distinct regions."
                  )})
        lof_results[drive] = m
        for k in agg["LOF"]:
            v = m.get(k)
            if isinstance(v,(int,float)): agg["LOF"][k].append(v)
        logger.info("  F1=%.4f AUC=%s Recall=%.4f Caught=%.1f%% Train=%.1fs",
                    m["f1_score"], m.get("roc_auc"), m["recall"], m["detection_rate"], tt)
    except Exception as e:
        logger.error("  LOF failed: %s", e)
        lof_results[drive] = {"error": str(e)}

    # ──────────────────────────────────────────────────────────────
    # 2. ELLIPTIC ENVELOPE
    # ──────────────────────────────────────────────────────────────
    logger.info("[EllipticEnvelope] Training ...")
    try:
        # EE needs n_samples > n_features^2/2 and struggles with high-dim data
        # Use PCA to reduce to 30 components first to keep it tractable
        from sklearn.decomposition import PCA
        n_components = min(30, X_tr_healthy.shape[1], X_tr_healthy.shape[0]-1)
        pca = PCA(n_components=n_components, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_healthy)
        X_te_pca = pca.transform(X_te_sc)

        contamination_ee = min(0.49, failure_rate * 1.5)
        t0 = time.perf_counter()
        ee = EllipticEnvelope(contamination=contamination_ee,
                              support_fraction=0.9, random_state=42)
        ee.fit(X_tr_pca)
        tt = round(time.perf_counter()-t0, 2)

        t0 = time.perf_counter()
        # mahalanobis score: more negative = more anomalous (outside envelope)
        raw_scores = -ee.decision_function(X_te_pca)
        lat = round((time.perf_counter()-t0)/len(X_te_pca)*1000, 4)

        y_pred_ee = (ee.predict(X_te_pca) == -1).astype(int)
        m, _ = metrics_from_scores(y_te, raw_scores, y_pred_ee)
        m.update({"train_time_sec": tt, "infer_latency_ms": lat,
                  "pca_components": n_components,
                  "contamination": round(contamination_ee,4),
                  "approach": "Unsupervised (Elliptic Envelope, Mahalanobis distance)",
                  "window_type": "Flat 10-Day Window -> PCA(30) -> Gaussian fit",
                  "description": (
                      "Fits a robust multivariate Gaussian (Minimum Covariance Determinant) "
                      "to the PCA-reduced healthy drive SMART distribution. "
                      "Flags drives whose Mahalanobis distance from the Gaussian center exceeds "
                      "the fitted threshold. PCA(30) applied first to handle high-dimensional "
                      "SMART feature space (190 features). "
                      "Formally tests: 'does this drive follow the same statistical distribution "
                      "as healthy drives?' — the cleanest statistical formulation of normality."
                  )})
        ee_results[drive] = m
        for k in agg["EllipticEnvelope"]:
            v = m.get(k)
            if isinstance(v,(int,float)): agg["EllipticEnvelope"][k].append(v)
        logger.info("  F1=%.4f AUC=%s Recall=%.4f Caught=%.1f%% Train=%.1fs",
                    m["f1_score"], m.get("roc_auc"), m["recall"], m["detection_rate"], tt)
    except Exception as e:
        logger.error("  EllipticEnvelope failed: %s", e)
        ee_results[drive] = {"error": str(e)}

    # ──────────────────────────────────────────────────────────────
    # 3. AUTOENCODER
    # ──────────────────────────────────────────────────────────────
    logger.info("[Autoencoder] Training ...")
    try:
        import tensorflow as tf
        from tensorflow import keras
        tf.get_logger().setLevel("ERROR")

        n_feats = X_tr_sc.shape[1]
        # Encoder-decoder: compress healthy SMART fingerprint to bottleneck
        enc_dim  = max(8, n_feats // 8)
        enc_dim2 = max(4, n_feats // 16)

        inputs  = keras.Input(shape=(n_feats,))
        encoded = keras.layers.Dense(enc_dim*4, activation="relu")(inputs)
        encoded = keras.layers.Dense(enc_dim*2, activation="relu")(encoded)
        bottleneck = keras.layers.Dense(enc_dim, activation="relu")(encoded)
        decoded = keras.layers.Dense(enc_dim*2, activation="relu")(bottleneck)
        decoded = keras.layers.Dense(enc_dim*4, activation="relu")(decoded)
        outputs = keras.layers.Dense(n_feats, activation="linear")(decoded)

        ae = keras.Model(inputs, outputs)
        ae.compile(optimizer="adam", loss="mse")

        t0 = time.perf_counter()
        ae.fit(X_tr_healthy, X_tr_healthy,
               epochs=AE_EPOCHS, batch_size=256,
               validation_split=0.1, verbose=0)
        tt = round(time.perf_counter()-t0, 2)

        t0 = time.perf_counter()
        X_te_recon = ae.predict(X_te_sc, verbose=0)
        # Reconstruction error per sample = mean squared error
        recon_errors = np.mean((X_te_sc - X_te_recon)**2, axis=1)
        lat = round((time.perf_counter()-t0)/len(X_te_sc)*1000, 4)

        # Threshold = 95th percentile of reconstruction errors on healthy train set
        train_recon  = ae.predict(X_tr_healthy, verbose=0)
        train_errors = np.mean((X_tr_healthy - train_recon)**2, axis=1)
        threshold_ae = float(np.percentile(train_errors, 95))
        y_pred_ae    = (recon_errors > threshold_ae).astype(int)

        m, _ = metrics_from_scores(y_te, recon_errors, y_pred_ae)
        m.update({"train_time_sec": tt, "infer_latency_ms": lat,
                  "epochs": AE_EPOCHS, "bottleneck_dim": enc_dim,
                  "threshold_percentile": 95,
                  "reconstruction_threshold": round(threshold_ae, 6),
                  "approach": "Unsupervised (Autoencoder — Neural Reconstruction Error)",
                  "window_type": "Flat 10-Day Window (z-score normalised)",
                  "description": (
                      f"Feed-forward autoencoder (encoder: {n_feats}->{enc_dim*4}->{enc_dim*2}->{enc_dim}, "
                      f"decoder: mirror). Trained for {AE_EPOCHS} epochs exclusively on healthy drives. "
                      "Anomaly score = mean squared reconstruction error per drive. "
                      "Threshold set at 95th percentile of healthy training reconstruction errors. "
                      "Failing drives exhibit SMART patterns the network never learned, producing "
                      "high reconstruction error. Most interpretable neural anomaly detection approach."
                  )})
        ae_results[drive] = m
        for k in agg["Autoencoder"]:
            v = m.get(k)
            if isinstance(v,(int,float)): agg["Autoencoder"][k].append(v)
        logger.info("  F1=%.4f AUC=%s Recall=%.4f Caught=%.1f%% Train=%.1fs",
                    m["f1_score"], m.get("roc_auc"), m["recall"], m["detection_rate"], tt)
    except Exception as e:
        logger.error("  Autoencoder failed: %s", e)
        ae_results[drive] = {"error": str(e)}

# ── Aggregate averages ──
avg = {}
for mname, vals in agg.items():
    avg[mname] = {k: round(float(np.mean(v)),4) if v else None for k,v in vals.items()}

# ── Append to existing metrics JSON ──
existing["local_outlier_factor"]  = {"aggregate_averages": avg["LOF"],         "per_dataset": lof_results}
existing["elliptic_envelope"]     = {"aggregate_averages": avg["EllipticEnvelope"], "per_dataset": ee_results}
existing["autoencoder"]           = {"aggregate_averages": avg["Autoencoder"],  "per_dataset": ae_results}
existing["generated_at"]          = pd.Timestamp.now().isoformat()

metrics_path.write_text(json.dumps(existing, indent=2))
logger.info("All results saved to %s", metrics_path)

# ── Final summary table ──
def row(name, avg_d):
    return (f"  {name:<25} "
            f"{avg_d.get('f1_score',0):>7.4f} "
            f"{str(avg_d.get('roc_auc','N/A')):>7} "
            f"{avg_d.get('precision',0):>7.4f} "
            f"{avg_d.get('recall',0):>7.4f} "
            f"{avg_d.get('train_time_sec',0):>10.2f} "
            f"{avg_d.get('infer_latency_ms',0):>13.4f}")

print("\n" + "="*95)
print("  SentinelDC — Full Unsupervised Anomaly Detection Comparison (Avg across 6 datasets)")
print("="*95)
print(f"  {'Method':<25} {'F1':>7} {'AUC':>7} {'Prec':>7} {'Recall':>7} {'Train(s)':>10} {'Latency(ms)':>13}")
print("-"*95)

# Pull all anomaly detection results for comparison
all_methods = {
    "Isolation Forest":  existing.get("aggregate_averages",{}).get("IsolationForest",{}),
    "One-Class SVM":     existing.get("one_class_svm",{}).get("aggregate_averages",{}),
    "LOF":               avg["LOF"],
    "Elliptic Envelope": avg["EllipticEnvelope"],
    "Autoencoder":       avg["Autoencoder"],
}
for name, d in all_methods.items():
    if d: print(row(name, d))
print("="*95)
