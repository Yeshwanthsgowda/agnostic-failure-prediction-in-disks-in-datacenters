import json, logging, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, precision_recall_curve)
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEQUENCE_LEN   = 10
TEST_RATIO     = 0.30
OC_SVM_SUBSAMPLE = 5000   # Max healthy drives for OC-SVM training (kernel tractability)
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

results = {}
aggregate = {"f1_score":[], "roc_auc":[], "precision":[], "recall":[],
             "train_time_sec":[], "infer_latency_ms":[], "subsample_used":[]}

for csv_path in DATASETS:
    p = Path(csv_path)
    if not p.exists(): logger.warning("MISSING: %s", csv_path); continue
    drive = p.stem.replace("_last_10_day_all_q_raw","").replace("_"," ").strip()
    logger.info("=== %s ===", drive)

    try:
        X, y, n_feats, n_fail, n_good = load_and_flatten(csv_path)
    except Exception as e:
        logger.error("Load failed: %s", e); continue

    if n_fail < 10:
        logger.warning("Skipping — too few failures (%d)", n_fail); continue

    # 70/30 split
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_RATIO, random_state=42, stratify=y)
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    # Normalise (critical for OC-SVM with RBF kernel)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # Subsample healthy training drives only (OC-SVM is unsupervised)
    healthy_mask = y_tr == 0
    X_tr_healthy = X_tr_sc[healthy_mask]
    n_healthy = len(X_tr_healthy)
    subsample = min(OC_SVM_SUBSAMPLE, n_healthy)
    if n_healthy > OC_SVM_SUBSAMPLE:
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(n_healthy, OC_SVM_SUBSAMPLE, replace=False)
        X_tr_fit = X_tr_healthy[idx_sub]
        logger.info("  Subsampling %d -> %d healthy drives for OC-SVM", n_healthy, OC_SVM_SUBSAMPLE)
    else:
        X_tr_fit = X_tr_healthy

    # nu = expected fraction of anomalies (set to true failure rate + small buffer)
    failure_rate = max(0.01, float(y_tr.mean()))
    nu = min(0.5, failure_rate * 2)  # generous upper bound

    logger.info("  nu=%.4f | training on %d healthy drives | %d features", nu, len(X_tr_fit), n_feats*SEQUENCE_LEN)

    t0 = time.perf_counter()
    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    ocsvm.fit(X_tr_fit)
    train_time = round(time.perf_counter()-t0, 2)

    # Anomaly score: decision_function returns signed distance from boundary
    # More negative = more anomalous. We negate and normalise to [0,1].
    t0 = time.perf_counter()
    raw_scores = ocsvm.decision_function(X_te_sc)
    lat = round((time.perf_counter()-t0)/len(X_te_sc)*1000, 4)

    # Invert so higher = more anomalous, then normalise to [0,1]
    anomaly_scores = -raw_scores
    s_min, s_max = anomaly_scores.min(), anomaly_scores.max()
    y_prob = (anomaly_scores - s_min) / max(s_max - s_min, 1e-9)

    # Binary prediction: OC-SVM returns +1 (normal) or -1 (anomaly)
    raw_pred = ocsvm.predict(X_te_sc)
    y_pred   = (raw_pred == -1).astype(int)

    # Metrics
    acc  = round(float(accuracy_score(y_te, y_pred)), 4)
    prec = round(float(precision_score(y_te, y_pred, zero_division=0)), 4)
    rec  = round(float(recall_score(y_te, y_pred, zero_division=0)), 4)
    f1   = round(float(f1_score(y_te, y_pred, zero_division=0)), 4)
    try:
        auc = round(float(roc_auc_score(y_te, y_prob)), 4)
    except Exception:
        auc = None

    # PR curve
    try:
        p_arr, r_arr, _ = precision_recall_curve(y_te, y_prob)
        step = max(1, len(p_arr)//100)
        pr_prec = [round(float(v),4) for v in p_arr[::step]]
        pr_rec  = [round(float(v),4) for v in r_arr[::step]]
    except Exception:
        pr_prec, pr_rec = [], []

    # Detection breakdown
    tp = int(((y_pred==1)&(y_te==1)).sum())
    fp = int(((y_pred==1)&(y_te==0)).sum())
    fn = int(((y_pred==0)&(y_te==1)).sum())
    tn = int(((y_pred==0)&(y_te==0)).sum())
    n_test_fail = int(y_te.sum())

    logger.info("  F1=%.4f | AUC=%s | Prec=%.4f | Recall=%.4f | Train=%.1fs",
                f1, auc, prec, rec, train_time)
    logger.info("  Detection: %d/%d failures caught | %d false alarms | TP=%d FP=%d FN=%d TN=%d",
                tp, n_test_fail, fp, tp, fp, fn, tn)

    m = {
        "approach":            "Unsupervised (One-Class SVM, RBF kernel)",
        "window_type":         "Flat 10-Day Window (z-score normalised, healthy drives only)",
        "nu":                  round(nu, 4),
        "subsample_used":      len(X_tr_fit),
        "healthy_train_total": n_healthy,
        "threshold":           "decision boundary",
        "accuracy":            acc,
        "precision":           prec,
        "recall":              rec,
        "f1_score":            f1,
        "roc_auc":             auc,
        "train_time_sec":      train_time,
        "infer_latency_ms":    lat,
        "pr_curve_prec":       pr_prec,
        "pr_curve_rec":        pr_rec,
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "detection_rate":      round(tp/max(n_test_fail,1)*100, 2),
        "false_alarm_rate":    round(fp/max(tn+fp,1)*100, 2),
        "description": (
            "Kernel-based one-class classifier trained exclusively on healthy drive data. "
            "Fits a high-dimensional RBF hypersphere around the normal SMART distribution. "
            "Test drives falling outside the boundary are flagged as anomalies. "
            f"Training subsampled to {len(X_tr_fit)} healthy drives for kernel tractability. "
            "Features z-score normalised (required for RBF kernel distance calculations). "
            "nu parameter set to 2x empirical failure rate as the expected contamination fraction."
        ),
    }
    results[drive] = m

    for k in aggregate:
        v = m.get(k)
        if isinstance(v, (int, float)) and v is not None:
            aggregate[k].append(v)

# Aggregate averages
avg = {k: round(float(np.mean(v)), 4) if v else None for k, v in aggregate.items()}

# Load existing research_metrics.json and add OC-SVM section
metrics_path = Path("models/research_metrics.json")
if metrics_path.exists():
    with open(metrics_path) as f:
        existing = json.load(f)
else:
    existing = {}

existing["one_class_svm"] = {
    "generated_at":       pd.Timestamp.now().isoformat(),
    "method":             "One-Class SVM (RBF kernel, trained on healthy drives only)",
    "subsample_cap":      OC_SVM_SUBSAMPLE,
    "aggregate_averages": avg,
    "per_dataset":        results,
}

with open(metrics_path, "w") as f:
    json.dump(existing, f, indent=2)
logger.info("OC-SVM results appended to %s", metrics_path)

# Final comparison table
print("\n" + "="*90)
print("  SentinelDC - One-Class SVM Anomaly Detection Results (Per Drive Family)")
print("="*90)
print(f"  {'Dataset':<30} {'F1':>7} {'AUC':>7} {'Prec':>7} {'Recall':>7} {'Caught%':>9} {'FalseAlm%':>11}")
print("-"*90)
for drive, m in results.items():
    print(f"  {drive:<30} {m['f1_score']:>7.4f} {str(m['roc_auc']):>7} "
          f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
          f"{m['detection_rate']:>8.1f}% {m['false_alarm_rate']:>10.1f}%")
print("-"*90)
print(f"  {'AVERAGE':<30} {avg['f1_score']:>7.4f} {avg['roc_auc']:>7.4f} "
      f"{avg['precision']:>7.4f} {avg['recall']:>7.4f}")
print("="*90)

# Compare OC-SVM vs Isolation Forest from existing results
print("\n  Anomaly Detection Head-to-Head: OC-SVM vs Isolation Forest")
print("  " + "-"*50)
if "aggregate_averages" in existing:
    if_avg = existing.get("aggregate_averages",{}).get("IsolationForest",{})
    print(f"  {'Metric':<20} {'One-Class SVM':>15} {'Isolation Forest':>17}")
    print("  " + "-"*50)
    for k in ["f1_score","roc_auc","precision","recall"]:
        ocsvm_val = avg.get(k, "N/A")
        if_val    = if_avg.get(k, "N/A") if isinstance(if_avg, dict) else "N/A"
        print(f"  {k:<20} {str(ocsvm_val):>15} {str(if_val):>17}")
print("="*90)
