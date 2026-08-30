"""
Smoke test for v2 DriveTelemetryRouter (stateful rolling window).
Simulates feeding 10 daily logs one at a time for a real drive from the demo stream.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
from pipeline.router import DriveTelemetryRouter

router = DriveTelemetryRouter(models_dir="models")
print("Registered models:", router.list_models())
print()

# Load real demo stream
demo = pd.read_csv("models/ST12000NM0007_demo_stream.csv", parse_dates=["date"])

# Pick one failing serial and one healthy serial
fail_sn  = demo[demo["failure"] == 1]["serial_number"].iloc[0]
good_sn  = demo[demo["failure"] == 0]["serial_number"].iloc[0]

print("=" * 60)
print("SIMULATING FAILING DRIVE:", fail_sn)
print("=" * 60)
router.clear_cache()

for day_num, (_, row) in enumerate(
    demo[demo["serial_number"] == fail_sn].sort_values("date").iterrows(), start=1
):
    result = router.ingest_log(row.to_dict())
    if result["status"] == "accumulating":
        print(f"  Day {day_num:2d}: Buffering ... ({result['days_buffered']}/10 days cached)")
    else:
        print(
            f"  Day {day_num:2d}: PREDICTION -> "
            f"Prob={result['probability_pct']:5.2f}%  "
            f"Risk={result['risk_level'].upper():8}  "
            f"Prediction={'FAILING' if result['prediction'] else 'HEALTHY'}"
        )

print()
print("=" * 60)
print("SIMULATING HEALTHY DRIVE:", good_sn)
print("=" * 60)
router.clear_cache()

for day_num, (_, row) in enumerate(
    demo[demo["serial_number"] == good_sn].sort_values("date").iterrows(), start=1
):
    result = router.ingest_log(row.to_dict())
    if result["status"] == "accumulating":
        print(f"  Day {day_num:2d}: Buffering ... ({result['days_buffered']}/10 days cached)")
    else:
        print(
            f"  Day {day_num:2d}: PREDICTION -> "
            f"Prob={result['probability_pct']:5.2f}%  "
            f"Risk={result['risk_level'].upper():8}  "
            f"Prediction={'FAILING' if result['prediction'] else 'HEALTHY'}"
        )

print()
print("=" * 60)
print("RCA REPORT FOR FAILING DRIVE:", fail_sn)
print("=" * 60)

# Re-feed failing drive to regenerate its cache
router.clear_cache()
for _, row in demo[demo["serial_number"] == fail_sn].sort_values("date").iterrows():
    router.ingest_log(row.to_dict())

rca = router.generate_rca(fail_sn, "ST12000NM0007", top_n=4)
print(rca["rca_text"])
print()
print("ACTION:", rca["action_required"])
print()
print("Top Risk Factors (grouped by SMART attribute):")
for f in rca["top_risk_factors"]:
    print(
        f"  #{f['rank']}  {f['smart_id']:12} | {f['name']:35} | "
        f"last_val={f['last_value']:>12,.0f} | shap={f['total_shap']:+.4f} | "
        f"trend={f['trend']:6} | {f['impact']}"
    )
