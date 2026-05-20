"""dashboard_exporter.py — aggregates all agent reports into web/dashboard.json for the frontend."""
import json
import os

import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
WEB_DIR    = os.path.join(BASE_DIR, "web")

REPORTS = {
    "model_evaluation":   os.path.join(OUTPUT_DIR, "model_evaluation.json"),
    "data_quality":       os.path.join(OUTPUT_DIR, "data_quality_report.json"),
    "feature_drift":      os.path.join(OUTPUT_DIR, "feature_drift_report.json"),
    "prediction_audit":   os.path.join(OUTPUT_DIR, "prediction_audit_report.json"),
    "retraining":         os.path.join(OUTPUT_DIR, "retraining_report.json"),
}
DASHBOARD_JSON = os.path.join(WEB_DIR, "dashboard.json")


def _load(key):
    path = REPORTS[key]
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def export():
    ev   = _load("model_evaluation")
    dq   = _load("data_quality")
    fd   = _load("feature_drift")
    pa   = _load("prediction_audit")
    rt   = _load("retraining")

    # ── Model metrics ─────────────────────────────────────────────────────
    model = None
    if ev:
        model = {
            "overall_accuracy":  ev.get("overall_accuracy"),
            "baseline_accuracy": ev.get("baseline_accuracy"),
            "vs_baseline":       ev.get("vs_baseline"),
            "total_games":       ev.get("total_games"),
            "last_7_days":       ev.get("last_7_days"),
            "last_30_days":      ev.get("last_30_days"),
            "calibration":       ev.get("calibration", []),
            "alert":             ev.get("alert_model_below_baseline", False),
        }

    # ── Health per component ──────────────────────────────────────────────
    health = {
        "data_quality": {
            "passed":   dq.get("passed", True)   if dq else None,
            "issues":   len(dq.get("issues", [])) if dq else 0,
            "warnings": len(dq.get("warnings", [])) if dq else 0,
            "generated_at": dq.get("generated_at") if dq else None,
        },
        "feature_drift": {
            "passed":   fd.get("passed", True)   if fd else None,
            "issues":   len(fd.get("issues", [])) if fd else 0,
            "warnings": len(fd.get("warnings", [])) if fd else 0,
            "generated_at": fd.get("generated_at") if fd else None,
        },
        "prediction_audit": {
            "passed":      pa.get("passed", True)      if pa else None,
            "predictions": pa.get("predictions", 0)    if pa else 0,
            "flags":       len(pa.get("flags", []))    if pa else 0,
            "generated_at": pa.get("generated_at")     if pa else None,
        },
        "retraining": {
            "retrained":    rt.get("retrained", False)  if rt else None,
            "reason":       rt.get("reason", "")        if rt else "",
            "test_accuracy": rt.get("test_accuracy") or rt.get("new_test_accuracy") if rt else None,
            "generated_at": rt.get("generated_at")      if rt else None,
        },
    }

    # ── Observations (all flags/issues across all reports) ────────────────
    observations = []

    if pa:
        for flag in pa.get("flags", []):
            observations.append({
                "source":   "prediction_audit",
                "severity": flag.get("severity", "low"),
                "title":    flag.get("game", ""),
                "detail":   f"[{flag.get('type', '')}] {flag.get('detail', '')}",
            })

    if fd:
        for issue in fd.get("issues", []):
            observations.append({
                "source":   "feature_drift",
                "severity": "high",
                "title":    "Feature Drift erkannt",
                "detail":   issue,
            })
        for warn in fd.get("warnings", []):
            observations.append({
                "source":   "feature_drift",
                "severity": "low",
                "title":    "Neue Features verfuegbar",
                "detail":   warn,
            })

    if dq:
        for issue in dq.get("issues", []):
            observations.append({
                "source":   "data_quality",
                "severity": "high",
                "title":    "Datenproblem",
                "detail":   issue,
            })

    if ev and ev.get("alert_model_below_baseline"):
        observations.append({
            "source":   "model_evaluator",
            "severity": "high",
            "title":    "Modell schlechter als Baseline",
            "detail":   f"Accuracy {ev.get('overall_accuracy', 0):.1%} < Baseline {ev.get('baseline_accuracy', 0):.1%}",
        })

    # ── Overall status ────────────────────────────────────────────────────
    high_count = sum(1 for o in observations if o["severity"] == "high")
    medium_count = sum(1 for o in observations if o["severity"] == "medium")
    if high_count > 0:
        overall_status = "critical"
    elif medium_count > 0:
        overall_status = "warning"
    else:
        overall_status = "ok"

    dashboard = {
        "generated_at":   pd.Timestamp.now(tz="US/Eastern").tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S"),
        "overall_status": overall_status,
        "model":          model,
        "health":         health,
        "observations":   observations,
    }

    os.makedirs(WEB_DIR, exist_ok=True)
    with open(DASHBOARD_JSON, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)

    status_label = {"ok": "OK", "warning": "WARNUNG", "critical": "KRITISCH"}.get(overall_status, overall_status)
    print(f"\n{'='*55}")
    print(f"  DASHBOARD EXPORT  --  {status_label}")
    print(f"{'='*55}")
    print(f"  Observations:  {len(observations)}  ({high_count} kritisch, {medium_count} Warnungen)")
    print(f"  Output:        {DASHBOARD_JSON}")
    print(f"{'='*55}\n")

    return dashboard


if __name__ == "__main__":
    export()
