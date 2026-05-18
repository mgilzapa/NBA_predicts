"""pipeline_runner.py — runs the full daily pipeline with per-step timing and a JSON report."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_PATH = os.path.join(BASE_DIR, "output", "pipeline_report.json")

SCRIPTS = [
    "src/nba_api_test.py",
    "src/scrape_upcoming_games.py",
    "src/injury_reports.py",
    "src/fetch_player_stats.py",
    "src/tabelle.py",
    "src/feature_engineering.py",
    "src/predict.py",
    "src/clean_excel.py",
    "src/create_excel.py",
    "src/fetch_odds.py",
    "src/export_json.py",
    "src/fetch_bracket.py",
]


def run():
    if not os.environ.get("ODDS_API_KEY"):
        os.environ["ODDS_API_KEY"] = "373d3a5bd6b6dad8581de9490f258177"

    started_at = datetime.now()
    results = []
    failed = False

    print(f"\n{'='*55}")
    print(f"  NBA PIPELINE  —  {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    for script in SCRIPTS:
        script_path = os.path.join(BASE_DIR, script)
        if not os.path.isfile(script_path):
            print(f"  ERR  {script}  —  Datei nicht gefunden")
            results.append({"script": script, "status": "not_found", "duration_s": 0})
            failed = True
            break

        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=BASE_DIR,
        )
        duration = round(time.monotonic() - t0, 2)

        if proc.returncode == 0:
            print(f"  OK   {script:<45}  {duration}s")
            results.append({"script": script, "status": "ok", "duration_s": duration})
        else:
            print(f"  ERR  {script:<45}  {duration}s")
            print(f"\n{'─'*40}")
            print(proc.stderr.strip()[-2000:])
            print(f"{'─'*40}\n")
            results.append({
                "script": script,
                "status": "error",
                "duration_s": duration,
                "stderr": proc.stderr.strip()[-2000:],
            })
            failed = True
            break

    finished_at = datetime.now()
    total_s = round((finished_at - started_at).total_seconds(), 2)

    report = {
        "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished_at": finished_at.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_duration_s": total_s,
        "success": not failed,
        "steps": results,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    label = "ERFOLGREICH" if not failed else "FEHLGESCHLAGEN"
    completed = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'='*55}")
    print(f"  {label}  —  {completed}/{len(SCRIPTS)} Schritte  —  {total_s}s gesamt")
    print(f"  Report: {REPORT_PATH}")
    print(f"{'='*55}\n")

    return not failed


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
