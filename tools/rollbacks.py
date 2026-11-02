"""Step 0a: list every rollback in every run, and mark runs excluded from early-checkpoint comparisons.

A rollback (train.py) restores the latest checkpoint after non-finite losses and re-seeds the
noise generator; events.jsonl records {"event": "rollback", "failed_at_iteration",
"to_iteration", "new_seed"}. As an independent check, every metrics.jsonl row carries the
running "rollbacks" count: the last row's count must equal the number of rollback events.

Rule: a run with a rollback that failed at or below --early (default 50,000) is excluded from
early-checkpoint comparisons. Both the failure iteration and the restored checkpoint are listed.

Usage (repository root):
    python tools/rollbacks.py runs/chtc/run_*
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def scan(run_dir: Path) -> dict:
    events = _jsonl(run_dir / "events.jsonl")
    rollbacks = [e for e in events if e.get("event") == "rollback"]
    metrics = _jsonl(run_dir / "metrics.jsonl")
    counted = metrics[-1].get("rollbacks") if metrics else None
    return {"run": run_dir.name, "rollbacks": rollbacks, "metrics_count": counted,
            "has_events": (run_dir / "events.jsonl").exists(), "last_iteration": metrics[-1]["iteration"] if metrics else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--early", type=int, default=50_000, help="exclusion threshold on the failure iteration")
    args = ap.parse_args(argv)

    results = [scan(Path(r)) for r in sorted(args.runs) if Path(r).is_dir()]
    problems = 0
    print(f"{'run':<40} {'failed at':>10} {'restored to':>12} {'new seed':>9}")
    any_rb = False
    for res in results:
        for e in res["rollbacks"]:
            any_rb = True
            print(f"{res['run']:<40} {e['failed_at_iteration']:>10,} {e['to_iteration']:>12,} {e['new_seed']:>9}")
    if not any_rb:
        print("none")

    print(f"\n{len(results)} runs scanned")
    excluded = []
    for res in results:
        n = len(res["rollbacks"])
        if not res["has_events"]:
            print(f"  WARNING {res['run']}: no events.jsonl")
            problems += 1
        if res["metrics_count"] is not None and res["metrics_count"] != n:
            print(f"  MISMATCH {res['run']}: {n} rollback events but metrics.jsonl counts {res['metrics_count']}")
            problems += 1
        if any(e["failed_at_iteration"] <= args.early for e in res["rollbacks"]):
            excluded.append(res["run"])
    print(f"excluded from early-checkpoint comparisons (rollback failing at or below {args.early:,}): "
          + (", ".join(excluded) if excluded else "none"))
    print("cross-check against metrics.jsonl: " + ("all consistent" if problems == 0 else f"{problems} problem(s) above"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
