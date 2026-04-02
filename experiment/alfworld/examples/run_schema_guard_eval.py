"""
Example: Run Experiment 2 schema-guard evaluation.

Usage:
    python run_schema_guard_eval.py [--data-root PATH] [--output-dir PATH] [--n-states N]

This script:
1. Generates synthetic captured states (or loads real ones from a JSONL file)
2. Builds injection cases for all 5 injection types
3. Runs kernel_node with C2 on and C2 off for each case
4. Prints and saves aggregate metrics
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.alfworld.evaluators.schema_guard_evaluator import (
    SchemaGuardEvaluator,
    compute_schema_guard_metrics,
    flatten_case_results,
)
from experiment.alfworld.utils.injection import build_injection_cases


# ---------------------------------------------------------------------------
# Synthetic state generator (used when no real states are available)
# ---------------------------------------------------------------------------

_SYNTHETIC_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "action": {"type": "string"},
        "task_type": {"type": "string", "enum": ["pick_and_place", "heat_then_place", "cool_then_place"]},
        "step_count": {"type": "integer"},
    },
    "required": ["observation", "action"],
}

_SYNTHETIC_DOMAIN_STATE = {
    "observation": "You are in the kitchen. There is a tomato on the counter.",
    "action": "go to counter 1",
    "task_type": "pick_and_place",
    "step_count": 3,
}


def make_synthetic_states(n: int) -> list:
    states = []
    for i in range(n):
        states.append({
            "episode_id": f"synthetic_ep_{i:03d}",
            "gamefile": "",
            "step_id": i,
            "domain_state": dict(_SYNTHETIC_DOMAIN_STATE, step_count=i),
            "data_schema": _SYNTHETIC_SCHEMA,
        })
    return states


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load newline-delimited JSON records from disk."""
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 2: schema guard evaluation")
    parser.add_argument("--states-file", default=None,
                        help="JSONL file with captured states (one JSON object per line)")
    parser.add_argument("--n-states", type=int, default=10,
                        help="Number of synthetic states to generate if --states-file not given")
    parser.add_argument("--output-dir", default="outputs/exp2",
                        help="Directory to write results JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load or generate states
    if args.states_file:
        states_path = Path(args.states_file)
        if not states_path.is_file():
            raise FileNotFoundError(
                f"Captured states file not found: {states_path}. "
                "This usually means ALFWorld state capture selected zero runnable gamefiles "
                "or produced zero captured states."
            )
        captured_states = load_jsonl(states_path)
        print(f"Loaded {len(captured_states)} states from {states_path}")
    else:
        captured_states = make_synthetic_states(args.n_states)
        print(f"Generated {len(captured_states)} synthetic states")

    if not captured_states:
        raise ValueError(
            "Schema-guard evaluation needs at least one captured state, but zero states were loaded."
        )

    # Build injection cases
    cases = build_injection_cases(captured_states)
    print(f"Built {len(cases)} injection cases")

    # Evaluate
    evaluator = SchemaGuardEvaluator()
    results = evaluator.evaluate_cases(cases)
    metrics = compute_schema_guard_metrics(results)

    # Print summary
    print("\n=== Schema Guard Evaluation Results ===")
    print(f"Total cases:              {metrics.n_cases}")
    print(f"C2 on  interception rate: {metrics.c2_on_interception_rate:.3f}")
    print(f"C2 off interception rate: {metrics.c2_off_interception_rate:.3f}")
    print(f"Delta (C2 adds):          {metrics.delta_interception_rate:+.3f}")
    print("\nBy injection type:")
    for t in sorted(metrics.n_by_type):
        n = metrics.n_by_type[t]
        on_r = metrics.full.by_injection_type.get(t, {}).get("interception_rate", 0.0)
        off_r = metrics.ablate_c2.by_injection_type.get(t, {}).get("interception_rate", 0.0)
        on_apply = metrics.full.by_injection_type.get(t, {}).get("apply_rate", 0.0)
        off_apply = metrics.ablate_c2.by_injection_type.get(t, {}).get("apply_rate", 0.0)
        print(
            f"  {t:<30s} n={n:3d}  "
            f"C2_on(intercept={on_r:.2f}, apply={on_apply:.2f})  "
            f"C2_off(intercept={off_r:.2f}, apply={off_apply:.2f})"
        )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "schema_guard_metrics.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_cases": metrics.n_cases,
            "n_by_type": metrics.n_by_type,
            "full": asdict(metrics.full),
            "ablate_c2": asdict(metrics.ablate_c2),
            "c2_on_interception_rate": metrics.c2_on_interception_rate,
            "c2_off_interception_rate": metrics.c2_off_interception_rate,
            "delta_interception_rate": metrics.delta_interception_rate,
            "c2_on_interception_by_type": metrics.c2_on_interception_by_type,
            "c2_off_interception_by_type": metrics.c2_off_interception_by_type,
        }, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    # Save per-case results
    cases_path = os.path.join(args.output_dir, "schema_guard_cases.jsonl")
    rows = flatten_case_results(results)
    with open(cases_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Per-case results saved to {cases_path}")


if __name__ == "__main__":
    main()
