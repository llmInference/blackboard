"""Run heuristic or OpenAI-compatible communication judging on Experiment 1 records."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.common.communication_analysis import load_jsonl, write_jsonl


def _build_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    content = (
        "You are grading communication consistency between a sender and a receiver action.\n"
        "Return JSON only with keys score and reason.\n"
        "score must be 1 if the receiver action is consistent with the sender intent/message, else 0.\n\n"
        "Formatting rules:\n"
        "- Output exactly one JSON object\n"
        "- Do not use markdown fences\n"
        "- Do not add explanation before or after JSON\n"
        "- score must be an integer 0 or 1\n\n"
        f"Sender intent: {record.get('sender_intent', '')}\n"
        f"Sender message: {record.get('sender_message', '')}\n"
        f"Receiver action: {record.get('receiver_action', '')}\n"
        f"Expected action: {record.get('expected_action', '')}\n"
    )
    return [{"role": "user", "content": content}]


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Judge returned empty content.")

    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    candidates: list[str] = []
    if cleaned:
        candidates.append(cleaned)
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start : end + 1].strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise ValueError(f"Judge response is not valid JSON: {raw[:200]}")


def _heuristic_score(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "judge_score": int(float(record.get("heuristic_score", 0.0)) >= 0.5),
        "judge_reason": "heuristic_proxy",
    }


def _openai_score(record: Dict[str, Any], *, model: str, base_url: str, api_key: str, timeout: float) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=_build_messages(record),
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = str(response.choices[0].message.content or "{}")
    payload = _extract_json_object(content)
    return {
        "judge_score": int(payload.get("score", 0)),
        "judge_reason": str(payload.get("reason", "") or ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Experiment 1 communication judge records")
    parser.add_argument("--input-path", required=True, help="Path to comm_judge.jsonl")
    parser.add_argument("--output-path", default="", help="Optional output JSONL path for scored rows")
    parser.add_argument("--summary-path", default="", help="Optional output JSON path for score summary")
    parser.add_argument(
        "--backend",
        choices=["heuristic", "openai_compatible"],
        default="heuristic",
        help="Judge backend",
    )
    parser.add_argument("--model", default="", help="Model name for openai_compatible backend")
    parser.add_argument("--base-url", default="", help="Base URL for openai_compatible backend")
    parser.add_argument("--api-key", default="", help="API key for openai_compatible backend")
    parser.add_argument("--model-env", default="OPENAI_MODEL_NAME", help="Model env var fallback")
    parser.add_argument("--base-url-env", default="OPENAI_API_BASE", help="Base URL env var fallback")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="API key env var fallback")
    parser.add_argument("--timeout", type=float, default=60.0, help="Network timeout in seconds")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of rows to score")
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=1.0,
        help="Seconds to sleep after each scored row for backend throttling",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep scoring later rows when one request fails",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    output_path = Path(args.output_path) if args.output_path else Path(args.input_path).with_name("comm_judge_scored.jsonl")
    summary_path = Path(args.summary_path) if args.summary_path else Path(args.input_path).with_name("comm_judge_summary.json")

    scored_rows: List[Dict[str, Any]] = []
    if args.backend == "heuristic":
        for idx, row in enumerate(rows, 1):
            print(f"[judge] {idx}/{len(rows)} heuristic", flush=True)
            scored_rows.append({**row, **_heuristic_score(row), "judge_backend": "heuristic"})
    else:
        model = args.model or os.getenv(args.model_env, "")
        base_url = args.base_url or os.getenv(args.base_url_env, "")
        api_key = args.api_key or os.getenv(args.api_key_env, "")
        if not model or not base_url or not api_key:
            raise ValueError("openai_compatible backend requires model, base_url, and api_key.")
        for idx, row in enumerate(rows, 1):
            judge_id = str(row.get("judge_id", f"row_{idx}"))
            print(f"[judge] {idx}/{len(rows)} {judge_id}", flush=True)
            try:
                score_payload = _openai_score(
                    row,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    timeout=args.timeout,
                )
            except Exception as exc:
                if not args.continue_on_error:
                    print(f"[judge] failed at {idx}/{len(rows)} {judge_id}: {exc}", flush=True)
                    raise
                print(f"[judge] failed at {idx}/{len(rows)} {judge_id}: {exc}", flush=True)
                scored_rows.append(
                    {
                        **row,
                        "judge_score": None,
                        "judge_reason": "",
                        "judge_backend": "openai_compatible",
                        "judge_model": model,
                        "judge_error": str(exc),
                    }
                )
                if idx < len(rows) and args.sleep_between_requests > 0:
                    time.sleep(args.sleep_between_requests)
                continue

            scored_rows.append(
                {
                    **row,
                    **score_payload,
                    "judge_backend": "openai_compatible",
                    "judge_model": model,
                }
            )
            if idx < len(rows) and args.sleep_between_requests > 0:
                time.sleep(args.sleep_between_requests)

    write_jsonl(output_path, scored_rows)
    valid_rows = [row for row in scored_rows if row.get("judge_score") is not None]
    failed_rows = [row for row in scored_rows if row.get("judge_score") is None]
    if valid_rows:
        score_sum = sum(int(row.get("judge_score", 0)) for row in valid_rows)
        summary = {
            "n_rows": len(scored_rows),
            "n_valid_rows": len(valid_rows),
            "n_failed_rows": len(failed_rows),
            "judge_alignment_rate": float(score_sum / len(valid_rows)),
            "judge_score_sum": score_sum,
            "backend": args.backend,
            "input_path": str(Path(args.input_path)),
            "output_path": str(output_path),
        }
    else:
        summary = {
            "n_rows": 0,
            "n_valid_rows": 0,
            "n_failed_rows": len(failed_rows),
            "judge_alignment_rate": 0.0,
            "judge_score_sum": 0,
            "backend": args.backend,
            "input_path": str(Path(args.input_path)),
            "output_path": str(output_path),
        }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Scored judge rows written to: {output_path}")
    print(f"Judge summary written to: {summary_path}")


if __name__ == "__main__":
    main()
