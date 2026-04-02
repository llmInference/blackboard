"""Tests for ALFWorld system comparison runner."""
from __future__ import annotations

import json

from examples.run_system_compare import prepare_resume_gamefiles, run_system_compare
from experiment.alfworld.utils.result_writer import ResultWriter


def test_run_system_compare_reuses_same_gamefiles_and_writes_summary(tmp_path):
    call_log: list[tuple[str, str]] = []
    gamefiles = ["gf1", "gf2", "gf3"]
    systems = ["blackboard", "langgraph", "autogen"]

    def fake_run_system_suite(system: str, system_output_dir: str):
        call_log.append((system, system_output_dir))
        return {
            "system": system,
            "system_id": f"{system}_id",
            "system_family": system,
            "workflow_mode": "planner_action",
            "summary": {
                "success_rate": 0.25,
                "mean_steps": 5.0,
                "mean_total_tokens": 123.0,
            },
            "episodes_path": str(tmp_path / f"{system}_episodes.jsonl"),
        }

    consolidated = run_system_compare(
        gamefiles=gamefiles,
        systems=systems,
        output_dir=str(tmp_path),
        run_system_suite=fake_run_system_suite,
        workflow_mode="planner_action",
        architect_mode="deterministic",
    )

    assert [system for system, _ in call_log] == systems

    selected_gamefiles = json.loads((tmp_path / "selected_gamefiles.json").read_text(encoding="utf-8"))
    assert selected_gamefiles == gamefiles

    loaded = json.loads((tmp_path / "system_compare_summary.json").read_text(encoding="utf-8"))
    assert loaded["systems"] == systems
    assert loaded["n_gamefiles"] == 3
    assert loaded["workflow_mode"] == "planner_action"
    assert loaded["architect_mode"] == "deterministic"
    assert loaded["system_summaries"]["autogen"]["result"]["summary"]["mean_steps"] == 5.0
    assert consolidated["system_summaries"]["blackboard"]["system"] == "blackboard"


def test_prepare_resume_gamefiles_dedupes_existing_episode_results(tmp_path):
    writer = ResultWriter(
        str(tmp_path / "blackboard"),
        "blackboard_full",
        system_id="blackboard_planner_llm_action_full",
        system_family="blackboard",
        env_name="alfworld",
    )
    rows = [
        {"episode_id": "ep1", "gamefile": "gf1", "success": False},
        {"episode_id": "ep2", "gamefile": "gf1", "success": True},
        {"episode_id": "ep3", "gamefile": "gf2", "success": True},
    ]
    for path in (writer.episodes_path, writer.standard_episodes_path):
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    resume_state = prepare_resume_gamefiles(
        gamefiles=["gf1", "gf2", "gf3"],
        writer=writer,
    )

    assert resume_state["completed_count"] == 2
    assert resume_state["remaining_gamefiles"] == ["gf3"]

    deduped = [json.loads(line) for line in writer.episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(deduped) == 2
    assert deduped[0]["gamefile"] == "gf1"
    assert deduped[0]["success"] is True
    assert deduped[1]["gamefile"] == "gf2"
