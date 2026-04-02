from __future__ import annotations

import json

from experiment.webarena.utils.task_loader import load_task_by_id, load_task_config, load_task_dataset, parse_task


def test_load_task_config_single_task(tmp_path):
    task_file = tmp_path / "task.json"
    task_file.write_text(
        json.dumps(
            {
                "task_id": 44,
                "sites": ["gitlab"],
                "intent": "Open my todos page",
                "start_urls": ["__GITLAB__"],
                "eval": [{"evaluator": "AgentResponseEvaluator"}],
            }
        ),
        encoding="utf-8",
    )

    config = load_task_config(task_file)

    assert config["task_id"] == 44
    assert config["sites"] == ["gitlab"]


def test_load_task_config_dataset_summary(tmp_path):
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(
        json.dumps(
            [
                {"task_id": 7, "sites": ["map"]},
                {"task_id": 44, "sites": ["gitlab"]},
                {"task_id": 45, "sites": ["gitlab"]},
            ]
        ),
        encoding="utf-8",
    )

    config = load_task_config(dataset_file)

    assert config["dataset"] is True
    assert config["n_tasks"] == 3
    assert config["sites"] == ["gitlab", "map"]


def test_load_task_dataset_and_load_task_by_id(tmp_path):
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(
        json.dumps(
            [
                {"task_id": 7, "sites": ["map"], "intent": "Map task"},
                {"task_id": 44, "sites": ["gitlab"], "intent": "Open my todos page"},
            ]
        ),
        encoding="utf-8",
    )

    tasks = load_task_dataset(dataset_file)
    task = load_task_by_id(dataset_file, 44)

    assert len(tasks) == 2
    assert task["task_id"] == 44
    assert task["intent"] == "Open my todos page"


def test_parse_task_normalizes_fields(tmp_path):
    task_file = tmp_path / "task.json"
    task_file.write_text(
        json.dumps(
            {
                "task_id": 44,
                "sites": ["gitlab"],
                "intent": "Open my todos page",
                "start_urls": ["__GITLAB__"],
                "eval": [
                    {"evaluator": "AgentResponseEvaluator"},
                    {"evaluator": "NetworkEventEvaluator"},
                ],
            }
        ),
        encoding="utf-8",
    )

    task = parse_task(task_file)

    assert task.task_id == "44"
    assert task.sites == ("gitlab",)
    assert task.n_evaluators == 2
    assert task.requires_text_response is True
