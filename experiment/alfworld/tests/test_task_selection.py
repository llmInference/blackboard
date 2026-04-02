"""Tests for shared ALFWorld task-selection helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.alfworld.utils.dataset_sampler import task_type_from_gamefile
from experiment.alfworld.utils.task_selection import select_gamefiles
from experiment.common.task_registry import resolve_task_set


def _write_gamefile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_resolve_task_set_group_sample_by_alfworld_type(tmp_path):
    data_root = tmp_path / "alfworld_data"
    _write_gamefile(data_root / "pick_and_place_task_1" / "trial" / "game.tw-pddl")
    _write_gamefile(data_root / "pick_and_place_task_2" / "trial" / "game.tw-pddl")
    _write_gamefile(data_root / "look_at_obj_task_1" / "trial" / "game.tw-pddl")
    _write_gamefile(data_root / "look_at_obj_task_2" / "trial" / "game.tw-pddl")

    manifest_path = tmp_path / "alfworld_debug_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "env_name": "alfworld",
                "task_set": "debug",
                "root_dir": str(data_root),
                "selection": {
                    "mode": "group_sample",
                    "patterns": ["**/game.tw-pddl"],
                    "group_by": "alfworld_task_type",
                    "n_per_group": 1,
                    "seed": 7,
                },
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_task_set(manifest_path)

    assert resolved["env_name"] == "alfworld"
    assert resolved["task_set"] == "debug"
    assert resolved["selection_mode"] == "group_sample"
    assert resolved["n_tasks"] == 2
    assert [task_type_from_gamefile(path) for path in resolved["task_paths"]] == [
        "pick_and_place",
        "look_at_obj",
    ]


def test_select_gamefiles_uses_task_manifest_without_data_root(tmp_path):
    data_root = tmp_path / "alfworld_data"
    first_gamefile = data_root / "pick_and_place_task_1" / "trial" / "game.tw-pddl"
    second_gamefile = data_root / "pick_and_place_task_2" / "trial" / "game.tw-pddl"
    _write_gamefile(first_gamefile)
    _write_gamefile(second_gamefile)

    manifest_path = tmp_path / "alfworld_explicit_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "env_name": "alfworld",
                "task_set": "smoke",
                "root_dir": str(data_root),
                "selection": {
                    "mode": "explicit",
                    "items": [
                        "pick_and_place_task_1/trial/game.tw-pddl",
                        "pick_and_place_task_2/trial/game.tw-pddl",
                    ],
                    "seed": 5,
                },
            }
        ),
        encoding="utf-8",
    )

    gamefiles, selection_meta = select_gamefiles(
        data_root="",
        split="debug",
        seed=11,
        limit=1,
        task_set_file=str(manifest_path),
    )

    assert gamefiles == [str(first_gamefile.resolve())]
    assert selection_meta["selection_source"] == "task_set_manifest"
    assert selection_meta["task_set_name"] == "smoke"
    assert selection_meta["selection_seed"] == 11


def test_select_gamefiles_raises_for_empty_task_manifest(tmp_path):
    empty_root = tmp_path / "alfworld_data"
    empty_root.mkdir(parents=True, exist_ok=True)

    manifest_path = tmp_path / "alfworld_empty_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "env_name": "alfworld",
                "task_set": "empty",
                "root_dir": str(empty_root),
                "selection": {
                    "mode": "group_sample",
                    "patterns": ["**/game.tw-pddl"],
                    "group_by": "alfworld_task_type",
                    "n_per_group": 1,
                    "seed": 7,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="resolved zero ALFWorld gamefiles"):
        select_gamefiles(
            data_root="",
            split="debug",
            seed=11,
            task_set_file=str(manifest_path),
        )
