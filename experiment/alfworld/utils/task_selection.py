"""Shared task-selection logic for ALFWorld experiment entrypoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from experiment.alfworld.utils.dataset_sampler import DatasetSampler
from experiment.common.task_registry import resolve_task_set


def _raise_for_empty_selection(
    *,
    source: str,
    task_set_file: str = "",
    selection_root_dir: str = "",
    selection_mode: str = "",
    gamefiles_file: str = "",
    data_root: str = "",
    split: str = "",
) -> None:
    """Raise a clear error when ALFWorld selection resolves no runnable gamefiles."""
    if source == "task_set_manifest":
        raise ValueError(
            "Task set resolved zero ALFWorld gamefiles: "
            f"{Path(task_set_file).resolve()} "
            f"(root_dir={selection_root_dir or '<unset>'}, selection_mode={selection_mode or '<unset>'}). "
            "Check ALFWORLD_DATA and the manifest patterns."
        )
    if source == "explicit_gamefiles_file":
        raise ValueError(
            "Explicit gamefiles file resolved zero ALFWorld gamefiles: "
            f"{Path(gamefiles_file).resolve()}. "
            "Check the upstream selection step that produced this file."
        )
    raise ValueError(
        "Dataset sampler resolved zero ALFWorld gamefiles: "
        f"data_root={data_root or '<unset>'}, split={split}. "
        "Check ALFWORLD_DATA and that the dataset contains game.tw-pddl files."
    )


def select_gamefiles(
    *,
    data_root: str,
    split: str,
    seed: int,
    limit: int = 0,
    gamefiles_file: str = "",
    task_set_file: str = "",
) -> tuple[List[str], Dict[str, Any]]:
    """Resolve ALFWorld gamefiles from a manifest, explicit JSON, or sampler."""
    if task_set_file:
        task_set = resolve_task_set(task_set_file, limit=limit, seed=seed)
        gamefiles = list(task_set["task_paths"])
        selection_meta = {
            "selection_source": "task_set_manifest",
            "task_set_file": str(Path(task_set_file).resolve()),
            "task_set_name": task_set.get("task_set", ""),
            "selection_mode": task_set.get("selection_mode", ""),
            "selection_seed": task_set.get("selection_seed", seed),
            "selection_root_dir": task_set.get("root_dir", ""),
        }
        if not gamefiles:
            _raise_for_empty_selection(
                source="task_set_manifest",
                task_set_file=task_set_file,
                selection_root_dir=selection_meta["selection_root_dir"],
                selection_mode=selection_meta["selection_mode"],
            )
        return gamefiles, selection_meta

    if gamefiles_file:
        gamefiles = json.loads(Path(gamefiles_file).read_text(encoding="utf-8"))
        if not isinstance(gamefiles, list):
            raise ValueError(f"gamefiles_file must contain a JSON array: {gamefiles_file}")
        if limit > 0:
            gamefiles = gamefiles[:limit]
        resolved_gamefiles = [str(gamefile) for gamefile in gamefiles]
        if not resolved_gamefiles:
            _raise_for_empty_selection(
                source="explicit_gamefiles_file",
                gamefiles_file=gamefiles_file,
            )
        return resolved_gamefiles, {
            "selection_source": "explicit_gamefiles_file",
            "gamefiles_file": str(Path(gamefiles_file).resolve()),
        }

    if not data_root:
        raise ValueError("data_root is required when task_set_file and gamefiles_file are not provided")

    sampler = DatasetSampler(data_root)
    if split == "debug":
        gamefiles = sampler.debug_split(seed=seed)
    elif split == "formal":
        gamefiles = sampler.formal_split(seed=seed)
    else:
        raise ValueError(f"Unsupported ALFWorld split: {split}")
    if limit > 0:
        gamefiles = gamefiles[:limit]
    if not gamefiles:
        _raise_for_empty_selection(
            source="dataset_sampler",
            data_root=data_root,
            split=split,
        )
    return gamefiles, {
        "selection_source": "dataset_sampler",
        "data_root": data_root,
        "split": split,
        "selection_seed": seed,
    }
