"""Task-set manifest helpers shared across experiment environments."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


_ALFWORLD_TASK_TYPE_ORDER = {
    "pick_and_place": 0,
    "pick_clean_then_place": 1,
    "pick_heat_then_place": 2,
    "pick_cool_then_place": 3,
    "look_at_obj": 4,
    "pick_two_obj": 5,
    "unknown": 99,
}


def _expand_path(value: str, *, base_dir: Path) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(value))
    path = Path(expanded)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _manifest_root(manifest: Dict[str, Any], *, manifest_path: Path) -> Path:
    root_dir = manifest.get("root_dir", ".")
    return _expand_path(str(root_dir), base_dir=manifest_path.parent)


def _load_manifest(manifest_path: str | Path) -> tuple[Path, Dict[str, Any]]:
    resolved_path = Path(manifest_path).resolve()
    manifest = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Task manifest must be a JSON object: {resolved_path}")
    return resolved_path, manifest


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen: set[str] = set()
    deduped: List[Path] = []
    for path in paths:
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path.resolve())
    return deduped


def _resolve_explicit_items(items: Iterable[str], *, root_dir: Path) -> List[Path]:
    return _dedupe_paths(_expand_path(item, base_dir=root_dir) for item in items)


def _resolve_glob_items(
    *,
    root_dir: Path,
    patterns: Iterable[str],
    exclude_patterns: Iterable[str],
) -> List[Path]:
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(path.resolve() for path in root_dir.glob(pattern) if path.is_file())

    excluded: set[str] = set()
    for pattern in exclude_patterns:
        excluded.update(str(path.resolve()) for path in root_dir.glob(pattern) if path.is_file())

    return [path for path in _dedupe_paths(sorted(candidates)) if str(path) not in excluded]


def _alfworld_task_type(path: Path) -> str:
    from experiment.alfworld.utils.dataset_sampler import task_type_from_gamefile

    return task_type_from_gamefile(str(path))


def _webarena_site_group(path: Path) -> str:
    from experiment.webarena.utils.task_loader import load_task_config

    config = load_task_config(path)
    sites = config.get("sites", [])
    if not isinstance(sites, list) or not sites:
        return "unknown"
    return "+".join(str(site) for site in sites)


def _group_key_for_path(path: Path, *, grouper: str) -> str:
    if grouper == "alfworld_task_type":
        return _alfworld_task_type(path)
    if grouper == "webarena_site":
        return _webarena_site_group(path)
    if grouper == "file_parent":
        return path.parent.name
    if grouper == "file_stem":
        return path.stem
    raise ValueError(f"Unsupported task group_by value: {grouper}")


def _sort_group_names(group_names: Iterable[str], *, grouper: str) -> List[str]:
    names = list(group_names)
    if grouper == "alfworld_task_type":
        return sorted(names, key=lambda name: (_ALFWORLD_TASK_TYPE_ORDER.get(name, 999), name))
    return sorted(names)


def _group_sample(
    candidates: List[Path],
    *,
    group_by: str,
    n_per_group: int,
    seed: int,
) -> List[Path]:
    grouped: Dict[str, List[Path]] = {}
    for candidate in candidates:
        group = _group_key_for_path(candidate, grouper=group_by)
        grouped.setdefault(group, []).append(candidate)

    rng = random.Random(seed)
    sampled: List[Path] = []
    for group in _sort_group_names(grouped, grouper=group_by):
        group_candidates = list(sorted(grouped[group]))
        rng.shuffle(group_candidates)
        sampled.extend(group_candidates[:n_per_group])
    return sampled


def resolve_task_set(
    manifest_path: str | Path,
    *,
    limit: int = 0,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Resolve a JSON task-set manifest into explicit task paths."""
    resolved = preview_task_set(manifest_path, limit=limit, seed=seed)
    missing_paths = list(resolved.get("missing_task_paths", []))
    if missing_paths:
        preview = ", ".join(missing_paths[:5])
        suffix = " ..." if len(missing_paths) > 5 else ""
        raise ValueError(f"Task manifest resolved missing files: {preview}{suffix}")
    return resolved


def preview_task_set(
    manifest_path: str | Path,
    *,
    limit: int = 0,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Resolve a manifest without failing when selected files are still missing."""
    resolved_manifest_path, manifest = _load_manifest(manifest_path)
    root_dir = _manifest_root(manifest, manifest_path=resolved_manifest_path)
    selection = manifest.get("selection", {})
    if not isinstance(selection, dict):
        raise ValueError(f"selection must be an object: {resolved_manifest_path}")

    mode = str(selection.get("mode", "explicit"))
    items = selection.get("items", [])
    patterns = selection.get("patterns", [])
    exclude_patterns = selection.get("exclude", [])
    if not isinstance(items, list) or not isinstance(patterns, list) or not isinstance(exclude_patterns, list):
        raise ValueError(f"items/patterns/exclude must be arrays: {resolved_manifest_path}")

    if items:
        candidates = _resolve_explicit_items(items, root_dir=root_dir)
    else:
        pattern_list = patterns or ["**/*"]
        candidates = _resolve_glob_items(
            root_dir=root_dir,
            patterns=[str(pattern) for pattern in pattern_list],
            exclude_patterns=[str(pattern) for pattern in exclude_patterns],
        )

    resolved_seed = int(seed if seed is not None else selection.get("seed", 42))
    if mode == "explicit":
        task_paths = candidates
    elif mode == "glob":
        task_paths = candidates
    elif mode == "group_sample":
        group_by = str(selection.get("group_by", "")).strip()
        n_per_group = int(selection.get("n_per_group", 0))
        if not group_by or n_per_group <= 0:
            raise ValueError(
                f"group_sample requires group_by and positive n_per_group: {resolved_manifest_path}"
            )
        task_paths = _group_sample(
            candidates,
            group_by=group_by,
            n_per_group=n_per_group,
            seed=resolved_seed,
        )
    else:
        raise ValueError(f"Unsupported task selection mode: {mode}")

    if limit > 0:
        task_paths = task_paths[:limit]

    existing_paths = [str(path) for path in task_paths if path.exists() and path.is_file()]
    missing_paths = [str(path) for path in task_paths if not path.exists() or not path.is_file()]

    return {
        "env_name": manifest.get("env_name", ""),
        "task_set": manifest.get("task_set", resolved_manifest_path.stem),
        "manifest_path": str(resolved_manifest_path),
        "root_dir": str(root_dir),
        "selection_mode": mode,
        "selection_seed": resolved_seed,
        "n_tasks": len(task_paths),
        "task_paths": [str(path) for path in task_paths],
        "existing_task_paths": existing_paths,
        "missing_task_paths": missing_paths,
        "metadata": manifest.get("metadata", {}),
    }


def load_task_paths(
    manifest_path: str | Path,
    *,
    limit: int = 0,
    seed: int | None = None,
) -> List[str]:
    """Resolve a task-set manifest and return only the task path list."""
    resolved = resolve_task_set(manifest_path, limit=limit, seed=seed)
    return list(resolved["task_paths"])
