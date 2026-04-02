"""
ALFWorld Environment Wrapper

Provides a clean interface to ALFWorld environments with configuration management.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

import yaml


class ALFWorldEnvWrapper:
    """Wrapper for ALFWorld environment with configuration support."""

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize ALFWorld environment wrapper.

        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to config_path)
        """
        self.env = None
        self.alfworld_env = None
        self.config = self._load_config(config_path, config_dict)
        self._default_batch_size = 1
        self._active_batch_size = 1
        self._gamefiles_override: tuple[str, ...] | None = None

    @staticmethod
    def _default_config_path() -> Path:
        root = Path(__file__).resolve().parents[3]
        return root / "alfworld" / "configs" / "base_config.yaml"

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = ALFWorldEnvWrapper._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _ensure_alfworld_data_env() -> None:
        if os.environ.get("ALFWORLD_DATA"):
            return

        try:
            import alfworld.info  # noqa: F401
        except ImportError:
            return

    def _load_config(self, config_path: Optional[str], config_dict: Optional[Dict]) -> Dict:
        """Load configuration from file or dict."""
        self._ensure_alfworld_data_env()

        base_config: Dict[str, Any] = {}
        default_config_path = self._default_config_path()
        if default_config_path.exists():
            with open(default_config_path, "r", encoding="utf-8") as f:
                base_config = yaml.safe_load(f) or {}

        loaded_config = base_config

        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = yaml.safe_load(f) or {}

        if config_dict:
            if loaded_config:
                loaded_config = self._deep_merge(loaded_config, config_dict)
            else:
                loaded_config = deepcopy(config_dict)

        if loaded_config:
            env_cfg = loaded_config.setdefault("env", {})
            env_cfg.setdefault("type", "AlfredTWEnv")
            env_cfg.setdefault("train_eval", "eval_out_of_distribution")
            return loaded_config

        # Fallback for environments where the repository copy of ALFWorld is absent.
        data_root = os.environ.get("ALFWORLD_DATA", "")
        return {
            "dataset": {
                "data_path": f"{data_root}/json_2.1.1/train",
                "eval_id_data_path": f"{data_root}/json_2.1.1/valid_seen",
                "eval_ood_data_path": f"{data_root}/json_2.1.1/valid_unseen",
                "num_train_games": -1,
                "num_eval_games": -1,
            },
            "logic": {
                "domain": f"{data_root}/logic/alfred.pddl",
                "grammar": f"{data_root}/logic/alfred.twl2",
            },
            "env": {
                "type": "AlfredTWEnv",
                "train_eval": "eval_out_of_distribution",
                "task_types": [1, 2, 3, 4, 5, 6],
                "domain_randomization": False,
                "expert_type": "handcoded",
                "goal_desc_human_anns_prob": 0.0,
            },
            "general": {"training_method": "dagger"},
            "dagger": {"training": {"max_nb_steps_per_episode": 50}},
        }

    def init(self, batch_size: int = 1) -> None:
        """
        Initialize the ALFWorld environment.

        Args:
            batch_size: Number of parallel environments
        """
        self._default_batch_size = batch_size
        self._build_env(batch_size=batch_size, gamefiles=None)

    def _build_env(self, batch_size: int, gamefiles: Optional[List[str]]) -> None:
        from alfworld.agents.environment import get_environment

        if self.env is not None and hasattr(self.env, "close"):
            self.env.close()

        env_config = self.config.get("env", {})
        env_type = env_config.get("type", "AlfredTWEnv")
        train_eval = env_config.get("train_eval", "eval_out_of_distribution")

        env_cls = get_environment(env_type)
        self.alfworld_env = env_cls(config=self.config, train_eval=train_eval)

        if gamefiles:
            self._apply_gamefiles_override(env_type, gamefiles)

        self.env = self.alfworld_env.init_env(batch_size=batch_size)
        self._active_batch_size = batch_size
        self._gamefiles_override = tuple(gamefiles) if gamefiles else None

    def _apply_gamefiles_override(self, env_type: str, gamefiles: List[str]) -> None:
        if env_type == "AlfredTWEnv":
            self.alfworld_env.game_files = list(gamefiles)
            self.alfworld_env.num_games = len(gamefiles)
            return

        if env_type == "AlfredThorEnv":
            task_files = [self._normalize_gamefile_for_env(env_type, gamefile) for gamefile in gamefiles]
            self.alfworld_env.json_file_list = task_files
            self.alfworld_env.num_games = len(task_files)
            return

        raise NotImplementedError(f"gamefile-controlled reset is not supported for env type: {env_type}")

    @staticmethod
    def _normalize_gamefile_for_env(env_type: str, gamefile: str) -> str:
        path = Path(gamefile)
        if env_type == "AlfredThorEnv" and path.name == "game.tw-pddl":
            return str(path.with_name("traj_data.json"))
        return str(path)

    def reset(self, gamefile: Optional[str] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            gamefile: Optional ALFWorld task file to force for the next episode.
                For `AlfredTWEnv`, pass a `game.tw-pddl` path.
                For `AlfredThorEnv`, both `game.tw-pddl` and `traj_data.json` paths
                are accepted.

        Returns:
            obs: List of observation strings
            infos: Dictionary containing environment info
        """
        if self.env is None and gamefile is None:
            raise RuntimeError("Environment not initialized. Call init() first.")

        if gamefile is not None:
            env_type = self.config.get("env", {}).get("type", "AlfredTWEnv")
            normalized = self._normalize_gamefile_for_env(env_type, gamefile)
            desired_override = (normalized,)

            if self.alfworld_env is None or self._active_batch_size != 1:
                # First call or batch size mismatch: full rebuild
                self._build_env(batch_size=1, gamefiles=[normalized])
            elif self._gamefiles_override != desired_override:
                # Reuse AlfredTWEnv (skip expensive collect_game_files),
                # but re-register the gym env with the new gamefile.
                if self.env is not None and hasattr(self.env, "close"):
                    self.env.close()
                self._apply_gamefiles_override(env_type, [normalized])
                self.env = self.alfworld_env.init_env(batch_size=1)
                self._gamefiles_override = desired_override
        elif self._gamefiles_override is not None:
            self._build_env(batch_size=self._default_batch_size, gamefiles=None)

        return self.env.reset()

    def step(self, actions: List[str]) -> Tuple[List[str], List[float], List[bool], Dict[str, Any]]:
        """
        Execute actions in the environment.

        Args:
            actions: List of action strings

        Returns:
            obs: List of observation strings
            rewards: List of rewards
            dones: List of done flags
            infos: Dictionary containing environment info
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call init() first.")

        return self.env.step(actions)

    def close(self) -> None:
        """Close the environment."""
        if self.env is not None:
            if hasattr(self.env, "close"):
                self.env.close()
            self.env = None
        self.alfworld_env = None
