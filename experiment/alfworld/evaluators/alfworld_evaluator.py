"""ALFWorld evaluator."""
from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, Optional

from .base_evaluator import BaseEvaluator


class ALFWorldEvaluator(BaseEvaluator):
    """
    Runs ALFWorld episodes via ALFWorldKernelAdapter and logs results.

    Gamefile control: ALFWorldEnvWrapper.reset(gamefile=...) pins each episode
    to a specific task file. Pass gamefile to run_episode() or run_batch() to
    use DatasetSampler-provided paths directly.
    """

    def __init__(self, adapter, result_writer=None, token_callback: Optional[Callable] = None):
        """
        Args:
            adapter: ALFWorldKernelAdapter instance
            result_writer: Optional ResultWriter; if provided, each episode is
                           written immediately after completion
            token_callback: Optional callable(result) -> dict passed to adapter
        """
        self.adapter = adapter
        self.result_writer = result_writer
        self.token_callback = token_callback

    def run_episode(
        self,
        gamefile: str = "",
        run_id: str = "",
        experiment_id: str = "",
        episode_id: str = "",
        model_name: str = "",
        max_steps: int = 50,
        capture_states: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run one episode, optionally pinned to a specific gamefile.
        """
        if not episode_id:
            episode_id = str(uuid.uuid4())[:8]

        result = self.adapter.run_episode(
            max_steps=max_steps,
            episode_id=episode_id,
            run_id=run_id,
            experiment_id=experiment_id,
            model_name=model_name,
            token_callback=self.token_callback,
            gamefile=gamefile or None,
            capture_states=capture_states,
        )

        # Prefer gamefile from infos over the hint argument
        if not result.get("gamefile") and gamefile:
            result["gamefile"] = gamefile

        if self.result_writer is not None:
            self.result_writer.write_episode(result)

        return result

    def run_batch(self, gamefiles, **kwargs) -> list:
        import os
        import time

        results = []
        total = len(gamefiles)
        system_label = kwargs.get("run_id", "") or kwargs.get("experiment_id", "") or "?"
        for idx, gf in enumerate(gamefiles, 1):
            task_name = os.path.basename(os.path.dirname(gf)) or os.path.basename(gf)
            print(f"[{system_label}] episode {idx}/{total}  {task_name}", flush=True)
            t0 = time.monotonic()
            result = self.run_episode(gamefile=gf, **kwargs)
            elapsed = time.monotonic() - t0
            success = result.get("success", False)
            steps = result.get("steps", "?")
            status = "✓" if success else "✗"
            print(
                f"[{system_label}] episode {idx}/{total}  {status}  steps={steps}  {elapsed:.1f}s",
                flush=True,
            )
            results.append(result)
            if idx < total:
                time.sleep(4.5)
        return results

    def finalize(self) -> Dict[str, Any]:
        """Compute and write summary for all episodes written so far."""
        if self.result_writer is None:
            return {}
        episodes = []
        path = self.result_writer.episodes_path
        if path.exists():
            import json
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        episodes.append(json.loads(line))
        summary = self.aggregate(episodes)
        self.result_writer.write_summary(summary)
        return summary
