"""
Unit tests for ALFWorldEnvWrapper.

Tests the environment wrapper functionality.
"""
import pytest
from environments.env_wrapper import ALFWorldEnvWrapper


def test_env_wrapper_init_with_dict():
    """Test initializing wrapper with config dictionary."""
    config = {
        "env": {
            "type": "AlfredTWEnv",
            "data_path": "/test/path",
            "train_eval": "eval_out_of_distribution"
        }
    }

    wrapper = ALFWorldEnvWrapper(config_dict=config)
    assert wrapper.config["env"]["type"] == "AlfredTWEnv"
    assert wrapper.config["env"]["data_path"] == "/test/path"
    assert wrapper.config["env"]["train_eval"] == "eval_out_of_distribution"
    assert "dataset" in wrapper.config
    assert wrapper.env is None


def test_env_wrapper_default_config():
    """Test wrapper uses default config when none provided."""
    wrapper = ALFWorldEnvWrapper()
    assert "env" in wrapper.config
    assert wrapper.config["env"]["type"] == "AlfredTWEnv"
    assert "dataset" in wrapper.config


def test_env_wrapper_not_initialized():
    """Test that methods raise error when env not initialized."""
    wrapper = ALFWorldEnvWrapper()

    with pytest.raises(RuntimeError, match="Environment not initialized"):
        wrapper.reset()

    with pytest.raises(RuntimeError, match="Environment not initialized"):
        wrapper.step(["look"])


def test_env_wrapper_init_uses_get_environment(monkeypatch):
    calls = {}

    class FakeBatchEnv:
        def reset(self):
            return ["obs"], {"admissible_commands": [["look"]]}

        def step(self, actions):
            return ["obs"], [0.0], [False], {"admissible_commands": [["look"]]}

        def close(self):
            calls["closed"] = True

    class FakeEnv:
        def __init__(self, config, train_eval):
            calls["config"] = config
            calls["train_eval"] = train_eval

        def init_env(self, batch_size):
            calls["batch_size"] = batch_size
            return FakeBatchEnv()

    monkeypatch.setattr(
        "alfworld.agents.environment.get_environment",
        lambda env_type: FakeEnv,
    )

    wrapper = ALFWorldEnvWrapper(config_dict={"env": {"type": "AlfredTWEnv", "train_eval": "eval_out_of_distribution"}})
    wrapper.init(batch_size=2)

    assert calls["train_eval"] == "eval_out_of_distribution"
    assert calls["batch_size"] == 2
    assert wrapper.reset()[0] == ["obs"]


def test_env_wrapper_reset_with_tw_gamefile_reuses_env(monkeypatch):
    """Switching gamefiles reuses AlfredTWEnv (no collect_game_files), but
    re-registers the gym env via init_env(). Only batch-size mismatch or first
    call triggers a full _build_env (new AlfredTWEnv instance)."""
    builds = []
    init_env_calls = []

    class FakeBatchEnv:
        def __init__(self, owner):
            self.owner = owner

        def reset(self):
            gamefile = self.owner.game_files[0] if getattr(self.owner, "game_files", None) else "random"
            return [f"obs:{gamefile}"], {
                "admissible_commands": [["look"]],
                "extra.gamefile": [gamefile],
            }

        def step(self, actions):
            return ["obs"], [0.0], [False], {"admissible_commands": [["look"]]}

        def close(self):
            pass

    class FakeEnv:
        def __init__(self, config, train_eval):
            self.config = config
            self.train_eval = train_eval
            self.game_files = ["random"]
            builds.append(self)

        def init_env(self, batch_size):
            self.batch_size = batch_size
            init_env_calls.append((self, batch_size))
            return FakeBatchEnv(self)

    monkeypatch.setattr(
        "alfworld.agents.environment.get_environment",
        lambda env_type: FakeEnv,
    )

    wrapper = ALFWorldEnvWrapper(
        config_dict={"env": {"type": "AlfredTWEnv", "train_eval": "eval_out_of_distribution"}}
    )
    wrapper.init(batch_size=2)
    assert len(builds) == 1
    assert len(init_env_calls) == 1

    gamefile = "/tmp/task/game.tw-pddl"
    obs, infos = wrapper.reset(gamefile=gamefile)

    # First gamefile reset: batch_size mismatch (2→1) triggers a full rebuild
    assert obs == [f"obs:{gamefile}"]
    assert len(builds) == 2           # new AlfredTWEnv created
    assert len(init_env_calls) == 2   # init_env called again
    assert builds[1].game_files == [gamefile]

    # Second gamefile reset: reuses AlfredTWEnv, but calls init_env again
    gamefile2 = "/tmp/task2/game.tw-pddl"
    obs2, infos2 = wrapper.reset(gamefile=gamefile2)
    assert obs2 == [f"obs:{gamefile2}"]
    assert len(builds) == 2           # still 2 — no new AlfredTWEnv
    assert len(init_env_calls) == 3   # init_env called again for new gamefile
    assert builds[1].game_files == [gamefile2]

    # Reset without gamefile: full rebuild with original batch_size
    wrapper.reset()
    assert len(builds) == 3
    assert builds[2].batch_size == 2


def test_env_wrapper_reset_with_thor_gamefile_converts_to_traj_data(monkeypatch):
    builds = []

    class FakeBatchEnv:
        def __init__(self, owner):
            self.owner = owner

        def reset(self):
            task_file = self.owner.json_file_list[0] if getattr(self.owner, "json_file_list", None) else "random"
            return [f"obs:{task_file}"], {
                "admissible_commands": [["look"]],
                "extra.gamefile": [task_file],
            }

        def step(self, actions):
            return ["obs"], [0.0], [False], {"admissible_commands": [["look"]]}

        def close(self):
            self.owner.close_calls += 1

    class FakeEnv:
        def __init__(self, config, train_eval):
            self.config = config
            self.train_eval = train_eval
            self.json_file_list = ["default_traj_data.json"]
            self.close_calls = 0
            builds.append(self)

        def init_env(self, batch_size):
            self.batch_size = batch_size
            return FakeBatchEnv(self)

    monkeypatch.setattr(
        "alfworld.agents.environment.get_environment",
        lambda env_type: FakeEnv,
    )

    init_env_calls = []
    orig_init_env = None

    wrapper = ALFWorldEnvWrapper(
        config_dict={"env": {"type": "AlfredThorEnv", "train_eval": "eval_out_of_distribution"}}
    )
    wrapper.init(batch_size=1)
    assert len(builds) == 1

    # Patch init_env on the existing alfworld_env to track calls
    orig_init_env = builds[0].init_env
    def tracked_init_env(batch_size):
        init_env_calls.append(batch_size)
        return orig_init_env(batch_size)
    builds[0].init_env = tracked_init_env

    gamefile = "/tmp/task/game.tw-pddl"
    expected_task_file = "/tmp/task/traj_data.json"
    obs, infos = wrapper.reset(gamefile=gamefile)

    # batch_size already 1 and alfworld_env exists: reuse AlfredTWEnv,
    # but re-register gym env via init_env()
    assert obs == [f"obs:{expected_task_file}"]
    assert infos["extra.gamefile"] == [expected_task_file]
    assert len(builds) == 1           # no new AlfredThorEnv — reused
    assert len(init_env_calls) == 1   # init_env called once for the new gamefile
    assert builds[0].json_file_list == [expected_task_file]


def test_env_wrapper_reset_with_unsupported_env_type_raises(monkeypatch):
    class FakeBatchEnv:
        def reset(self):
            return ["obs"], {"admissible_commands": [["look"]]}

        def step(self, actions):
            return ["obs"], [0.0], [False], {"admissible_commands": [["look"]]}

        def close(self):
            return None

    class FakeEnv:
        def __init__(self, config, train_eval):
            self.config = config
            self.train_eval = train_eval

        def init_env(self, batch_size):
            return FakeBatchEnv()

    monkeypatch.setattr(
        "alfworld.agents.environment.get_environment",
        lambda env_type: FakeEnv,
    )

    wrapper = ALFWorldEnvWrapper(
        config_dict={"env": {"type": "AlfredHybrid", "train_eval": "eval_out_of_distribution"}}
    )
    wrapper.init(batch_size=1)

    with pytest.raises(NotImplementedError, match="gamefile-controlled reset is not supported"):
        wrapper.reset(gamefile="/tmp/task/game.tw-pddl")


# Note: Full integration tests with actual ALFWorld environment
# require ALFWorld to be installed and configured with data.
# These would be added in integration test suite.
