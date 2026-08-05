from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from onnx_world_model import WorldModel, WorldModelError


def _inputs(batch_size: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((batch_size, 4), dtype=np.float32),
        np.zeros((batch_size, 2), dtype=np.float32),
        np.zeros((batch_size, 3), dtype=np.float32),
    )


def test_loads_mobius_contract(world_model_path: Path):
    model = WorldModel(world_model_path)

    assert [spec.name for spec in model.metadata.inputs] == [
        "observation",
        "action",
        "state",
    ]
    assert [spec.name for spec in model.metadata.outputs] == [
        "next_state",
        "observation_prediction",
        "reward",
        "continuation",
    ]
    assert model.metadata.inputs[0].shape == (-1, 4)


def test_stateless_step_matches_reference(world_model_path: Path):
    model = WorldModel(world_model_path)
    observation, action, state = _inputs(batch_size=2)

    output = model.step(observation, action, state)

    np.testing.assert_allclose(
        output.next_state,
        np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        output.observation_prediction,
        np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(output.reward, np.full((2, 1), 0.5, np.float32))
    np.testing.assert_array_equal(
        output.continuation,
        np.full((2, 1), 0.5, np.float32),
    )


def test_rollout_preserves_and_resets_state(world_model_path: Path):
    model = WorldModel(world_model_path)
    rollout = model.create_rollout()
    observation, action, _ = _inputs()

    first = rollout.step(observation, action)
    second = rollout.step(observation, action)

    np.testing.assert_allclose(first.next_state, [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(second.next_state, [[0.2, 0.4, 0.6]])
    np.testing.assert_allclose(rollout.state, second.next_state)

    rollout.reset(batch_size=1)
    np.testing.assert_array_equal(rollout.state, np.zeros((1, 3), np.float32))
    rollout.reset()
    assert rollout.state is None


def test_rejects_incorrect_input_dtype(world_model_path: Path):
    model = WorldModel(world_model_path)
    observation, action, state = _inputs()

    with pytest.raises(WorldModelError, match="data type"):
        model.step(observation.astype(np.float64), action, state)
