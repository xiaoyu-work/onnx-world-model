from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from onnx_world_model import (
    LatentDynamicsModel,
    OnnxModel,
    WorldModelError,
    available_execution_providers,
    supported_pipeline_capabilities,
)


def _inputs(batch_size: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((batch_size, 4), dtype=np.float32),
        np.zeros((batch_size, 2), dtype=np.float32),
        np.zeros((batch_size, 3), dtype=np.float32),
    )


def test_loads_mobius_contract(world_model_path: Path):
    model = LatentDynamicsModel(world_model_path)

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


def test_generic_named_tensor_model(world_model_path: Path):
    model = OnnxModel(world_model_path)
    observation, action, state = _inputs(batch_size=2)

    output = model.run(
        {
            "observation": observation,
            "action": action,
            "state": state,
        }
    )

    assert set(output) == {
        "next_state",
        "observation_prediction",
        "reward",
        "continuation",
    }
    np.testing.assert_allclose(output["next_state"], [[0.1, 0.2, 0.3]] * 2)
    assert model.metadata.execution_providers == ("CPUExecutionProvider",)


def test_configures_cpu_execution_provider(world_model_path: Path):
    model = OnnxModel(
        world_model_path,
        providers=["cpu"],
        provider_options={"cpu": {"use_arena": False}},
    )

    assert model.metadata.execution_providers == ("CPUExecutionProvider",)


def test_explicit_provider_fallback(world_model_path: Path):
    model = OnnxModel(world_model_path, providers=["cuda", "cpu"])

    assert model.metadata.execution_providers == ("CPUExecutionProvider",)


def test_rejects_unavailable_provider(world_model_path: Path):
    with pytest.raises(WorldModelError, match="requested execution providers"):
        OnnxModel(world_model_path, providers=["cuda"])


def test_lists_available_execution_providers():
    providers = available_execution_providers()

    assert "CPUExecutionProvider" in providers


def test_advertises_supported_pipeline_capabilities():
    capabilities = supported_pipeline_capabilities()

    assert {
        "classifier_free_guidance",
        "conditioned_diffusion",
        "iterative_scheduler",
        "loop_carried_state",
        "packed_sequence_program",
    } <= set(capabilities)
    assert "streaming" not in capabilities


def test_registers_available_non_cpu_provider(world_model_path: Path):
    if "AzureExecutionProvider" not in available_execution_providers():
        pytest.skip("Azure execution provider is unavailable")

    model = OnnxModel(world_model_path, providers=["azure", "cpu"])
    observation, action, state = _inputs()
    output = model.run(
        {"observation": observation, "action": action, "state": state}
    )

    assert model.metadata.execution_providers == (
        "AzureExecutionProvider",
        "CPUExecutionProvider",
    )
    np.testing.assert_allclose(output["next_state"], [[0.1, 0.2, 0.3]])


def test_generic_model_rejects_missing_input(world_model_path: Path):
    model = OnnxModel(world_model_path)
    observation, action, _ = _inputs()

    with pytest.raises(WorldModelError, match="missing input tensor 'state'"):
        model.run({"observation": observation, "action": action})


def test_stateless_step_matches_reference(world_model_path: Path):
    model = LatentDynamicsModel(world_model_path)
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
    model = LatentDynamicsModel(world_model_path)
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
    model = LatentDynamicsModel(world_model_path)
    observation, action, state = _inputs()

    with pytest.raises(WorldModelError, match="data type"):
        model.step(observation.astype(np.float64), action, state)
