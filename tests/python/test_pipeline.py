from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from onnx_world_model import Pipeline, WorldModelError


def _inputs(batch_size: int = 1) -> dict[str, np.ndarray]:
    return {
        "observation": np.zeros((batch_size, 4), dtype=np.float32),
        "action": np.zeros((batch_size, 2), dtype=np.float32),
        "state": np.zeros((batch_size, 3), dtype=np.float32),
    }


def test_loads_pipeline_contract(pipeline_path: Path):
    pipeline = Pipeline(pipeline_path)

    assert pipeline.profile == {"name": "latent-dynamics", "version": "1.0"}
    assert [spec.name for spec in pipeline.inputs] == [
        "action",
        "observation",
        "state",
    ]
    assert {spec.name for spec in pipeline.outputs} == {
        "next_state",
        "observation_prediction",
        "reward",
        "continuation",
    }
    assert pipeline.stages[0].name == "step"


def test_runs_pipeline_stage(pipeline_path: Path):
    session = Pipeline(pipeline_path).create_session()

    output = session.run_stage("step", _inputs(batch_size=2))

    np.testing.assert_allclose(output["next_state"], [[0.1, 0.2, 0.3]] * 2)
    np.testing.assert_array_equal(
        output["observation_prediction"],
        [[1.0, 2.0, 3.0, 4.0]] * 2,
    )


def test_runs_selected_pipeline(pipeline_path: Path):
    session = Pipeline(pipeline_path).create_session()

    output = session.run(_inputs(), stages=["step"])

    np.testing.assert_allclose(output["reward"], [[0.5]])


def test_pipeline_rejects_unknown_input(pipeline_path: Path):
    session = Pipeline(pipeline_path).create_session()

    with pytest.raises(WorldModelError, match="Unknown pipeline input"):
        session.run_stage("step", {"unknown": np.zeros((1,), dtype=np.float32)})
