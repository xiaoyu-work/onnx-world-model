# @agent-file
# @agent-purpose: Tests the Pipeline and PipelineSession wrappers: manifest contract exposure, provider selection, graph-optimization levels, single-stage runs, stage subsets, and unknown-input rejection.
# @agent-public-api: none
# @agent-invariants: Every test needs the `pipeline_path` fixture, so the whole module skips unless the `mobius` exporter is installed. It asserts that an incompatible provider raises while a requested CPU fallback is preserved.
# @agent-side-effects: Loads an exported pipeline package from a pytest temporary directory and runs ONNX Runtime inference.

from __future__ import annotations

import json
import shutil
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
    assert pipeline.execution_providers == {
        "dynamics": ("CPUExecutionProvider",)
    }


def test_pipeline_provider_selection(pipeline_path: Path):
    pipeline = Pipeline(
        pipeline_path,
        providers=["cuda", "cpu"],
        provider_options={"cpu": {"use_arena": True}},
    )

    assert pipeline.execution_providers["dynamics"] == ("CPUExecutionProvider",)


def test_pipeline_rejects_incompatible_provider(pipeline_path: Path):
    with pytest.raises(WorldModelError, match="no execution provider compatible"):
        Pipeline(pipeline_path, providers=["cuda"])


def test_pipeline_keeps_requested_cpu_fallback(tmp_path: Path, pipeline_path: Path):
    """A component preferring one accelerator must still accept a CPU fallback.

    Exporting with ``mobius build --ep cuda`` writes
    ``preferred_execution_providers: ["cuda"]`` for every component. Those are
    placement preferences, not an allowlist, so an explicitly requested CPU
    provider has to survive; dropping it makes ONNX Runtime refuse any node the
    accelerator does not implement.
    """
    package = tmp_path / "package"
    shutil.copytree(pipeline_path, package)
    manifest_path = package / "pipeline.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    for component in document["manifest"]["components"]:
        component["preferred_execution_providers"] = ["cuda"]
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    pipeline = Pipeline(package, providers=["cuda", "cpu"])

    for providers in pipeline.execution_providers.values():
        assert "CPUExecutionProvider" in providers


def test_pipeline_rejects_unknown_graph_optimization(pipeline_path: Path):
    with pytest.raises(ValueError, match="graph_optimization"):
        Pipeline(pipeline_path, graph_optimization="aggressive")


def test_pipeline_accepts_graph_optimization_levels(pipeline_path: Path):
    for level in ("disabled", "basic", "extended", "all"):
        pipeline = Pipeline(pipeline_path, graph_optimization=level)
        assert pipeline.execution_providers


def test_runs_pipeline_stage(pipeline_path: Path):
    session = Pipeline(
        pipeline_path,
        device_outputs=True,
    ).create_session()

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
