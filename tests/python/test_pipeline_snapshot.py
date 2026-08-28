"""In-memory snapshot, restore, fork, and named checkpoints of a ``PipelineSession``.

The component is a tiny hand-built graph, so the recurrent-state round trip
runs through the actual runtime and ONNX Runtime rather than a stub. Named
checkpoints are in-memory transaction markers over the same capture; they are
not paged KV blocks and are never written to disk.
"""

# @agent-file
# @agent-purpose: Tests the PipelineSessionSnapshot wrapper, the PipelineSession snapshot, restore, and fork methods, and the named in-memory checkpoint methods against a small counter package built with onnx_ir.
# @agent-public-api: none
# @agent-invariants: The opaque-handle tests need no package and always run. The state tests build their graph in-process with `pytest.importorskip("onnx_ir")`, so they skip rather than fail when that optional dependency is absent, and they never need the Mobius exporter. Restoring a snapshot into a session from a second, separately loaded Pipeline must raise WorldModelError even though both read the same directory. Named checkpoints are asserted through the public session surface only: empty and unknown names raise WorldModelError, reset clears them, and a fork starts without any.
# @agent-side-effects: Writes an ONNX model and package files into pytest temporary directories and runs ONNX Runtime inference.

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from onnx_world_model import (
    Pipeline,
    PipelineSessionSnapshot,
    WorldModelError,
    _native,
)

_OPSET = 18

_MANIFEST = {
    "format": "mobius-pipeline",
    "schema_version": "1.1",
    "manifest": {
        "schema_version": "1.1",
        "components": [
            {
                "name": "counter",
                "role": "dynamics",
                "run_on": "step",
                "inputs": [
                    {"name": "state", "dtype": "FLOAT", "shape": [1]}
                ],
                "outputs": [
                    {"name": "next_state", "dtype": "FLOAT", "shape": [1]}
                ],
                "preferred_execution_providers": ["cpu"],
                "parameter_dtype": "FLOAT",
            }
        ],
        "connections": [
            {
                "source": "counter.next_state",
                "target": "counter.state",
                "recurrent": True,
            }
        ],
        "stages": [
            {
                "name": "transition",
                "kind": "state_transition",
                "components": ["counter"],
                "run_on": "step",
                "options": {"state_names": ["counter_state"]},
                "capabilities": ["loop_carried_state"],
            }
        ],
        "inputs": [
            {
                "port": "counter.state",
                "kind": "generated",
                "required": True,
                "semantic": "state.initial",
                "generator": {"kind": "zeros"},
            }
        ],
        "outputs": [{"state": "counter_state", "alias": "value"}],
        "profile": {"name": "counter-world", "version": "1.0"},
        "states": [
            {
                "name": "counter_state",
                "kind": "recurrent",
                "input": "counter.state",
                "output": "counter.next_state",
                "lifetime": "session",
                "release_after": "transition",
            }
        ],
        "required_capabilities": ["loop_carried_state"],
    },
    "component_files": {"counter": "model.onnx"},
}


@pytest.fixture
def counter_package(tmp_path: Path) -> Path:
    """A one-component package whose stage adds one to its recurrent state."""
    ir = pytest.importorskip("onnx_ir")
    state = ir.Value(
        name="state",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([1]),
    )
    one = ir.Node(
        "",
        "Constant",
        inputs=[],
        attributes=[
            ir.AttrTensor(
                "value",
                ir.tensor(np.ones((1,), dtype=np.float32), name="one"),
            )
        ],
        num_outputs=1,
    )
    increment = ir.Node(
        "", "Add", inputs=[state, one.outputs[0]], num_outputs=1
    )
    increment.outputs[0].name = "next_state"
    increment.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    increment.outputs[0].shape = ir.Shape([1])
    graph = ir.Graph(
        inputs=[state],
        outputs=[increment.outputs[0]],
        nodes=[one, increment],
        name="counter",
        opset_imports={"": _OPSET},
    )
    ir.save(
        ir.Model(graph, ir_version=10, producer_name="onnx-world-model-test"),
        tmp_path / "model.onnx",
    )
    (tmp_path / "pipeline.json").write_text(
        json.dumps(_MANIFEST),
        encoding="utf-8",
    )
    return tmp_path


def test_native_snapshot_has_no_public_constructor() -> None:
    """A snapshot can only come from a session, never from user code."""
    with pytest.raises(TypeError):
        _native.PipelineSessionSnapshot()


def test_snapshot_restores_the_recurrent_state(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()
    session.run_stage("transition")
    session.run_stage("transition")

    snapshot = session.snapshot()
    assert isinstance(snapshot, PipelineSessionSnapshot)
    assert snapshot.valid

    session.run_stage("transition")
    np.testing.assert_allclose(session.state("counter_state"), [3.0])

    session.restore(snapshot)

    np.testing.assert_allclose(session.state("counter_state"), [2.0])
    np.testing.assert_allclose(session.outputs["value"], [2.0])
    np.testing.assert_allclose(session.run_stage("transition")["value"], [3.0])


def test_snapshot_restores_a_reset_session(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()
    session.run_stage("transition")
    snapshot = session.snapshot()

    session.reset()
    assert session.state("counter_state") is None

    session.restore(snapshot)

    np.testing.assert_allclose(session.state("counter_state"), [1.0])


def test_fork_evolves_independently(counter_package: Path) -> None:
    parent = Pipeline(counter_package).create_session()
    parent.run_stage("transition")
    parent.run_stage("transition")

    child = parent.fork()
    assert child.pipeline is parent.pipeline
    np.testing.assert_allclose(child.state("counter_state"), [2.0])

    child.run_stage("transition")
    np.testing.assert_allclose(child.state("counter_state"), [3.0])
    np.testing.assert_allclose(parent.state("counter_state"), [2.0])

    parent.run_stage("transition")
    parent.run_stage("transition")
    np.testing.assert_allclose(parent.state("counter_state"), [4.0])
    np.testing.assert_allclose(child.state("counter_state"), [3.0])

    child.reset()
    assert child.state("counter_state") is None
    np.testing.assert_allclose(parent.state("counter_state"), [4.0])


def test_restore_rejects_a_snapshot_from_another_pipeline(
    counter_package: Path,
) -> None:
    """Package identity decides the fit, not the manifest text."""
    session = Pipeline(counter_package).create_session()
    session.run_stage("transition")
    snapshot = session.snapshot()

    stranger = Pipeline(counter_package).create_session()

    with pytest.raises(WorldModelError, match="different pipeline package"):
        stranger.restore(snapshot)
    assert stranger.state("counter_state") is None


def test_restore_accepts_a_shallow_copied_pipeline(
    counter_package: Path,
) -> None:
    pipeline = Pipeline(counter_package)
    copied = copy.copy(pipeline)
    source = pipeline.create_session()
    destination = copied.create_session()
    source.run_stage("transition")

    destination.restore(source.snapshot())

    np.testing.assert_allclose(destination.state("counter_state"), [1.0])


def test_checkpoint_creates_and_queries_a_name(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()
    session.run_stage("transition")

    assert session.has_checkpoint("before") is False

    session.checkpoint("before")

    assert session.has_checkpoint("before") is True
    assert session.has_checkpoint("other") is False


def test_checkpoint_survives_stage_execution_and_rewinds(
    counter_package: Path,
) -> None:
    session = Pipeline(counter_package).create_session()
    session.run_stage("transition")
    session.run_stage("transition")
    session.checkpoint("before")

    session.run_stage("transition")
    session.run_stage("transition")
    np.testing.assert_allclose(session.state("counter_state"), [4.0])
    assert session.has_checkpoint("before")

    session.restore_checkpoint("before")

    np.testing.assert_allclose(session.state("counter_state"), [2.0])
    np.testing.assert_allclose(session.outputs["value"], [2.0])
    assert session.has_checkpoint("before"), "restoring does not consume the name"
    np.testing.assert_allclose(session.run_stage("transition")["value"], [3.0])


def test_checkpoint_replaces_the_same_name(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()
    session.run_stage("transition")
    session.checkpoint("mark")

    session.run_stage("transition")
    session.run_stage("transition")
    session.checkpoint("mark")
    session.run_stage("transition")

    session.restore_checkpoint("mark")

    np.testing.assert_allclose(session.state("counter_state"), [3.0])


def test_drop_checkpoint_removes_the_name(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()
    session.run_stage("transition")
    session.checkpoint("mark")

    session.drop_checkpoint("mark")

    assert session.has_checkpoint("mark") is False
    with pytest.raises(WorldModelError, match="no checkpoint named"):
        session.restore_checkpoint("mark")
    np.testing.assert_allclose(session.state("counter_state"), [1.0])


def test_unknown_checkpoint_names_fail_loudly(counter_package: Path) -> None:
    """Restore and drop report a missing name; drop is not a silent no-op."""
    session = Pipeline(counter_package).create_session()
    session.run_stage("transition")

    with pytest.raises(WorldModelError, match="no checkpoint named"):
        session.restore_checkpoint("missing")
    with pytest.raises(WorldModelError, match="no checkpoint named"):
        session.drop_checkpoint("missing")
    np.testing.assert_allclose(session.state("counter_state"), [1.0])


def test_empty_checkpoint_names_are_rejected(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()

    for operation in (
        session.checkpoint,
        session.restore_checkpoint,
        session.drop_checkpoint,
        session.has_checkpoint,
    ):
        with pytest.raises(WorldModelError, match="must not be empty"):
            operation("")


def test_reset_clears_every_checkpoint(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()
    session.run_stage("transition")
    session.checkpoint("first")
    session.checkpoint("second")

    session.reset()

    assert session.has_checkpoint("first") is False
    assert session.has_checkpoint("second") is False
    with pytest.raises(WorldModelError, match="no checkpoint named"):
        session.restore_checkpoint("first")


def test_fork_starts_without_checkpoint_names(counter_package: Path) -> None:
    """A fork inherits execution state but not the checkpoint namespace."""
    parent = Pipeline(counter_package).create_session()
    parent.run_stage("transition")
    parent.checkpoint("first")
    parent.checkpoint("second")

    child = parent.fork()

    np.testing.assert_allclose(child.state("counter_state"), [1.0])
    assert child.has_checkpoint("first") is False
    assert child.has_checkpoint("second") is False
    assert parent.has_checkpoint("first") is True
    assert parent.has_checkpoint("second") is True
    with pytest.raises(WorldModelError, match="no checkpoint named"):
        child.restore_checkpoint("first")


def test_restore_preserves_the_target_checkpoints(counter_package: Path) -> None:
    """A snapshot carries execution state, never a checkpoint namespace."""
    pipeline = Pipeline(counter_package)
    source = pipeline.create_session()
    destination = pipeline.create_session()
    source.run_stage("transition")
    source.checkpoint("source_only")
    destination.run_stage("transition")
    destination.run_stage("transition")
    destination.checkpoint("destination_only")

    destination.restore(source.snapshot())

    np.testing.assert_allclose(destination.state("counter_state"), [1.0])
    assert destination.has_checkpoint("destination_only") is True
    assert destination.has_checkpoint("source_only") is False
