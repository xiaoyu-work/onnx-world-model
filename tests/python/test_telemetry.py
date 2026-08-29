"""Opt-in runtime telemetry through the Python ``Pipeline``.

Nothing here asserts a duration, a rate, or timing between threads. Wall-clock
magnitudes are a property of the machine, and the concurrent behaviour itself
is proven deterministically in ``tests/cpp/pipeline_telemetry_test.cpp`` with
gated stub backends. What this module protects is the Python surface: the
default, the shape and immutability of a reading, the exact counts and byte
totals a single-threaded run produces, the epoch reset, and the argument check
that happens before conversion.
"""

# @agent-file
# @agent-purpose: Tests the keyword-only enable_telemetry argument on Pipeline and WorldModel, the Pipeline.telemetry_snapshot reading and its frozen dataclasses and read-only mappings, the counts one single-threaded run produces for components, stages, admission, and device transfers, and the reset_telemetry epoch.
# @agent-public-api: none
# @agent-invariants: Every assertion is about configuration, shape, immutability, or an exact count that comes from tensor byte sizes rather than from a clock, so nothing here is timing-dependent; durations are only ever compared against each other -- a maximum never exceeds its total -- never against a magnitude. The package fixture builds its graph in-process with `pytest.importorskip("onnx_ir")` and never needs the Mobius exporter. Telemetry is off by default and off means no collector, so a disabled reading is `enabled=False`, epoch `0`, and empty mappings rather than a map of zeros; an enabled reading always carries every manifest component, every manifest stage, and all six executable stage kinds. `enable_telemetry` must be a real bool -- `1` is rejected for the same reason `True` is rejected as a concurrency limit. One `run_stage` of a one-pass stage is one execution, one step, and one completion; a direct `step_stage` is one execution and one step with no completion, because it emits no terminal event; and an unlimited stage kind records no admission at all while its executions are still measured. WorldModel is checked by signature rather than by loading a package, because it needs an exported Mobius package this suite does not build.
# @agent-side-effects: Writes ONNX models and package files into pytest temporary directories and runs ONNX Runtime inference.

from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx_world_model import (
    Pipeline,
    PipelineAdmissionStats,
    PipelineComponentStats,
    PipelineStageStats,
    PipelineTelemetrySnapshot,
    PipelineTransferStats,
    WorldModel,
)

_OPSET = 18

# The one-component package below moves a single float32 value, so every byte
# total a call produces is exactly this.
_STATE_BYTES = 4

_STAGE_KINDS = frozenset(
    {
        "single_pass",
        "autoregressive",
        "iterative",
        "state_transition",
        "composite",
        "on_demand",
    }
)

_COUNTER_MANIFEST = {
    "format": "mobius-pipeline",
    "schema_version": "1.1",
    "manifest": {
        "schema_version": "1.1",
        "components": [
            {
                "name": "counter",
                "role": "dynamics",
                "run_on": "step",
                "inputs": [{"name": "state", "dtype": "FLOAT", "shape": [1]}],
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
                "value", ir.tensor(np.ones((1,), dtype=np.float32), name="one")
            )
        ],
        num_outputs=1,
    )
    increment = ir.Node("", "Add", inputs=[state, one.outputs[0]], num_outputs=1)
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
        json.dumps(_COUNTER_MANIFEST),
        encoding="utf-8",
    )
    return tmp_path


def test_telemetry_is_off_by_default(counter_package: Path) -> None:
    """Off means no collector at all, which is what the reading has to say."""
    pipeline = Pipeline(counter_package)
    session = pipeline.create_session()
    assert session.run_stage("transition")

    snapshot = pipeline.telemetry_snapshot

    assert pipeline.telemetry_enabled is False
    assert snapshot.enabled is False
    assert snapshot.epoch == 0
    assert dict(snapshot.components) == {}
    assert dict(snapshot.stages) == {}
    assert dict(snapshot.admission_by_stage_kind) == {}
    assert snapshot.transfers == PipelineTransferStats(
        device_to_host_copies=0,
        device_to_host_bytes=0,
        component_input_bytes_device_resident=0,
        component_input_bytes_host=0,
    )


def test_resetting_a_disabled_pipeline_does_nothing(
    counter_package: Path,
) -> None:
    """A caller can reset unconditionally without testing first."""
    pipeline = Pipeline(counter_package)

    pipeline.reset_telemetry()

    assert pipeline.telemetry_snapshot.enabled is False
    assert pipeline.telemetry_snapshot.epoch == 0


def test_enable_telemetry_must_be_a_bool(counter_package: Path) -> None:
    """A switch is not a count: `1` quietly meaning "on" would be worse."""
    with pytest.raises(TypeError):
        Pipeline(counter_package, enable_telemetry=1)  # type: ignore[arg-type]


def test_an_idle_enabled_reading_is_fully_populated(
    counter_package: Path,
) -> None:
    """Every key exists before anything runs, so a caller never tests for one."""
    pipeline = Pipeline(counter_package, enable_telemetry=True)

    snapshot = pipeline.telemetry_snapshot

    assert pipeline.telemetry_enabled is True
    assert snapshot.enabled is True
    assert snapshot.epoch == 1
    assert set(snapshot.components) == {"counter"}
    assert set(snapshot.stages) == {"transition"}
    assert set(snapshot.admission_by_stage_kind) == _STAGE_KINDS
    assert snapshot.components["counter"] == PipelineComponentStats(
        successful_calls=0,
        failed_calls=0,
        cancelled_calls=0,
        deadline_exceeded_calls=0,
        total_duration_ns=0,
        max_duration_ns=0,
        input_bytes=0,
        output_bytes=0,
    )
    assert snapshot.stages["transition"] == PipelineStageStats(
        successful_executions=0,
        failed_executions=0,
        cancelled_executions=0,
        deadline_exceeded_executions=0,
        steps=0,
        completions=0,
        total_execution_duration_ns=0,
        max_execution_duration_ns=0,
    )
    assert snapshot.admission_by_stage_kind["state_transition"] == (
        PipelineAdmissionStats(
            queued_acquisitions=0,
            admitted_acquisitions=0,
            cancelled_while_queued=0,
            deadline_while_queued=0,
            total_wait_ns=0,
            max_wait_ns=0,
        )
    )


def test_a_reading_is_a_frozen_detached_value(counter_package: Path) -> None:
    """A reading is a value, not a view: it cannot be edited or updated."""
    pipeline = Pipeline(counter_package, enable_telemetry=True)
    session = pipeline.create_session()
    assert session.run_stage("transition")
    snapshot = pipeline.telemetry_snapshot

    with pytest.raises(FrozenInstanceError):
        snapshot.epoch = 7  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.components["counter"].successful_calls = 7  # type: ignore[misc]
    with pytest.raises(TypeError):
        snapshot.components["other"] = snapshot.components["counter"]  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.stages["other"] = snapshot.stages["transition"]  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.admission_by_stage_kind["other"] = (  # type: ignore[index]
            snapshot.admission_by_stage_kind["state_transition"]
        )

    assert session.run_stage("transition")
    # The reading already returned keeps the counts it was taken with.
    assert snapshot.stages["transition"].successful_executions == 1
    assert pipeline.telemetry_snapshot.stages["transition"].successful_executions == 2
    assert pipeline.telemetry_snapshot is not snapshot
    assert isinstance(snapshot, PipelineTelemetrySnapshot)


def test_one_run_stage_is_one_execution_one_step_one_completion(
    counter_package: Path,
) -> None:
    """Executions, steps, and completions are three different measurements."""
    pipeline = Pipeline(counter_package, enable_telemetry=True)
    session = pipeline.create_session()

    assert session.run_stage("transition")

    stage = pipeline.telemetry_snapshot.stages["transition"]
    assert stage.successful_executions == 1
    assert stage.steps == 1
    assert stage.completions == 1
    assert stage.failed_executions == 0
    assert stage.cancelled_executions == 0
    assert stage.deadline_exceeded_executions == 0
    # A duration is never compared against a magnitude, only against itself.
    assert stage.max_execution_duration_ns <= stage.total_execution_duration_ns


def test_component_calls_report_exact_byte_totals(
    counter_package: Path,
) -> None:
    """Byte totals come from the tensors themselves, so they are exact."""
    pipeline = Pipeline(counter_package, enable_telemetry=True)
    session = pipeline.create_session()

    assert session.run_stage("transition")
    assert session.run_stage("transition")

    component = pipeline.telemetry_snapshot.components["counter"]
    assert component.successful_calls == 2
    assert component.failed_calls == 0
    assert component.input_bytes == 2 * _STATE_BYTES
    assert component.output_bytes == 2 * _STATE_BYTES
    assert component.max_duration_ns <= component.total_duration_ns


def test_a_cpu_only_package_materializes_nothing(
    counter_package: Path,
) -> None:
    """Residency is presentation: these inputs never left the host."""
    pipeline = Pipeline(counter_package, enable_telemetry=True)
    session = pipeline.create_session()

    assert session.run_stage("transition")

    transfers = pipeline.telemetry_snapshot.transfers
    assert transfers.device_to_host_copies == 0
    assert transfers.device_to_host_bytes == 0
    assert transfers.component_input_bytes_device_resident == 0
    assert transfers.component_input_bytes_host == _STATE_BYTES


def test_an_incremental_run_counts_each_step_as_its_own_execution(
    counter_package: Path,
) -> None:
    """The terminal step is an execution that completes rather than steps."""
    pipeline = Pipeline(counter_package, enable_telemetry=True)
    session = pipeline.create_session()

    with session.begin_stage("transition") as run:
        assert pipeline.telemetry_snapshot.stages["transition"].successful_executions == 0
        first = run.step()
        assert not first.finished
        stage = pipeline.telemetry_snapshot.stages["transition"]
        assert stage.successful_executions == 1
        assert stage.steps == 1
        assert stage.completions == 0

        terminal = run.step()
        assert terminal.finished

    stage = pipeline.telemetry_snapshot.stages["transition"]
    assert stage.successful_executions == 2
    assert stage.steps == 1
    assert stage.completions == 1


def test_a_cached_finish_changes_no_counter(counter_package: Path) -> None:
    """A completed run has nothing left to execute, so it counts nothing."""
    pipeline = Pipeline(counter_package, enable_telemetry=True)
    session = pipeline.create_session()

    with session.begin_stage("transition") as run:
        assert run.finish()
        before = pipeline.telemetry_snapshot.stages["transition"]
        assert run.finish()
        after = pipeline.telemetry_snapshot.stages["transition"]

    assert after == before


def test_a_direct_step_counts_a_step_and_never_a_completion(
    counter_package: Path,
) -> None:
    """``step_stage`` bypasses the event state machine, so it never completes."""
    pipeline = Pipeline(counter_package, enable_telemetry=True)
    session = pipeline.create_session()

    assert session.step_stage("transition")
    assert session.step_stage("transition")

    stage = pipeline.telemetry_snapshot.stages["transition"]
    assert stage.successful_executions == 2
    assert stage.steps == 2
    assert stage.completions == 0


def test_an_unlimited_stage_kind_records_no_admission(
    counter_package: Path,
) -> None:
    """An unconstrained kind takes no permit, so there is nothing to report."""
    pipeline = Pipeline(counter_package, enable_telemetry=True)
    session = pipeline.create_session()

    assert session.run_stage("transition")

    admission = pipeline.telemetry_snapshot.admission_by_stage_kind
    assert admission["state_transition"].admitted_acquisitions == 0
    assert admission["state_transition"].queued_acquisitions == 0
    assert admission["state_transition"].total_wait_ns == 0
    # The execution itself is still measured.
    assert pipeline.telemetry_snapshot.stages["transition"].successful_executions == 1


def test_a_constrained_kind_records_an_immediate_grant(
    counter_package: Path,
) -> None:
    """Nothing else is running, so the permit is granted without a wait.

    This is single-threaded on purpose: what a queued acquisition reports
    belongs to the C++ telemetry test, which can park a backend. What Python
    proves is that a constrained kind is counted at all, and that an
    uncontended grant is admitted rather than queued.
    """
    pipeline = Pipeline(
        counter_package,
        enable_telemetry=True,
        max_concurrent_executions=1,
    )
    session = pipeline.create_session()

    assert session.run_stage("transition")

    admission = pipeline.telemetry_snapshot.admission_by_stage_kind[
        "state_transition"
    ]
    assert admission.admitted_acquisitions == 1
    assert admission.queued_acquisitions == 0
    assert admission.cancelled_while_queued == 0
    assert admission.deadline_while_queued == 0
    assert admission.total_wait_ns == 0
    assert admission.max_wait_ns == 0


def test_reset_starts_a_new_epoch_that_keeps_collecting(
    counter_package: Path,
) -> None:
    """Reset is a new beginning, not an edit and not a shutdown."""
    pipeline = Pipeline(counter_package, enable_telemetry=True)
    session = pipeline.create_session()
    assert session.run_stage("transition")

    pipeline.reset_telemetry()

    snapshot = pipeline.telemetry_snapshot
    assert snapshot.epoch == 2
    assert snapshot.enabled is True
    assert set(snapshot.components) == {"counter"}
    assert set(snapshot.stages) == {"transition"}
    assert set(snapshot.admission_by_stage_kind) == _STAGE_KINDS
    assert snapshot.stages["transition"].successful_executions == 0
    assert snapshot.components["counter"].input_bytes == 0
    assert snapshot.transfers.component_input_bytes_host == 0

    assert session.run_stage("transition")
    after = pipeline.telemetry_snapshot
    assert after.epoch == 2
    assert after.stages["transition"].successful_executions == 1


def test_a_forked_session_records_into_the_same_collector(
    counter_package: Path,
) -> None:
    """One collector per pipeline, shared by every session it produced."""
    pipeline = Pipeline(counter_package, enable_telemetry=True)
    session = pipeline.create_session()
    assert session.run_stage("transition")

    forked = session.fork()
    assert forked.run_stage("transition")

    assert pipeline.telemetry_snapshot.stages["transition"].successful_executions == 2


def test_a_separate_pipeline_counts_only_its_own_work(
    counter_package: Path,
) -> None:
    """Two pipelines over the same package have independent collectors."""
    first = Pipeline(counter_package, enable_telemetry=True)
    second = Pipeline(counter_package, enable_telemetry=True)
    session = first.create_session()

    assert session.run_stage("transition")

    assert first.telemetry_snapshot.stages["transition"].successful_executions == 1
    assert second.telemetry_snapshot.stages["transition"].successful_executions == 0


def test_world_model_forwards_the_telemetry_option() -> None:
    """WorldModel needs an exported package, so its surface is checked here."""
    parameters = inspect.signature(WorldModel.__init__).parameters

    assert "enable_telemetry" in parameters
    assert parameters["enable_telemetry"].default is False
    assert parameters["enable_telemetry"].kind is inspect.Parameter.KEYWORD_ONLY
    assert isinstance(WorldModel.telemetry_snapshot, property)
    assert callable(WorldModel.reset_telemetry)


def test_the_telemetry_values_are_exported() -> None:
    """The five frozen values are part of the documented public surface."""
    import onnx_world_model

    exported: list[Any] = [
        PipelineComponentStats,
        PipelineStageStats,
        PipelineAdmissionStats,
        PipelineTransferStats,
        PipelineTelemetrySnapshot,
    ]
    for value in exported:
        assert value.__name__ in onnx_world_model.__all__
        assert getattr(onnx_world_model, value.__name__) is value
