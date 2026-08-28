"""Shared stage-aware admission scheduling through the Python ``Pipeline``.

Nothing here asserts timing between threads. The concurrency behaviour itself
is proven deterministically in ``tests/cpp/pipeline_scheduler_test.cpp`` with
gated stub backends; what this module protects is the Python surface: the
defaults, the values the constructor accepts and echoes back, the argument
checks that happen before conversion, the stage-kind names the native layer
rejects, that a configured pipeline still executes a stage normally, and that
a ``StageRun`` stopped during admission still closes and gives the session's
run slot back.
"""

# @agent-file
# @agent-purpose: Tests the keyword-only max_concurrent_executions and max_concurrent_by_stage_kind arguments on Pipeline and WorldModel -- defaults, accepted values, bool/non-int/negative rejection, unknown and empty stage-kind rejection, that a limited pipeline still runs a stage -- the Pipeline.scheduling_stats reading of the admission scheduler, and that a step or finish stopped by cancellation or a deadline during admission closes its run and leaves the session reusable.
# @agent-public-api: none
# @agent-invariants: Every assertion is about configuration, error handling, or single-threaded run-slot bookkeeping, never about wall-clock behaviour between threads, so nothing here races; the concurrency guarantees themselves belong to the C++ scheduler test. The package fixture builds its graph in-process with `pytest.importorskip("onnx_ir")` and never needs the Mobius exporter. A count must be a plain non-negative int -- `True` is rejected because bool is an int subclass and silently meaning "one at a time" would be worse than failing -- and a per-kind key must be a string naming a stage kind the runtime executes. Limits of 0 and stage kinds left out of the mapping mean unlimited, which is the default and the behaviour every earlier release had. The admission-cancellation tests deliberately configure a limit, because only a constrained stage kind takes a queue ticket and therefore checks the token before the session lock; `session.snapshot()` is the assertion that the slot came back, since it is refused while a run is active. The scheduling-stats assertions are single-threaded and therefore timing-free: they check the shape of a reading -- all six executable stage kinds present in both mappings, a frozen dataclass, read-only mappings, a detached value rather than a live view -- and that a limited pipeline settles back to zero active and zero queued once an execution finishes, which is a leak check rather than a concurrency claim. An unlimited kind takes no permit at all, so an unconfigured pipeline reports zeros however much it has run. WorldModel is checked by signature rather than by loading a package, because it needs an exported Mobius package this suite does not build.
# @agent-side-effects: Writes ONNX models and package files into pytest temporary directories and runs ONNX Runtime inference.

from __future__ import annotations

import inspect
import json
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx_world_model import (
    CancellationSource,
    CancelledError,
    DeadlineExceededError,
    Pipeline,
    WorldModel,
    WorldModelError,
)

_OPSET = 18

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


def test_a_pipeline_is_unlimited_by_default(counter_package: Path) -> None:
    pipeline = Pipeline(counter_package)

    assert pipeline.max_concurrent_executions == 0
    assert dict(pipeline.max_concurrent_by_stage_kind) == {}


def test_configured_limits_are_reported_back(counter_package: Path) -> None:
    pipeline = Pipeline(
        counter_package,
        max_concurrent_executions=2,
        max_concurrent_by_stage_kind={"state_transition": 1, "iterative": 3},
    )

    assert pipeline.max_concurrent_executions == 2
    assert dict(pipeline.max_concurrent_by_stage_kind) == {
        "state_transition": 1,
        "iterative": 3,
    }


def test_the_reported_stage_kind_mapping_is_read_only(
    counter_package: Path,
) -> None:
    """The limits are fixed when the pipeline is built, so the view is too."""
    pipeline = Pipeline(
        counter_package,
        max_concurrent_by_stage_kind={"state_transition": 1},
    )

    with pytest.raises(TypeError):
        pipeline.max_concurrent_by_stage_kind["state_transition"] = 5  # type: ignore[index]


def test_the_caller_mapping_is_copied(counter_package: Path) -> None:
    """Mutating the caller's mapping afterwards cannot change the ceiling."""
    requested = {"state_transition": 1}
    pipeline = Pipeline(
        counter_package, max_concurrent_by_stage_kind=requested
    )

    requested["state_transition"] = 9
    requested["single_pass"] = 4

    assert dict(pipeline.max_concurrent_by_stage_kind) == {
        "state_transition": 1
    }


def test_a_limited_pipeline_still_runs_its_stage(counter_package: Path) -> None:
    pipeline = Pipeline(
        counter_package,
        max_concurrent_executions=1,
        max_concurrent_by_stage_kind={"state_transition": 1},
    )
    session = pipeline.create_session()

    outputs = session.run_stage("transition")

    assert float(np.asarray(outputs["value"]).reshape(-1)[0]) == 1.0
    # A second execution proves the first one returned its permit.
    outputs = session.run_stage("transition")
    assert float(np.asarray(outputs["value"]).reshape(-1)[0]) == 2.0


def test_a_zero_limit_means_unlimited(counter_package: Path) -> None:
    pipeline = Pipeline(
        counter_package,
        max_concurrent_executions=0,
        max_concurrent_by_stage_kind={"state_transition": 0},
    )
    session = pipeline.create_session()

    assert session.run_stage("transition")


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


def test_an_idle_pipeline_reports_nothing_active_or_queued(
    counter_package: Path,
) -> None:
    """Nothing has run, so every counter is zero but every kind is present."""
    pipeline = Pipeline(counter_package, max_concurrent_executions=1)

    stats = pipeline.scheduling_stats

    assert stats.active_executions == 0
    assert stats.queued_executions == 0
    assert set(stats.active_by_stage_kind) == _STAGE_KINDS
    assert set(stats.queued_by_stage_kind) == _STAGE_KINDS
    assert set(stats.active_by_stage_kind.values()) == {0}
    assert set(stats.queued_by_stage_kind.values()) == {0}


def test_scheduling_stats_are_a_frozen_detached_value(
    counter_package: Path,
) -> None:
    """A reading is a value, not a view: it cannot be edited or updated."""
    pipeline = Pipeline(counter_package, max_concurrent_executions=1)
    stats = pipeline.scheduling_stats

    with pytest.raises(FrozenInstanceError):
        stats.active_executions = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        stats.active_by_stage_kind["single_pass"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        stats.queued_by_stage_kind["single_pass"] = 3  # type: ignore[index]

    assert pipeline.scheduling_stats is not stats


def test_a_configured_pipeline_returns_its_permits_after_a_stage(
    counter_package: Path,
) -> None:
    """A completed execution gives its permit back, so the reading is zero.

    This is single-threaded on purpose: what a permit does *while* it is held
    belongs to the C++ scheduler test, which can park a backend. What Python
    proves is that a limited pipeline settles back to an idle reading, so no
    execution here leaked one.
    """
    pipeline = Pipeline(
        counter_package,
        max_concurrent_executions=2,
        max_concurrent_by_stage_kind={"state_transition": 1},
    )
    session = pipeline.create_session()

    assert session.run_stage("transition")

    stats = pipeline.scheduling_stats
    assert stats.active_executions == 0
    assert stats.queued_executions == 0
    assert stats.active_by_stage_kind["state_transition"] == 0
    assert stats.queued_by_stage_kind["state_transition"] == 0


def test_an_unlimited_pipeline_reports_zeros(counter_package: Path) -> None:
    """An unconstrained kind takes no permit, so there is nothing to count."""
    pipeline = Pipeline(counter_package)
    session = pipeline.create_session()

    assert session.run_stage("transition")

    assert pipeline.scheduling_stats.active_executions == 0
    assert pipeline.scheduling_stats.queued_executions == 0


def test_a_step_cancelled_at_admission_releases_the_session(    counter_package: Path,
) -> None:
    """A configured pipeline queues, so cancellation can precede the step.

    Under a limit the scheduler checks the token before it takes a queue
    ticket, so a ``step`` on a token cancelled after ``begin_stage`` fails
    during admission -- before the session lock is ever taken. That failure
    still has to close the handle and release the session's single run slot,
    which is what the reuse below proves.
    """
    pipeline = Pipeline(counter_package, max_concurrent_executions=1)
    session = pipeline.create_session()
    source = CancellationSource()
    run = session.begin_stage("transition", cancellation=source.token())
    assert not run.done

    source.cancel()
    with pytest.raises(CancelledError):
        run.step()

    assert run.done
    # A snapshot is refused while a run is active, so taking one proves the
    # slot was released rather than leaked to a run that never executed.
    assert session.snapshot().valid
    outputs = session.run_stage("transition")
    assert float(np.asarray(outputs["value"]).reshape(-1)[0]) == 1.0


def test_a_finish_stopped_at_admission_releases_the_session(
    counter_package: Path,
) -> None:
    """The same claim for ``finish`` under an already-passed deadline."""
    pipeline = Pipeline(counter_package, max_concurrent_executions=1)
    session = pipeline.create_session()
    run = session.begin_stage("transition", timeout=0.05)
    time.sleep(0.1)

    with pytest.raises(DeadlineExceededError):
        run.finish()

    assert run.done
    assert session.snapshot().valid
    # A fresh incremental run works too, so the slot is genuinely free.
    with session.begin_stage("transition") as fresh:
        outputs = fresh.finish()
    assert float(np.asarray(outputs["value"]).reshape(-1)[0]) == 1.0


def test_a_stale_run_does_not_release_a_newer_runs_slot(
    counter_package: Path,
) -> None:
    """Closing a run that already lost its slot must not free a newer one."""
    pipeline = Pipeline(counter_package, max_concurrent_executions=1)
    session = pipeline.create_session()
    source = CancellationSource()
    stale = session.begin_stage("transition", cancellation=source.token())
    source.cancel()
    with pytest.raises(CancelledError):
        stale.step()

    fresh = session.begin_stage("transition")
    stale.close()

    with pytest.raises(WorldModelError):
        session.snapshot()
    assert float(np.asarray(fresh.finish()["value"]).reshape(-1)[0]) == 1.0


def test_every_supported_stage_kind_is_accepted(counter_package: Path) -> None:
    limits = {
        "single_pass": 1,
        "autoregressive": 2,
        "iterative": 3,
        "state_transition": 4,
        "composite": 5,
        "on_demand": 6,
    }

    pipeline = Pipeline(counter_package, max_concurrent_by_stage_kind=limits)

    assert dict(pipeline.max_concurrent_by_stage_kind) == limits


@pytest.mark.parametrize("kind", ["", "batching", "Single_Pass", "single pass"])
def test_an_unsupported_stage_kind_is_rejected(
    counter_package: Path, kind: str
) -> None:
    with pytest.raises(WorldModelError):
        Pipeline(counter_package, max_concurrent_by_stage_kind={kind: 1})


def test_a_negative_global_limit_is_rejected(counter_package: Path) -> None:
    with pytest.raises(ValueError):
        Pipeline(counter_package, max_concurrent_executions=-1)


def test_a_negative_per_kind_limit_is_rejected(counter_package: Path) -> None:
    with pytest.raises(ValueError):
        Pipeline(
            counter_package, max_concurrent_by_stage_kind={"iterative": -1}
        )


@pytest.mark.parametrize("value", [True, False, 1.0, "2", None])
def test_a_non_integer_global_limit_is_rejected(
    counter_package: Path, value: Any
) -> None:
    """``True`` is an int in Python, and silently meaning 1 would be worse."""
    with pytest.raises(TypeError):
        Pipeline(counter_package, max_concurrent_executions=value)


@pytest.mark.parametrize("value", [True, 2.5, "3", None])
def test_a_non_integer_per_kind_limit_is_rejected(
    counter_package: Path, value: Any
) -> None:
    with pytest.raises(TypeError):
        Pipeline(
            counter_package, max_concurrent_by_stage_kind={"iterative": value}
        )


def test_a_non_string_stage_kind_key_is_rejected(counter_package: Path) -> None:
    with pytest.raises(TypeError):
        Pipeline(counter_package, max_concurrent_by_stage_kind={1: 2})


def test_the_scheduling_options_are_keyword_only() -> None:
    parameters = inspect.signature(Pipeline.__init__).parameters

    for name in ("max_concurrent_executions", "max_concurrent_by_stage_kind"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    assert parameters["max_concurrent_executions"].default == 0
    assert parameters["max_concurrent_by_stage_kind"].default is None


def test_world_model_forwards_the_scheduling_options() -> None:
    """Loading a package needs an exporter, so the contract is the signature."""
    parameters = inspect.signature(WorldModel.__init__).parameters

    for name in ("max_concurrent_executions", "max_concurrent_by_stage_kind"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    assert parameters["max_concurrent_executions"].default == 0
    assert parameters["max_concurrent_by_stage_kind"].default is None
