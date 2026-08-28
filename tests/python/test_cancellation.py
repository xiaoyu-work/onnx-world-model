"""Explicit cancellation and deadlines for pipeline stages and generic models.

The token, source, and exception tests need no package and no optional
dependency, so they always run. The pipeline tests build a small counter
package in-process with ``onnx_ir`` and cancel at step boundaries rather than
racing a graph that finishes in microseconds, so nothing here is timing
dependent. The blocking-wait tests are bounded by a worker thread and a
generous safety-net deadline, so a watchdog that never fires fails the suite
instead of hanging it.
"""

# @agent-file
# @agent-purpose: Tests the Python CancellationSource and CancellationToken wrappers, their blocking wait and the shared deadline watchdog behind it, the CancelledError and DeadlineExceededError hierarchy, and the cancellation and timeout arguments on PipelineSession, StageRun, and OnnxModel.
# @agent-public-api: none
# @agent-invariants: The source, token, wait, exception-hierarchy, and mutual-exclusion tests use no package and no optional dependency, so they always run. The pipeline tests build their graph in-process with `pytest.importorskip("onnx_ir")` and never need the Mobius exporter. Cancellation is always requested at a step boundary or before a call, never by racing an in-flight graph, so no assertion depends on timing. Every `wait` runs on a worker thread through `_wait_bounded`, whose source is cancelled once its budget is exhausted, and the GIL test carries a long safety-net deadline, so no test can hang the suite and none asserts an upper bound on watchdog latency. A cancelled run must leave the state it already applied in place and must free the session's run slot, a `timeout` and a `cancellation` token may never be supplied together, and any finite non-negative `timeout` -- including one far past the steady clock's range -- must saturate rather than become an already-expired deadline.
# @agent-side-effects: Writes ONNX models and package files into pytest temporary directories, runs ONNX Runtime inference, and starts short-lived worker threads that block on a cancellation token.

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx_world_model import (
    CancellationSource,
    CancellationToken,
    CancelledError,
    DeadlineExceededError,
    OnnxModel,
    Pipeline,
    WorldModelError,
    _native,
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
                "name": "iterate",
                "kind": "iterative",
                "components": ["counter"],
                "run_on": "step",
                "options": {
                    "scheduler": {
                        "kind": "FlowMatchEulerDiscreteScheduler",
                        "config_asset": "scheduler.json",
                    },
                    "default_steps": 4,
                    "timestep": {},
                    "state_inputs": ["counter.state"],
                },
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
                "release_after": "iterate",
            }
        ],
        "assets": [{"path": "scheduler.json"}],
        "required_capabilities": ["loop_carried_state"],
    },
    "component_files": {"counter": "model.onnx"},
}


def _write_counter_model(path: Path) -> None:
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
            ir.AttrTensor("value", ir.tensor(np.ones((1,), dtype=np.float32), name="one"))
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
        path,
    )


@pytest.fixture
def counter_package(tmp_path: Path) -> Path:
    """A one-component package whose iterative stage adds one per step."""
    _write_counter_model(tmp_path / "model.onnx")
    (tmp_path / "scheduler.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pipeline.json").write_text(
        json.dumps(_COUNTER_MANIFEST),
        encoding="utf-8",
    )
    return tmp_path


def _counter_value(session: Any) -> float:
    return float(np.asarray(session.state("counter_state")).reshape(-1)[0])


# --- Source, token, and exception contract (no package required) -----------


def test_native_token_has_no_public_constructor() -> None:
    """A token can only come from a source, never from user code."""
    with pytest.raises(TypeError):
        _native.CancellationToken()


def test_a_fresh_source_hands_out_an_uncancelled_token() -> None:
    source = CancellationSource()

    token = source.token()

    assert isinstance(token, CancellationToken)
    assert token.cancellable is True
    assert token.cancelled is False
    assert token.reason == "none"
    assert source.cancelled is False
    assert source.reason == "none"


def test_cancel_is_visible_through_every_token() -> None:
    source = CancellationSource()
    first = source.token()
    second = source.token()

    source.cancel()
    source.cancel()

    assert first.cancelled is True
    assert second.cancelled is True
    assert first.reason == "cancelled"
    assert source.reason == "cancelled"


def test_zero_timeout_is_already_exceeded() -> None:
    source = CancellationSource(timeout=0)

    assert source.cancelled is True
    assert source.reason == "deadline_exceeded"
    assert source.token().reason == "deadline_exceeded"


def test_a_future_timeout_is_not_cancelled() -> None:
    source = CancellationSource(timeout=3600)

    assert source.cancelled is False
    assert source.token().reason == "none"


@pytest.mark.parametrize("timeout", [1.0e10, 1.0e30, 1.0e300, 1.7976931348623157e308])
def test_a_huge_finite_timeout_is_not_immediately_cancelled(timeout: float) -> None:
    """A timeout past the clock's range saturates rather than wrapping into the past."""
    source = CancellationSource(timeout=timeout)

    assert source.cancelled is False
    assert source.reason == "none"
    assert source.token().reason == "none"


@pytest.mark.parametrize("timeout", [-1.0, float("nan"), float("inf")])
def test_invalid_timeouts_are_rejected(timeout: float) -> None:
    with pytest.raises(ValueError):
        CancellationSource(timeout=timeout)


def test_cancellation_errors_derive_from_world_model_error() -> None:
    """Existing ``except WorldModelError`` handlers still catch a cancellation."""
    assert issubclass(CancelledError, WorldModelError)
    assert issubclass(DeadlineExceededError, WorldModelError)
    assert CancelledError is not DeadlineExceededError


# --- Blocking waits and the shared deadline watchdog ------------------------

_WAIT_BUDGET = 5.0


def _wait_bounded(source: CancellationSource, wait: Any) -> str:
    """Run ``wait`` on a worker thread with a hard time bound.

    A watchdog that never fires must fail the suite rather than hang it, so an
    exhausted budget cancels ``source`` — the only other thing that can
    release the waiter — and then reports the failure.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(wait)
        try:
            return str(future.result(timeout=_WAIT_BUDGET))
        except FutureTimeoutError:
            source.cancel()
            future.result(timeout=_WAIT_BUDGET)
            raise AssertionError("wait was never released") from None


def test_a_timeout_wakes_a_waiting_thread() -> None:
    """Nothing polls the token, so only the watchdog can release the wait."""
    source = CancellationSource(timeout=0.02)
    token = source.token()

    assert _wait_bounded(source, token.wait) == "deadline_exceeded"
    assert token.reason == "deadline_exceeded"
    assert source.reason == "deadline_exceeded"


def test_a_source_can_be_waited_on_directly() -> None:
    source = CancellationSource(timeout=0.02)

    assert _wait_bounded(source, source.wait) == "deadline_exceeded"


def test_waiting_on_an_already_stopped_source_returns_immediately() -> None:
    cancelled = CancellationSource()
    cancelled.cancel()
    expired = CancellationSource(timeout=0)

    assert _wait_bounded(cancelled, cancelled.token().wait) == "cancelled"
    assert _wait_bounded(expired, expired.token().wait) == "deadline_exceeded"


def test_cancel_wakes_a_waiting_thread() -> None:
    """An explicit cancel from another thread releases a blocked waiter."""
    source = CancellationSource()
    token = source.token()
    waiting = threading.Event()

    def waiter() -> str:
        waiting.set()
        return token.wait()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(waiter)
        assert waiting.wait(timeout=_WAIT_BUDGET)
        source.cancel()
        assert future.result(timeout=_WAIT_BUDGET) == "cancelled"


def test_wait_releases_the_gil() -> None:
    """A blocked wait must leave the interpreter usable by other threads.

    The deadline here is only a safety net: if ``wait`` held the GIL the main
    thread below could never run, and this would deadlock rather than fail.
    """
    source = CancellationSource(timeout=30.0)
    token = source.token()
    waiting = threading.Event()

    def waiter() -> str:
        waiting.set()
        return token.wait()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(waiter)
            assert waiting.wait(timeout=_WAIT_BUDGET)
            # Ordinary Python work executed while the worker is blocked.
            executed = sum(1 for _ in range(10_000))
            source.cancel()
            assert future.result(timeout=_WAIT_BUDGET) == "cancelled"
    finally:
        source.cancel()

    assert executed == 10_000


def test_a_cancel_before_a_deadline_is_not_demoted_by_it() -> None:
    """A later deadline firing proves the earlier one passed without demoting."""
    cancelled = CancellationSource(timeout=0.04)
    token = cancelled.token()
    cancelled.cancel()

    assert _wait_bounded(cancelled, token.wait) == "cancelled"

    sentinel = CancellationSource(timeout=0.08)
    assert _wait_bounded(sentinel, sentinel.token().wait) == "deadline_exceeded"
    assert token.reason == "cancelled"


# --- Pipeline cancellation --------------------------------------------------


def test_request_cancellation_stops_the_next_step(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()

    with session.begin_stage("iterate") as run:
        run.step()
        run.request_cancellation()
        with pytest.raises(CancelledError):
            run.step()

    assert _counter_value(session) == pytest.approx(1.0)


def test_request_cancellation_is_callable_from_another_thread(
    counter_package: Path,
) -> None:
    """The request never takes the session lock, so a second thread may make it."""
    session = Pipeline(counter_package).create_session()

    with session.begin_stage("iterate") as run:
        run.step()
        requester = threading.Thread(target=run.request_cancellation)
        requester.start()
        requester.join()
        with pytest.raises(CancelledError):
            run.finish()

    assert _counter_value(session) == pytest.approx(1.0)


def test_an_external_source_cancels_a_run_it_started(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()
    source = CancellationSource()

    with session.begin_stage("iterate", cancellation=source.token()) as run:
        run.step()
        source.cancel()
        with pytest.raises(CancelledError):
            run.step()

    assert source.reason == "cancelled"


def test_a_cancelled_run_keeps_what_it_applied_and_frees_the_session(
    counter_package: Path,
) -> None:
    session = Pipeline(counter_package).create_session()
    source = CancellationSource()

    run = session.begin_stage("iterate", cancellation=source.token())
    run.step()
    run.step()
    source.cancel()
    with pytest.raises(CancelledError):
        run.finish()

    # Nothing was rolled back, and the run slot is free for the next run.
    assert _counter_value(session) == pytest.approx(2.0)
    outputs = session.run_stage("iterate")
    assert float(np.asarray(outputs["value"]).reshape(-1)[0]) == pytest.approx(4.0)


def test_an_expired_timeout_raises_deadline_exceeded(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()

    with pytest.raises(DeadlineExceededError):
        session.run_stage("iterate", timeout=0)

    # The failed run never claimed the session, and never ran a step.
    assert session.state("counter_state") is None
    assert session.run_stage("iterate")["value"] is not None


def test_a_huge_timeout_still_runs_a_stage(counter_package: Path) -> None:
    """A timeout beyond the clock's range must not become an expired deadline."""
    session = Pipeline(counter_package).create_session()

    outputs = session.run_stage("iterate", timeout=1.0e10)

    assert float(np.asarray(outputs["value"]).reshape(-1)[0]) == pytest.approx(4.0)


def test_a_pre_cancelled_source_fails_before_the_stage_starts(
    counter_package: Path,
) -> None:
    session = Pipeline(counter_package).create_session()
    source = CancellationSource()
    source.cancel()

    with pytest.raises(CancelledError):
        session.run_stage("iterate", cancellation=source.token())
    with pytest.raises(CancelledError):
        session.begin_stage("iterate", cancellation=source.token())
    with pytest.raises(CancelledError):
        session.step_stage("iterate", cancellation=source.token())

    assert session.state("counter_state") is None
    # A reused cancelled token stays cancelled, so a fresh one is required.
    assert session.run_stage("iterate", cancellation=CancellationSource().token())


def test_step_stage_and_run_honor_their_tokens(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()
    source = CancellationSource()

    session.step_stage("iterate", cancellation=source.token())
    assert _counter_value(session) == pytest.approx(1.0)

    source.cancel()
    with pytest.raises(CancelledError):
        session.step_stage("iterate", cancellation=source.token())
    assert _counter_value(session) == pytest.approx(1.0)

    with pytest.raises(DeadlineExceededError):
        session.run({}, stages=["iterate"], timeout=0)


def test_iter_stage_accepts_a_token(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()
    source = CancellationSource()

    events = session.iter_stage("iterate", cancellation=source.token())
    assert next(events).kind == "iteration"
    source.cancel()
    with pytest.raises(CancelledError):
        next(events)


@pytest.mark.parametrize(
    "call",
    ["run_stage", "step_stage", "begin_stage", "iter_stage", "run"],
)
def test_cancellation_and_timeout_are_mutually_exclusive(
    counter_package: Path,
    call: str,
) -> None:
    session = Pipeline(counter_package).create_session()
    token = CancellationSource().token()
    arguments: tuple[Any, ...] = ({},) if call == "run" else ("iterate",)

    with pytest.raises(ValueError, match="not both"):
        getattr(session, call)(*arguments, cancellation=token, timeout=1.0)


def test_close_and_request_cancellation_are_different_operations(
    counter_package: Path,
) -> None:
    """``close`` abandons the handle; ``request_cancellation`` stops the work."""
    session = Pipeline(counter_package).create_session()

    run = session.begin_stage("iterate")
    run.step()
    run.close()
    # close released the run slot, so the session accepts another run and the
    # closed handle refuses to step.
    with pytest.raises(WorldModelError):
        run.step()
    second = session.begin_stage("iterate")
    # A stale handle's request cannot disturb the newer run.
    run.request_cancellation()
    assert second.step().kind == "iteration"
    second.close()


# --- Generic model cancellation ---------------------------------------------


def test_onnx_model_run_honors_cancellation(tmp_path: Path) -> None:
    _write_counter_model(tmp_path / "model.onnx")
    model = OnnxModel(tmp_path / "model.onnx")
    inputs = {"state": np.zeros((1,), dtype=np.float32)}

    assert model.run(inputs)["next_state"] == pytest.approx([1.0])
    assert model.run(inputs, timeout=3600)["next_state"] == pytest.approx([1.0])

    source = CancellationSource()
    source.cancel()
    with pytest.raises(CancelledError):
        model.run(inputs, cancellation=source.token())
    with pytest.raises(DeadlineExceededError):
        model.run(inputs, timeout=0)
    with pytest.raises(ValueError, match="not both"):
        model.run(inputs, cancellation=source.token(), timeout=1.0)
