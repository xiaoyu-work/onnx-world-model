"""Incremental stage execution through ``begin_stage``, ``StageRun``, and ``StageEvent``.

The wrapper protocol tests use a stand-in for the native handle so they always
run; the parity tests build a real package in-process so the events come from
the actual runtime rather than a mock.
"""

# @agent-file
# @agent-purpose: Tests the Python StageEvent dataclass, the StageRun iterator and lifetime protocol, and PipelineSession.begin_stage and iter_stage against a state-transition package and an autoregressive package built with onnx_ir.
# @agent-public-api: none
# @agent-invariants: The wrapper-protocol tests drive a local _FakeStageRunCore rather than the native handle, so they need no package, no onnx_ir, and no ONNX Runtime and always run. The package tests build their graphs in-process with `pytest.importorskip("onnx_ir")` and never need the Mobius exporter. Parity is asserted against a second session created from the same Pipeline, because equality with run_stage is the contract this file protects. An active run must make every state-mutating session method raise WorldModelError while outputs, state, and has_checkpoint stay readable.
# @agent-side-effects: Writes ONNX models and package files into pytest temporary directories and runs ONNX Runtime inference.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx_world_model import (
    Pipeline,
    StageEvent,
    StageRun,
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

_DECODER_MANIFEST = {
    "format": "mobius-pipeline",
    "schema_version": "1.1",
    "manifest": {
        "schema_version": "1.1",
        "components": [
            {
                "name": "decoder",
                "role": "decoder",
                "run_on": "decode",
                "inputs": [
                    {
                        "name": "tokens",
                        "dtype": "INT64",
                        "shape": ["batch", "sequence"],
                    }
                ],
                "outputs": [
                    {
                        "name": "logits",
                        "dtype": "FLOAT",
                        "shape": ["batch", "sequence", 4],
                    }
                ],
                "preferred_execution_providers": ["cpu"],
                "parameter_dtype": "FLOAT",
            }
        ],
        "connections": [],
        "stages": [
            {
                "name": "decode",
                "kind": "autoregressive",
                "components": ["decoder"],
                "run_on": "decode",
                "options": {
                    "tokenizer_asset": "tokenizer.json",
                    "sampling": {"do_sample": False},
                    "stop": {"kind": "token_ids", "eos_token_ids": [3]},
                    "max_tokens": {"default": 8, "limit": 8},
                },
            }
        ],
        "inputs": [
            {
                "port": "decoder.tokens",
                "kind": "external",
                "required": True,
                "semantic": "text.token_ids",
            }
        ],
        "outputs": [{"port": "decoder.logits"}],
        "profile": {"name": "decoder-world", "version": "1.0"},
        "assets": [{"path": "tokenizer.json"}],
    },
    "component_files": {"decoder": "model.onnx"},
}


def _constant(name: str, array: np.ndarray[Any, Any]) -> Any:
    ir = pytest.importorskip("onnx_ir")
    return ir.Node(
        "",
        "Constant",
        inputs=[],
        attributes=[ir.AttrTensor("value", ir.tensor(array, name=name))],
        num_outputs=1,
    )


@pytest.fixture
def counter_package(tmp_path: Path) -> Path:
    """A one-component package whose stage adds one to its recurrent state."""
    ir = pytest.importorskip("onnx_ir")
    state = ir.Value(
        name="state",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([1]),
    )
    one = _constant("one", np.ones((1,), dtype=np.float32))
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


@pytest.fixture
def decoder_package(tmp_path: Path) -> Path:
    """A decoder whose greedy prediction is one past each lane's last token.

    ``logits[..., v] = -(v - (last_token + 1))**2`` peaks at ``last + 1``, so a
    prompt of ``[0]`` decodes ``1, 2, 3`` and stops on the declared
    end-of-sequence token ``3``.
    """
    ir = pytest.importorskip("onnx_ir")
    tokens = ir.Value(
        name="tokens",
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape(["batch", "sequence"]),
    )
    cast = ir.Node(
        "",
        "Cast",
        inputs=[tokens],
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
        num_outputs=1,
    )
    axes = _constant("axes", np.asarray([2], dtype=np.int64))
    column = ir.Node(
        "", "Unsqueeze", inputs=[cast.outputs[0], axes.outputs[0]], num_outputs=1
    )
    one = _constant("one", np.ones((1, 1, 1), dtype=np.float32))
    target = ir.Node(
        "", "Add", inputs=[column.outputs[0], one.outputs[0]], num_outputs=1
    )
    vocabulary = _constant(
        "vocabulary", np.arange(4, dtype=np.float32).reshape(1, 1, 4)
    )
    difference = ir.Node(
        "",
        "Sub",
        inputs=[vocabulary.outputs[0], target.outputs[0]],
        num_outputs=1,
    )
    squared = ir.Node(
        "",
        "Mul",
        inputs=[difference.outputs[0], difference.outputs[0]],
        num_outputs=1,
    )
    logits = ir.Node("", "Neg", inputs=[squared.outputs[0]], num_outputs=1)
    logits.outputs[0].name = "logits"
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape(["batch", "sequence", 4])
    graph = ir.Graph(
        inputs=[tokens],
        outputs=[logits.outputs[0]],
        nodes=[
            cast,
            axes,
            column,
            one,
            target,
            vocabulary,
            difference,
            squared,
            logits,
        ],
        name="decoder",
        opset_imports={"": _OPSET},
    )
    ir.save(
        ir.Model(graph, ir_version=10, producer_name="onnx-world-model-test"),
        tmp_path / "model.onnx",
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pipeline.json").write_text(
        json.dumps(_DECODER_MANIFEST),
        encoding="utf-8",
    )
    return tmp_path


class _FakeStageRunCore:
    """A stand-in for ``_native.StageRun`` with the same small protocol.

    It lets the wrapper's iteration, closing, and context-manager rules be
    asserted without a package, so those rules are covered even where the
    optional graph-building dependency is missing.
    """

    def __init__(self, steps: int) -> None:
        self._steps = steps
        self._index = 0
        self._finished = False
        self.cancels = 0

    @property
    def stage(self) -> str:
        return "transition"

    @property
    def done(self) -> bool:
        return self._finished

    @property
    def iteration(self) -> int:
        return self._index

    def step(self) -> dict[str, Any]:
        if self._finished:
            raise WorldModelError("stage run already completed")
        if self._index < self._steps:
            event = {
                "kind": "transition",
                "stage": "transition",
                "iteration": self._index,
                "token_ids": None,
                "outputs": {"value": np.asarray([float(self._index)], np.float32)},
                "finished": False,
            }
            self._index += 1
            return event
        self._finished = True
        return {
            "kind": "completed",
            "stage": "transition",
            "iteration": self._index,
            "token_ids": None,
            "outputs": {"value": np.asarray([float(self._index)], np.float32)},
            "finished": True,
        }

    def finish(self) -> dict[str, Any]:
        self._finished = True
        return {"value": np.asarray([float(self._steps)], np.float32)}

    def cancel(self) -> None:
        self.cancels += 1


def _fake_run(steps: int) -> tuple[StageRun, _FakeStageRunCore]:
    core = _FakeStageRunCore(steps)
    return StageRun(object(), core), core  # type: ignore[arg-type]


def test_native_stage_run_has_no_public_constructor() -> None:
    """A run can only come from a session, never from user code."""
    with pytest.raises(TypeError):
        _native.StageRun()


def test_step_converts_the_native_payload_into_a_stage_event() -> None:
    run, _ = _fake_run(1)

    event = run.step()

    assert isinstance(event, StageEvent)
    assert event.kind == "transition"
    assert event.stage == "transition"
    assert event.iteration == 0
    assert event.token_ids is None
    assert event.finished is False
    np.testing.assert_allclose(event.outputs["value"], [0.0])


def test_iteration_yields_completed_exactly_once() -> None:
    run, _ = _fake_run(2)

    events = list(run)

    assert [event.kind for event in events] == [
        "transition",
        "transition",
        "completed",
    ]
    assert [event.finished for event in events] == [False, False, True]
    assert run.done is True
    with pytest.raises(StopIteration):
        next(run)


def test_close_is_idempotent_and_cancels_once() -> None:
    run, core = _fake_run(2)
    run.step()

    run.close()
    run.close()

    assert core.cancels == 1
    assert run.done is True
    with pytest.raises(WorldModelError):
        run.step()
    with pytest.raises(WorldModelError):
        run.finish()


def test_context_manager_closes_an_unfinished_run() -> None:
    run, core = _fake_run(5)

    with run as active:
        for event in active:
            if event.iteration == 1:
                break

    assert core.cancels == 1
    assert run.done is True


def test_finish_marks_the_run_exhausted() -> None:
    run, _ = _fake_run(3)

    outputs = run.finish()

    np.testing.assert_allclose(outputs["value"], [3.0])
    with pytest.raises(StopIteration):
        next(run)


def test_begin_stage_streams_a_transition_then_completes(
    counter_package: Path,
) -> None:
    session = Pipeline(counter_package).create_session()

    run = session.begin_stage("transition")

    assert run.stage == "transition"
    assert run.done is False
    events = list(run)
    assert [event.kind for event in events] == ["transition", "completed"]
    assert [event.iteration for event in events] == [0, 1]
    assert all(event.token_ids is None for event in events)
    np.testing.assert_allclose(events[0].outputs["value"], [1.0])
    np.testing.assert_allclose(events[-1].outputs["value"], [1.0])
    assert run.done is True


def test_finish_matches_run_stage(counter_package: Path) -> None:
    pipeline = Pipeline(counter_package)
    expected = pipeline.create_session().run_stage("transition")

    session = pipeline.create_session()
    run = session.begin_stage("transition")
    run.step()

    np.testing.assert_allclose(run.finish()["value"], expected["value"])
    np.testing.assert_allclose(session.state("counter_state"), [1.0])


def test_an_active_run_blocks_state_changing_session_methods(
    counter_package: Path,
) -> None:
    session = Pipeline(counter_package).create_session()
    session.checkpoint("mark")
    snapshot = session.snapshot()

    with session.begin_stage("transition") as run:
        run.step()

        for call in (
            lambda: session.begin_stage("transition"),
            lambda: session.run_stage("transition"),
            lambda: session.step_stage("transition"),
            session.snapshot,
            lambda: session.restore(snapshot),
            session.fork,
            lambda: session.checkpoint("other"),
            lambda: session.restore_checkpoint("mark"),
            lambda: session.drop_checkpoint("mark"),
            session.reset,
            lambda: session.release_stage("transition"),
        ):
            with pytest.raises(WorldModelError, match="stage run is active"):
                call()

        # Reads observe state instead of changing it, so they stay legal.
        np.testing.assert_allclose(session.outputs["value"], [1.0])
        np.testing.assert_allclose(session.state("counter_state"), [1.0])
        assert session.has_checkpoint("mark") is True

    np.testing.assert_allclose(session.run_stage("transition")["value"], [2.0])


def test_closing_a_run_releases_the_session(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()
    run = session.begin_stage("transition")
    run.step()

    run.close()

    # The abandoned step stays applied; only the run slot is released.
    np.testing.assert_allclose(session.state("counter_state"), [1.0])
    np.testing.assert_allclose(session.run_stage("transition")["value"], [2.0])


def test_iter_stage_closes_the_run_when_the_iterator_closes(
    counter_package: Path,
) -> None:
    session = Pipeline(counter_package).create_session()

    events = session.iter_stage("transition")
    first = next(events)
    assert first.kind == "transition"
    with pytest.raises(WorldModelError, match="stage run is active"):
        session.run_stage("transition")

    events.close()

    np.testing.assert_allclose(session.run_stage("transition")["value"], [2.0])


def test_iter_stage_completes_and_releases(counter_package: Path) -> None:
    session = Pipeline(counter_package).create_session()

    kinds = [event.kind for event in session.iter_stage("transition")]

    assert kinds == ["transition", "completed"]
    np.testing.assert_allclose(session.run_stage("transition")["value"], [2.0])


def test_iter_stage_reports_an_unknown_stage_before_iterating(
    counter_package: Path,
) -> None:
    session = Pipeline(counter_package).create_session()

    with pytest.raises(WorldModelError):
        session.iter_stage("missing")


def test_token_events_carry_the_generated_tokens(decoder_package: Path) -> None:
    pipeline = Pipeline(decoder_package)
    prompt = {"text.token_ids": np.asarray([[0]], dtype=np.int64)}
    expected = pipeline.create_session().run_stage("decode", prompt)

    session = pipeline.create_session()
    events = list(session.iter_stage("decode", prompt))

    assert [event.kind for event in events] == [
        "token",
        "token",
        "token",
        "completed",
    ]
    tokens = [event.token_ids for event in events if event.kind == "token"]
    for token in tokens:
        assert token is not None
        assert token.dtype == np.int64
        assert token.shape == (1, 1)
    assert [int(token[0, 0]) for token in tokens] == [1, 2, 3]
    assert events[-1].token_ids is None
    assert events[-1].finished is True
    np.testing.assert_array_equal(
        events[-1].outputs["generated_token_ids"],
        expected["generated_token_ids"],
    )
    np.testing.assert_array_equal(
        expected["generated_token_ids"], np.asarray([[1, 2, 3]], dtype=np.int64)
    )


def test_streamed_decoding_can_stop_early(decoder_package: Path) -> None:
    session = Pipeline(decoder_package).create_session()
    prompt = {"text.token_ids": np.asarray([[0]], dtype=np.int64)}

    with session.begin_stage("decode", prompt) as run:
        first = run.step()
        assert first.kind == "token"
        assert run.iteration == 1

    # Stopping early keeps the tokens already decoded and frees the session.
    outputs = session.run_stage("decode", options={"max_tokens": 1})
    np.testing.assert_array_equal(
        outputs["generated_token_ids"], np.asarray([[2]], dtype=np.int64)
    )
