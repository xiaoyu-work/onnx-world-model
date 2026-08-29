"""Per-run ONNX Runtime node traces through the Python API.

Unlike ``test_telemetry.py``, these tests really do run ONNX Runtime and read
what it wrote: the trace file is the artefact under test. Nothing here asserts
a duration, a node count, or a file name, because ONNX Runtime chooses the
name, the timestamp inside it, and how many events a graph produces. What is
asserted is the contract this runtime owns: the file appears, exactly one
record describes it, the record says how the call ended, records are capped
without stopping the files, a reset clears records and keeps files, and a
misconfiguration is refused while the pipeline loads.
"""

# @agent-file
# @agent-purpose: Tests the OnnxModel.run profile_prefix argument against a real ONNX Runtime graph and the Pipeline telemetry_trace_directory and max_trace_records arguments, the PipelineTraceRecord values a traced run publishes, the retention cap, the epoch reset, and the configuration errors.
# @agent-public-api: none
# @agent-invariants: The package and model fixtures build their graph in-process with `pytest.importorskip("onnx_ir")` and never need the Mobius exporter, and every file is written under pytest's `tmp_path`, so no test touches a shared location. Trace files are found by listing the configured directory for the `.json` files ONNX Runtime wrote, never by predicting a name, because the timestamp in that name belongs to ONNX Runtime. The only trace content asserted is that the file parses as a JSON list of event objects containing the `model_run` event and at least one node event, which is ONNX Runtime's documented shape rather than a byte count. Concurrency is asserted through distinct paths and record identifiers produced by threads that really run, never through timing. Tracing requires `enable_telemetry`, so a trace directory alone raises `WorldModelError` while the pipeline loads; `max_trace_records` must be a non-bool int greater than zero. A retention cap drops records and never files, `reset_telemetry` clears records and never deletes a file, and a pre-cancelled run is asserted only for what it may do -- no record at all, or a cancelled record whose path is empty when the trace failed -- because whether the component scope opens before the model call is an implementation detail this suite must not freeze.
# @agent-side-effects: Writes ONNX models, package files, and ONNX Runtime trace files into pytest temporary directories and runs ONNX Runtime inference.

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx_world_model import (
    CancellationSource,
    OnnxModel,
    Pipeline,
    PipelineTraceRecord,
    WorldModelError,
)

_OPSET = 18

_COUNTER_MANIFEST: dict[str, Any] = {
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


def _write_counter_model(destination: Path) -> Path:
    """Writes the one-node graph both fixtures run: ``next_state = state + 1``."""
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
        destination,
    )
    return destination


@pytest.fixture
def counter_model(tmp_path: Path) -> Path:
    """A standalone ONNX file for the generic :class:`OnnxModel` surface."""
    return _write_counter_model(tmp_path / "model.onnx")


@pytest.fixture
def counter_package(tmp_path: Path) -> Path:
    """A one-component package whose stage adds one to its recurrent state."""
    _write_counter_model(tmp_path / "model.onnx")
    (tmp_path / "pipeline.json").write_text(
        json.dumps(_COUNTER_MANIFEST),
        encoding="utf-8",
    )
    return tmp_path


def _trace_files(directory: Path) -> list[Path]:
    """Every trace file in ``directory``, found by suffix rather than by name."""
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def _events(path: Path) -> list[dict[str, Any]]:
    """The event list ONNX Runtime wrote, parsed by the test rather than the runtime."""
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, list)
    return document


def test_model_run_writes_one_trace_for_that_call(
    counter_model: Path, tmp_path: Path
) -> None:
    """``profile_prefix`` traces one call and still returns only the outputs."""
    directory = tmp_path / "traces"
    directory.mkdir()
    model = OnnxModel(counter_model)

    outputs = model.run(
        {"state": np.zeros((1,), dtype=np.float32)},
        profile_prefix=directory / "generic",
    )

    assert set(outputs) == {"next_state"}
    assert outputs["next_state"] == pytest.approx(np.ones((1,), dtype=np.float32))
    files = _trace_files(directory)
    assert len(files) == 1
    # ONNX Runtime owns the name; the test only requires that it starts with
    # the prefix it was given.
    assert files[0].name.startswith("generic_")
    events = _events(files[0])
    assert any(event.get("name") == "model_run" for event in events)
    assert any(event.get("cat") == "Node" for event in events)
    assert all("dur" in event for event in events)


def test_model_run_without_a_prefix_writes_nothing(
    counter_model: Path, tmp_path: Path
) -> None:
    """The historical call is unchanged: no prefix means no profiling."""
    directory = tmp_path / "traces"
    directory.mkdir()
    model = OnnxModel(counter_model)

    model.run({"state": np.zeros((1,), dtype=np.float32)})

    assert _trace_files(directory) == []


def test_two_calls_on_one_model_write_two_traces(
    counter_model: Path, tmp_path: Path
) -> None:
    """Profiling is per run, so a second call is traced separately."""
    directory = tmp_path / "traces"
    directory.mkdir()
    model = OnnxModel(counter_model)

    model.run(
        {"state": np.zeros((1,), dtype=np.float32)},
        profile_prefix=directory / "first",
    )
    model.run({"state": np.zeros((1,), dtype=np.float32)})
    model.run(
        {"state": np.zeros((1,), dtype=np.float32)},
        profile_prefix=directory / "second",
    )

    names = [path.name for path in _trace_files(directory)]
    assert len(names) == 2
    assert sum(name.startswith("first_") for name in names) == 1
    assert sum(name.startswith("second_") for name in names) == 1


def test_a_traced_pipeline_records_one_trace_per_component_call(
    counter_package: Path, tmp_path: Path
) -> None:
    """One component call is one file and one record that points at it."""
    directory = tmp_path / "pipeline-traces"
    pipeline = Pipeline(
        counter_package,
        enable_telemetry=True,
        telemetry_trace_directory=directory,
    )
    session = pipeline.create_session()

    assert session.run_stage("transition")

    snapshot = pipeline.telemetry_snapshot
    assert pipeline.telemetry_trace_directory == directory
    assert pipeline.max_trace_records == 256
    assert len(snapshot.traces) == 1
    assert snapshot.dropped_traces == 0
    assert snapshot.failed_traces == 0
    record = snapshot.traces[0]
    assert isinstance(record, PipelineTraceRecord)
    assert record.component == "counter"
    assert record.epoch == 1
    assert record.trace_id > 0
    assert record.outcome == "success"
    assert record.profiling_failed is False
    assert isinstance(record.path, Path)
    assert record.path.is_absolute()
    assert record.path.exists()
    assert record.size_bytes == record.path.stat().st_size > 0
    assert record.duration_ns > 0
    events = _events(record.path)
    assert any(event.get("name") == "model_run" for event in events)
    assert any(event.get("cat") == "Node" for event in events)


def test_a_counters_only_pipeline_writes_no_trace_file(
    counter_package: Path, tmp_path: Path
) -> None:
    """Counters stay free: without a directory nothing is written or recorded."""
    directory = tmp_path / "unused"
    directory.mkdir()
    pipeline = Pipeline(counter_package, enable_telemetry=True)
    session = pipeline.create_session()

    assert session.run_stage("transition")

    snapshot = pipeline.telemetry_snapshot
    assert pipeline.telemetry_trace_directory is None
    assert snapshot.traces == ()
    assert snapshot.dropped_traces == 0
    assert snapshot.failed_traces == 0
    assert snapshot.components["counter"].successful_calls == 1
    assert _trace_files(directory) == []


def test_concurrent_calls_write_distinct_traces(
    counter_package: Path, tmp_path: Path
) -> None:
    """Every call owns its prefix, so concurrent calls cannot collide."""
    workers = 4
    directory = tmp_path / "concurrent"
    pipeline = Pipeline(
        counter_package,
        enable_telemetry=True,
        telemetry_trace_directory=directory,
    )
    ready = threading.Barrier(workers)

    def run() -> None:
        session = pipeline.create_session()
        ready.wait(timeout=30)
        session.run_stage("transition")

    threads = [threading.Thread(target=run) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive()

    snapshot = pipeline.telemetry_snapshot
    assert len(snapshot.traces) == workers
    assert snapshot.failed_traces == 0
    assert snapshot.dropped_traces == 0
    assert len({record.path for record in snapshot.traces}) == workers
    assert len({record.trace_id for record in snapshot.traces}) == workers
    assert len(_trace_files(directory)) == workers
    assert all(record.size_bytes > 0 for record in snapshot.traces)


def test_independent_pipelines_share_a_trace_directory_without_collisions(
    counter_package: Path, tmp_path: Path
) -> None:
    directory = tmp_path / "shared-directory"
    first = Pipeline(
        counter_package,
        enable_telemetry=True,
        telemetry_trace_directory=directory,
    )
    second = Pipeline(
        counter_package,
        enable_telemetry=True,
        telemetry_trace_directory=directory,
    )

    first.create_session().run_stage("transition")
    second.create_session().run_stage("transition")

    first_record = first.telemetry_snapshot.traces[0]
    second_record = second.telemetry_snapshot.traces[0]
    assert first_record.profiling_failed is False
    assert second_record.profiling_failed is False
    assert first_record.path is not None
    assert second_record.path is not None
    assert first_record.path != second_record.path
    assert len(_trace_files(directory)) == 2


def test_the_retention_cap_drops_records_and_never_files(
    counter_package: Path, tmp_path: Path
) -> None:
    """Retention bounds memory: the files past the cap are still written."""
    directory = tmp_path / "capped"
    pipeline = Pipeline(
        counter_package,
        enable_telemetry=True,
        telemetry_trace_directory=directory,
        max_trace_records=2,
    )
    session = pipeline.create_session()
    for _ in range(3):
        assert session.run_stage("transition")

    snapshot = pipeline.telemetry_snapshot
    assert pipeline.max_trace_records == 2
    assert len(snapshot.traces) == 2
    assert snapshot.dropped_traces == 1
    assert snapshot.failed_traces == 0
    assert len(_trace_files(directory)) == 3
    assert snapshot.traces[0].trace_id < snapshot.traces[1].trace_id


def test_reset_clears_records_and_leaves_the_files(
    counter_package: Path, tmp_path: Path
) -> None:
    """A reset is a new epoch, not a cleanup."""
    directory = tmp_path / "reset"
    pipeline = Pipeline(
        counter_package,
        enable_telemetry=True,
        telemetry_trace_directory=directory,
    )
    session = pipeline.create_session()
    assert session.run_stage("transition")
    before = _trace_files(directory)
    assert len(before) == 1
    first_trace_id = pipeline.telemetry_snapshot.traces[0].trace_id

    pipeline.reset_telemetry()

    snapshot = pipeline.telemetry_snapshot
    assert snapshot.epoch == 2
    assert snapshot.traces == ()
    assert snapshot.dropped_traces == 0
    assert snapshot.failed_traces == 0
    assert _trace_files(directory) == before

    assert session.run_stage("transition")
    after = pipeline.telemetry_snapshot
    assert len(after.traces) == 1
    assert after.traces[0].epoch == 2
    # Trace identifiers are collector-wide, so a reset can never reissue one
    # an in-flight call is still writing under.
    assert after.traces[0].trace_id > first_trace_id
    assert len(_trace_files(directory)) == 2


def test_a_trace_record_is_frozen(counter_package: Path, tmp_path: Path) -> None:
    """A record is a detached value like every other telemetry reading."""
    from dataclasses import FrozenInstanceError

    pipeline = Pipeline(
        counter_package,
        enable_telemetry=True,
        telemetry_trace_directory=tmp_path / "frozen",
    )
    session = pipeline.create_session()
    assert session.run_stage("transition")

    record = pipeline.telemetry_snapshot.traces[0]
    with pytest.raises(FrozenInstanceError):
        record.component = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        pipeline.telemetry_snapshot.traces[0] = record  # type: ignore[index]


def test_a_trace_directory_without_telemetry_is_rejected(
    counter_package: Path, tmp_path: Path
) -> None:
    """Tracing is a second opt-in: it cannot be turned on by itself."""
    directory = tmp_path / "rejected"

    with pytest.raises(WorldModelError):
        Pipeline(counter_package, telemetry_trace_directory=directory)

    assert not directory.exists()


def test_max_trace_records_is_validated(
    counter_package: Path, tmp_path: Path
) -> None:
    """A cap is a count, and zero would keep nothing while writing everything."""
    directory = tmp_path / "invalid"

    with pytest.raises(TypeError):
        Pipeline(
            counter_package,
            enable_telemetry=True,
            telemetry_trace_directory=directory,
            max_trace_records=True,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        Pipeline(
            counter_package,
            enable_telemetry=True,
            telemetry_trace_directory=directory,
            max_trace_records="8",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        Pipeline(
            counter_package,
            enable_telemetry=True,
            telemetry_trace_directory=directory,
            max_trace_records=0,
        )


def test_a_trace_directory_that_is_a_file_is_rejected(
    counter_package: Path, tmp_path: Path
) -> None:
    """An unusable path fails while loading rather than on the first run."""
    occupied = tmp_path / "occupied"
    occupied.write_text("not a directory", encoding="utf-8")

    with pytest.raises(WorldModelError):
        Pipeline(
            counter_package,
            enable_telemetry=True,
            telemetry_trace_directory=occupied,
        )


def test_the_trace_directory_is_created_while_loading(
    counter_package: Path, tmp_path: Path
) -> None:
    """A caller may name a directory that does not exist yet."""
    directory = tmp_path / "created" / "nested"

    pipeline = Pipeline(
        counter_package,
        enable_telemetry=True,
        telemetry_trace_directory=directory,
    )

    assert directory.is_dir()
    session = pipeline.create_session()
    assert session.run_stage("transition")
    assert len(_trace_files(directory)) == 1


def test_a_cancelled_run_never_reports_a_successful_trace(
    counter_package: Path, tmp_path: Path
) -> None:
    """A cancelled call may be traced or not, but never as a success.

    Whether the component scope opens before the model call -- and therefore
    whether a pre-cancelled run produces a failed trace record or no record at
    all -- is an implementation detail. What the contract fixes is that no
    record claims a successful call, and that a record with no usable file
    says so.
    """
    directory = tmp_path / "cancelled"
    pipeline = Pipeline(
        counter_package,
        enable_telemetry=True,
        telemetry_trace_directory=directory,
    )
    session = pipeline.create_session()
    source = CancellationSource()
    source.cancel()

    with pytest.raises(WorldModelError):
        session.run_stage("transition", cancellation=source.token())

    snapshot = pipeline.telemetry_snapshot
    assert all(record.outcome != "success" for record in snapshot.traces)
    for record in snapshot.traces:
        assert record.profiling_failed is (record.path is None)
        if record.profiling_failed:
            assert record.size_bytes == 0
    assert snapshot.failed_traces == sum(
        record.profiling_failed for record in snapshot.traces
    )
