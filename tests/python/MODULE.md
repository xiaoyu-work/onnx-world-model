# tests/python

## Purpose

Holds the pytest suite that exercises the installed `onnx_world_model` Python
package, including the compiled `_native` extension, through its public API.

## Responsibilities

- Verify the typed wrappers in `_api.py`: metadata, provider selection and
  registration rejection, device-output forwarding, named-tensor validation,
  and latent-dynamics stepping.
- Verify pipeline loading and staged execution through `Pipeline` and
  `PipelineSession`.
- Verify incremental stage execution: the `StageEvent` dataclass, the
  `StageRun` iterator and lifetime protocol, `begin_stage`, and `iter_stage`.
- Verify explicit cancellation and deadlines: the `CancellationSource` and
  `CancellationToken` wrappers, the blocking `wait` on both and the shared
  native deadline watchdog that releases it while the GIL is free, the
  `CancelledError` and `DeadlineExceededError` hierarchy under
  `WorldModelError`, the keyword-only `cancellation` and `timeout` arguments
  and their mutual exclusion, and the guarantee that a cancelled run frees the
  session without rolling anything back.
- Verify the in-memory `PipelineSessionSnapshot` wrapper: snapshot, restore,
  fork, the named-checkpoint methods, and the pipeline-identity check that
  mirrors the native one.
- Verify the admission-scheduling arguments on `Pipeline` and `WorldModel`:
  the unlimited defaults, the values the constructor accepts and echoes back,
  bool, non-integer, and negative rejection before conversion, the stage-kind
  names the native layer rejects, and that a limited pipeline still runs a
  stage. Verify the `scheduling_stats` reading the same way: an idle pipeline
  reports zeros with all six stage kinds present, the value is a frozen
  dataclass over read-only mappings and is detached rather than live, and a
  configured pipeline settles back to zero once its execution finishes. The
  concurrency behaviour itself belongs to
  `tests/cpp/pipeline_scheduler_test.cpp`, so nothing here asserts timing.
- Verify the load-time placement arguments on `Pipeline` and `WorldModel`:
  the `ComponentPlacementSpec` and plain-mapping forms of
  `component_placement`, `allow_unpreferred_providers`, the argument shapes
  the binding rejects as `TypeError` or `ValueError`, the component-name,
  repeated-provider, and unselected-provider-option rules the native layer
  rejects as `WorldModelError`, the `DeviceSpec` a CPU port reports through
  `ModelMetadata`, and the frozen `Pipeline.transfer_plan` — one `direct`
  transfer per connection for an all-CPU package, and a
  `device_outputs_enabled` flag that mirrors the constructor without rewriting
  any kind.
- Verify the opt-in telemetry surface on `Pipeline` and `WorldModel`: the
  disabled default and what a disabled reading says, the non-bool
  `enable_telemetry` rejection, the fully populated idle reading, the frozen
  dataclasses and read-only mappings of a detached snapshot, the exact
  execution, step, completion, call, and byte counts one single-threaded run
  produces, the admission counters an unlimited and a constrained kind report,
  the `reset_telemetry` epoch, and which pipelines share a collector. Nothing
  here asserts a duration magnitude or timing between threads; those belong to
  `tests/cpp/pipeline_telemetry_test.cpp`.
- Verify the per-run ONNX Runtime tracing surface end to end against real
  trace files: `OnnxModel.run(profile_prefix=...)` writing one parseable JSON
  event list per call and none without a prefix, a traced `Pipeline`
  publishing one `PipelineTraceRecord` per component call whose path exists
  and whose size matches the file, concurrent calls producing distinct files
  and records, a counters-only pipeline writing nothing, the retention cap
  dropping records while files keep being written, `reset_telemetry` clearing
  records and leaving files, the `WorldModelError` for a trace directory
  without `enable_telemetry` or an unusable path, the `TypeError` and
  `ValueError` for a bad `max_trace_records`, and a pre-cancelled run never
  reporting a successful trace. No assertion there depends on an ONNX Runtime
  file name, a node count, or timing.
- Verify the precision report surface: the immutable `precision_report` tuple
  and its frozen `PrecisionPort` and `ComponentPrecisionReport` values, manifest
  component order, a declared `float16` and a declared `int8` dtype reported
  unchanged, `None` for a component that declares none, the real port dtypes of
  the loaded graph reported beside — and never reconciled with — that
  declaration, the providers ONNX Runtime selected, and the read-only
  `WorldModel` pass-through. Nothing there asserts that quantization was
  detected, because no field reports it.
- Verify the preprocessing and media layers against explicit reference
  implementations of the runtime tensor layouts.
- Verify classifier-free guidance and the image-to-video entry point.

## Key Files

| File | Coverage |
|---|---|
| `conftest.py` | Shared fixtures that build throwaway packages, tokenizers, and chat templates. |
| `test_api.py` | `_api.py` wrappers, providers, `LatentDynamicsModel`, `Rollout`. |
| `test_pipeline.py` | `Pipeline` and `PipelineSession` contract and stage execution. |
| `test_pipeline_snapshot.py` | `PipelineSessionSnapshot` plus `PipelineSession.snapshot`, `restore`, `fork`, `checkpoint`, `restore_checkpoint`, `drop_checkpoint`, and `has_checkpoint` on a counter package built with `onnx_ir`. |
| `test_pipeline_stream.py` | `StageEvent`, `StageRun`, `PipelineSession.begin_stage`, and `iter_stage`: payload conversion and iteration against a local fake native handle, plus event, parity, active-run, and token-shape assertions on counter and decoder packages built with `onnx_ir`. |
| `test_cancellation.py` | `CancellationSource`, `CancellationToken`, the blocking `wait` on both and the GIL release around it, the `CancelledError` and `DeadlineExceededError` hierarchy, `StageRun.request_cancellation`, and the `cancellation` and `timeout` arguments on `PipelineSession` and `OnnxModel`, using a counter package built with `onnx_ir`. |
| `test_scheduling.py` | `Pipeline`'s `max_concurrent_executions` and `max_concurrent_by_stage_kind`: unlimited defaults, accepted values and the read-only properties that echo them, the copied caller mapping, bool, non-integer, and negative rejection, unknown, empty, and wrong-case stage kinds, `WorldModel`'s keyword-only forwarding, the `scheduling_stats` reading — all six stage kinds present in both mappings, a frozen dataclass over read-only mappings, a detached value rather than a live view, zeros for an unlimited pipeline, and a limited pipeline settling back to zero active and zero queued after an execution — and — under a configured limit, where a stage kind takes a queue ticket — a `StageRun.step` cancelled and a `StageRun.finish` past its deadline failing during admission while still closing the run and returning the session's run slot, on a counter package built with `onnx_ir`. |
| `test_placement.py` | `Pipeline`'s `component_placement` and `allow_unpreferred_providers`: the `ComponentPlacementSpec` and plain-mapping forms, provider and provider-option overrides, the binding's `TypeError`/`ValueError` shape checks — a mapping per component, a provider list that is not a bare string, string provider names, scalar option values, an unknown placement key, and non-bool, non-integer, and negative thread counts — the native `WorldModelError` rules for an unknown or empty component name, a repeated provider, and options for a provider a component does not run on, the `DeviceSpec` a CPU port reports through `ModelMetadata`, and the frozen `Pipeline.transfer_plan`: one `direct`, bind-eligible, reasonless transfer for the all-CPU package, a stable identical value, and a `device_outputs_enabled` flag that mirrors the constructor without rewriting a kind. `WorldModel`, `OnnxModel`, and `LatentDynamicsModel` are checked by signature, on a two-component package built with `onnx_ir`. |
| `test_telemetry.py` | `Pipeline`'s `enable_telemetry`: the off-by-default reading — `enabled=False`, epoch `0`, empty mappings, zero transfers — the no-op reset of a disabled pipeline, the `TypeError` for a non-bool switch, the fully populated idle reading with all six stage kinds, a frozen detached snapshot over read-only mappings, one `run_stage` as one execution, one step, and one completion, exact component call and byte totals, zero device materializations for a CPU-only package, an incremental run counting each `step` as its own execution with the terminal one adding the completion, a cached `finish` changing nothing, a direct `step_stage` counting a step and never a completion, no admission counters for an unlimited kind and an immediate grant for a constrained one, the `reset_telemetry` epoch, a forked session sharing the collector while a separate `Pipeline` does not, `WorldModel`'s keyword-only forwarding checked by signature, and the exported dataclasses, on a counter package built with `onnx_ir`. |
| `test_trace.py` | Per-run ONNX Runtime traces against real files: `OnnxModel.run(profile_prefix=...)` writing exactly one parseable JSON event list containing `model_run` and at least one node event, no file without a prefix, and one file per traced call; a `Pipeline` with `telemetry_trace_directory` publishing one frozen `PipelineTraceRecord` per component call with an absolute existing path, a size equal to the file's, `outcome="success"`, and a positive duration; four threaded calls producing four distinct paths, identifiers, and files; a counters-only pipeline writing nothing and reporting `telemetry_trace_directory is None`; `max_trace_records=2` keeping two records, counting one `dropped_traces`, and still writing three files; `reset_telemetry` advancing the epoch, clearing records, keeping every file, and issuing a higher `trace_id` afterwards; a frozen record and an immutable `traces` tuple; `WorldModelError` for a trace directory without `enable_telemetry` and for a path that is a file; `TypeError` and `ValueError` for a bool, non-integer, and zero `max_trace_records`; the trace directory being created while loading; and a pre-cancelled run never reporting a successful trace, on a counter package built with `onnx_ir`. |
| `test_precision.py` | `Pipeline.precision_report`: an immutable tuple of frozen `ComponentPrecisionReport` values in manifest component order, frozen `PrecisionPort` entries, a declared `float16` and a declared `int8` parameter dtype reported verbatim, `None` for a profile-less component that declares none, the live graph dtypes -- `int64` in, `float32` out -- reported independently of that declaration, empty state lists for a stateless package, providers matching `Pipeline.execution_providers`, the absence of any quantization or verification field, and the read-only `WorldModel.precision_report` property checked by shape, on a two-component package built with `onnx_ir`. |
| `test_preprocessing.py` | `preprocessing.py` and `media.py`, including latent-token round trips. |
| `test_guided_generation.py` | Classifier-free guidance on graphs synthesized with `onnx_ir`. |
| `test_image_to_video_smoke.py` | The `tools/image_to_video_smoke.py` dry run and generation path. |

## Dependencies

- The installed `onnx_world_model` package and its compiled `_native` module;
  these tests import the package rather than the source tree.
- `pyproject.toml` sets `pythonpath = ["."]` and `testpaths =
  ["tests/python"]`, so `tools/` is importable from a test.
- Optional dependencies gate parts of the suite:
  - `mobius` — `world_model_path` and `pipeline_path` call
    `pytest.importorskip`, so `test_api.py` and `test_pipeline.py` skip
    without it;
  - `onnx_ir` — required by `test_guided_generation.py`, and by the package
    fixtures in `test_pipeline_snapshot.py`, `test_pipeline_stream.py`,
    `test_cancellation.py`, `test_scheduling.py`, `test_placement.py`,
    `test_telemetry.py`, `test_trace.py`, and `test_precision.py`, which call
    `pytest.importorskip` so the rest of those modules still runs;
  - an exported image-to-video package — guards the generation test in
    `test_image_to_video_smoke.py` with `skipif`.

Fixtures that build a package in `tmp_path` need no exporter and no network, so
`test_preprocessing.py`, `test_guided_generation.py`,
`test_pipeline_snapshot.py`, `test_pipeline_stream.py`,
`test_cancellation.py`, `test_scheduling.py`, `test_placement.py`,
`test_telemetry.py`, `test_trace.py`, and `test_precision.py` always run.
`test_trace.py` is the
only module that writes files ONNX Runtime produced, and it writes them under
`tmp_path` like everything else.

## Tests

```console
.venv\Scripts\python.exe -m pytest tests/python -q
.venv\Scripts\python.exe -m pytest tests/python/test_preprocessing.py -q
```

Reinstall the package after changing C++ sources so the tests exercise the
rebuilt extension:

```console
uv pip install -e ".[test]" --reinstall-package onnx-world-model
```
