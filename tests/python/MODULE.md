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
    `test_cancellation.py`, `test_scheduling.py`, and `test_placement.py`,
    which call
    `pytest.importorskip` so the rest of those modules still runs;
  - an exported image-to-video package — guards the generation test in
    `test_image_to_video_smoke.py` with `skipif`.

Fixtures that build a package in `tmp_path` need no exporter and no network, so
`test_preprocessing.py`, `test_guided_generation.py`,
`test_pipeline_snapshot.py`, `test_pipeline_stream.py`,
`test_cancellation.py`, `test_scheduling.py`, and `test_placement.py` always
run.

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
