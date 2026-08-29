# python/onnx_world_model

## Purpose

Implements the installed Python package: it wraps the compiled `_native`
extension in typed classes and adds the preprocessing and generation layers
that turn prompts, images, and videos into the tensors a Mobius package
expects.

## Responsibilities

- Locate and load the ONNX Runtime shared library before the extension is used,
  including the CUDA libraries an `onnxruntime-gpu` wheel resolves by soname.
- Register explicit execution-provider libraries and forward the opt-in
  device-output policy through every model and pipeline wrapper.
- Present the manifest as typed `PipelineInputSpec`, `PipelineOutputSpec`, and
  `PipelineStageSpec` values, and each model port as a `TensorSpec` whose
  `device` is the `DeviceSpec` ONNX Runtime actually assigned, or `None` when
  the backend reports no placement.
- Forward the pipeline-only load-time placement options —
  `component_placement`, which accepts a `ComponentPlacementSpec` or an
  equivalent mapping per component, and `allow_unpreferred_providers` — from
  `Pipeline` and `WorldModel` to the native layer, and expose the resulting
  `PipelineTransferPlan` as the frozen, inspection-only `Pipeline.transfer_plan`.
  `OnnxModel` and `LatentDynamicsModel` do not accept them, because they are
  single graphs rather than pipelines.
- Prepare model inputs from package configuration: chat templates, tokenizers,
  image and video patch packing, latent-token layout, and conditioning frames.
- Offer the modality-oriented `WorldModel` API (`text`, `image`, `video`,
  `action`) and report the capabilities the loaded package actually declares.
- Forward the pipeline-only admission-scheduling options —
  `max_concurrent_executions` and `max_concurrent_by_stage_kind` — from
  `Pipeline` and `WorldModel` to the native layer, and report the accepted
  values back as read-only properties. `Pipeline.scheduling_stats` reads the
  same controller and returns a frozen `PipelineSchedulingStats` with the
  admitted and queued counts, in total and per stage kind. `OnnxModel` and
  `LatentDynamicsModel` do not accept them, because they are not pipelines.
- Forward the pipeline-only `enable_telemetry` switch from `Pipeline` and
  `WorldModel` to the native layer, which rejects anything that is not a real
  bool, and expose the resulting counters as the frozen
  `Pipeline.telemetry_snapshot` and `WorldModel.telemetry_snapshot`, with
  `reset_telemetry()` on both to start a new epoch. A pipeline that was not
  asked to collect reads as `enabled=False` with empty mappings rather than
  raising. This is the runtime's own measurement — component calls, stage
  executions, admission waits, and device materializations — and is separate
  from the per-request wall-clock `timings` each generator returns.
- Forward the pipeline-only `telemetry_trace_directory` and
  `max_trace_records` arguments the same way, so each component call also
  writes one ONNX Runtime node trace and publishes one frozen
  `PipelineTraceRecord` in `telemetry_snapshot.traces`, with a usable path
  presented as a `pathlib.Path`, a failed trace as `None`, and its outcome as
  a `PipelineCallOutcome` string. Tracing requires `enable_telemetry` and raises `WorldModelError`
  without it. `OnnxModel.run` takes the same profiling per call as
  `profile_prefix` and still returns only the outputs.
- Keep the low-level `OnnxModel`, `Pipeline`, `PipelineSession`,
  `PipelineSessionSnapshot`, `StageRun`, `CancellationSource`,
  `CancellationToken`, `LatentDynamicsModel`, and `Rollout` APIs available.
- Expose cooperative cancellation and per-call deadlines: a keyword-only
  `cancellation` token or `timeout` in seconds on the pipeline and model
  entry points, `StageRun.request_cancellation` for stopping work already
  running, the blocking `CancellationToken.wait` and `CancellationSource.wait`
  that release the GIL until an explicit cancel or the shared native deadline
  watchdog claims a reason, and the `CancelledError` and
  `DeadlineExceededError` subclasses of `WorldModelError`.

## Key Files

| File | Responsibility |
|---|---|
| `__init__.py` | Public package surface; re-exports only, with a sorted `__all__`. |
| `_api.py` | ORT and EP library discovery/registration, `_native` wrappers, manifest spec dataclasses, the `DeviceSpec`, `ComponentPlacementSpec`, `PipelineTransfer`, and `PipelineTransferPlan` placement values, device-output options, `OnnxModel` with its per-call `profile_prefix`, `Pipeline` with its admission-scheduling limits, its frozen `PipelineSchedulingStats` reading, its opt-in `enable_telemetry` with the frozen `PipelineComponentStats`, `PipelineStageStats`, `PipelineAdmissionStats`, `PipelineTransferStats`, `PipelineTraceRecord`, and `PipelineTelemetrySnapshot` reading and `reset_telemetry`, its `telemetry_trace_directory` and `max_trace_records` tracing options, its load-time `component_placement`, and its frozen `transfer_plan`, `PipelineSession` with its `PipelineSessionSnapshot`, named-checkpoint, and incremental `begin_stage`/`iter_stage` methods, the `StageEvent` value type and the `StageRun` iterator, the `CancellationSource` and `CancellationToken` wrappers and the shared `cancellation`/`timeout` argument handling, latent-dynamics API. |
| `media.py` | Image and video decoding, grid-aligned resizing, and patch-token packing. |
| `preprocessing.py` | Chat templating, tokenization, latent-token packing, and reasoner and world-model input assembly. |
| `generation.py` | `WorldModel` plus the text, image, video, and action generators and their output dataclasses. |

`WorldModelPipeline` and `LegacyWorldModel` are compatibility aliases for
`Pipeline` and `LatentDynamicsModel`. The modality `generate()` methods in
`generation.py` do not accept a cancellation token yet; a caller who needs one
drives `PipelineSession` directly.

## Dependencies

Intra-package imports are one-way and acyclic:

```text
_native (compiled) <- _api <- generation
                     media <- preprocessing <- generation
```

`media.py` must not import `preprocessing.py`; `preprocessing.py` imports and
re-exports the media symbols so `onnx_world_model.preprocessing` remains the
single documented import path.

Third-party runtime dependencies come from `pyproject.toml`: `numpy`,
`onnxruntime`, `tokenizers`, `Jinja2`, `Pillow`, `imageio[ffmpeg]`, and
`ml_dtypes`.

The compiled `_native` module is produced from `bindings/python_module.cpp` by
scikit-build-core and installed next to these files; it is not present in the
source tree.

## Tests

```console
.venv\Scripts\python.exe -m pytest tests/python -q
uvx ruff check python
```

`tests/python/test_api.py` covers `_api.py`,
`tests/python/test_pipeline_snapshot.py` covers the session snapshot and
named-checkpoint wrappers,
`tests/python/test_pipeline_stream.py` covers `StageEvent`, `StageRun`,
`begin_stage`, and `iter_stage`,
`tests/python/test_cancellation.py` covers `CancellationSource`,
`CancellationToken`, the exception hierarchy, and the `cancellation` and
`timeout` arguments,
`tests/python/test_scheduling.py` covers the `max_concurrent_executions` and
`max_concurrent_by_stage_kind` arguments on `Pipeline` and `WorldModel` plus
the shape and immutability of the `Pipeline.scheduling_stats` reading,
`tests/python/test_placement.py` covers the `component_placement` and
`allow_unpreferred_providers` arguments, the `DeviceSpec` a CPU port reports,
and the frozen `Pipeline.transfer_plan`,
`tests/python/test_telemetry.py` covers the `enable_telemetry` argument, the
shape and immutability of the `Pipeline.telemetry_snapshot` reading, the exact
counts one single-threaded run produces, and `reset_telemetry`,
`tests/python/test_trace.py` covers `OnnxModel.run(profile_prefix=...)`, the
`telemetry_trace_directory` and `max_trace_records` arguments, the frozen
`PipelineTraceRecord` values a traced run publishes against real ONNX Runtime
trace files, the retention cap, and the configuration errors,
`tests/python/test_preprocessing.py` covers `preprocessing.py` and `media.py`,
and `tests/python/test_guided_generation.py` and
`tests/python/test_image_to_video_smoke.py` cover `generation.py`. Tests that
need a Mobius export skip unless the `mobius` package is installed.
