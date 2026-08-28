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
  `PipelineStageSpec` values.
- Prepare model inputs from package configuration: chat templates, tokenizers,
  image and video patch packing, latent-token layout, and conditioning frames.
- Offer the modality-oriented `WorldModel` API (`text`, `image`, `video`,
  `action`) and report the capabilities the loaded package actually declares.
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
| `_api.py` | ORT and EP library discovery/registration, `_native` wrappers, manifest spec dataclasses, device-output options, `Pipeline`, `PipelineSession` with its `PipelineSessionSnapshot`, named-checkpoint, and incremental `begin_stage`/`iter_stage` methods, the `StageEvent` value type and the `StageRun` iterator, the `CancellationSource` and `CancellationToken` wrappers and the shared `cancellation`/`timeout` argument handling, latent-dynamics API. |
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
`tests/python/test_preprocessing.py` covers `preprocessing.py` and `media.py`,
and `tests/python/test_guided_generation.py` and
`tests/python/test_image_to_video_smoke.py` cover `generation.py`. Tests that
need a Mobius export skip unless the `mobius` package is installed.
