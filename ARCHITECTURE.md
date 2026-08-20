# ONNX World Model Runtime Architecture

## Purpose

This repository is a C++20 and Python runtime that executes validated
world-model packages exported by [Mobius](https://github.com/onnxruntime/mobius).
A package is a directory whose `pipeline.json` manifest declares neural
components, dataflow, generated inputs, control flow, recurrent state,
scheduler behavior, assets, and public outputs. The runtime loads that manifest,
validates it against what it can actually execute, and drives ONNX Runtime for
text, image, video, and action generation.

The guiding rule is that unknown or unsupported semantics fail during loading or
execution rather than silently selecting a fallback.

## System Context

```text
Mobius exporter ──▶ pipeline package on disk ──▶ this runtime ──▶ generated
                    (pipeline.json,                               text, images,
                     component ONNX models,                       video, actions
                     tokenizer.json, scheduler_config.json,
                     preprocessor and recipe configuration)

                              │
                              ▼
                    ONNX Runtime shared library
                    (loaded dynamically at run time)
```

External systems:

- **Mobius packages** — the only supported input format. The loader accepts
  pipeline schema major version 1.
- **ONNX Runtime** — resolved at run time, never linked. The Python package
  finds the library inside the installed `onnxruntime` wheel, or at the path in
  `ONNX_RUNTIME_LIBRARY_PATH`. Available execution providers therefore depend on
  the ORT build present on the machine.
- **Build-time header downloads** — `cmake/FetchOrtHeaders.cmake` and
  `cmake/FetchJsonHeader.cmake` fetch and SHA256-verify the ONNX Runtime and
  nlohmann/json headers, unless `ONNXRUNTIME_INCLUDE_DIR` and
  `NLOHMANN_JSON_INCLUDE_DIR` point at local copies.

There is no server, database, or outbound network call at run time.

## Components

| Component | Location | Responsibility |
|---|---|---|
| Public C++ API | `include/onnx_world_model/` | Installed declarations: `Tensor`, `Error`, `Model`, `Pipeline`, `PipelineSession`, `WorldModel`, `Rollout`. |
| Core library | `src/` | ORT loading, tensor marshalling, manifest parsing and validation, staged execution. |
| Python binding | `bindings/python_module.cpp` | The `_native` pybind11 module and NumPy-to-`Tensor` conversion. |
| Python package | `python/onnx_world_model/` | Typed wrappers, preprocessing, media handling, and the modality-oriented generation API. |
| C++ tests | `tests/cpp/` | Stub-backend tests registered with CTest. |
| Python tests | `tests/python/` | Pytest suite over the installed package. |
| Tools | `tools/` | Test-package exporter and an image-to-video smoke script. |
| Documentation | `docs/` | Pipeline API, low-level APIs, Cosmos3 Edge validation results. |

Within `src/` the runtime is layered:

- `dynamic_library` loads the ORT shared library and binds `OrtApi` once per
  process.
- `ort_backend` is the only other translation unit that touches ORT; it builds
  sessions and marshals tensors.
- `model` validates named tensors against graph signatures.
- `pipeline` parses `pipeline.json`; `pipeline_manifest_validation` checks the
  parsed manifest's semantics; `pipeline_manifest_common` holds the checks both
  share.
- `pipeline_session` executes stages and owns per-trajectory state.
- `world_model` provides the fixed latent-dynamics compatibility API.

## Dependency Rules

- `include/onnx_world_model/` depends only on the C++ standard library. It must
  not include ONNX Runtime or nlohmann/json headers, so consumers do not need
  them on their include path.
- ONNX Runtime headers appear only in `src/dynamic_library.cpp` and
  `src/ort_backend.cpp`. Every other file reaches ORT through
  `detail::CreateOrtBackend`.
- `src/` internal headers live in `onnx_world_model::detail` and are not
  installed.
- `pipeline_manifest_validation` receives an already-parsed `PipelineManifest`
  and never parses a raw document; `pipeline.cpp` never re-implements semantic
  checks.
- `bindings/` depends on `include/`, never on `src/` internals.
- Python intra-package imports are one-way:
  `_native <- _api <- generation` and `media <- preprocessing <- generation`.
  `media.py` must not import `preprocessing.py`.
- Tests depend on public surfaces only: `tests/cpp/` includes public headers,
  and `tests/python/` imports the installed package.
- The library links against no ONNX Runtime binary at build time.

## Data Flow

Loading:

1. `Pipeline.__init__` reads `pipeline.json` and calls `_native.Pipeline`.
2. `PipelinePackage::Load` resolves component paths inside the package root,
   parses the manifest, and runs `detail::ValidateManifest`.
3. Each component ONNX file is opened as a `Model`, and its signature is checked
   against the manifest.
4. `Pipeline` holds the resulting components immutably behind a `shared_ptr`.

Generation, using video as the example:

1. `WorldModel.video.generate(...)` forwards to `_GenerationRuntime.generate`.
2. `WorldModelPreprocessor` renders the chat template, tokenizes the prompt, and
   prepares conditioning frames and latents, delegating image and video decoding
   and patch packing to `media.py`.
3. `Pipeline.create_session()` produces a `PipelineSession` that owns this
   request's KV cache, diffusion latent, guidance values, stage cursors, and
   recurrent state.
4. `PipelineSession.run_stage` calls into `PipelineSession::Impl`, which resolves
   each component's inputs from external values, generated-input programs,
   transforms, defaults, and recurrent state, then runs the component through
   ONNX Runtime.
5. Stage kinds drive control flow: `single_pass`, `autoregressive`, `iterative`
   with a diffusion scheduler, and `state_transition`.
6. `ReleaseStage` frees state whose declared `release_after` names that stage;
   `outputs()` collects the manifest's public outputs.
7. The generator unpacks latent tokens and returns a modality-specific output
   dataclass.

Tensors cross the language boundary as NumPy arrays converted to `Tensor` values
with copy-on-write storage; `float16` and `bfloat16` cross as raw two-byte views.

## Entry Points

Python:

- `onnx_world_model.WorldModel.from_pretrained(package_path)` — the primary
  generation API, exposing `.text`, `.image`, `.video`, and `.action`, each with
  a `generate()` method.
- `onnx_world_model.Pipeline` and `PipelineSession` — direct stage execution.
- `onnx_world_model.OnnxModel` — one ONNX graph with named tensors.
- `onnx_world_model.LatentDynamicsModel` and `Rollout` — the fixed
  latent-dynamics API.
- `onnx_world_model.available_execution_providers` and
  `supported_pipeline_capabilities` — capability queries.

C++:

- `onnx_world_model::Pipeline::Load` then `Pipeline::CreateSession` and
  `PipelineSession::RunStage` or `StepStage`.
- `onnx_world_model::Model::Load` and `Model::Run`.
- `onnx_world_model::WorldModel::Load`, `WorldModel::Step`, and `Rollout`.

Command line:

- `python tools/image_to_video_smoke.py PACKAGE IMAGE [--dry-run]`
- `python tools/export_mobius_test_model.py OUTPUT_DIR`

Build and install targets: the `onnx_world_model` CMake library, the exported
package `onnx_world_model::onnx_world_model`, the `_native` Python extension,
and the `onnx-world-model` wheel built by scikit-build-core.

## Cross-Cutting Constraints

- **Fail fast, never fall back.** Unsupported capabilities, unknown manifest
  fields, unknown stage or generator kinds, unavailable execution providers, and
  mismatched tensor signatures raise instead of degrading silently.
- **Single error type.** All C++ failures throw `onnx_world_model::Error` with an
  `ErrorCode`; pybind11 surfaces it as `WorldModelError`.
- **Validate at the boundary.** `Model::Run` checks every input and output
  tensor; manifest semantics are validated once at load time.
- **Ownership split.** `Pipeline` is immutable and shareable; `PipelineSession`
  is move-only and owns exactly one request or trajectory. Session state is
  guarded by `impl_->mutex`, and `Rollout` guards its state by its own mutex.
- **Value semantics.** `Tensor` copies are cheap and copy-on-write, so a shared
  buffer is cloned before mutation.
- **Portable, contained packages.** Every manifest path must stay inside the
  package root, and component names must be safe portable path segments.
- **Deterministic sampling.** A `PipelineSession` seeds its own random engine, so
  a given seed reproduces a generation.
- **Run-time ORT binding.** ORT is initialized once per process from a single
  library path; a second, different path is an error.
- **Compatibility aliases.** `WorldModelPipeline` and `LegacyWorldModel` must
  keep pointing at `Pipeline` and `LatentDynamicsModel`.

## Current Scope

Text, image, video, and action generation from Mobius packages; one image or
video per text-generation request; image-to-video conditioning and
classifier-free guidance for packages that declare them. Output media encoding
is not included, and fixed-step stochastic FlowMatch schedules are not yet
supported.
