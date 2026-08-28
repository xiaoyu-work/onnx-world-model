# src

## Purpose

Implements the `onnx_world_model` C++ library declared in
`include/onnx_world_model`: ONNX Runtime loading and session management, tensor
marshalling, Mobius pipeline manifest parsing and validation, staged pipeline
execution, and the fixed latent-dynamics compatibility layer.

## Responsibilities

- Confine every ONNX Runtime dependency to `dynamic_library.*` and
  `ort_backend.*` so the rest of the library stays independent of a specific
  ORT build and links against no ORT binary.
- Turn `pipeline.json` into a `PipelineManifest` and reject anything the
  runtime cannot execute at load time rather than at inference time.
- Execute pipeline stages, including generated inputs, transforms, diffusion
  schedulers, classifier-free guidance, autoregressive decoding, and recurrent
  state lifecycles.
- Keep `Pipeline` immutable and shareable while `PipelineSession` owns one
  request's mutable state.
- Preserve device-backed tensor storage through the session and materialize a
  device source to CPU only once, at the outer boundary of each host-evaluated
  transform, scheduler, sampler, or generated-input program.

## Key Files

| File | Responsibility |
|---|---|
| `dynamic_library.hpp/.cpp` | RAII shared-library handle; binds the `OrtApi` table once per process. |
| `ort_backend.hpp/.cpp` | The only ORT-facing translation unit pair: shares the process-wide ORT environment, builds sessions, applies providers, and retains I/O-bound outputs in ORT-owned device buffers. |
| `tensor.cpp` | Canonical tensor devices, owned CPU buffers, checked shape arithmetic, explicit CPU materialization, and copy-on-write mutation. |
| `model.cpp` | `Model` facade, provider-name normalization, tensor-versus-signature validation. |
| `world_model.cpp` | `WorldModel` contract enforcement and `Rollout` recurrent state. |
| `pipeline.cpp` | `pipeline.json` parsing and `PipelinePackage` / `Pipeline` loading. |
| `pipeline_manifest_common.hpp/.cpp` | JSON field, token, and portable-name checks shared by parsing and validation. |
| `pipeline_manifest_validation.hpp/.cpp` | Semantic validation of a parsed manifest: dataflow, programs, stage options, capabilities, state lifecycles. |
| `pipeline_session.cpp` | `PipelineSession::Impl`, the staged execution engine, all per-trajectory state, and the device-versus-host materialization boundaries. |

`dynamic_library.hpp`, `ort_backend.hpp`, `pipeline_manifest_common.hpp`, and
`pipeline_manifest_validation.hpp` are internal: they live in
`onnx_world_model::detail` and are not installed.

## Dependencies

- Public declarations come from `include/onnx_world_model`; this directory adds
  no public type of its own.
- Third-party: ONNX Runtime C/C++ headers (loaded at run time, resolved by
  `cmake/FetchOrtHeaders.cmake`) and nlohmann/json (header only, resolved by
  `cmake/FetchJsonHeader.cmake`).
- Internal direction is one-way and acyclic:

```text
dynamic_library -> ort_backend -> model -> world_model
                                    `-> pipeline -> pipeline_session
pipeline_manifest_common -> pipeline_manifest_validation -> pipeline
```

`pipeline_manifest_validation.cpp` never parses raw JSON documents itself; it
receives an already-populated `PipelineManifest` from `pipeline.cpp`.

## Tests

```console
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

`tests/cpp/tensor_test.cpp`, `tests/cpp/model_test.cpp`,
`tests/cpp/pipeline_test.cpp`, and `tests/cpp/pipeline_device_test.cpp` cover
this directory directly; `tests/cpp/pipeline_test.cpp` is the primary coverage
for manifest parsing and validation, and `tests/cpp/pipeline_device_test.cpp`
is the primary coverage for the device-versus-host materialization boundaries
in `pipeline_session.cpp`. `tests/python/` exercises the same code through the
`_native` extension module.
