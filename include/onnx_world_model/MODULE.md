# include/onnx_world_model

## Purpose

Declares the installed public C++20 API of the `onnx_world_model` library: the
tensor value type, the error taxonomy, model and pipeline session contracts,
and the fixed latent-dynamics compatibility types.

## Responsibilities

- Define every type, function, and enumeration that a C++ consumer or the
  pybind11 binding layer is allowed to use.
- Keep implementation details out of the installed surface: `PipelineSession`
  hides its state behind a private `Impl` pointer, and `Model` and `WorldModel`
  hold abstract backend pointers rather than ONNX Runtime types.
- Stay free of ONNX Runtime and nlohmann/json includes so that consumers do not
  need those headers on their include path.

## Key Files

| File | Contents |
|---|---|
| `error.hpp` | `ErrorCode` categories and the `Error` exception thrown by every entry point. |
| `tensor.hpp` | `DataType`, canonical `TensorDevice` identities, the ORT-independent `TensorBuffer` contract, and the device-aware copy-on-write `Tensor`. |
| `backend.hpp` | `TensorSpec`, `ModelMetadata`, `ValidateTensor`, `StepInput`, `StepOutput`, `Backend`. |
| `model.hpp` | `RuntimeOptions`, `GraphOptimizationLevel`, provider-name helpers, `NamedTensors`, `ModelBackend`, `Model`. |
| `pipeline.hpp` | Manifest value types, `PipelineManifest`, `PipelinePackage`, `Pipeline`, `PipelineSession`, `PipelineRunOptions`. |
| `world_model.hpp` | `WorldModel` and `Rollout`, the fixed three-input/four-output latent-dynamics API. |
| `onnx_world_model.hpp` | Umbrella header that includes all of the above. |

## Dependencies

Internal include order is strictly layered and acyclic:

```text
error.hpp <- tensor.hpp <- backend.hpp <- model.hpp <- pipeline.hpp
                                              `<- world_model.hpp
```

Outside this directory these headers depend only on the C++ standard library.
`src/` implements them and `bindings/python_module.cpp` consumes them through
`onnx_world_model.hpp`; nothing here depends on either.

`CMakeLists.txt` installs this directory verbatim, so a change to any
declaration is a change to the installed ABI and to the exported CMake package
`onnx_world_model::onnx_world_model`.

## Tests

Header declarations are exercised through the implementation:

```console
cmake --build --preset dev
ctest --preset dev
```

`tests/cpp/tensor_test.cpp` covers `tensor.hpp`, `tests/cpp/model_test.cpp`
covers `model.hpp` and `backend.hpp`, and `tests/cpp/pipeline_test.cpp` covers
`pipeline.hpp`. `tests/python/` reaches the same declarations through the
`_native` extension module.
