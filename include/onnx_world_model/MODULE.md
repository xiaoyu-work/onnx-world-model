# include/onnx_world_model

## Purpose

Declares the installed public C++20 API of the `onnx_world_model` library: the
tensor value type, the error taxonomy, cancellation and deadline types, model
and pipeline session contracts, and the fixed latent-dynamics compatibility
types.

## Responsibilities

- Define every type, function, and enumeration that a C++ consumer or the
  pybind11 binding layer is allowed to use.
- Keep implementation details out of the installed surface: `PipelineSession`,
  `PipelineSessionSnapshot`, and `StageRun` hide their state behind a private
  `Impl` pointer, `CancellationToken` and `CancellationSource` hide theirs
  behind an opaque `detail::CancellationState`, `Pipeline` holds its shared
  admission controller as an opaque `detail::PipelineScheduler` pointer and its
  shared telemetry collector as an opaque `detail::PipelineTelemetry` pointer,
  and `Model` and `WorldModel` hold abstract backend pointers rather than ONNX
  Runtime types.
- Stay free of ONNX Runtime and nlohmann/json includes so that consumers do not
  need those headers on their include path.

## Key Files

| File | Contents |
|---|---|
| `error.hpp` | `ErrorCode` categories and the `Error` exception thrown by every entry point. |
| `cancellation.hpp` | `CancellationReason`, the copyable observer `CancellationToken` with its boundary `ThrowIfCancellationRequested` and blocking `WaitForCancellation`, and the move-only `CancellationSource` that owns the cancellable state and its optional deadline. |
| `tensor.hpp` | `DataType`, canonical `TensorDevice` identities, the ORT-independent `TensorBuffer` contract, and the device-aware copy-on-write `Tensor`. |
| `backend.hpp` | `TensorSpec` with its optional runtime `device`, `ModelMetadata`, `ValidateTensor`, `StepInput`, `StepOutput`, `Backend`. |
| `model.hpp` | `RuntimeOptions`, device-output policy, provider discovery and library registration, `NamedTensors`, `ModelRunOptions` with its cancellation token and optional ONNX Runtime profile-file prefix, `ModelBackend` with its default cancellable and default `ModelRunOptions` `Run` overloads, and `Model`. |
| `pipeline.hpp` | Manifest value types, `PipelineManifest`, `ComponentPlacement` and `PipelinePlacementOptions`, the `PipelineTransferKind`, `PipelineTransfer`, and `PipelineTransferPlan` classification of every connection, the `PrecisionPort` and `ComponentPrecisionReport` inspection values behind `PipelinePackage::precision_report` and `Pipeline::precision_report`, `PipelinePackage`, `Pipeline` with its shared `PipelineSchedulingOptions` admission limits and its `PipelineSchedulingStats` reading of them, its opt-in `PipelineTelemetryOptions` -- counters plus the optional `trace_directory` and `max_trace_records` that add per-run ONNX Runtime node traces -- and the `PipelineComponentStats`, `PipelineStageStats`, `PipelineAdmissionStats`, `PipelineTransferStats`, `PipelineCallOutcome`, `PipelineTraceRecord`, and `PipelineTelemetrySnapshot` reading of those counters and records, `PipelineSession` with its incremental `BeginStage` and named-checkpoint methods, `PipelineSessionSnapshot`, `StageEventKind`, `StageEvent`, `StageRun`, `PipelineRunOptions`. |
| `world_model.hpp` | `WorldModel` and `Rollout`, the fixed three-input/four-output latent-dynamics API. |
| `onnx_world_model.hpp` | Umbrella header that includes all of the above. |

## Dependencies

Internal include order is strictly layered and acyclic:

```text
error.hpp <- cancellation.hpp <- model.hpp <- pipeline.hpp
error.hpp <- tensor.hpp <- backend.hpp <- model.hpp
                                   `<- world_model.hpp
```

Outside this directory these headers depend only on the C++ standard library.
`src/` implements them and `bindings/python_module.cpp` consumes them through
`onnx_world_model.hpp`; nothing here depends on either.

`CMakeLists.txt` installs this directory verbatim, so a change to any
declaration is a change to the installed ABI and to the exported CMake package
`onnx_world_model::onnx_world_model`. The cancellation surface added in
version 0.3.0 was such a change: `CancellationToken` is a new member of
`PipelineRunOptions`, `ModelBackend` gained a virtual method, and `StageRun`
gained `RequestCancellation`. Version 0.4.0 adds
`CancellationToken::WaitForCancellation` and
`CancellationSource::WaitForCancellation`. Version 0.5.0 adds
`PipelineSchedulingOptions`, a second `Pipeline` constructor parameter, a
third `Pipeline::Load` parameter, the `PipelineSchedulingStats` value type
with `Pipeline::scheduling_stats` that reports the shared controller's
admitted and queued counts, and a `Pipeline` data member holding that
controller, which changes the class layout; the library
therefore carries `SOVERSION 0.5` and a C++ consumer built against 0.4 must be
recompiled. Version 0.6.0 adds a final `std::optional<TensorDevice> device`
member to `TensorSpec`, the `ComponentPlacement` and `PipelinePlacementOptions`
load-time placement types with a final defaulted parameter on
`PipelinePackage::Load` and `Pipeline::Load`, the `PipelineTransferKind`,
`PipelineTransfer`, and `PipelineTransferPlan` value types with
`PipelinePackage::transfer_plan` and `Pipeline::transfer_plan`, a final
defaulted `device_outputs_enabled` parameter on the `PipelinePackage`
constructor, and a `PipelinePackage` data member holding the computed plan;
`TensorSpec` and `PipelinePackage` therefore both change layout, so the library
carries `SOVERSION 0.6` and a consumer built against 0.5 must be recompiled.
Placement is deliberately absent from `Pipeline(PipelinePackage, scheduling)`,
whose sessions are already built. Version 0.7.0 adds the opt-in
`PipelineTelemetryOptions` as a final defaulted third `Pipeline` constructor
parameter and a final defaulted fifth `Pipeline::Load` parameter, the
`PipelineComponentStats`, `PipelineStageStats`, `PipelineAdmissionStats`,
`PipelineTransferStats`, and `PipelineTelemetrySnapshot` value types with
`Pipeline::telemetry_snapshot` and `Pipeline::ResetTelemetry`, a third
parameter on the private `PipelineSession` constructor, and a `Pipeline` data
member holding the shared collector; `Pipeline` therefore changes layout again,
so the library carries `SOVERSION 0.7` and a consumer built against 0.6 must be
recompiled. Version 0.8.0 adds `ModelRunOptions` -- the per-call value carrying
a `CancellationToken` and an optional `profile_file_prefix` -- a third virtual
`ModelBackend::Run` overload taking it with a default implementation that
forwards to the cancellable one, a matching `Model::Run` overload the other two
delegate to, the `trace_directory` and `max_trace_records` members on
`PipelineTelemetryOptions`, the `PipelineCallOutcome` enumeration and the
`PipelineTraceRecord` value type, and the `traces`, `dropped_traces`, and
`failed_traces` members on `PipelineTelemetrySnapshot`; `ModelBackend`'s vtable
and both telemetry structs therefore change layout, so the library carries
`SOVERSION 0.8` and a consumer built against 0.7 must be recompiled. Every one
of those parameters is defaulted and every new virtual has a default
implementation, so source that already compiled keeps compiling and keeps its
unlimited, unplaced, unmeasured, untraced behavior. Version 0.9.0 adds the `PrecisionPort` and `ComponentPrecisionReport` value types and the `PipelinePackage::precision_report` and `Pipeline::precision_report` member functions that return them by value; no existing type changes layout and no member is added to either class, because the report is computed on demand from the manifest and the loaded session metadata, but a declaration was added to installed headers, so the library carries `SOVERSION 0.9`. The report is inspection only: it adds no precision policy, never verifies a declared parameter dtype against weights, and cannot observe quantized operators or initializers through the ONNX Runtime session API.

## Tests

Header declarations are exercised through the implementation:

```console
cmake --build --preset dev
ctest --preset dev
```

`tests/cpp/tensor_test.cpp` covers `tensor.hpp`, `tests/cpp/model_test.cpp`
covers `model.hpp` and `backend.hpp`,
`tests/cpp/cancellation_test.cpp` covers `cancellation.hpp`, and
`tests/cpp/pipeline_test.cpp`, `tests/cpp/pipeline_device_test.cpp`,
`tests/cpp/pipeline_snapshot_test.cpp`,
`tests/cpp/pipeline_stream_test.cpp`,
`tests/cpp/pipeline_cancellation_test.cpp`,
`tests/cpp/pipeline_scheduler_test.cpp`,
`tests/cpp/pipeline_telemetry_test.cpp`,
`tests/cpp/pipeline_trace_test.cpp`, and
`tests/cpp/pipeline_precision_test.cpp` cover `pipeline.hpp`;
`tests/cpp/pipeline_device_test.cpp` is the primary coverage for
`PipelineTransferPlan` and for the API shape that keeps
`PipelinePlacementOptions` off the already-built-package constructor,
`tests/cpp/pipeline_telemetry_test.cpp` is the primary coverage for
`PipelineTelemetrySnapshot` and for what its counters mean, and
`tests/cpp/pipeline_trace_test.cpp` is the primary coverage for
`ModelRunOptions::profile_file_prefix`, `PipelineTraceRecord`, and the trace
fields of `PipelineTelemetryOptions` and `PipelineTelemetrySnapshot`, and
`tests/cpp/pipeline_precision_test.cpp` is the primary coverage for
`PrecisionPort`, `ComponentPrecisionReport`, and the two `precision_report`
accessors.
`tests/python/` reaches the same declarations through the `_native` extension
module.
