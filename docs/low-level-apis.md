# Low-level APIs

## Generic ONNX model

`OnnxModel` executes one ONNX graph with named tensor inputs and outputs:

```python
from onnx_world_model import OnnxModel

model = OnnxModel("component/model.onnx")
outputs = model.run(
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
)
```

It accepts the same `providers` and `provider_options` arguments as
`WorldModel`.

`run()` also accepts a keyword-only `cancellation` token or a `timeout` in
seconds, which are mutually exclusive:

```python
from onnx_world_model import CancellationSource, DeadlineExceededError

source = CancellationSource()
outputs = model.run(inputs, cancellation=source.token())   # source.cancel() stops it

try:
    outputs = model.run(inputs, timeout=5.0)
except DeadlineExceededError:
    ...
```

The token is checked before the graph runs and after its outputs are
validated, and the runtime terminates an in-flight `Session::Run` through a
fresh per-call `Ort::RunOptions`. A deadline reaches that termination callback
from one shared process-wide watchdog rather than from the next boundary, so a
`timeout` stops a call that is already blocked. ONNX Runtime honors the flag
between graph nodes, so a single long-running kernel still finishes before the
call unwinds. See
[Cancellation and deadlines](pipeline-api.md#cancellation-and-deadlines) for
the full contract.

In C++ this is `Model::Run(inputs, cancellation)`. `ModelBackend` declares the
cancellable overload virtually with a default implementation that checks the
token around the historical one-argument `Run`, so an external backend keeps
compiling and still stops at those boundaries. A backend that can block with no
boundary of its own can park on `CancellationToken::WaitForCancellation()`
instead of polling.

### Per-call node profiling

`run()` also accepts a keyword-only `profile_prefix`, which turns on ONNX
Runtime's node-level profiling for that one call:

```python
outputs = model.run(inputs, profile_prefix="traces/encoder")
# traces/encoder_2026-01-31_12-00-00_123.json
```

It is a *prefix*, not a file name: ONNX Runtime appends its own local
timestamp and `.json`, so the file is found by listing that directory for the
prefix rather than by predicting a name. Profiling is enabled on the call's own
run options, so the session is never rebuilt and a concurrent call that asked
for no trace still gets none. The return value is the outputs and nothing else,
and a trace that fails to appear never fails the call.

In C++ the three `Model::Run` overloads collapse onto one:
`Model::Run(inputs, ModelRunOptions{.cancellation = token,
.profile_file_prefix = prefix})`. `ModelBackend` declares that overload
virtually too, with a default implementation that drops the prefix and
forwards to the cancellable overload, so a backend that cannot profile keeps
compiling and simply produces no file. The prefix reaches ONNX Runtime in the
platform's native path characters, so a non-ASCII directory is not narrowed on
the way. Pipelines wire the same mechanism up automatically: see
[ONNX Runtime node traces](pipeline-api.md#onnx-runtime-node-traces).

## Device-aware C++ tensors

The installed C++ `Tensor` API can wrap an immutable `TensorBuffer` supplied by
an execution backend. A buffer declares a canonical device name and ID, whether
host access is available, its raw address and byte size, and a synchronous
`CopyToCpu` operation. Calling `bytes()`, `values<T>()`, or `mutable_bytes()` on
device-only storage raises; call `Tensor::CopyToCpu()` at an explicit host
boundary.

Owned tensor constructors still allocate CPU storage and preserve copy-on-write
mutation. `Model::Run` uses ORT I/O binding and binds outputs to CPU by default.
C++ callers may register an EP library with
`RegisterExecutionProviderLibrary`, set `RuntimeOptions.device_outputs = true`,
and receive outputs in the memory location assigned by graph partitioning. An
ORT-backed output can bind directly as a later model input; foreign device
buffers stage through CPU. EP registration is required because device-to-CPU
materialization uses the process-wide ORT data-transfer registry.

Python `OnnxModel.run()` always returns independent NumPy arrays and therefore
materializes device outputs. Python callers configure the same behavior with
`register_execution_provider_library(...)` and `device_outputs=True` on
`OnnxModel`, `Pipeline`, `LatentDynamicsModel`, or `WorldModel`.

Each model port also reports where ONNX Runtime placed it. In C++ that is
`TensorSpec::device`, a `std::optional<TensorDevice>` read once per session
from the graph partitioner's memory plan; in Python it is `TensorSpec.device`,
a `DeviceSpec(type, id)` or `None`:

```python
model = OnnxModel("component/model.onnx")
print(model.metadata.outputs[0].device)   # DeviceSpec(type='cpu', id=0)
```

`None` means the backend does not report placement, which is what a custom
`ModelBackend` written before this member existed reports. The device is
runtime placement rather than part of the graph signature, so it never takes
part in tensor validation.

`PipelineSession::RunStage` and `StepStage` preserve device storage. Caller
inputs, overrides, component outputs, recurrent state, and public outputs keep
whatever buffer the producer supplied, and a transform-free connection hands
the identical `TensorBuffer` to the next component. Rank adaptation of an
external input and the `reshape` transform only relabel axes, so they reuse the
same buffer as well. A transform this runtime evaluates on the host — `cast`,
scheduler steps, guidance combination, packed video and audio finalization,
token sampling, and the generated-input programs that read tensor values —
materializes each device operand once at its own boundary and then reads only
host memory. A public output can therefore be device-only, so call
`Tensor::CopyToCpu()` before `bytes()` or `values<T>()`.

Every one of those materializations goes through a single boundary inside the
session, so a pipeline with telemetry enabled counts each real copy once, with
its byte total, in
`PipelineTelemetrySnapshot::transfers.device_to_host_copies` and
`device_to_host_bytes`. A source that is already ordinary CPU memory is handed
back without a copy and is never counted. There is deliberately no
host-to-device counter: uploads happen inside ONNX Runtime when it binds an
input, so this runtime cannot measure them. What it can measure exactly is
where each component's inputs lived when they were presented, which is what
`component_input_bytes_device_resident` and `component_input_bytes_host`
report; that is presentation rather than transfer, so a tensor presented twice
is counted twice. See
[Telemetry](pipeline-api.md#telemetry) for the full contract.

## Latent dynamics

`LatentDynamicsModel` preserves the original fixed single-graph contract:

```text
observation + action + state
  -> next_state + observation_prediction + reward + continuation
```

```python
from onnx_world_model import LatentDynamicsModel

model = LatentDynamicsModel("model.onnx")
result = model.step(observation, action, state)

rollout = model.create_rollout()
result = rollout.step(observation, action)
rollout.reset(batch_size=1)
```

`LegacyWorldModel` is an explicit alias for `LatentDynamicsModel`.

This fixed API does not accept a cancellation token in this milestone; use
`Pipeline` and `PipelineSession` when a call must be interruptible. It also
takes no concurrency limits, no placement, no telemetry, and no profiling:
`max_concurrent_executions`, `max_concurrent_by_stage_kind`,
`component_placement`, `allow_unpreferred_providers`, `enable_telemetry`,
`telemetry_trace_directory`, and `max_trace_records` are pipeline-only
options, so `OnnxModel` and `LatentDynamicsModel` do not accept them —
`OnnxModel.run` takes its own per-call `profile_prefix` instead. A
`Model::Run` a caller makes directly is not a pipeline component call, so it
is never counted by pipeline telemetry and never produces a
`PipelineTraceRecord`.

## Pipeline

For package-level tensor and stage execution — including the admission
scheduling that caps concurrent executions, the `scheduling_stats` reading of
it, the opt-in `enable_telemetry` and the `telemetry_snapshot` reading of it,
the `telemetry_trace_directory` that adds one ONNX Runtime node trace per
component call, the per-component `component_placement` overrides, the
`transfer_plan` they produce, and the `precision_report` that states what the
runtime can honestly observe about numeric precision — see the
[Pipeline API](pipeline-api.md).

### Precision is not observable from a single graph

There is deliberately no precision inspection on `OnnxModel`. Its
`metadata.inputs` and `metadata.outputs` already report the real port dtypes,
and that is the whole of what a loaded ONNX Runtime session exposes: the
`Session` public API this runtime uses does not expose ordinary initializers or
nodes, so nothing here can see a weight tensor, a MatMulNBits block, a QDQ
pair, or a QLinear operator. A declared parameter dtype exists only in a
package manifest, which is why the precision report lives on `Pipeline` and
`PipelinePackage` and is explicit about being an unverified producer claim.
