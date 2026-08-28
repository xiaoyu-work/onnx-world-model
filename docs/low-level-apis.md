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
materializes device outputs.

Pipeline execution currently materializes component outputs before applying its
CPU transforms. Device-to-device pipeline connections are enabled separately by
the connection planner.

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

## Pipeline

For package-level tensor and stage execution, see the
[Pipeline API](pipeline-api.md).
