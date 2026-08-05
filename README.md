# ONNX World Model Runtime

A small C++20 inference engine for world models exported by
[Mobius](https://github.com/onnxruntime/mobius). ONNX Runtime is the first
execution backend; the public model and rollout APIs do not expose ORT types.

## Model contract

The engine validates this single-step contract when loading a model:

```text
inputs
  observation  [batch, ...]
  action       [batch, ...]
  state        [batch, ...]

outputs
  next_state             same dtype and shape as state
  observation_prediction same dtype and shape as observation
  reward                 [batch, 1]
  continuation           [batch, 1]
```

Input and output names are part of the contract. A model with missing or extra
values is rejected before inference.

## Architecture

```text
Python API ── pybind11 ─┐
                       ├─ WorldModel (stateless) ─ Backend ─ OrtBackend
C++ API / CLI ─────────┘           │
                                   └─ Rollout (per-trajectory state)
```

- `WorldModel` owns an immutable backend session and executes steps with
  explicit state. It can be shared by independent callers.
- `Rollout` owns the recurrent state for one trajectory and serializes updates.
- `Tensor` has value semantics with copy-on-write storage.
- `Backend` is independent of ONNX Runtime, allowing additional backends later.
- ORT is loaded dynamically, so the C++ library does not link against a
  particular ONNX Runtime binary.

## Build the C++ library and CLI

Requirements:

- CMake 3.24+
- A C++20 compiler
- Internet access during first configure, or `ONNXRUNTIME_INCLUDE_DIR`
  pointing to ONNX Runtime headers

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

The build downloads and SHA256-verifies the pinned ONNX Runtime 1.28 C/C++
headers. It does not download or link an ORT runtime library.

### Inspect a model

```bash
onnx-world-model inspect model.onnx \
  --ort-library /path/to/onnxruntime.dll
```

### Execute one step

The CLI infers tensor shapes from the model and replaces the leading dynamic
dimension with `--batch`. CSV input currently supports float32 models.

```bash
onnx-world-model step model.onnx \
  --ort-library /path/to/onnxruntime.dll \
  --observation "0,0,0,0" \
  --action "0,0"
```

Use `--state` to provide an explicit flattened state. Without it, the CLI
creates a zero state.

## Python API

The Python package builds the same C++ core and locates the shared library from
an installed `onnxruntime` wheel on Windows. On other platforms, set
`ONNX_RUNTIME_LIBRARY_PATH` if the wheel does not contain `libonnxruntime`.

```bash
pip install -e ".[test]"
```

```python
import numpy as np

from onnx_world_model import WorldModel

model = WorldModel("model.onnx")

observation = np.zeros((1, 4), dtype=np.float32)
action = np.zeros((1, 2), dtype=np.float32)
state = np.zeros((1, 3), dtype=np.float32)

# Stateless execution with explicit recurrent state.
result = model.step(observation, action, state)

# One independent state context per trajectory.
rollout = model.create_rollout()
first = rollout.step(observation, action)
second = rollout.step(observation, action)
rollout.reset(batch_size=1)
```

Model loading and inference release the Python GIL. NumPy inputs are copied
into engine-owned storage before execution; outputs are returned as independent
NumPy arrays.

## C++ API

```cpp
#include <array>
#include <span>

#include "onnx_world_model/onnx_world_model.hpp"

using namespace onnx_world_model;

RuntimeOptions options;
options.ort_library_path = "/path/to/onnxruntime.dll";
WorldModel model = WorldModel::Load("model.onnx", options);

const std::array<float, 4> observation_values{};
const std::array<float, 2> action_values{};
const std::array<float, 3> state_values{};

Tensor observation =
    Tensor::FromValues<float>({1, 4}, std::span(observation_values));
Tensor action = Tensor::FromValues<float>({1, 2}, std::span(action_values));
Tensor state = Tensor::FromValues<float>({1, 3}, std::span(state_values));

StepOutput output = model.Step(observation, action, state);
```

## Testing with Mobius

The integration fixture is exported through Mobius rather than handwritten:

```bash
python tools/export_mobius_test_model.py build/testdata
pytest
```

The tests cover model-contract discovery, stateless inference, stateful rollout,
reset behavior, dtype rejection, and PyTorch/Mobius-compatible outputs.

## Current scope

- ONNX Runtime CPU/default execution provider
- Dense tensor inputs and outputs
- Fixed non-batch state dimensions for automatic zero-state creation
- One deterministic step graph; sampling policies remain outside the engine
- CLI CSV input for float32 models
