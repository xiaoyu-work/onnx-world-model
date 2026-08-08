# ONNX World Model Runtime

A C++20 and Python runtime for validated world-model packages exported by
[Mobius](https://github.com/onnxruntime/mobius). ONNX Runtime executes each
neural component; `pipeline.json` defines dataflow, generated inputs, control
flow, recurrent state, scheduler behavior, assets, and public outputs.

## Package contract

The preferred input is a Mobius pipeline directory:

```text
cosmos3-edge/
├── pipeline.json
├── generator/model.onnx
├── reasoner_decoder/model.onnx
├── reasoner_embedding/model.onnx
├── reasoner_vision_encoder/model.onnx
├── video_decoder/model.onnx
├── video_encoder/model.onnx
├── scheduler/scheduler_config.json
└── tokenizer.json
```

The loader supports Mobius pipeline schema 1.x and validates:

- component files and ONNX signatures;
- portable paths, assets, stages, connections, and acyclic dataflow;
- external, generated, defaulted, and stateful input sources;
- transform parameters and runtime references;
- recurrent state lifecycle and public state outputs;
- profile versions, capabilities, dtypes, and execution-provider hints.

Unknown or unsupported semantics fail during loading or execution instead of
silently selecting a fallback.

## Architecture

```text
Python API ── pybind11 ─┐
                       ├─ Pipeline ─ PipelineSession ─ named ONNX Model sessions
C++ API / CLI ─────────┤                  │
                       │                  ├─ generated-input programs
                       │                  ├─ stage strategies and schedulers
                       │                  └─ per-trajectory recurrent state
                       └─ WorldModel / Rollout compatibility API
```

- `Model` runs any ONNX graph using named tensors.
- `Pipeline` owns immutable component sessions and can be shared by callers.
- `PipelineSession` owns one request/trajectory's KV cache, diffusion latent,
  action state, outputs, and stage cursors.
- `WorldModel` and `Rollout` preserve the original latent-dynamics API.
- `Tensor` has value semantics with copy-on-write storage.
- ORT is loaded dynamically, so the library does not link to one ORT binary.

## Build

Requirements:

- CMake 3.24+
- A C++20 compiler
- Internet access during first configure, or local header paths

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

The build SHA256-verifies downloaded ONNX Runtime 1.28 and nlohmann/json
headers. Offline builds can set `ONNXRUNTIME_INCLUDE_DIR` and
`NLOHMANN_JSON_INCLUDE_DIR`.

## Python pipeline API

The Python package locates the shared library from an installed `onnxruntime`
wheel on Windows. On other platforms, set `ONNX_RUNTIME_LIBRARY_PATH` when the
wheel does not ship `libonnxruntime`.

```bash
pip install -e ".[test]"
```

```python
import numpy as np

from onnx_world_model import Pipeline

pipeline = Pipeline("output/cosmos3-edge")
print(pipeline.profile)
print(pipeline.inputs)
print(pipeline.stages)

session = pipeline.create_session()

# Supplying a semantic name can feed multiple compatible component ports.
inputs = {
    "text.token_ids": np.asarray([[1, 42, 7]], dtype=np.int64),
    "vision.pixel_values": pixel_values,
    "diffusion.initial_vision_latent": vision_noise,
    "diffusion.initial_action_latent": action_noise,
}

session.run_stage("reasoner_prompt", inputs)
reasoning = session.run_stage(
    "reasoner_decode",
    options={"max_tokens": 64, "seed": 1234},
)
session.release_stage("reasoner_decode")
world = session.run_stage(
    "world_generation",
    options={
        "mode": "action",
        "action_domain": "droid_lerobot",
        "num_inference_steps": 50,
        "video_latent_frames": latent_frames,
        "video_latent_height": latent_height,
        "video_latent_width": latent_width,
    },
)
session.release_stage("world_generation")
video = session.run_stage(
    "decode_video",
    options={
        "video_latent_frames": latent_frames,
        "video_latent_height": latent_height,
        "video_latent_width": latent_width,
    },
)

action = world["action"]
frames = video["video"]
session.release_stage("decode_video")
```

`run_stage()` executes the strategy declared by the manifest: one pass,
autoregressive decoding, or all iterative scheduler steps. `step_stage()`
executes exactly one pass for applications that manage loops themselves.
Generated tensors or transformed targets can be supplied through `overrides`
using their semantic or qualified port name.

`PipelineSession.run()` executes a selected stage list in declaration order and
releases state according to the manifest:

```python
result = session.run(
    inputs,
    stages=["world_generation", "decode_video"],
    options={
        "num_inference_steps": 50,
        "video_latent_frames": latent_frames,
        "video_latent_height": latent_height,
        "video_latent_width": latent_width,
    },
)
```

Model loading and inference release the Python GIL. Inputs and outputs are
copied between NumPy and engine-owned storage.

## Generic ONNX API

```python
from onnx_world_model import OnnxModel

model = OnnxModel("component/model.onnx")
outputs = model.run({"input_ids": input_ids, "attention_mask": attention_mask})
```

## Latent-dynamics compatibility API

Single-graph exports with this fixed contract remain supported:

```text
observation + action + state
  -> next_state + observation_prediction + reward + continuation
```

```python
from onnx_world_model import WorldModel

model = WorldModel("model.onnx")
result = model.step(observation, action, state)

rollout = model.create_rollout()
result = rollout.step(observation, action)
rollout.reset(batch_size=1)
```

The CLI continues to inspect and execute this compatibility contract:

```bash
onnx-world-model inspect model.onnx --ort-library /path/to/onnxruntime.dll
onnx-world-model step model.onnx --observation "0,0,0,0" --action "0,0"
```

## C++ pipeline API

```cpp
#include "onnx_world_model/onnx_world_model.hpp"

using namespace onnx_world_model;

RuntimeOptions runtime;
runtime.ort_library_path = "/path/to/onnxruntime.dll";

Pipeline pipeline = Pipeline::Load("output/cosmos3-edge", runtime);
PipelineSession session = pipeline.CreateSession();

NamedTensors outputs = session.RunStage(
    "world_generation",
    inputs,
    overrides,
    PipelineRunOptions{
        .strings = {{"mode", "action"}, {"action_domain", "droid_lerobot"}},
        .integers = {{"num_inference_steps", 50}},
    });
```

## Current scope

- ONNX Runtime CPU execution provider; a profiled component must permit CPU.
- Dense tensor inputs and outputs, including FP16 and BF16 host transforms.
- Single-pass, on-demand, state-transition, iterative, and autoregressive
  stages.
- KV-cache, request, sequence, iteration, and session state lifecycles.
- FlowMatch Euler and flow-prediction UniPC order-1/order-2 schedulers.
- Greedy and temperature/top-k/top-p autoregressive sampling.
- Packed layout, attention mask, multimodal position, scheduler timestep, and
  action-domain generated-input programs.
- Scheduler, cast, reshape, packed video finalization, and audio finalization
  transforms.
- Fixed-step stochastic FlowMatch schedules are rejected until their SDE
  sampler is implemented.
