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
                       └─ LatentDynamicsModel / Rollout compatibility API
```

- `Model` runs any ONNX graph using named tensors.
- `Pipeline` owns immutable component sessions and can be shared by callers.
- `PipelineSession` owns one request/trajectory's KV cache, diffusion latent,
  action state, outputs, and stage cursors.
- `LatentDynamicsModel` and `Rollout` preserve the original fixed
  latent-dynamics API.
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

## Generation API

`WorldModel` is the public generation entry point. APIs are grouped by
modality; each modality uses `generate()` and returns its own output type:

```python
from onnx_world_model import WorldModel

model = WorldModel.from_pretrained(
    "output/cosmos3-edge",
    providers=["cuda", "cpu"],
    provider_options={"cuda": {"device_id": 0}},
)
print(model.capabilities)  # ("text", "image", "video", "action")

# Text / visual-language generation.
answer = model.text.generate(
    "What is shown in this image?",
    image="frame.png",
    max_tokens=64,
    do_sample=False,
)
print(answer.text)

# Video understanding uses the same text interface.
summary = model.text.generate(
    "Summarize what happens in this video.",
    video="clip.mp4",
    video_sample_fps=2,
    max_tokens=128,
)
print(summary.text)

# Video generation.
video = model.video.generate(
    "A robot moves the red block to the left.",
    frames=17,
    height=256,
    width=256,
    num_inference_steps=50,
    seed=1234,
)
print(video.video.shape)

# Action generation.
action = model.action.generate(
    "Move the red block to the left.",
    domain="droid_lerobot",
    steps=16,
    num_inference_steps=50,
    seed=1234,
)
print(action.actions.shape)

# Image generation for packages whose VAE supports one latent frame.
image = model.image.generate(
    "A red robot on a white background.",
    height=256,
    width=256,
)
print(image.images.shape)
```

The modality objects are also public (`TextGenerator`, `ImageGenerator`,
`VideoGenerator`, and `ActionGenerator`), but they normally come from one
loaded `WorldModel` so component sessions and preprocessing assets are shared.

Raw `image=` belongs to `model.text.generate()` and conditions the visual
Reasoner. `video=` accepts a video path, a frame sequence, or a
`[T,H,W,C]`/`[T,C,H,W]` NumPy array. Packages with a declared video-
understanding contract apply frame sampling, timestamps, packed patchification,
video-token expansion, and `grid_thw` automatically. `image` and `video` are
mutually exclusive in one request.

Video sampling options:

- `video_fps`: source FPS when an array/frame sequence has no container
  metadata;
- `video_sample_fps`: target sampling rate, defaulting to the package contract;
- `video_num_frames`: exact uniform sample count, mutually exclusive with
  `video_sample_fps`.

Video understanding requires a package that declares
`metadata.vision_understanding.routing.video` and exposes
`reasoner_embedding.video_features`. Older Cosmos3 Edge exports without this
contract must be re-exported with a current Mobius version; the runtime does
not silently reinterpret video frames as images.

World generation currently starts from packed Gaussian noise, or from
explicitly supplied initial latent tensors. Image-to-video conditioning is not
silently inferred from an image; it requires the model-specific VAE
conditioning/masking contract.

Results retain model-boundary array layouts:

- `TextOutput.text` is decoded text and `token_ids` is
  `[batch, generated_tokens]`;
- `ActionOutput.actions` is sliced from padded action state to the selected
  domain's raw width;
- `ImageOutput.images` is float NCHW;
- `VideoOutput.video` is float NCTHW and still needs application-specific
  clipping/range conversion and encoding.

### Standalone preprocessing

Preprocessing is public and can be used without loading any ONNX sessions:

```python
from onnx_world_model.preprocessing import WorldModelPreprocessor

processor = WorldModelPreprocessor("output/cosmos3-edge")

reasoner = processor.prepare_reasoner(
    "Describe this image.",
    image="frame.png",
)
print(reasoner.input_ids.shape)
print(reasoner.pixel_values.shape)

world = processor.prepare_world(
    "Predict what happens next.",
    frames=17,
    height=256,
    width=256,
    action_steps=16,
    action_domain="droid_lerobot",
    include_action=True,
    seed=1234,
)
print(world.vision_tokens.shape)
print(world.options)
```

`TextPreprocessor`, `ImagePreprocessor`, `PackedImagePreprocessor`,
`PackedVideoPreprocessor`, `PreparedReasonerInputs`, `PreparedVideo`, and
`PreparedWorldInputs` are also public for applications that want to replace
only part of the preprocessing stack. The processor supports fixed NCHW image
graphs and Mobius's variable-resolution Cosmos3 Edge image/video contract
(`smart_resize` + frame sampling + block-major patchification + `grid_thw`).

## Python pipeline API

The Python package locates the shared library from an installed `onnxruntime`
wheel on Windows. On other platforms, set `ONNX_RUNTIME_LIBRARY_PATH` when the
wheel does not ship `libonnxruntime`.

```bash
pip install -e ".[test]"
```

### Execution providers

`providers` is an ordered preference list. Provider names are
case-insensitive and may use short names such as `cuda`, `dml`, `qnn`, and
`cpu`, or ORT names such as `CUDAExecutionProvider`. CPU must be last because
ORT uses it as the fallback provider.

```python
from onnx_world_model import Pipeline, available_execution_providers

print(available_execution_providers())

pipeline = Pipeline(
    "output/cosmos3-edge",
    providers=["cuda", "cpu"],
    provider_options={
        "cuda": {
            "device_id": 0,
            "gpu_mem_limit": 24 * 1024**3,
            "use_tf32": False,
        },
        "cpu": {"use_arena": True},
    },
)

# Actual providers selected for each component after applying manifest hints
# and filtering against the loaded ORT build.
print(pipeline.execution_providers)
```

For `Pipeline`, the requested order is intersected with each component's
`preferred_execution_providers`. When `providers` is omitted, the component's
manifest order is used; a profileless component defaults to CPU. An unavailable
provider is skipped only when a later requested fallback is available, so
`["cuda"]` fails on a CPU-only ORT build while `["cuda", "cpu"]` selects CPU.
Use an ORT build or wheel that contains the requested EP—changing
`ort_library_path` alone does not add an EP to a CPU-only runtime.

`OnnxModel` and `LatentDynamicsModel` accept the same arguments and default to
CPU. The generation `WorldModel` follows `Pipeline` provider selection:

```python
model = OnnxModel(
    "model.onnx",
    providers=["dml", "cpu"],
    provider_options={"dml": {"performance_preference": "high_performance"}},
)
print(model.metadata.execution_providers)
```

`Pipeline` is a tensor runtime, not a raw-media processor. The caller must
tokenize text, apply the exported image/video processor, expand multimodal
placeholder tokens, and create packed diffusion latents before calling a
stage. Inspect `pipeline.inputs`, `pipeline.stages`, `pipeline.metadata`, and
the packaged processor/tokenizer assets instead of hard-coding one
checkpoint's shapes.

```python
import numpy as np

from onnx_world_model import Pipeline

pipeline = Pipeline("output/cosmos3-edge")
print(pipeline.profile)
print(pipeline.inputs)
print(pipeline.stages)

session = pipeline.create_session()

# All values below are already tokenized, normalized, or packed by the host.
# A semantic name can feed multiple compatible component ports.
reasoner_inputs = {
    "text.token_ids": reasoner_input_ids,
    "vision.pixel_values": pixel_values,
}

session.run_stage("reasoner_prompt", reasoner_inputs)
reasoning = session.run_stage(
    "reasoner_decode",
    options={"max_tokens": 64, "seed": 1234},
)
session.release_stage("reasoner_decode")

world_inputs = {
    "text.token_ids": generator_input_ids,
    "diffusion.initial_vision_latent": packed_vision_noise,
    "diffusion.initial_action_latent": packed_action_noise,
}
world = session.run_stage(
    "world_generation",
    world_inputs,
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

`reasoning["generated_token_ids"]` contains token IDs, not decoded text.
Likewise, component outputs retain their model boundary representation:
Cosmos3 Edge action state remains padded to width 64 and must be sliced using
`pipeline.metadata["action"]["raw_dimensions"]`; decoded video is float NCTHW
and still needs the deployment's clipping/range and frame conversion.

For the tested Cosmos3 Edge export, the fixed vision encoder consumes
normalized NCHW `[1, 3, 256, 256]` pixels and produces 64 feature rows. The
reasoner token sequence must therefore contain 64 image-placeholder tokens,
one per feature row. Its chat template emits one `<|image_pad|>` marker; the
host processor must expand that marker before inference. These values are
checkpoint-specific—read the component signatures and processor config for
other exports.

`run_stage()` executes the strategy declared by the manifest: one pass,
autoregressive decoding, or all iterative scheduler steps. `step_stage()`
executes exactly one pass for applications that manage loops themselves.
Generated tensors or transformed targets can be supplied through `overrides`
using their semantic or qualified port name.

`PipelineSession.run()` executes a selected stage list in declaration order and
releases state according to the manifest. It is primarily a convenience for
pipelines whose selected stages share one prepared input set; heterogeneous
Cosmos flows are clearer as explicit `run_stage()` calls:

```python
result = session.run(
    world_inputs,
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

## Cosmos3 Edge verification status

The runtime was exercised against an actual 23.35 GB
`cosmos3-edge-f32-export` package on ONNX Runtime CPU:

| Check | Result |
|---|---|
| Parse and validate Mobius schema 1.1 | Passed |
| Load all six ONNX component sessions | Passed, approximately 25–30 seconds |
| Text-only Reasoner control (`2 + 2`) | Passed, generated `4` |
| Vision encoder, embedding, and Reasoner execution | Passed without runtime or non-finite tensor errors |
| One-step Generator and action state | Passed |
| UniPC scheduler and packed state update | Passed |
| Wan VAE decoder micro-run | Passed, produced `[1, 3, 5, 32, 32]` |
| Cosmos3 Edge image-description semantic parity | **Not established** |

The image path executes, but a natural cat-image prompt did not produce a
correct description. The same decoder succeeds on text-only inference, and
Mobius currently documents its Cosmos3 Edge visual implementation as L1
graph-build only: projector pixel-shuffle ordering and numerical parity have
not been verified against an authoritative NVIDIA implementation. Therefore
the runtime is proven to load and execute this package, but the README does not
claim that Cosmos3 Edge visual semantics are correct.

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
from onnx_world_model import LatentDynamicsModel

model = LatentDynamicsModel("model.onnx")
result = model.step(observation, action, state)

rollout = model.create_rollout()
result = rollout.step(observation, action)
rollout.reset(batch_size=1)
```

`LegacyWorldModel` is an explicit compatibility alias for
`LatentDynamicsModel`; `WorldModel` now consistently means the generation
package API.

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
runtime.providers = {"cuda", "cpu"};
runtime.provider_options["cuda"] = {
    {"device_id", "0"},
    {"gpu_mem_limit", "25769803776"},
};

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

- Ordered execution-provider configuration with per-component manifest
  selection. CUDA and TensorRT use their V2 option APIs; CPU, DML, OpenVINO,
  VitisAI, QNN, XNNPACK, WebGPU, Azure, and CoreML use their applicable ORT
  registration APIs. Provider availability still depends on the loaded ORT
  build.
- All component sessions are loaded eagerly. Pipeline scheduling and host
  transforms currently operate on CPU tensors; ORT performs device transfers
  at component boundaries.
- The generation API supports exported chat templates, fast
  tokenizers, fixed NCHW and variable-resolution packed image preprocessing,
  video path/array decoding, frame sampling, image/video placeholder
  expansion, text decoding, packed Gaussian world-state initialization,
  domain-aware action slicing, and automatic stage/state lifecycle.
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
- Frame-level image/video mRoPE positioning for fixed-square and
  variable-resolution Cosmos3 Edge contracts. Multiple independent images or
  videos in one prompt are not yet supported.
- The low-level API accepts tensors. The generation API adds raw text
  tokenization, fixed/packed image preprocessing, video container/array
  decoding and sampling, and diffusion-noise initialization. It does not
  encode output video files or implement arbitrary undeclared media
  processors.
- Conditioning handoffs recorded only in manifest metadata are not executed
  automatically; callers must supply the corresponding packed initial latent.
- Unsupported general transforms require an explicit target `override`.
- Fixed-step stochastic FlowMatch and FlowMatch beta-sigma schedules are
  rejected rather than approximated. UniPC support is limited to the
  flow-prediction order-1/order-2 contract exercised by Cosmos3 Edge.
