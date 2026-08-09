# Pipeline API

`Pipeline` is the low-level tensor and stage execution API. Use `WorldModel`
for raw text, image, video, and action generation.

## Load a package

```python
from onnx_world_model import Pipeline

pipeline = Pipeline("output/cosmos3-edge")

print(pipeline.profile)
print(pipeline.inputs)
print(pipeline.outputs)
print(pipeline.stages)
print(pipeline.metadata)
```

All component sessions are loaded eagerly. `Pipeline` owns immutable model
sessions and can be shared; each `PipelineSession` owns one request or
trajectory's state.

## Execution providers

Provider names are case-insensitive and may use short names such as `cuda`,
`dml`, `qnn`, and `cpu`, or ONNX Runtime names such as
`CUDAExecutionProvider`. CPU must be last because it is the fallback provider.

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

print(pipeline.execution_providers)
```

The requested order is intersected with each component's manifest preferences.
An unavailable provider is skipped only when a later fallback is available.
The loaded ONNX Runtime build must contain the requested provider.

## Input contract

Callers provide tokenized, normalized, and packed tensors using semantic or
qualified port names. Inspect package metadata instead of hard-coding one
checkpoint's shapes.

```python
session = pipeline.create_session()

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
```

World generation uses packed initial state:

```python
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
session.release_stage("decode_video")
```

## Stage execution

- `run_stage()` executes the strategy declared by the manifest: single pass,
  autoregressive generation, or all iterative scheduler steps.
- `step_stage()` executes exactly one pass for callers that manage a loop.
- `overrides` supplies generated tensors or transformed targets explicitly.
- `release_stage()` releases state according to the manifest lifecycle.
- `reset()` clears all session state.

`PipelineSession.run()` executes selected stages in declaration order and
releases each stage after execution. Explicit `run_stage()` calls are clearer
when stages need different prepared inputs.

## Outputs

Outputs remain at their model boundary:

- `generated_token_ids` contains token IDs, not decoded text;
- action state may retain the model's padded width;
- decoded video is a float NCTHW tensor.

The modality APIs on `WorldModel` add preprocessing and structured output
handling.

Model loading and inference release the Python GIL. NumPy inputs are copied
into engine-owned storage, and outputs are returned as independent arrays.

## C++ API

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

## Runtime scope

- Dense FP16, BF16, FP32, integer, and boolean tensors.
- Single-pass, on-demand, state-transition, iterative, and autoregressive
  stages.
- KV-cache, request, sequence, iteration, and session state lifecycles.
- FlowMatch Euler and flow-prediction UniPC order-1/order-2 schedulers.
- Greedy and temperature/top-k/top-p autoregressive sampling.
- Packed layout, attention masks, multimodal positions, scheduler timesteps,
  and action-domain generated inputs.
- Scheduler, cast, reshape, packed video finalization, and audio finalization
  transforms.

Pipeline scheduling and host transforms operate on CPU tensors. ONNX Runtime
performs device transfers at component boundaries.
