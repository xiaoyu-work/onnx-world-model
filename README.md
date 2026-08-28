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

Image-to-video conditioning and classifier-free guidance are described by
additional stage options and metadata; see the manifest contract in the
[Pipeline API](docs/pipeline-api.md).

## Architecture

```text
Python API ── pybind11 ─┐
                       ├─ Pipeline ─ PipelineSession ─ named ONNX Model sessions
C++ API ───────────────┤                  │
                       │                  ├─ generated-input programs
                       │                  ├─ stage strategies and schedulers
                       │                  └─ per-trajectory recurrent state
                       └─ LatentDynamicsModel / Rollout compatibility API
```

- `Model` runs any ONNX graph using named tensors.
- `Model` uses ONNX Runtime I/O binding and can opt into device-resident C++
  outputs after registering the corresponding EP library; Python results are
  materialized as independent NumPy arrays.
- `Pipeline` owns immutable component sessions and can be shared by callers.
- `PipelineSession` owns one request/trajectory's KV cache, diffusion latent,
  action state, outputs, and stage cursors, and preserves device-backed tensors
  across component connections, recurrent state, and public outputs.
- `PipelineSession.begin_stage()` runs any stage one step at a time and reports
  each step as a `StageEvent`, so a caller can stream decoded tokens or
  diffusion steps. Stepping is synchronous — one call blocks for one model or
  scheduler step — and `run_stage()` is that same run drained to the end, so
  both paths produce identical results.
- Cancellation is explicit and cooperative. Pass a `CancellationToken` or a
  `timeout` in seconds to `run_stage()`, `step_stage()`, `begin_stage()`,
  `iter_stage()`, `run()`, or `OnnxModel.run()`, or call
  `StageRun.request_cancellation()` from another thread to stop a step already
  running. The interrupted call raises `CancelledError` or
  `DeadlineExceededError` — both subclasses of `WorldModelError` — releases
  the session, and keeps everything it already applied. A deadline is claimed
  by one shared background watchdog, so it fires while work is blocked rather
  than at the next boundary, and `CancellationToken.wait()` blocks — with the
  GIL released — until that happens.
- `PipelineSession.snapshot()` captures all of that mutable state in memory,
  and `restore()` and `fork()` rewind or branch a trajectory from it without
  copying tensor data or leaving the process. `checkpoint(name)`,
  `restore_checkpoint(name)`, `drop_checkpoint(name)`, and
  `has_checkpoint(name)` add named in-memory transaction markers over the same
  capture. This is in-memory transaction support only: it is not paged KV and
  nothing is serialized to disk or crosses a process boundary. An unfinished
  stage run holds the session, so these calls and every other state-mutating
  call fail while one is active.
- Concurrency is bounded by admission scheduling, not by batching. Pass
  `max_concurrent_executions` and `max_concurrent_by_stage_kind` to
  `Pipeline` or `WorldModel` to cap how many executions run at once, globally
  and per stage kind; everything else queues fairly and enters oldest-first,
  and a full stage kind never blocks a different one. Nothing is merged,
  split, reordered, or preempted. A queued request still honors its
  `cancellation` token and `timeout`, so it can be stopped before it ever
  starts. `Pipeline.scheduling_stats` reads how many executions are admitted
  and how many are queued right now, per stage kind as well as in total.
- Component placement is per component and load-time. Pass
  `component_placement` to `Pipeline` or `WorldModel` to give one component its
  own execution providers, provider options (including `device_id`), graph
  optimization level, and thread counts, layered over the pipeline-wide
  options; `allow_unpreferred_providers` lets a component that names its own
  providers run somewhere its manifest does not prefer. Every port then reports
  where ONNX Runtime actually placed it, and `Pipeline.transfer_plan`
  classifies each manifest connection from those placements — `direct`,
  `upload`, `download`, `host_staged`, `host_transform`, or `unknown` — with a
  reason for every non-direct one. The plan is inspection only: nothing
  executes from it, and warm-up, lazy loading, offload and eviction, and
  peer-to-peer transfers are not included.
- `LatentDynamicsModel` and `Rollout` preserve the original fixed
  latent-dynamics API.
- Generic ONNX and latent-dynamics APIs are documented in
  [Low-level APIs](docs/low-level-apis.md).
- `Tensor` has value semantics with copy-on-write CPU storage and an
  ORT-independent device-buffer contract; host access to accelerator storage
  requires explicit CPU materialization.
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

Version 0.2 introduces the device-aware `TensorBuffer` ABI, version 0.3
adds the cancellation surface — a `CancellationToken` member on
`PipelineRunOptions`, a virtual cancellable `ModelBackend::Run`, and
`StageRun::RequestCancellation` — and version 0.5 changes the `Pipeline`
layout by giving it a shared admission scheduler and a defaulted
`PipelineSchedulingOptions` parameter on its constructor and on
`Pipeline::Load`. C++ applications built against an earlier version must be
recompiled when upgrading; source that already compiled keeps compiling,
because every new parameter is defaulted.

## Install

The Python package locates the shared library from an installed `onnxruntime`
wheel on Windows. On other platforms, set `ONNX_RUNTIME_LIBRARY_PATH` when the
wheel does not ship `libonnxruntime`.

```bash
pip install -e ".[test]"
```

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

# Video generation. Add image= for image-to-video; omit it for text-to-video.
video = model.video.generate(
    "A robot moves the red block to the left.",
    image="frame.png",
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

To keep compatible intermediate tensors on an accelerator, first register the
provider library with ONNX Runtime's process-wide environment, then opt into
device outputs:

```python
from onnx_world_model import register_execution_provider_library

register_execution_provider_library(
    "CUDAExecutionProvider",
    "/path/to/onnxruntime_providers_cuda.dll",
)
model = WorldModel.from_pretrained(
    "output/cosmos3-edge",
    providers=["cuda", "cpu"],
    device_outputs=True,
)
```

Python results remain independent NumPy arrays; only compatible internal
pipeline connections and recurrent state stay device-resident.

`image=` accepts an image path, PIL image, or NumPy array. `video=` accepts a
video path, frame sequence, or THWC/TCHW NumPy array. They are mutually
exclusive in one text-generation request. Image-to-video requires a package
that declares the corresponding recipe.

Low-level tensor and stage execution is documented in the
[Pipeline API](docs/pipeline-api.md). Cosmos3 Edge results are in
[Cosmos3 Edge validation](docs/cosmos3-edge-validation.md).

## Current scope

- Text, image, video, and action generation from Mobius packages.
- Configurable execution providers, subject to the loaded ONNX Runtime build,
  pipeline-wide and per component.
- Per-component load-time placement — execution providers, provider options,
  graph optimization level, and thread counts — plus an inspection-only
  `transfer_plan` over the ports ONNX Runtime actually assigned. Session
  warm-up, lazy or deferred component loading, offload and eviction,
  peer-to-peer device-to-device transfers, and executing from the plan are
  **not** included.
- One image or video per text-generation request.
- Image-to-video conditioning and classifier-free guidance for packages that
  declare them; output media encoding is not yet included.
- Explicit cancellation and deadlines on `PipelineSession`, `StageRun`, and
  `OnnxModel`. One shared process-wide watchdog claims every deadline, so one
  fires while a call is blocked rather than at the next boundary; inside ONNX
  Runtime that claim is honored between graph nodes, so a single long kernel
  can overrun the deadline; and the `WorldModel` modality `generate()` methods
  and the fixed `LatentDynamicsModel` API do not take a token yet.
- Shared admission scheduling: a global and a per-stage-kind cap on how many
  executions run at once, with cancellation-aware fair queuing for the rest,
  plus a `scheduling_stats` reading of what is admitted and what is queued.
  Continuous or dynamic batching is **not** included and is not planned by
  this change; it would additionally need request compatibility keys, tensor
  concatenation and result splitting, per-lane recurrent state and RNG
  streams, and a KV-cache manager that can admit and evict lanes mid-stage.
- Fixed-step stochastic FlowMatch schedules are not yet supported.
