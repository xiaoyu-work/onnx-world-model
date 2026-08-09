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
- `Pipeline` owns immutable component sessions and can be shared by callers.
- `PipelineSession` owns one request/trajectory's KV cache, diffusion latent,
  action state, outputs, and stage cursors.
- `LatentDynamicsModel` and `Rollout` preserve the original fixed
  latent-dynamics API.
- Generic ONNX and latent-dynamics APIs are documented in
  [Low-level APIs](docs/low-level-apis.md).
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

`image=` accepts an image path, PIL image, or NumPy array. `video=` accepts a
video path, frame sequence, or THWC/TCHW NumPy array. They are mutually
exclusive in one text-generation request.

### Image-to-video

`model.video.generate(prompt, image=...)` runs the image-to-video recipe the
package declares. Text-to-video and text-to-image are unchanged: image
conditioning is never inferred, and a package without that recipe rejects
`image=`.

```python
video = model.video.generate(
    "A robot picks up the red block.",
    image="frame.png",
    negative_prompt=model.video.default_negative_prompt("image_to_video"),
    guidance_scale=5.0,
    frames=121,
    height=480,
    width=832,
    num_inference_steps=50,
    seed=1234,
)
```

The conditioning frame is normalized to the encoder's range, encoded by the
declared encoder stage, packed into generator token rows, and written over the
noise of the conditioned latent frames. Only the remaining rows carry a
timestep, an MSE index, and a scheduler update, so the conditioned frame is
bit-identical from the first step to the last. Each step then runs the
generator once for the prompt and once for the negative prompt and combines the
velocities as `unconditional + scale * (conditional - unconditional)`.

`guidance_scale=1` skips the unconditional pass. `negative_prompt` defaults to
the empty string unless the recipe declares `"negative_default": "asset"`, and
`negative_input_ids` supplies unconditional token IDs directly — it is mutually
exclusive with `negative_prompt`, and using either without active guidance is
an error. `conditioned_latent_frames=`, the metadata templates
(`add_resolution_template`, `add_duration_template`), and the system prompt
(`use_system_prompt`, `system_prompt`) all default to the recipe's values; JSON
prompts are never rewritten by the templates. Scheduler modes stay explicit:
`mode="image_to_video"` is selected when an image is supplied, and a call that
selects no mode keeps the stage defaults.

Low-level tensor and stage execution is documented in the
[Pipeline API](docs/pipeline-api.md). Cosmos3 Edge results are in
[Cosmos3 Edge validation](docs/cosmos3-edge-validation.md).

## Current scope

- Text, image, video, and action generation from Mobius packages.
- Configurable execution providers, subject to the loaded ONNX Runtime build.
- One image or video per text-generation request.
- Image-to-video conditioning and classifier-free guidance for packages that
  declare them; output media encoding is not yet included.
- Fixed-step stochastic FlowMatch schedules are not yet supported.
