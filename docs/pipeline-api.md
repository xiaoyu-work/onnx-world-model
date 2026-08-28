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

Device-resident component handoff is opt-in. Register the EP's shared library
once before constructing a pipeline, then enable `device_outputs`:

```python
from onnx_world_model import Pipeline, register_execution_provider_library

register_execution_provider_library(
    "CUDAExecutionProvider",
    "/path/to/onnxruntime_providers_cuda.dll",
)
pipeline = Pipeline(
    "output/cosmos3-edge",
    providers=["cuda", "cpu"],
    device_outputs=True,
)
```

Direct connections, recurrent state, public outputs, and shape-only views
retain their device buffers. Host-authored programs and numeric transforms
materialize each device source once at the transform boundary. Python return
values are always independent NumPy arrays.

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

## Conditioned, guided generation contract

Image-to-video generation is described entirely by the manifest. An iterative
stage that supports it declares the capabilities
`classifier_free_guidance` and `conditioned_diffusion` and these options:

```json
{
  "name": "world_generation",
  "kind": "iterative",
  "capabilities": ["loop_carried_state", "classifier_free_guidance",
                   "conditioned_diffusion"],
  "options": {
    "guidance": {
      "kind": "classifier_free",
      "conditioning_input": "generator.input_ids",
      "scale_option": "guidance_scale",
      "default_scale": 1.0,
      "combine": "unconditional + scale * (conditional - unconditional)"
    },
    "conditioning": {
      "vision": {
        "encoder_stage": "encode_video",
        "encoder_input": "video_encoder.sample",
        "encoder_output": "video_encoder.latent",
        "state": "vision_state",
        "conditioned_latent_frames_option": "vision_conditioned_latent_frames",
        "default_conditioned_latent_frames": [],
        "packing": {
          "spatial_patch_size": 2,
          "temporal_patch_size": 1,
          "input_layout": "BCTHW",
          "output_layout": "NC",
          "channel_order": "patch_height_patch_width_channel"
        }
      }
    }
  }
}
```

`guidance` accepts two optional fields: `unconditional_input` names the input
the host uses for the unconditional value, and `outputs` lists the guided
predictions. Without them the runtime accepts `unconditional:<port>` (also
`unconditional:<semantic>` and `unconditional:<alias>`) and guides every
scheduler-driven recurrent output of the stage. `combine` is validated, not
interpreted: only the formula above is supported. `scale_option` is resolved
from the run options first, then from the selected scheduler mode override,
then from `default_scale`; a scale of 1 runs one pass, any other scale requires
an unconditional value.

The two capabilities are part of the contract in both directions: a stage that
declares `guidance` must advertise `classifier_free_guidance`, a stage that
declares `conditioning` must advertise `conditioned_diffusion`, and a stage
that advertises either without the matching option is rejected. The runtime
publishes everything it implements, and a `required_capabilities` entry is
honored when the runtime implements it or when a component or stage in the same
manifest declares that it provides it; only names nothing can satisfy fail to
load:

```python
from onnx_world_model import supported_pipeline_capabilities

print(supported_pipeline_capabilities())
# ('action_domain_program', 'attention_mask_program', 'classifier_free_guidance',
#  'conditioned_diffusion', 'iterative_scheduler', 'loop_carried_state', ...)
```

The C++ equivalent is `PipelineManifest::SupportedCapabilities()`.

`conditioning.<modality>.packing` must describe the same packing the runtime's
video unpatchify inverts (`BCTHW` -> `NC` with
`patch_height_patch_width_channel` channels and `temporal_patch_size` 1);
anything else is rejected at load time, as is a `spatial_patch_size` that
disagrees with `metadata.packing.latent_patch_size`. `encoder_output` must also
be a public pipeline output so the host can read the encoded latent back from
the encoder stage.

`conditioning.<modality>` may add an optional `preprocessing` block describing
how conditioning frames reach the encoder:

```json
"preprocessing": {
  "resize": "stretch_to_target",
  "resample": "bilinear",
  "convert_rgb": true,
  "rescale_factor": 0.00392156862745098,
  "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
}
```

`resample` names the filter (`bilinear`, `bicubic`, `nearest`, or `lanczos`);
`resize` may name either the filter or the only supported strategy,
`stretch_to_target`, which scales frames straight to the requested output
geometry. `rescale_factor` must be `1/255` and `convert_rgb` must be true,
since that is what the runtime does. Omitting the block keeps the default
bilinear resize into the signed `[-1, 1]` range.

Generation recipes and prompt packaging live in the manifest metadata:

```json
"metadata": {
  "generation_recipes": {
    "image_to_video": {
      "conditioning": {"modality": "image", "encoder_stage": "encode_video",
                       "conditioned_latent_frames": [0]},
      "prompt": {"positive": "json_or_text",
                 "negative_asset": "assets/negative_prompt.json",
                 "add_resolution_template": false,
                 "add_duration_template": false,
                 "use_system_prompt": false},
      "height": 480, "width": 832, "frames": 121, "fps": 24.0
    }
  },
  "generator_prompt": {
    "chat": {
      "add_generation_prompt": true,
      "add_vision_id": false,
      "enable_thinking": true
    },
    "suffix_token_ids": [151645, 151652],
    "system_prompts": {"image": "...", "video": "..."},
    "templates": {
      "duration": "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS.",
      "inverse_duration": "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS.",
      "image_resolution": "This image is of {height}x{width} resolution.",
      "inverse_image_resolution": "This image is not of {height}x{width} resolution.",
      "video_resolution": "This video is of {height}x{width} resolution.",
      "inverse_video_resolution": "This video is not of {height}x{width} resolution."
    }
  }
}
```

`generation_recipes.<mode>` supplies the default geometry, fps, prompt flags,
and conditioned latent frames for `mode`; the mode name is also the scheduler
`mode_overrides` key, which is where `num_inference_steps`, `flow_shift`,
`use_karras_sigmas`, and `guidance_scale` come from. `generator_prompt` is
optional and backward compatible: `suffix_token_ids` (or `suffix_tokens` with
token strings) are appended after the chat template, `system_prompts` selects a
per-modality system message, and `templates` supplies the metadata sentences. A
missing section means the runtime performs that step exactly as it did before,
and requesting a template or system prompt the package does not declare is an
error rather than a guess.

A recipe's `prompt` block may restate any `generator_prompt` field (`chat`,
`system_prompts`, `templates`, `suffix_token_ids`, `suffix_tokens`) to override
it for that mode only, and `"negative_default": "empty" | "asset"` selects what
an omitted `negative_prompt` means. The default is `"empty"`, matching the
reference pipeline; `"asset"` uses the shipped `negative_asset`. That asset may
hold the prompt text, a wrapper object with a `negative_prompt`/`default`/
mode-named string, or — as Cosmos3 Edge ships it — the structured JSON document
itself, which is passed through verbatim as a JSON string.

`chat.add_generation_prompt`, `chat.add_vision_id`, and
`chat.enable_thinking` are passed to the exported chat template, so a package
selects the official behavior; any other field under `chat` is rejected.

## Stage execution

- `run_stage()` executes the strategy declared by the manifest: single pass,
  autoregressive generation, or all iterative scheduler steps.
- `begin_stage()` executes the same strategy one step at a time and reports
  each step as a `StageEvent`; `run_stage()` is exactly that run drained to
  completion.
- `step_stage()` executes exactly one pass for callers that manage a loop.
- `overrides` supplies generated tensors or transformed targets explicitly.
- `release_stage()` releases state according to the manifest lifecycle.
- `reset()` clears all session state.

`PipelineSession.run()` executes selected stages in declaration order and
releases each stage after execution. Explicit `run_stage()` calls are clearer
when stages need different prepared inputs.

## Incremental stage execution

`begin_stage()` returns a `StageRun`: a handle over one execution of one stage
that reports what each model or scheduler step produced.

```python
session = pipeline.create_session()

with session.begin_stage("reasoner_decode", options={"max_tokens": 64}) as run:
    for event in run:
        if event.kind == "token":
            print(event.token_ids)          # int64 [batch, 1]
        elif event.finished:
            reasoning = event.outputs       # what run_stage() would return
```

`session.iter_stage(...)` is the same loop without the handle, for callers that
consume every event:

```python
for event in session.iter_stage("world_generation", world_inputs, options=opts):
    progress(event.iteration)
```

Each event carries:

| Field | Meaning |
|---|---|
| `kind` | `"token"` for autoregressive decoding, `"iteration"` for an iterative scheduler step, `"transition"` for the single pass of every other stage kind, `"completed"` for the terminal event. |
| `stage` | The stage this run drives. |
| `iteration` | Zero-based index of this step event within this run; the terminal event reports how many step events preceded it. |
| `token_ids` | The tokens this step generated, on `"token"` events only. |
| `outputs` | The stage's public outputs as of this event, as NumPy arrays. |
| `finished` | True on the terminal `"completed"` event and nowhere else. |

The rules the API guarantees:

- **Parity.** `run.finish()` returns exactly what `run_stage()` returns for the
  same arguments, because `run_stage()` is `begin_stage(...).finish()`. The
  token budget, sampling configuration, seed, end-of-sequence tokens, and
  iterative target are resolved once, when the run begins.
- **Synchronous steps.** One `step()` blocks until one model or scheduler step
  finishes. Events are results, not notifications; there is no background
  thread. A step can be stopped cooperatively — see
  [Cancellation and deadlines](#cancellation-and-deadlines) — but not
  preempted.
- **One terminal event.** A stopping condition — the end-of-sequence token in
  every lane, the token budget, or the last scheduler step — is reported by the
  *next* `step()` as the `"completed"` event, so every run ends the same way.
  Stepping past it raises `WorldModelError`, and `finish()` keeps returning the
  cached result.
- **One run per session.** While a run is unfinished the session raises
  `WorldModelError` from `begin_stage()`, `run_stage()`, `step_stage()`,
  `snapshot()`, `restore()`, `fork()`, `checkpoint()`, `restore_checkpoint()`,
  `drop_checkpoint()`, `reset()`, and `release_stage()`. An in-flight decode
  loop is not part of a snapshot, so the runtime refuses to capture one rather
  than hand back a snapshot that silently drops it. Reading `outputs`,
  `state()`, and `has_checkpoint()` stays legal.
- **No implicit rewind.** `close()`, the context manager, a failed step, and
  dropping the handle all release the session where the run stopped and keep
  everything it already applied. Use `snapshot()` before the run, or
  `checkpoint()`, when a caller wants to rewind.
- **Python materializes.** `token_ids` and `outputs` are independent NumPy
  arrays, exactly as `run_stage()` returns them; in C++ a device-backed output
  stays on its device.

Stopping early releases the session only when the run is closed. A `with`
block or an explicit `close()` does that deterministically; a bare `break` out
of `iter_stage()` waits for the generator to be closed or collected.

The C++ API is the same shape: `PipelineSession::BeginStage` returns a
move-only `StageRun` with `Step()`, `Finish()`, `RequestCancellation()`, and
`Cancel()`, and a `StageEvent` there keeps its device-resident tensors.

## Cancellation and deadlines

Every execution entry point accepts either a `cancellation` token or a
`timeout` in seconds. They are mutually exclusive: a `timeout` builds its own
source, so passing both raises `ValueError`.

```python
import threading

from onnx_world_model import CancellationSource, CancelledError, DeadlineExceededError

# Stop on demand, from anywhere.
source = CancellationSource()
threading.Timer(5.0, source.cancel).start()
try:
    outputs = session.run_stage("world_generation", inputs, cancellation=source.token())
except CancelledError:
    partial = session.outputs        # everything the stage already applied

# Or bound one call by wall-clock time.
try:
    outputs = session.run_stage("world_generation", inputs, timeout=30.0)
except DeadlineExceededError:
    ...
```

`StageRun` adds `request_cancellation()`, the one method that is safe to call
from another thread while `step()` or `finish()` is running, because it takes
no session lock:

```python
run = session.begin_stage("reasoner_decode", prompt)
watchdog = threading.Timer(10.0, run.request_cancellation)
watchdog.start()
try:
    for event in run:
        consume(event)
finally:
    watchdog.cancel()
    run.close()
```

The rules the API guarantees:

- **Two outcomes, one base.** An explicit cancellation raises `CancelledError`
  and an expired deadline raises `DeadlineExceededError`. Both derive from
  `WorldModelError`, so an existing `except WorldModelError` handler still
  catches them, and the C++ `ErrorCode` — `cancelled` or `deadline_exceeded` —
  decides which one, never the message text.
- **First reason wins.** A token is a one-way latch. Once a reason is claimed
  it never changes, and a cancelled token stays cancelled: reusing it fails the
  next call immediately. Build a new `CancellationSource` per request.
- **Never a rollback.** A cancelled call stops at its next boundary, releases
  the session's run slot, and leaves everything it already applied in place —
  exactly like a failed call. The one thing it does undo is internal scratch
  state that was never a result: a guided step that is stopped during its
  unconditional pass restores the conditional conditioning input before it
  unwinds, so the next step is not silently conditioned on the unconditional
  value. Take a `snapshot()` or a `checkpoint()` first if you want to rewind.
- **`request_cancellation()` is not `close()`.** `request_cancellation()`
  signals work that is running and does not take the session lock.
  `close()` takes the lock, so it waits for an in-flight step, and then only
  abandons the handle and releases the slot. A stale handle can do neither to
  a newer run.
- **Boundaries, not preemption.** The token is checked before and after each
  component, between the two classifier-free-guidance passes, around guidance
  combination and the state transforms, around token sampling, and at each
  step. Inside one ONNX Runtime call the runtime terminates the run through a
  fresh per-call `Ort::RunOptions`, which ONNX Runtime honors *between graph
  nodes* — a single long-running kernel still finishes first.
- **Deadlines fire on time, from one shared watchdog.** A deadline still in
  the future is armed on a single lazily started, process-wide watchdog
  thread. There is no thread and no timer per request: it holds every armed
  source weakly, sleeps until the earliest deadline, and claims it there. So a
  deadline stops work that is already blocked inside an ONNX Runtime call
  rather than waiting for the next boundary — bounded only by ORT's
  between-nodes termination check, so a single long kernel can overrun it. A
  deadline that has *already* passed when the source is created is left to the
  next poll instead, which is what lets `source.cancel()` immediately after
  `CancellationSource(timeout=0)` still win the first-reason race.
- **`wait()` blocks without polling.** `CancellationToken.wait()` and
  `CancellationSource.wait()` block until a reason is claimed and return it —
  `"cancelled"` or `"deadline_exceeded"` — rather than raising. They release
  the GIL, so another Python thread can still cancel, and they are what work
  with no boundary of its own uses instead of a polling loop. Waiting on a
  token nothing can cancel raises `WorldModelError` instead of blocking
  forever.
- **One deadline per `run()`.** `PipelineSession.run(..., timeout=...)` builds
  its source once, so a single absolute deadline covers the whole stage
  sequence rather than restarting at each stage.
- **Not yet covered.** The `WorldModel` modality `generate()` methods and the
  fixed `LatentDynamicsModel`, `WorldModel::Step`, and `Rollout` APIs do not
  take a token; drive `PipelineSession` directly when you need one.

## Session snapshots and named checkpoints

A session can capture everything it has accumulated and rewind or branch from
that capture:

```python
session = pipeline.create_session()
session.run_stage("world_generation", inputs, options={"num_inference_steps": 20})

checkpoint = session.snapshot()

# Explore one continuation, then rewind and try another.
first = session.run_stage("world_generation", options={"num_inference_steps": 20})
session.restore(checkpoint)
second = session.run_stage("world_generation", options={"num_inference_steps": 20})

# Or branch, and let both sides advance independently.
branch = session.fork()
```

- `snapshot()` returns an immutable `PipelineSessionSnapshot` holding the
  session's external, endpoint, recurrent-state, and guidance tensors, its
  stage cursors, scheduler histories, position cursors, and random engine.
- `restore()` replaces the session's state with the captured one. It either
  succeeds completely or changes nothing.
- `fork()` returns an independent session on the same pipeline, started from a
  snapshot of this one. Later runs, releases, and resets on either side never
  reach the other.

Snapshots are in-memory values only. They share tensor storage
copy-on-write, so capturing and forking copy no tensor data and never move a
device-resident tensor to the host. They are not written to disk, are not
paged out, and cannot be sent to another process. A snapshot stays bound to
the `Pipeline` it was taken from: restoring it into a session from a
separately loaded `Pipeline` raises `WorldModelError` even when both packages
are byte-for-byte identical. An unfinished `StageRun` also blocks every one of
these calls, because a decode or scheduler loop in flight is not part of the
capture.

### Named checkpoints

When a caller would otherwise track snapshot handles itself, the session can
hold them by name instead:

```python
session.checkpoint("before_branch")

session.run_stage("world_generation", options={"num_inference_steps": 20})
session.restore_checkpoint("before_branch")   # rewind and try again

assert session.has_checkpoint("before_branch")
session.drop_checkpoint("before_branch")
```

- `checkpoint(name)` captures exactly what `snapshot()` captures and stores it
  under `name`, replacing any checkpoint already stored there. The capture and
  the store happen together, so the checkpoint is the state the session held at
  one instant.
- `restore_checkpoint(name)` rewinds the session to that checkpoint with the
  same all-or-nothing behaviour as `restore()`, and leaves the checkpoint
  available for reuse.
- `drop_checkpoint(name)` discards it. Dropping a name the session does not
  hold raises `WorldModelError` rather than silently succeeding.
- `has_checkpoint(name)` reports whether the name is currently held.
- An empty name raises `WorldModelError` from every one of the four methods.

Named checkpoints are control metadata that lives beside the execution state,
not inside it:

- they survive stage execution, `release_stage()`, and `restore()`;
- `restore()` replaces the execution state only, so the session receiving a
  snapshot keeps its own checkpoint names and gains none from the source;
- a snapshot never carries checkpoints, so `fork()` inherits the parent's
  execution state but starts with an empty checkpoint namespace;
- `reset()` clears every named checkpoint along with the session state.

This is in-memory transaction support. It is not paged KV attention, and it is
not disk serialization: no checkpoint is written to disk, paged out, or shared
across processes.

## Conditioned, guided stages

An iterative stage that declares `guidance` and `conditioning` adds two host
responsibilities: run the declared conditioning encoder stage first, and pass
the unconditional value of the guided input alongside the conditional one. The
runtime then runs the stage twice per step — once conditional, once
unconditional — and still applies exactly one scheduler update.

```python
encoded = session.run_stage(
    "encode_video",
    {"video_encoder.sample": conditioning_frames},  # [1, 3, T, H, W] in [-1, 1]
)
session.release_stage("encode_video")

world = session.run_stage(
    "world_generation",
    {
        "text.token_ids": conditional_input_ids,
        "unconditional:generator.input_ids": unconditional_input_ids,
        "diffusion.initial_vision_latent": packed_latents,  # conditioned rows anchored
    },
    overrides={
        # Only the noisy rows carry a timestep, an MSE index, and a scheduler
        # update; the generator still attends to every latent frame.
        "generator.vision_timestep_token_indexes": noisy_token_indexes,
    },
    options={
        "mode": "image_to_video",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "video_latent_frames": latent_frames,
        "video_latent_height": latent_height,
        "video_latent_width": latent_width,
    },
)
```

The unconditional value is supplied as `unconditional:<port>`, and the manifest
may name it explicitly through `guidance.unconditional_input`. A guidance scale
of 1 runs one pass; any other scale requires the unconditional value.

`WorldModelPreprocessor.prepare_world(..., image=...)` produces exactly those
tensors: `pipeline_inputs()` and `pipeline_overrides()` carry them, and
`with_conditioning_latent()` folds the encoder's latent into the packed rows.

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

PipelineSessionSnapshot checkpoint = session.Snapshot();
PipelineSession branch = session.Fork();
session.Restore(checkpoint);

session.Checkpoint("before_branch");
if (session.HasCheckpoint("before_branch")) {
  session.RestoreCheckpoint("before_branch");
  session.DropCheckpoint("before_branch");
}
```

`PipelineSessionSnapshot` has no public constructor, so it can only come from
`PipelineSession::Snapshot()`. `Restore` and `Fork` take the session lock, and
`Restore` throws `Error` with `ErrorCode::state` when the snapshot came from a
session on a different `PipelinePackage` instance.

`Checkpoint`, `RestoreCheckpoint`, `DropCheckpoint`, and `HasCheckpoint` take
that same lock. `Checkpoint` captures and publishes under one lock hold, so its
linearization point is that single acquisition; `RestoreCheckpoint` linearizes
twice — once where it copies the named checkpoint handle, once where `Restore`
commits — and never holds the lock across the delegation, so it cannot
deadlock. An empty name throws `ErrorCode::invalid_argument`; restoring or
dropping a name the session does not hold throws `ErrorCode::state` and changes
nothing.

Cancellation in C++ is the same contract:

```cpp
CancellationSource source;                      // or WithTimeout(30s)
PipelineRunOptions options;
options.cancellation = source.token();

StageRun run = session.BeginStage("world_generation", inputs, {}, options);
std::jthread watchdog([&run] { run.RequestCancellation(); });
try {
  NamedTensors outputs = run.Finish();
} catch (const Error& error) {
  // ErrorCode::cancelled or ErrorCode::deadline_exceeded.
}
```

`CancellationToken` has no public constructor beyond the default, inert one, so
a cancellable token can only come from a `CancellationSource`. The source is
move-only; the token is copyable and observes the same state. `Cancel()` is
`noexcept` and safe from any thread.

`CancellationToken::WaitForCancellation()` blocks until a reason is claimed and
returns it, so a backend with no boundary of its own can park on the token
instead of polling:

```cpp
NamedTensors MyBackend::Run(
    const NamedTensors& inputs,
    const CancellationToken& cancellation) const {
  StartTheWork();
  (void)cancellation.WaitForCancellation();   // released by cancel or deadline
  cancellation.ThrowIfCancellationRequested();
  ...
}
```

It reports the reason rather than throwing it, and it rejects a token that is
not cancellable with `ErrorCode::invalid_argument` instead of blocking forever.
`CancellationSource::WaitForCancellation()` is the same wait for the owner.

`Model::Run` has a second overload that takes a token, and `ModelBackend::Run`
has a virtual cancellable overload whose default implementation checks the
token before and after the historical one-argument `Run`. An external backend
therefore keeps compiling and still stops at those boundaries; only a backend
that can interrupt work already running needs to override it, as the ONNX
Runtime backend does.

## Runtime scope

- Dense FP16, BF16, FP32, integer, and boolean tensors.
- Single-pass, on-demand, state-transition, iterative, and autoregressive
  stages.
- Classifier-free guidance and conditioned diffusion on iterative stages,
  driven by the stage's `guidance` and `conditioning` options rather than by
  model names.
- KV-cache, request, sequence, iteration, and session state lifecycles.
- In-memory session snapshot, restore, fork, and named checkpoints. This is
  in-memory transaction support, not paged KV attention: nothing is serialized
  to disk and nothing crosses a process boundary.
- Explicit cancellation and deadlines on stage execution and generic model
  execution, enforced at execution boundaries, by one shared process-wide
  deadline watchdog, and inside an ONNX Runtime call at graph-node
  granularity — so a single long-running kernel can overrun its deadline.
  Cancellation on the `WorldModel` modality APIs and the fixed
  latent-dynamics API is not included.
- FlowMatch Euler and flow-prediction UniPC order-1/order-2 schedulers.
- Greedy and temperature/top-k/top-p autoregressive sampling.
- Packed layout, attention masks, multimodal positions, scheduler timesteps,
  and action-domain generated inputs.
- Scheduler, cast, reshape, packed video finalization, and audio finalization
  transforms.

Pipeline scheduling and host transforms operate on CPU tensors, but the session
no longer forces every tensor to CPU. Direct connections, recurrent state,
rank adaptation, and `reshape` keep the producer's buffer, so a device-resident
component output can reach the next component and the public outputs untouched.
Each host transform materializes its device operands once at its own boundary.
In C++ this means a stage output may be device-only; call
`Tensor::CopyToCpu()` before reading it. The Python API is unaffected because
it always converts results to independent NumPy arrays.

Scheduler modes are never inferred: `mode` selects a declared `mode_overrides`
entry, and an absent mode uses the stage defaults. Conditioning handoffs
recorded only in manifest metadata are not executed automatically; the
declared `conditioning` stage option is what drives image-to-video, and other
handoffs still require the caller to supply the packed initial latent.
