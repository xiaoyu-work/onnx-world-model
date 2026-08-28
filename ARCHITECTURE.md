# ONNX World Model Runtime Architecture

## Purpose

This repository is a C++20 and Python runtime that executes validated
world-model packages exported by [Mobius](https://github.com/onnxruntime/mobius).
A package is a directory whose `pipeline.json` manifest declares neural
components, dataflow, generated inputs, control flow, recurrent state,
scheduler behavior, assets, and public outputs. The runtime loads that manifest,
validates it against what it can actually execute, and drives ONNX Runtime for
text, image, video, and action generation.

The guiding rule is that unknown or unsupported semantics fail during loading or
execution rather than silently selecting a fallback.

## System Context

```text
Mobius exporter ──▶ pipeline package on disk ──▶ this runtime ──▶ generated
                    (pipeline.json,                               text, images,
                     component ONNX models,                       video, actions
                     tokenizer.json, scheduler_config.json,
                     preprocessor and recipe configuration)

                              │
                              ▼
                    ONNX Runtime shared library
                    (loaded dynamically at run time)
```

External systems:

- **Mobius packages** — the only supported input format. The loader accepts
  pipeline schema major version 1.
- **ONNX Runtime** — resolved at run time, never linked. The Python package
  finds the library inside the installed `onnxruntime` wheel, or at the path in
  `ONNX_RUNTIME_LIBRARY_PATH`. Available execution providers therefore depend on
  the ORT build present on the machine.
- **Build-time header downloads** — `cmake/FetchOrtHeaders.cmake` and
  `cmake/FetchJsonHeader.cmake` fetch and SHA256-verify the ONNX Runtime and
  nlohmann/json headers, unless `ONNXRUNTIME_INCLUDE_DIR` and
  `NLOHMANN_JSON_INCLUDE_DIR` point at local copies.

There is no server, database, or outbound network call at run time.

## Components

| Component | Location | Responsibility |
|---|---|---|
| Public C++ API | `include/onnx_world_model/` | Installed declarations: `Tensor`, `Error`, `CancellationToken`, `CancellationSource`, `Model`, `Pipeline`, `PipelineSchedulingOptions`, `PipelineSchedulingStats`, `ComponentPlacement`, `PipelinePlacementOptions`, `PipelineTransfer`, `PipelineTransferPlan`, `PipelineSession`, `PipelineSessionSnapshot`, `StageRun`, `StageEvent`, `WorldModel`, `Rollout`. |
| Core library | `src/` | ORT loading, tensor marshalling, cancellation state, manifest parsing and validation, component placement and connection transfer planning, admission scheduling, staged execution. |
| Python binding | `bindings/python_module.cpp` | The `_native` pybind11 module and NumPy-to-`Tensor` conversion. |
| Python package | `python/onnx_world_model/` | Typed wrappers, preprocessing, media handling, and the modality-oriented generation API. |
| C++ tests | `tests/cpp/` | Stub-backend tests registered with CTest. |
| Python tests | `tests/python/` | Pytest suite over the installed package. |
| Tools | `tools/` | Test-package exporter and an image-to-video smoke script. |
| Documentation | `docs/` | Pipeline API, low-level APIs, Cosmos3 Edge validation results. |

Within `src/` the runtime is layered:

- `cancellation` owns the state behind `CancellationToken` and
  `CancellationSource`: the claimed reason, the immutable deadline, the
  callback registry that lets a cancelling thread reach into work already
  running, the blocking `WaitForCancellation`, and the one process-wide
  deadline watchdog that claims a deadline on time without a thread per
  request.
- `dynamic_library` loads the ORT shared library and binds `OrtApi` once per
  process.
- `ort_backend` is the only other translation unit that touches ORT; it owns
  the process-wide ORT environment, builds component sessions, reads each
  session's per-port memory plan once and publishes it as
  `TensorSpec::device`, uses I/O binding to wrap ORT-owned outputs in
  device-aware tensors, and terminates an
  in-flight `Session::Run` through a per-call `Ort::RunOptions`.
- `model` validates named tensors against graph signatures.
- `pipeline` parses `pipeline.json`, resolves and validates per-component
  placement over the pipeline-wide `RuntimeOptions`, and classifies every
  connection into the package's conservative `PipelineTransferPlan`;
  `pipeline_manifest_validation` checks the
  parsed manifest's semantics; `pipeline_manifest_common` holds the checks both
  share.
- `pipeline_scheduler` is the shared admission controller: the supported
  stage-kind list and its validation, the per-kind permit buckets, the
  cancellation- and deadline-aware FIFO queue with its oldest-eligible pump,
  the RAII lease that returns a permit exactly once, and the consistent
  snapshot of that state a `Pipeline` reports. It decides only when an
  execution may start; it never runs, resumes, or inspects one.
- `pipeline_session` executes stages, owns per-trajectory state, drives one
  stage at a time through the `StageRun` state machine, captures or
  restores that state as an in-memory snapshot, and defines `Pipeline`, which
  hands every session it creates that one scheduler.
- `world_model` provides the fixed latent-dynamics compatibility API.

## Dependency Rules

- `include/onnx_world_model/` depends only on the C++ standard library. It must
  not include ONNX Runtime or nlohmann/json headers, so consumers do not need
  them on their include path.
- ONNX Runtime headers appear only in `src/dynamic_library.cpp` and
  `src/ort_backend.cpp`. Every other file reaches ORT through
  `detail::CreateOrtBackend`.
- `src/` internal headers live in `onnx_world_model::detail` and are not
  installed.
- `pipeline_manifest_validation` receives an already-parsed `PipelineManifest`
  and never parses a raw document; `pipeline.cpp` never re-implements semantic
  checks.
- `bindings/` depends on `include/`, never on `src/` internals.
- Python intra-package imports are one-way:
  `_native <- _api <- generation` and `media <- preprocessing <- generation`.
  `media.py` must not import `preprocessing.py`.
- Tests depend on public surfaces only: `tests/cpp/` includes public headers,
  and `tests/python/` imports the installed package.
- The library links against no ONNX Runtime binary at build time.

## Data Flow

Loading:

1. `Pipeline.__init__` reads `pipeline.json` and calls `_native.Pipeline`.
2. `PipelinePackage::Load` resolves component paths inside the package root,
   parses the manifest, runs `detail::ValidateManifest`, and then validates the
   requested `PipelinePlacementOptions` against that manifest, before any
   component model file is opened.
3. Each component ONNX file is opened as a `Model` with its placement layered
   over the pipeline-wide `RuntimeOptions`, and its signature is checked
   against the manifest. The ORT backend reads that session's per-port memory
   plan once and publishes it as each `TensorSpec::device`.
4. `PipelinePackage` classifies every manifest connection into its
   `PipelineTransferPlan` from those port devices.
5. `Pipeline` holds the resulting components immutably behind a `shared_ptr`.

Generation, using video as the example:

1. `WorldModel.video.generate(...)` forwards to `_GenerationRuntime.generate`.
2. `WorldModelPreprocessor` renders the chat template, tokenizes the prompt, and
   prepares conditioning frames and latents, delegating image and video decoding
   and patch packing to `media.py`.
3. `Pipeline.create_session()` produces a `PipelineSession` that owns this
   request's KV cache, diffusion latent, guidance values, stage cursors, and
   recurrent state.
4. `PipelineSession.run_stage` calls into `PipelineSession::Impl`, which resolves
   each component's inputs from external values, generated-input programs,
   transforms, defaults, and recurrent state, then runs the component through
   ONNX Runtime.
5. Stage kinds drive control flow: `single_pass`, `autoregressive`, `iterative`
   with a diffusion scheduler, and `state_transition`.
6. `ReleaseStage` frees state whose declared `release_after` names that stage;
   `outputs()` collects the manifest's public outputs.
7. The generator unpacks latent tokens and returns a modality-specific output
   dataclass.

Every stage runs through one state machine. `BeginStage` resolves the stage
kind, inputs, overrides, options, and all autoregressive configuration once and
returns a `StageRun`; each `Step()` takes the session lock, performs exactly
one component pass or scheduler step, and returns a `StageEvent` describing it;
the run ends with exactly one terminal `completed` event whose outputs are the
stage's result. `RunStage` and `StageRun` drain the same state machine, so
incremental and all-at-once execution cannot diverge. `RunStage` holds the
session lock for its entire drain, preserving whole-stage serialization;
explicit `Step()` releases it between events. Stepping is synchronous: an event
is the result of work already done, not a notification, and there is no background
thread.

Stopping that work is cooperative and explicit. `PipelineRunOptions` carries a
`CancellationToken`; `BeginStage` copies its deadline into the run's own
`CancellationSource`, links the caller's token into that source with a
reason-preserving registration, and publishes the resulting internal token as
the run's options, so every downstream call observes both the caller's
cancellation and `StageRun::RequestCancellation`. Boundaries — not per-element
loops — poll it: before and after each component, between the two guidance
passes, around guidance combination, around the state transforms, around token
sampling, and at each `Step`, `Finish`, and direct `StepStage` entry. Inside
one ONNX Runtime call, the ORT backend registers a callback that terminates a
fresh per-call `Ort::RunOptions`, which ORT honors between graph nodes. A
cancelled call throws `ErrorCode::cancelled` or `ErrorCode::deadline_exceeded`
through the same path a failure takes: the run slot is released, the handle
closes, and everything already applied stays applied. `RequestCancellation`
never takes the session lock, so it is the one operation a second thread can
perform while a step holds it; `Cancel()` does take the lock and only closes
the handle.

A deadline no longer waits for a boundary. `CancellationSource::WithDeadline`
arms any still-future deadline on one lazily created, process-wide watchdog:
a single detached thread over a `multimap` of deadlines that holds each state
weakly, sleeps until the earliest one, and then claims `deadline_exceeded`
itself. There is no thread per request and no per-request timer; a source that
is destroyed or claimed early disarms its own entry, and a deadline that was
already due when the source was created is deliberately left to the next poll
so an immediate `Cancel()` can still win the first-reason race. That claim is
what releases `CancellationToken::WaitForCancellation`, the blocking wait work
with no boundary of its own uses instead of polling, and what fires the ORT
termination callback while a `Session::Run` is in flight. ONNX Runtime checks
that flag between graph nodes, so a single long-running kernel still finishes
before the call unwinds.

A session has one run slot, held from `BeginStage` until the run completes, is
cancelled, or is dropped. While it is held, `BeginStage`, `RunStage`,
`StepStage`, `Snapshot`, `Restore`, `Fork`, `Checkpoint`, `RestoreCheckpoint`,
`DropCheckpoint`, `Reset`, and `ReleaseStage` throw `ErrorCode::state`, while
`outputs()`, `state()`, and `HasCheckpoint()` stay legal. That exclusion is
deliberate: a run's decode history, per-lane stop latch, and token budget live
on the run rather than in `SessionState`, so capturing or rewinding the session
mid-run would silently drop them. A failed or cancelled run releases the slot
without rolling anything back, exactly as a failed `RunStage` always did.

A session can capture all of that mutable execution state — external,
endpoint, recurrent-state and guidance tensors, stage cursors, scheduler
histories, position cursors, and the random engine — as an immutable
`PipelineSessionSnapshot`, restore itself from one, or fork an independent
session initialized from one. The same capture is also reachable by name
through `Checkpoint`, `RestoreCheckpoint`, `DropCheckpoint`, and
`HasCheckpoint`. This is an in-process, in-memory transaction capability: a
snapshot shares tensor storage copy-on-write, never materializes a device
buffer, is not paged KV attention, and is not serialized to disk or portable
across processes. A snapshot records the `PipelinePackage` instance it came
from, so restoring it into a session built on any other package instance fails
with `ErrorCode::state`.

Concurrency across sessions is bounded by one shared admission controller per
`Pipeline`. `PipelineSchedulingOptions` sets a global ceiling and an optional
per-stage-kind ceiling; `0`, and a stage kind absent from the map, mean
unlimited, which is the default. A key that is not one of the six stage kinds
the runtime executes — `single_pass`, `autoregressive`, `iterative`,
`state_transition`, `composite`, `on_demand` — is `ErrorCode::invalid_argument`
at construction, because a limit no manifest could ever match would silently
do nothing. `Pipeline` holds that controller beside its package and shares
both the same way: an implicit copy, every session it creates, every fork of
those sessions, and every `StageRun` they produce compete for the same
permits, while a separately constructed `Pipeline` gets its own.

An execution is exactly one of four things, and each takes exactly one permit
for its whole duration: one `RunStage`, one `StepStage`, one `StageRun::Step`,
or one `StageRun::Finish` that still has steps to drain. `BeginStage` executes
nothing, a `Finish` on an already-completed run returns a cached value, and
`Cancel`, `RequestCancellation`, `outputs()`, `state()`, `Snapshot`,
`Restore`, `Fork`, and the checkpoint methods are not executions, so none of
them take a permit and an idle `StageRun` between `Step` calls holds nothing.
The permit is a stack-scoped RAII lease that is never stored on a run or a
session, so a cancellation, a deadline, or a backend failure returns it on the
way out.

Admission is work conserving and fair within a stage kind. A caller that fits
is granted without blocking; one that does not takes a ticket. The pump walks
the queue oldest first, skips a waiter whose kind is full so a different
eligible kind still enters, and stops once the global ceiling is reached, so a
saturated kind never blocks the head of the queue and no later waiter of the
same kind passes an earlier one. A queued request keeps observing its own
token: an explicit cancel or a deadline claimed by the shared watchdog removes
it from the queue and throws `ErrorCode::cancelled` or
`ErrorCode::deadline_exceeded` without the execution ever starting. Admission
happens before the session lock, so a queued caller can still find a
conflicting active run once it is admitted and fail then; peeking at session
state before waiting would be a staler race, not a safer one.

This is admission scheduling, not batching. Nothing merges two requests into
one model call, splits one apart, reorders work within a stage, or preempts an
execution that already started. Continuous or dynamic batching would need
things this runtime does not have: request compatibility keys, tensor
concatenation and result splitting across lane boundaries, per-lane recurrent
state and RNG streams, and a KV-cache manager that can admit and evict lanes
mid-stage.

`Pipeline::scheduling_stats()` is the operational reading of that controller
and the only way to observe it. It returns a detached `PipelineSchedulingStats`
value — the admitted and queued totals plus both per-stage-kind breakdowns —
taken at one moment under the scheduler's own mutex, so its fields agree with
each other and none of them changes afterwards. Both maps always carry all six
executable stage kinds, so a caller never tests for a key. The counts are
permits rather than executions: an unlimited stage kind takes the lock-free
fast path and no permit at all, so an unconfigured `Pipeline` reports zeros
however much work is inside it. Reading takes no session lock, runs no
callback, admits nothing, and queues nothing, so observing admission cannot
perturb it. This is observability for admission only; it reports no timing,
no throughput, and nothing about batching.

Named checkpoints are control metadata held on the session beside that
execution state rather than inside it. A snapshot therefore never carries a
checkpoint namespace: checkpoints survive stage execution and ordinary
`Restore`, a fork inherits execution state but starts with no checkpoint
names, and `Reset` clears them all. An empty name is `ErrorCode::invalid_argument`
and an unknown name is `ErrorCode::state` — `DropCheckpoint` is not a no-op.

`Tensor` can retain an ORT-independent `TensorBuffer` on a named device. Owned
tensor constructors allocate copy-on-write CPU storage; device-only buffers
must be explicitly materialized before host access. Tensors cross the Python
language boundary as independent NumPy arrays, so the binding materializes
device storage to CPU before copying it; `float16` and `bfloat16` cross as raw
two-byte views. A `StageEvent` is no exception: its outputs keep their device
buffers in C++ and become NumPy arrays in Python.

The generic ORT backend uses I/O binding. By default it binds outputs to CPU;
when `RuntimeOptions.device_outputs` is enabled after registering the
corresponding EP library, it leaves each output on the device selected by graph
partitioning. An ORT-backed tensor can bind directly into a later model
invocation; foreign device buffers use explicit CPU staging. `PipelineSession`
preserves that storage: caller inputs, overrides, component outputs, recurrent
state, and public outputs keep the producing buffer, a transform-free
connection forwards the identical `TensorBuffer`, and rank adaptation and the
`reshape` transform reuse it because they only relabel axes. Each host-side
transform materializes its device operands exactly once at its own boundary.

## Entry Points

Python:

- `onnx_world_model.WorldModel.from_pretrained(package_path)` — the primary
  generation API, exposing `.text`, `.image`, `.video`, and `.action`, each with
  a `generate()` method.
- `onnx_world_model.Pipeline` and `PipelineSession` — direct stage execution,
  plus `begin_stage()` and `iter_stage()` for incremental execution,
  `snapshot()`, `restore()`, `fork()`, and the named `checkpoint()`,
  `restore_checkpoint()`, `drop_checkpoint()`, and `has_checkpoint()` methods
  for in-memory session branching, the keyword-only `cancellation` and
  `timeout` arguments on every execution method, the keyword-only
  `max_concurrent_executions` and `max_concurrent_by_stage_kind` constructor
  arguments that cap concurrent executions, the keyword-only
  `component_placement` and `allow_unpreferred_providers` constructor
  arguments that place each component's session, and the read-only
  `transfer_plan` those produce.
- `onnx_world_model.CancellationSource` and `CancellationToken` — explicit
  cancellation and deadlines, with `CancelledError` and
  `DeadlineExceededError` as the outcomes, both derived from
  `WorldModelError`, plus the blocking `wait()` that releases the GIL until a
  reason is claimed.
- `onnx_world_model.OnnxModel` — one ONNX graph with named tensors.
- `onnx_world_model.LatentDynamicsModel` and `Rollout` — the fixed
  latent-dynamics API.
- `onnx_world_model.available_execution_providers` and
  `supported_pipeline_capabilities` — capability queries.
- `onnx_world_model.register_execution_provider_library` — process-wide EP
  registration required before opting into device-resident outputs for that
  provider.

C++:

- `onnx_world_model::Pipeline::Load` then `Pipeline::CreateSession` and
  `PipelineSession::RunStage`, `BeginStage`, or `StepStage`. `Pipeline`'s
  constructor and `Load` take an optional `PipelineSchedulingOptions` that
  caps concurrent executions globally and per stage kind, and
  `Pipeline::scheduling_stats` reads how many are admitted and queued now.
  `Pipeline::Load` and `PipelinePackage::Load` take an optional
  `PipelinePlacementOptions` that places each component's session;
  `Pipeline::transfer_plan` and `PipelinePackage::transfer_plan` read the
  resulting connection classification.
- `onnx_world_model::PipelineSession::Snapshot`, `Restore`, and `Fork` for
  in-memory session branching, plus `Checkpoint`, `RestoreCheckpoint`,
  `DropCheckpoint`, and `HasCheckpoint` for named in-memory checkpoints.
- `onnx_world_model::Model::Load` and `Model::Run`, whose cancellable overload
  takes a `CancellationToken`.
- `onnx_world_model::CancellationSource` and `CancellationToken`, whose
  `WaitForCancellation` blocks until a reason is claimed, plus
  `StageRun::RequestCancellation` for stopping work already running.
- `onnx_world_model::WorldModel::Load`, `WorldModel::Step`, and `Rollout`.

Command line:

- `python tools/image_to_video_smoke.py PACKAGE IMAGE [--dry-run]`
- `python tools/export_mobius_test_model.py OUTPUT_DIR`

Build and install targets: the `onnx_world_model` CMake library, the exported
package `onnx_world_model::onnx_world_model`, the `_native` Python extension,
and the `onnx-world-model` wheel built by scikit-build-core.

## Cross-Cutting Constraints

- **Fail fast, never fall back.** Unsupported capabilities, unknown manifest
  fields, unknown stage or generator kinds, unavailable execution providers, and
  mismatched tensor signatures raise instead of degrading silently.
- **Single error type.** All C++ failures throw `onnx_world_model::Error` with an
  `ErrorCode`; pybind11 maps that code onto the Python exception hierarchy, so
  `ErrorCode::cancelled` becomes `CancelledError`,
  `ErrorCode::deadline_exceeded` becomes `DeadlineExceededError`, and every
  other code becomes their common base `WorldModelError`.
- **Cancellation is cooperative and never a rollback.** A token is a one-way
  latch: the first claimed reason wins, a cancelled token stays cancelled, and
  reusing one cancels the next call immediately. A stopped call throws at its
  next boundary, releases the session's run slot, and leaves everything it
  already applied in place; a caller who wants to rewind takes a snapshot or a
  checkpoint first. `StageRun::RequestCancellation` signals in-flight work and
  takes no session lock, while `StageRun::Cancel` takes the lock and only
  closes the handle. `CancellationToken::WaitForCancellation` blocks without
  polling and reports the claimed reason rather than throwing it.
- **One deadline watchdog for the process.** Deadlines are claimed by one
  lazily created, immortal service with a single detached worker, never by a
  thread or timer per request. It holds each state weakly, so arming a
  deadline never keeps a source alive, and it never runs a callback, a state
  destructor, or its own singleton accessor while holding its schedule lock.
- **Validate at the boundary.** `Model::Run` checks every input and output
  tensor; manifest semantics are validated once at load time.
- **Ownership split.** `Pipeline` is immutable and shareable; `PipelineSession`
  is move-only and owns exactly one request or trajectory. Session state is
  guarded by `impl_->mutex`, and `Rollout` guards its state by its own mutex.
  The session holds that state through a `shared_ptr` and a `StageRun` holds
  the same pointer, so an incremental run stays valid even if the session
  wrapper is moved or destroyed while the run is in flight.
  `PipelineSession::Snapshot`, `Restore`, `Fork`, and the named-checkpoint
  operations take that same lock; no public method calls `Snapshot` or
  `Restore` while already holding it, so they cannot deadlock. `Restore`
  copies every container before taking it and commits by swapping, so a
  failure leaves the target session unchanged.
- **One stage run at a time.** A session executes one stage at a time and says
  so: an active `StageRun` rejects every other execution and state-mutating
  call with `ErrorCode::state` instead of interleaving with it or capturing a
  half-executed stage. `RunStage` and `BeginStage` use the same internal state
  machine rather than two implementations that can drift apart; ordinary
  concurrent `RunStage` calls still serialize for the whole stage.
- **Admission scheduling, never batching.** The shared scheduler decides how
  many executions run at once and in what order queued ones enter, and that is
  all it decides. It never merges, splits, reorders, or preempts work, and the
  documentation must not describe it as batching of any kind. Its lock order
  is fixed and one-way: `CancellationState`'s callback mutex, then the
  scheduler mutex, then the session mutex, then ONNX Runtime. A cancellation
  registration is therefore never created or destroyed while the scheduler
  mutex is held, and every execution takes its permit before the session lock.
  `Pipeline::scheduling_stats` observes that controller and nothing else: it
  reports admitted and queued counts, never timings or throughput, and reading
  it cannot admit, queue, or block anything.
- **Placement is load-time, and the transfer plan only describes it.** Which
  execution providers, provider options, graph optimization level, and thread
  counts a component's session is built with is decided once, while that
  session is being created; `PipelinePlacementOptions` therefore appears only
  on `PipelinePackage::Load` and `Pipeline::Load` and never on the constructor
  that takes an already-built package. A component's own provider list beats
  the global order, which beats its manifest preferences, and those
  preferences still filter the result unless `allow_unpreferred_providers` is
  set *and* that component supplied its own list. Global provider options are
  a pipeline-wide default and are filtered to the providers a component runs
  on; a component's own options for a provider it does not run on are an error
  rather than a silent drop. `PipelineTransferPlan` is a conservative
  description of where ONNX Runtime placed each port — direct only for
  identical known devices with nothing in between, `unknown` whenever an
  endpoint device is unreported, and host-involving for every transform this
  runtime evaluates on the host, including any `reshape` whose declared source
  and target shapes are not identical. It is the configured physical plan, not
  the effective one: when `device_outputs` is off every output is still bound
  to the host, and the plan is deliberately not rewritten to say so. Nothing
  executes from the plan. Warm-up, lazy loading, offload and eviction, and
  peer-to-peer transfers are explicitly out of scope.
- **Value semantics.** `Tensor` copies are cheap and copy-on-write, so a shared
  CPU buffer is cloned before mutation. Device buffers expose immutable
  storage and an explicit synchronous CPU-copy operation.
- **Portable, contained packages.** Every manifest path must stay inside the
  package root, and component names must be safe portable path segments.
- **Deterministic sampling.** A `PipelineSession` seeds its own random engine, so
  a given seed reproduces a generation.
- **Run-time ORT binding.** ORT is initialized once per process from a single
  library path; a second, different path is an error.
- **Shared ORT environment.** Every component session uses one process-wide
  `Ort::Env`. Thread counts, logging severity, graph optimization, and
  execution providers remain session-specific.
- **ORT-owned device outputs.** Opt-in generic model execution binds outputs to
  the memory locations selected by ORT. Returned buffers retain their
  `Ort::Value` plus a flattened set of required session, binding, and aliased
  input lifetime roots, and use an environment-registered EP data transfer for
  explicit CPU materialization.
- **One materialization per host transform.** `PipelineSession` never copies a
  tensor to CPU implicitly. It materializes a device operand only where a
  host-evaluated transform, scheduler, sampler, or generated-input program
  needs the values, and does so once per source at that transform's outer
  boundary rather than inside a per-element helper.
- **Compatibility aliases.** `WorldModelPipeline` and `LegacyWorldModel` must
  keep pointing at `Pipeline` and `LatentDynamicsModel`.

## Current Scope

Text, image, video, and action generation from Mobius packages; one image or
video per text-generation request; image-to-video conditioning and
classifier-free guidance for packages that declare them. Incremental stage
execution is synchronous and single-consumer, and snapshotting a session
mid-run is not included.

Cancellation and deadlines cover `PipelineSession`, `StageRun`, and the
generic `Model`. Two limits are deliberate and documented rather than
hidden:

- One shared process-wide watchdog claims every armed deadline, so a deadline
  fires while work is blocked rather than at the next boundary. Inside an
  ONNX Runtime call that claim reaches ORT's termination flag, which ORT
  checks between graph nodes, so a single long-running kernel can overrun the
  deadline by however long that one kernel takes.
- The high-level modality APIs on `WorldModel` and the fixed
  `LatentDynamicsModel`, `WorldModel::Step`, and `Rollout` surfaces take no
  token yet.

Output media encoding is not included, and fixed-step stochastic FlowMatch
schedules are not yet supported.

Component placement covers where each component's session is built and a
conservative, inspectable description of what every connection would then have
to move. The following are deliberately out of scope in this milestone and are
not partially implemented anywhere:

- session warm-up and lazy or deferred component loading;
- offloading or evicting a component's session or weights;
- peer-to-peer device-to-device transfers, so two different non-CPU devices
  are always reported as staging through the host;
- rewriting execution from the transfer plan. Nothing reads it, so turning
  `device_outputs` on or off changes what the session does while the plan
  keeps describing the same physical placement.
