# src

## Purpose

Implements the `onnx_world_model` C++ library declared in
`include/onnx_world_model`: ONNX Runtime loading and session management, tensor
marshalling, cooperative cancellation and deadlines, Mobius pipeline manifest
parsing and validation, staged pipeline execution, and the fixed
latent-dynamics compatibility layer.

## Responsibilities

- Confine every ONNX Runtime dependency to `dynamic_library.*` and
  `ort_backend.*` so the rest of the library stays independent of a specific
  ORT build and links against no ORT binary.
- Turn `pipeline.json` into a `PipelineManifest` and reject anything the
  runtime cannot execute at load time rather than at inference time.
- Place each component's ONNX Runtime session where the caller asked: a
  per-component execution-provider order, provider options, graph optimization
  level, and thread counts layered over the pipeline-wide `RuntimeOptions`,
  validated against the manifest before a single model file is opened. From
  the ports ONNX Runtime actually assigned, classify every manifest connection
  into a conservative `PipelineTransferPlan`. That plan is inspection only:
  nothing executes from it, and warm-up, lazy loading, offload and eviction,
  and peer-to-peer transfers are deliberately not implemented.
- Answer, on demand, what the runtime can honestly observe about each
  component's numeric precision: the manifest's declared `parameter_dtype`
  copied verbatim, the loaded session's real port dtypes in graph order, the
  endpoint-qualified state ports that component carries, and the execution
  providers registered on the session. That report parses no file, calls no extra
  ONNX Runtime API, enforces nothing, and deliberately cannot say whether a
  component's weights are quantized.
- Execute pipeline stages, including generated inputs, transforms, diffusion
  schedulers, classifier-free guidance, autoregressive decoding, and recurrent
  state lifecycles, through one state machine that a caller can either drain
  in a single `RunStage` call or step through a `StageRun`. That machine owns
  a run's identity, slot, bindings, cancellation, step count, and terminal
  event; what one step of a kind actually does lives behind an internal stage
  executor instead, reached through a narrow host interface rather than
  through the session's internals.
- Stop that execution cooperatively: every boundary polls the run's
  `CancellationToken`, `StageRun::RequestCancellation` signals a step already
  in flight without taking the session lock, one shared process-wide deadline
  watchdog claims a deadline on time so blocked work does not wait for the
  next boundary, and the ORT backend terminates an in-flight `Session::Run`
  through a per-call `Ort::RunOptions`. A cancellation is never a rollback.
- Admit that execution fairly: one shared admission controller per `Pipeline`
  caps how many executions run at once, globally and per stage kind, and
  queues the rest oldest-first without letting a saturated kind block a
  different eligible one. This is admission scheduling only — nothing is
  merged, split, reordered, or preempted — and a queued request still
  observes its own cancellation token and deadline. That controller also
  reports its own admitted and queued counts, per stage kind and in total,
  as one consistent snapshot that reading cannot perturb.
- Measure that execution when, and only when, a caller opts in: one shared
  telemetry collector per `Pipeline` records per-component call counts, byte
  totals and durations, per-stage execution, step, and completion counts,
  per-stage-kind admission wait outcomes, and the device-to-host
  materializations the session performs, and reports them as an immutable
  `PipelineTelemetrySnapshot` under a reset-able epoch. Telemetry changes what
  is measured and never what is executed; with it disabled the `Pipeline`
  holds no collector and every recording site is one null-pointer test.
  Provider peak memory and exact host-to-device byte counts are deliberately
  not implemented, because only ONNX Runtime holds that data.
- Trace ONNX Runtime nodes per run when a caller opts in a second time, by
  configuring a trace directory: each component call is handed a unique
  profile-file prefix, ONNX Runtime writes that one call's trace, and the
  collector records one bounded `PipelineTraceRecord` naming the file, its
  size, and the call's outcome. Profiling is per run and never per session, a
  trace is discovered by prefix and never parsed here, a trace that cannot be
  found is counted as a profiling failure and never changes the call, and no
  trace file is ever deleted.
- Keep `Pipeline` immutable and shareable while `PipelineSession` owns one
  request's mutable state, and let a session capture that state as an
  immutable in-memory `PipelineSessionSnapshot` it can restore or fork from, or
  hold by name as a checkpoint.
- Preserve device-backed tensor storage through the session and materialize a
  device source to CPU only once, at the outer boundary of each host-evaluated
  transform, scheduler, sampler, or generated-input program.

## Key Files

| File | Responsibility |
|---|---|
| `cancellation.hpp/.cpp` | The cancellation state machine behind `CancellationToken` and `CancellationSource`: first-reason-wins claiming, deadline claiming from both a poll and the one process-wide `detail::DeadlineService` watchdog, the blocking `WaitForCancellation`, the race-free callback registry, the `detail::CancellationAccess` seam, and the RAII `detail::CancellationRegistration`. |
| `dynamic_library.hpp/.cpp` | RAII shared-library handle; binds the `OrtApi` table once per process. |
| `ort_backend.hpp/.cpp` | The only ORT-facing translation unit pair: shares the process-wide ORT environment, builds sessions, applies providers, reads each session's per-port memory plan once and publishes it as `TensorSpec::device`, retains I/O-bound outputs in ORT-owned device buffers, and terminates an in-flight `Session::Run` through a per-call `Ort::RunOptions` that it also enables per-run profiling on when the call supplied a profile-file prefix. |
| `tensor.cpp` | Canonical tensor devices, owned CPU buffers, checked shape arithmetic, explicit CPU materialization, and copy-on-write mutation. |
| `model.cpp` | `Model` facade with its one `ModelRunOptions` implementation the other two `Run` overloads delegate to, the default cancellable and default `ModelRunOptions` `ModelBackend::Run` overloads, provider-name normalization, tensor-versus-signature validation. |
| `world_model.cpp` | `WorldModel` contract enforcement and `Rollout` recurrent state. |
| `pipeline.cpp` | `pipeline.json` parsing, per-component placement resolution and validation, `PipelinePackage` loading, the connection transfer plan, and the on-demand per-component precision report. |
| `pipeline_manifest_common.hpp/.cpp` | JSON field, token, and portable-name checks shared by parsing and validation. |
| `stage_registry.hpp/.cpp` | The one source of truth for stage kinds: each kind's manifest name, the `StageExecutionStrategy` the stage state machine drives it with, and the exact option names its manifest options object may carry, plus `kStageKindCount` and the non-throwing `FindStageKind`/`StageKindIndex` lookups every other file resolves a kind through. |
| `stage_executor.hpp/.cpp` | The strategy objects `StageRun` steps a stage through: the narrow `StageExecutionHost` a strategy may reach the session by, the `StageStepContext` that hands a run's own inputs to the first step that actually executes, the `StageExecutor` interface whose absent step event means "complete", the one-pass executor that `single_pass`, `state_transition`, `composite`, and `on_demand` all share, and `MakeStageExecutor`, the exhaustive switch over the registry's strategies that builds one before a run claims the session's slot. |
| `stage_executor_autoregressive.cpp` | The autoregressive strategy: the sampling plan, seed, end-of-sequence set, and prompt-derived token budget resolved when the run is built, the decode-sample-substitute-feed-back step, the per-lane end-of-sequence latch that requests the stop the next step reports, and the batch-major `generated_token_ids` added to the result without displacing a manifest output of that name. |
| `stage_executor_iterative.cpp` | The iterative strategy: the absolute scheduler cursor read once when the run is built, the one scheduler step per `StepLocked` while the session's stage cursor is below it, and the last step's outputs -- or the session's current outputs for a stage that was already complete -- as the run's result. |
| `pipeline_manifest_validation.hpp/.cpp` | Semantic validation of a parsed manifest: dataflow, programs, stage options, capabilities, state lifecycles. |
| `pipeline_scheduler.hpp/.cpp` | The shared admission controller behind `PipelineSchedulingOptions`: the validation of its per-kind keys against `stage_registry`, the per-kind permit buckets that registry sizes and orders, the cancellation- and deadline-aware FIFO queue with its oldest-eligible pump, the RAII `detail::PipelineLease`, the per-stage-kind admission outcome each acquisition records into the telemetry collector, and `detail::SnapshotSchedulingStats`, the consistent reading of that state `Pipeline::scheduling_stats` returns. |
| `pipeline_telemetry.hpp/.cpp` | The opt-in telemetry collector behind `PipelineTelemetryOptions`: the immutable per-epoch counter slab pre-populated from the manifest, the relaxed-atomic counters and their compare-exchange maxima, the RAII component and stage recorders that classify an outcome by `ErrorCode`, the admission and device-transfer hooks, `detail::ValidatePipelineTelemetryOptions` and the trace directory it creates on the cold path, the sanitized unique per-call trace prefix and the `noexcept` discovery that turns the file ONNX Runtime wrote into one capped `PipelineTraceRecord`, the atomic slab publication that makes `Pipeline::ResetTelemetry` a new epoch rather than a barrier, and `detail::SnapshotPipelineTelemetry`, the reading `Pipeline::telemetry_snapshot` returns. |
| `pipeline_session.cpp` | `PipelineSession::Impl`, the staged execution engine, all per-trajectory state in one `SessionState` bundle, the `StageRun::Impl` state machine that both `RunStage` and `BeginStage` execute through and the nested `SessionHost` adapter that is the only path from a stage executor back into that session, its cancellation source and the link into a caller's token, the admission leases every execution takes before the session lock, the telemetry recorders those executions and their component calls are measured by, the snapshot, restore, fork, and named-checkpoint operations over that bundle, the single `MaterializeHost` device-versus-host materialization boundary, and `Pipeline` itself. |

`cancellation.hpp`, `dynamic_library.hpp`, `ort_backend.hpp`,
`pipeline_manifest_common.hpp`, `pipeline_manifest_validation.hpp`,
`pipeline_scheduler.hpp`, `pipeline_telemetry.hpp`, `stage_executor.hpp`, and
`stage_registry.hpp`
are internal: they live in `onnx_world_model::detail` and are not installed.

## Dependencies

- Public declarations come from `include/onnx_world_model`; this directory adds
  no public type of its own.
- Third-party: ONNX Runtime C/C++ headers (loaded at run time, resolved by
  `cmake/FetchOrtHeaders.cmake`) and nlohmann/json (header only, resolved by
  `cmake/FetchJsonHeader.cmake`).
- Internal direction is one-way and acyclic:

```text
cancellation -> ort_backend -> model -> world_model
dynamic_library -> ort_backend       `-> pipeline -> pipeline_session
cancellation -> pipeline_session
cancellation -> pipeline_scheduler -> pipeline_session
pipeline_telemetry -> pipeline_scheduler -> pipeline_session
pipeline_telemetry -> pipeline_session
stage_registry -> pipeline_manifest_validation
stage_registry -> pipeline_telemetry
stage_registry -> pipeline_scheduler
stage_registry -> pipeline_session
stage_registry -> stage_executor -> pipeline_session
pipeline_manifest_common -> pipeline_manifest_validation -> pipeline
```

`stage_registry` is the bottom of that graph: it depends on nothing but the
standard library, which is what lets manifest validation, the scheduler,
telemetry, and the session all read one stage-kind list without a cycle. A
stage kind exists there or it does not exist at all — its `kStageKindCount`
sizes every per-kind array in this directory, its order is the order of the
scheduler's buckets, telemetry's admission entries, and both public per-stage-
kind maps, and its `StageExecutionStrategy` is what `MakeStageExecutor`
switches on instead of comparing the kind string.

`stage_executor` sits between that registry and the session: it declares the
`StageExecutionHost` a strategy body may reach a session by and implements one
executor per strategy, and it depends on nothing in `pipeline_session.cpp` —
only on the public headers and the registry. The dependency runs the other
way, and it runs through that interface alone: `pipeline_session.cpp`
implements the host as a class nested in `StageRun::Impl`, so an executor
cannot see the session's mutex, run slot, admission scheduler, telemetry
collector, snapshots, or `SessionState`, and nothing in `stage_executor*.cpp`
takes a lock — `StageRun` holds the session lock across every executor call.

`pipeline_manifest_validation.cpp` never parses raw JSON documents itself; it
receives an already-populated `PipelineManifest` from `pipeline.cpp`.
`pipeline_scheduler.cpp` never executes, resumes, or inspects a stage; it only
decides when one may start, so it depends on nothing in `pipeline_session.cpp`.
Its `Stats` is the one const path: it takes the same mutex — which is `mutable`
for exactly that reason — copies counters into stack arrays, tallies the queue,
and builds its maps only after releasing the lock, so observing admission
allocates nothing under the lock and cannot block it.
`pipeline_telemetry.cpp` sits below both: it records and never reads back, it
takes its admission keys from `StageKindDefinitions()` in `stage_registry.hpp`
rather than from the scheduler, so the two agree without depending on each
other, and its counter paths touch nothing but their own
atomics. Its one exception is tracing, which is explicit about being different:
it takes a trace mutex that guards only its record vector, allocates one record
per traced call, and reads the configured trace directory to find the file ONNX
Runtime wrote. That work still happens under no other lock, calls back into no
other file, and cannot fail the call it measures.

The one lock order this directory must preserve is
`CancellationState`'s callback mutex, then the scheduler mutex, then the
session mutex, then ONNX Runtime. A cancellation registration is therefore
never created or destroyed while the scheduler mutex is held, and every
execution takes its admission lease before it takes the session lock.
Telemetry is outside that order entirely: it takes none of those locks, calls
back into none of those files, and only ever updates relaxed atomics, loads an
atomic `shared_ptr`, or takes its own trace mutex, so it can neither deadlock
nor reorder anything.

## Tests

```console
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

`tests/cpp/tensor_test.cpp`, `tests/cpp/model_test.cpp`,
`tests/cpp/cancellation_test.cpp`, `tests/cpp/pipeline_test.cpp`,
`tests/cpp/pipeline_device_test.cpp`,
`tests/cpp/pipeline_snapshot_test.cpp`,
`tests/cpp/pipeline_stream_test.cpp`,
`tests/cpp/pipeline_cancellation_test.cpp`,
`tests/cpp/pipeline_scheduler_test.cpp`,
`tests/cpp/pipeline_telemetry_test.cpp`,
`tests/cpp/pipeline_trace_test.cpp`, and
`tests/cpp/pipeline_precision_test.cpp` cover this directory directly;
`tests/cpp/pipeline_test.cpp` is the primary coverage for manifest parsing and
validation, `tests/cpp/cancellation_test.cpp` is the primary coverage for
`cancellation.cpp`, `tests/cpp/pipeline_device_test.cpp` is the primary
coverage for the device-versus-host materialization boundaries in
`pipeline_session.cpp` and for the transfer plan `pipeline.cpp` computes, `tests/cpp/pipeline_snapshot_test.cpp` is the primary
coverage for its snapshot, restore, fork, and named-checkpoint operations,
`tests/cpp/pipeline_stream_test.cpp` is the primary coverage for its
`StageRun` state machine and the `RunStage` parity that machine guarantees,
`tests/cpp/pipeline_cancellation_test.cpp` is the primary coverage for
that machine's cancellation and deadline boundaries,
`tests/cpp/pipeline_scheduler_test.cpp` is the primary coverage for
`pipeline_scheduler.cpp` and for which calls take an admission lease, and
`tests/cpp/pipeline_telemetry_test.cpp` is the primary coverage for
`pipeline_telemetry.cpp` and for what each recording site in
`pipeline_session.cpp` and `pipeline_scheduler.cpp` counts, and
`tests/cpp/pipeline_trace_test.cpp` is the primary coverage for its trace
prefix, file discovery, record retention, and configuration validation, and
`tests/cpp/pipeline_precision_test.cpp` is the primary coverage for the
precision report `pipeline.cpp` assembles.
`tests/python/` exercises the same code through the `_native` extension
module; `tests/python/test_placement.py` is where the load-time placement
overrides and their rejection cases are exercised end to end,
`tests/python/test_telemetry.py` is where the telemetry surface is,
`tests/python/test_precision.py` is where the precision report is read back
through Python, and
`tests/python/test_trace.py` is where real ONNX Runtime trace files are
produced and read back.
