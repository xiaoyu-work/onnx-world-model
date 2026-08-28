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
- Execute pipeline stages, including generated inputs, transforms, diffusion
  schedulers, classifier-free guidance, autoregressive decoding, and recurrent
  state lifecycles, through one state machine that a caller can either drain
  in a single `RunStage` call or step through a `StageRun`.
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
| `ort_backend.hpp/.cpp` | The only ORT-facing translation unit pair: shares the process-wide ORT environment, builds sessions, applies providers, retains I/O-bound outputs in ORT-owned device buffers, and terminates an in-flight `Session::Run` through a per-call `Ort::RunOptions`. |
| `tensor.cpp` | Canonical tensor devices, owned CPU buffers, checked shape arithmetic, explicit CPU materialization, and copy-on-write mutation. |
| `model.cpp` | `Model` facade, the default cancellable `ModelBackend::Run` overload, provider-name normalization, tensor-versus-signature validation. |
| `world_model.cpp` | `WorldModel` contract enforcement and `Rollout` recurrent state. |
| `pipeline.cpp` | `pipeline.json` parsing and `PipelinePackage` loading. |
| `pipeline_manifest_common.hpp/.cpp` | JSON field, token, and portable-name checks shared by parsing and validation. |
| `pipeline_manifest_validation.hpp/.cpp` | Semantic validation of a parsed manifest: dataflow, programs, stage options, capabilities, state lifecycles. |
| `pipeline_scheduler.hpp/.cpp` | The shared admission controller behind `PipelineSchedulingOptions`: the supported stage-kind list and its validation, the per-kind permit buckets, the cancellation- and deadline-aware FIFO queue with its oldest-eligible pump, the RAII `detail::PipelineLease`, and `detail::SnapshotSchedulingStats`, the consistent reading of that state `Pipeline::scheduling_stats` returns. |
| `pipeline_session.cpp` | `PipelineSession::Impl`, the staged execution engine, all per-trajectory state in one `SessionState` bundle, the `StageRun::Impl` state machine that both `RunStage` and `BeginStage` execute through, its cancellation source and the link into a caller's token, the admission leases every execution takes before the session lock, the snapshot, restore, fork, and named-checkpoint operations over that bundle, the device-versus-host materialization boundaries, and `Pipeline` itself. |

`cancellation.hpp`, `dynamic_library.hpp`, `ort_backend.hpp`,
`pipeline_manifest_common.hpp`, `pipeline_manifest_validation.hpp`, and
`pipeline_scheduler.hpp` are internal: they live in
`onnx_world_model::detail` and are not installed.

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
pipeline_manifest_common -> pipeline_manifest_validation -> pipeline
```

`pipeline_manifest_validation.cpp` never parses raw JSON documents itself; it
receives an already-populated `PipelineManifest` from `pipeline.cpp`.
`pipeline_scheduler.cpp` never executes, resumes, or inspects a stage; it only
decides when one may start, so it depends on nothing in `pipeline_session.cpp`.
Its `Stats` is the one const path: it takes the same mutex — which is `mutable`
for exactly that reason — copies counters into stack arrays, tallies the queue,
and builds its maps only after releasing the lock, so observing admission
allocates nothing under the lock and cannot block it.

The one lock order this directory must preserve is
`CancellationState`'s callback mutex, then the scheduler mutex, then the
session mutex, then ONNX Runtime. A cancellation registration is therefore
never created or destroyed while the scheduler mutex is held, and every
execution takes its admission lease before it takes the session lock.

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
`tests/cpp/pipeline_cancellation_test.cpp`, and
`tests/cpp/pipeline_scheduler_test.cpp` cover this directory directly;
`tests/cpp/pipeline_test.cpp` is the primary coverage for manifest parsing and
validation, `tests/cpp/cancellation_test.cpp` is the primary coverage for
`cancellation.cpp`, `tests/cpp/pipeline_device_test.cpp` is the primary
coverage for the device-versus-host materialization boundaries in
`pipeline_session.cpp`, `tests/cpp/pipeline_snapshot_test.cpp` is the primary
coverage for its snapshot, restore, fork, and named-checkpoint operations,
`tests/cpp/pipeline_stream_test.cpp` is the primary coverage for its
`StageRun` state machine and the `RunStage` parity that machine guarantees,
`tests/cpp/pipeline_cancellation_test.cpp` is the primary coverage for
that machine's cancellation and deadline boundaries, and
`tests/cpp/pipeline_scheduler_test.cpp` is the primary coverage for
`pipeline_scheduler.cpp` and for which calls take an admission lease.
`tests/python/` exercises the same code through the `_native` extension
module.
