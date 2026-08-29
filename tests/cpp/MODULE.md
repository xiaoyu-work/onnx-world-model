# tests/cpp

## Purpose

Holds the C++ test executables that verify the library through its public
headers, using in-process stub backends so the default run needs no ONNX
Runtime library and no real ONNX model.

## Responsibilities

- Cover the value semantics and error paths of `Tensor`.
- Cover `Model` metadata validation and execution-provider name normalization.
- Cover pipeline manifest parsing, manifest rejection cases, stage execution,
  schedulers, classifier-free guidance, and recurrent state.
- Cover device-tensor preservation across `PipelineSession`: which paths keep
  the producer's `TensorBuffer` and which materialize it, using a
  non-host-accessible stub buffer with a shared CPU-copy counter. The same
  executable covers the conservative `PipelineTransferPlan` a
  `PipelinePackage` computes, using stub backends that report whichever
  `TensorSpec::device` — or no device at all — each classification needs.
  This is a separate executable from the manifest and stage-execution tests.
- Cover the in-memory `PipelineSessionSnapshot` contract: recurrent-state
  round trips, parent and child independence after a fork, package identity,
  device-buffer sharing without materialization, and random-engine
  determinism. The same executable covers the named-checkpoint contract:
  create, query, replace, rewind, drop, empty and unknown names, `Reset`
  clearing, and a fork's empty checkpoint namespace. This is also a separate
  executable.
- Cover the incremental `StageRun` contract in its own executable: parity with
  `RunStage` for every stage kind, the single terminal event, the run slot's
  exclusion of conflicting session calls, cancellation, failure, handle and
  session ownership, and device-preserving stage events.
- Cover the cancellation primitives in their own executable: the inert default
  token, first-reason-wins claiming, deadline claiming from both a poll and
  the shared watchdog, the blocking `WaitForCancellation`, stable error
  codes, move semantics, and cancelling from another thread.
- Cover pipeline cancellation and deadlines in a separate executable:
  `RequestCancellation` against a step that holds the session lock, an
  externally supplied token, `deadline_exceeded` as an outcome distinct from
  `cancelled`, a deadline stopping a backend already blocked inside a
  component pass, run-slot ownership after a cancelled run, and the promise
  that cancelling neither rolls back applied state nor materializes a device
  buffer.
- Cover shared admission scheduling in its own executable: the unlimited
  default, a global ceiling that is reached but never exceeded and still
  admits everyone, a per-stage-kind ceiling that serializes its own kind
  without blocking a different eligible one, cancellation and deadlines while
  queued -- including a queued `StageRun::Step` or `StageRun::Finish` that
  must still close its handle and release the session's active-run slot so the
  session stays usable -- permit recovery after a backend failure, which calls
  are executions and which are not, and whether a copied, forked, or
  separately constructed `Pipeline` shares the ceiling. Every "it is queued"
  claim there is synchronized on `Pipeline::scheduling_stats`, never on a wait
  that timed out.
- Cover opt-in runtime telemetry in its own executable: the disabled default
  that collects nothing and reads as `enabled=false` with empty maps, the
  fully pre-populated idle reading of an enabled pipeline, per-component call
  counts, exact byte totals and durations, outcome classification by
  `ErrorCode` for a failure, a cancellation, and a deadline, execution versus
  step versus completion counts across `RunStage`, `StepStage`, and an
  incremental `StageRun`, a cached `Finish` that changes nothing, admission
  wait outcomes for a constrained stage kind — an immediate grant, a queued
  grant, a queued cancellation, and a queued deadline — and their complete
  absence for an unlimited one, one host materialization per device source
  with its exact byte count, component input residency, the epoch reset, and
  which `Pipeline` values share a collector. No check there asserts an
  absolute duration: a duration is compared only against zero and a maximum
  only against its own total.
- Cover per-run ONNX Runtime node tracing in its own executable, with stub
  backends that write the file ONNX Runtime would have written: the empty
  prefix a counters-only or disabled pipeline hands a component, the shape and
  sanitization of a real prefix, one record per component call with its path,
  size, outcome, and duration, prefixes that stay unique while four calls are
  outstanding at once, outcome classification for a failure, a cancellation,
  and a deadline, the three ways discovery fails and the counted profiling
  failure each produces without touching the call's own outcome, the retention
  cap that drops records while the files keep being written, the reset that
  clears records and leaves every file, and the configuration errors a
  `Pipeline` refuses to be built with. Every file it writes lives in one
  scratch directory it creates and removes itself.
- Cover the per-component precision report in its own executable: manifest
  component order, a declared `FLOAT16` and a declared `INT8` parameter dtype
  both reported unchanged -- which is what proves a non-floating declaration is
  not rejected -- a profile-less component with no declaration reporting
  `nullopt`, mixed graph port dtypes listed exactly and never reconciled with
  the declaration, the state ports two states associate with one component in
  manifest order with their shared carrier port listed once, the providers a
  backend selected, the empty list a backend with no providers reports, and a
  reading that can be edited without changing the package, the manifest, or
  the next reading.

## Key Files

| File | CTest name | Coverage |
|---|---|---|
| `tensor_test.cpp` | `tensor_test` | `Tensor` construction, typed views, overflow checks, copy-on-write. |
| `model_test.cpp` | `model_test` | `Model` input and output validation through the `AddOneBackend` stub. |
| `cancellation_test.cpp` | `cancellation_test` | `CancellationToken` and `CancellationSource`: default inertness, explicit cancel, first-reason-wins, zero, negative, future, and saturating out-of-range deadlines, `ErrorCode::cancelled` versus `ErrorCode::deadline_exceeded`, moved-from sources, concurrent cancels, cross-thread observation, and the blocking `WaitForCancellation` — released by the shared watchdog at its deadline, by an explicit cancel, rejected with `invalid_argument` for an uncancellable state, and never released before its own deadline. |
| `pipeline_test.cpp` | `pipeline_test` | Manifest parsing and rejection, stage execution, guidance, schedulers, state lifecycle. |
| `pipeline_device_test.cpp` | `pipeline_device_test` | Device-tensor preservation across `PipelineSession`: rank adaptation, transform-free connections, reshape, public outputs, and CPU-transform materialization. It also covers `PipelinePackage::transfer_plan`: one entry per manifest connection in manifest order with the recurrent edge flagged, `direct` and its `direct_bind_eligible` for identical known devices, `upload`, `download`, `host_staged` for two device ordinals and for two device types, `host_transform` for a cast and for a shape-changing reshape while an identical-shape reshape stays `direct`, `unknown` for a backend that reports no placement — which outranks a host transform — and the constructor's `device_outputs_enabled` flag reaching the plan without rewriting a kind. Two `static_assert`s in `main` are what assert that `PipelinePlacementOptions` cannot be handed to the already-built-package `Pipeline` constructor. |
| `pipeline_snapshot_test.cpp` | `pipeline_snapshot_test` | `PipelineSession::Snapshot`, `Restore`, `Fork`, and the named `Checkpoint`, `RestoreCheckpoint`, `DropCheckpoint`, and `HasCheckpoint`: state round trips, fork independence, package identity rejection, zero-copy device sharing, random-engine determinism, and checkpoint create/replace/rewind/drop plus empty and unknown name failures. |
| `pipeline_stream_test.cpp` | `pipeline_stream_test` | `PipelineSession::BeginStage` and `StageRun`: greedy, sampled, and per-lane early-stopping autoregressive parity with `RunStage`, the once-only token budget, iterative and single-pass parity, a `composite` stage sharing the one-pass strategy, one terminal `StageEvent`, `Finish` after partial stepping, the active-run exclusions, cancellation and destruction, failure without rollback, moved handles, session move and destruction safety, and device-preserving event outputs. |
| `pipeline_cancellation_test.cpp` | `pipeline_cancellation_test` | `StageRun::RequestCancellation` interrupting a `Finish` that holds the session lock, an external `CancellationSource`, no extra component pass after a cancellation between steps, partial state surviving, `deadline_exceeded` versus `cancelled`, an expired deadline never claiming the run slot, a stale handle not releasing a newer run, session reuse with a fresh token, a reused cancelled token failing immediately, direct `StepStage` honoring its token, a `~20 ms` deadline unblocking a `RunStage` and a `Step` whose backend is parked on `WaitForCancellation` with no polling at all, exception-safe restoration of the classifier-free guidance scratch state when the unconditional pass is cancelled, `Model::Run`'s cancellable overload, and no device materialization from cancelling. |
| `pipeline_scheduler_test.cpp` | `pipeline_scheduler_test` | `PipelineSchedulingOptions` through the public API only: the unlimited default admitting three sessions at once, a global ceiling of two that is reached but never exceeded and drains work-conservingly, a per-kind ceiling of one that serializes `single_pass` while an uncapped `state_transition` still enters, a queued execution cancelled or stopped by its deadline without reaching the backend and without leaving its queue position behind, a queued `StageRun::Step` cancelled and a queued `StageRun::Finish` stopped by its deadline each closing the handle and releasing the session's active-run slot so the same session immediately accepts a `Snapshot`, a fresh `BeginStage`, and a `RunStage`, a stale handle's `Cancel` not releasing a newer run's slot, a permit returned after a backend failure, `BeginStage` and an idle handle holding nothing while `Step`, `Finish`, and `StepStage` each hold one, a completed `Finish` returning its cached result without admission, the `PipelineSchedulingStats` reading -- all six stage kinds always present, zeros for an unlimited pipeline, the admitted and queued counts under a ceiling, and a drain back to zero -- which is also what every queued-request claim in the file synchronizes on, a copied and a forked session sharing the ceiling while two separately constructed `Pipeline`s do not, unknown, empty, and wrong-case stage-kind keys rejected with `invalid_argument`, and a limit of zero meaning unlimited. |
| `pipeline_telemetry_test.cpp` | `pipeline_telemetry_test` | `PipelineTelemetryOptions` and `Pipeline::telemetry_snapshot` through the public API only: a disabled pipeline reading as `enabled=false`, epoch `0`, empty maps, and zero transfers even after running a stage, and resetting one changing nothing; an enabled pipeline starting at epoch `1` already carrying every manifest component, every manifest stage, and all six stage kinds at zero; one component call's exact input and output byte totals and its positive duration; `failed_calls`, `cancelled_calls`, and `deadline_exceeded_calls` chosen by the `ErrorCode` a stub backend throws, with the matching stage bucket, the presented bytes still counted, and no step or completion; one `RunStage` of a one-pass stage as one execution, one step, and one completion and of a three-step iterative stage as one execution, three steps, and one completion; four `Step` calls as four executions with three steps and one completion; one `Step` plus a draining `Finish` as two executions; a `Finish` on a completed run changing nothing; two direct `StepStage` calls as two executions and two steps with no completion; no admission counters at all for an unlimited kind whose executions are still measured; an immediate grant admitted with a zero wait and not queued; a gated second execution counted as one queued acquisition, two admissions, and a positive wait; a queued cancellation and a queued deadline each counted as their own outcome, with a positive wait and no stage execution because they never got a permit; exactly one device-to-host copy with its exact byte count for a cast connection, with host and device-resident presentation totals; a reset advancing the epoch, zeroing every counter, keeping the maps populated, and continuing to collect; a `Pipeline` copy and a forked session sharing the collector while a separately constructed `Pipeline` does not; a moved-from `Pipeline` reporting the disabled reading; and a returned reading never updating itself. |
| `pipeline_trace_test.cpp` | `pipeline_trace_test` | `ModelRunOptions::profile_file_prefix`, `PipelineTelemetryOptions::trace_directory` and `max_trace_records`, and `PipelineTelemetrySnapshot::traces` through the public API only, with stub backends that record the prefix they were handed and write `<prefix>_stub.json` themselves: an empty prefix and no records for a counters-only pipeline and for one without telemetry; a prefix inside the configured directory whose file name starts with the sanitized component -- `vision encoder+1` becoming `vision_encoder_1` -- for a traced one; two calls never sharing a prefix and four concurrent calls, all outstanding at once behind a gate, producing four distinct prefixes, four records, and four files; one record per component call carrying epoch `1`, an increasing `trace_id`, `success`, an absolute existing path, a positive size, and a positive duration; a scripted `runtime_execution`, `cancelled`, and `deadline_exceeded` failure each recorded with its own outcome and its trace still found; an already cancelled run never calling the component, recording nothing, and writing no file; no file, two files, and a zero-byte file each counted once in `failed_traces` and recorded with an empty path, a zero size, and the call's own successful outcome unchanged; a cap of two keeping the first two records, counting one `dropped_traces`, and still leaving three files on disk; a reset advancing the epoch, clearing records and both counters, deleting no file, and continuing into the new epoch with a still-increasing `trace_id`; and `invalid_argument` for a trace directory without telemetry, for a zero cap, and for a directory that is really a file, with a rejected configuration creating nothing and an accepted one creating a nested directory. |
| `pipeline_precision_test.cpp` | `pipeline_precision_test` | `PrecisionPort`, `ComponentPrecisionReport`, and the `PipelinePackage::precision_report` and `Pipeline::precision_report` accessors through the public API only, with `StubBackend`s that report an explicit `ModelMetadata` and never run: one entry per manifest component in manifest order; a declared `FLOAT16` and a declared `INT8` parameter dtype each reported unchanged, which is what proves an integer declaration is not rejected; `nullopt` for a profile-less component that declares none; graph inputs and outputs listed in graph order with the live types -- `int64` token ids and `float32` logits beside a `float16` declaration -- and asserted to be reported independently of that declaration rather than reconciled with it; empty state lists for a stateless component; two states over one component reporting `carrier.state_a` and `carrier.state_b` in manifest order with the shared `carrier.next` output listed once; providers reported in the order the backend selected them and per component; an empty provider list for a backend that reports none; and an edited reading changing neither a later reading nor the manifest's own declared dtype. |

Each file is a self-contained `main()` with local `Check` and `CheckThrows`
helpers (and `CheckThrowsMessage` where a message is asserted, or
`CheckThrowsCode`, `CheckThrowsState`, and `CheckThrowsInvalidArgument` where a
specific `ErrorCode` is asserted), a file-local
`failures` counter, and a non-zero exit code on failure.
`pipeline_telemetry_test.cpp` additionally keeps its checks in one
`RunTelemetryChecks()` that `main` calls inside a `try`, so an unexpected
`Error` -- an embedded manifest the file got wrong, say -- is reported as a
failure instead of aborting the process with no message.
`pipeline_trace_test.cpp` does the same with `RunTraceChecks()` and adds the
one filesystem contract in this directory: it creates
`pipeline_trace_test_scratch` beside the working directory CTest runs it in,
removes any stale copy first so file counts cannot lie, writes every trace file
under it, and removes the whole tree on the way out whether the checks passed
or failed. `pipeline_precision_test.cpp` wraps its own checks in a `try` for
the same reason, so a manifest it got wrong is reported rather than aborting.
There is no
third-party test framework; a new test is a new check inside an existing
`main()`, or a new executable added to the
`foreach(test_name IN ITEMS ...)` list in the root `CMakeLists.txt`.

## Dependencies

- Links `onnx_world_model` and includes only public headers from
  `include/onnx_world_model`; it never includes `src/` internal headers.
- Components under test are stub implementations of
  `onnx_world_model::ModelBackend` defined inside each test file, plus
  `FakeDeviceBuffer`, a stub `onnx_world_model::TensorBuffer` in
  `pipeline_device_test.cpp` that reports a non-CPU device, refuses host
  access, and counts each `CopyToCpu`, and `PlacedBackend` in the same file,
  whose ports report whichever `TensorSpec::device` a transfer-plan check
  needs — including none, which is how "unknown" is exercised;
  `pipeline_snapshot_test.cpp`,
  `pipeline_stream_test.cpp`, and `pipeline_cancellation_test.cpp` have their
  own `CountingDeviceBuffer` with the same shape.
- Concurrency is expressed with condition variables, never with sleeps:
  `pipeline_stream_test.cpp` uses `BlockingControl` with
  `BlockingIncrementBackend`, `pipeline_cancellation_test.cpp` uses its
  own `BlockingControl` with `BlockingCancellableBackend`, a stub that
  overrides the cancellable `Run`, signals that it entered, waits to be
  released, and only then checks its token, and
  `pipeline_scheduler_test.cpp` uses one shared `Gate` with `GatedBackend`,
  which records the concurrency peak per component before parking, and
  `pipeline_telemetry_test.cpp` uses the same shape for its admission checks:
  a smaller shared `Gate` with its own `GatedBackend`, plus a `CountingBackend`
  whose shared failure cell makes the next call throw a chosen `ErrorCode` and
  its own `FakeDeviceBuffer` with a `DeviceCopyCounter`.
  `pipeline_trace_test.cpp` uses the same shape again for its uniqueness
  check: one `TracingBackend` that records every prefix it is handed under a
  mutex, writes the trace file its `TraceBehavior` describes, optionally parks
  on a shared `Gate` so four prefixes are provably outstanding at once, and
  optionally throws a chosen `ErrorCode` afterwards.
  None of them infers "this request is still queued" from a wait that timed out: it polls
  `Pipeline::scheduling_stats().queued_executions` until the queue holds
  exactly the expected number, bounded by a five-second budget, so the
  scheduler's own published state is the synchronization point and a broken
  scheduler fails a check rather than hanging. Its deadlines are long enough
  that the queue insertion is always observed first. Any call that could block
  on admission -- including the probes that must *not* queue -- runs on a
  worker thread under a finite budget, and a budget that expires opens the
  gate before joining and records the failure. A `Succeeded` helper reports a
  call that was expected to work but threw, so a run-slot regression fails a
  check instead of unwinding past a worker thread that is still parked inside
  the gate.
- Every CTest in `CMakeLists.txt` carries `TIMEOUT 60` as a last-resort hang
  guard. Each test bounds its own waits well inside that, so reaching it means
  a deadlock.
- `pipeline_test.cpp` writes temporary package and scheduler directories under
  the filesystem temporary directory, and optionally accepts a real package
  directory and ORT library path as `argv[1]` and `argv[2]`.
  `pipeline_trace_test.cpp` writes only inside the
  `pipeline_trace_test_scratch` directory it creates in the working directory
  and removes when it finishes.
- `cancellation_test.cpp`, `pipeline_device_test.cpp`,
  `pipeline_snapshot_test.cpp`, `pipeline_stream_test.cpp`,
  `pipeline_cancellation_test.cpp`, `pipeline_scheduler_test.cpp`,
  `pipeline_telemetry_test.cpp`, `pipeline_trace_test.cpp`, and
  `pipeline_precision_test.cpp` take no
  arguments; every one of them builds each `PipelinePackage` in
  memory from an embedded manifest string, and only `pipeline_trace_test.cpp`
  touches the filesystem at all.

## Tests

These files are the tests. Build and run them with:

```console
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

The `dev` preset sets `ONNX_WORLD_MODEL_BUILD_TESTS=ON`; the default build does
not compile this directory.
