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
  non-host-accessible stub buffer with a shared CPU-copy counter.
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

## Key Files

| File | CTest name | Coverage |
|---|---|---|
| `tensor_test.cpp` | `tensor_test` | `Tensor` construction, typed views, overflow checks, copy-on-write. |
| `model_test.cpp` | `model_test` | `Model` input and output validation through the `AddOneBackend` stub. |
| `cancellation_test.cpp` | `cancellation_test` | `CancellationToken` and `CancellationSource`: default inertness, explicit cancel, first-reason-wins, zero, negative, future, and saturating out-of-range deadlines, `ErrorCode::cancelled` versus `ErrorCode::deadline_exceeded`, moved-from sources, concurrent cancels, cross-thread observation, and the blocking `WaitForCancellation` — released by the shared watchdog at its deadline, by an explicit cancel, rejected with `invalid_argument` for an uncancellable state, and never released before its own deadline. |
| `pipeline_test.cpp` | `pipeline_test` | Manifest parsing and rejection, stage execution, guidance, schedulers, state lifecycle. |
| `pipeline_device_test.cpp` | `pipeline_device_test` | Device-tensor preservation across `PipelineSession`: rank adaptation, transform-free connections, reshape, public outputs, and CPU-transform materialization. |
| `pipeline_snapshot_test.cpp` | `pipeline_snapshot_test` | `PipelineSession::Snapshot`, `Restore`, `Fork`, and the named `Checkpoint`, `RestoreCheckpoint`, `DropCheckpoint`, and `HasCheckpoint`: state round trips, fork independence, package identity rejection, zero-copy device sharing, random-engine determinism, and checkpoint create/replace/rewind/drop plus empty and unknown name failures. |
| `pipeline_stream_test.cpp` | `pipeline_stream_test` | `PipelineSession::BeginStage` and `StageRun`: greedy, sampled, and per-lane early-stopping autoregressive parity with `RunStage`, the once-only token budget, iterative and single-pass parity, one terminal `StageEvent`, `Finish` after partial stepping, the active-run exclusions, cancellation and destruction, failure without rollback, moved handles, session move and destruction safety, and device-preserving event outputs. |
| `pipeline_cancellation_test.cpp` | `pipeline_cancellation_test` | `StageRun::RequestCancellation` interrupting a `Finish` that holds the session lock, an external `CancellationSource`, no extra component pass after a cancellation between steps, partial state surviving, `deadline_exceeded` versus `cancelled`, an expired deadline never claiming the run slot, a stale handle not releasing a newer run, session reuse with a fresh token, a reused cancelled token failing immediately, direct `StepStage` honoring its token, a `~20 ms` deadline unblocking a `RunStage` and a `Step` whose backend is parked on `WaitForCancellation` with no polling at all, exception-safe restoration of the classifier-free guidance scratch state when the unconditional pass is cancelled, `Model::Run`'s cancellable overload, and no device materialization from cancelling. |
| `pipeline_scheduler_test.cpp` | `pipeline_scheduler_test` | `PipelineSchedulingOptions` through the public API only: the unlimited default admitting three sessions at once, a global ceiling of two that is reached but never exceeded and drains work-conservingly, a per-kind ceiling of one that serializes `single_pass` while an uncapped `state_transition` still enters, a queued execution cancelled or stopped by its deadline without reaching the backend and without leaving its queue position behind, a queued `StageRun::Step` cancelled and a queued `StageRun::Finish` stopped by its deadline each closing the handle and releasing the session's active-run slot so the same session immediately accepts a `Snapshot`, a fresh `BeginStage`, and a `RunStage`, a stale handle's `Cancel` not releasing a newer run's slot, a permit returned after a backend failure, `BeginStage` and an idle handle holding nothing while `Step`, `Finish`, and `StepStage` each hold one, a completed `Finish` returning its cached result without admission, the `PipelineSchedulingStats` reading -- all six stage kinds always present, zeros for an unlimited pipeline, the admitted and queued counts under a ceiling, and a drain back to zero -- which is also what every queued-request claim in the file synchronizes on, a copied and a forked session sharing the ceiling while two separately constructed `Pipeline`s do not, unknown, empty, and wrong-case stage-kind keys rejected with `invalid_argument`, and a limit of zero meaning unlimited. |

Each file is a self-contained `main()` with local `Check` and `CheckThrows`
helpers (and `CheckThrowsMessage` where a message is asserted, or
`CheckThrowsCode`, `CheckThrowsState`, and `CheckThrowsInvalidArgument` where a
specific `ErrorCode` is asserted), a file-local
`failures` counter, and a non-zero exit code on failure. There is no
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
  access, and counts each `CopyToCpu`; `pipeline_snapshot_test.cpp`,
  `pipeline_stream_test.cpp`, and `pipeline_cancellation_test.cpp` have their
  own `CountingDeviceBuffer` with the same shape.
- Concurrency is expressed with condition variables, never with sleeps:
  `pipeline_stream_test.cpp` uses `BlockingControl` with
  `BlockingIncrementBackend`, `pipeline_cancellation_test.cpp` uses its
  own `BlockingControl` with `BlockingCancellableBackend`, a stub that
  overrides the cancellable `Run`, signals that it entered, waits to be
  released, and only then checks its token, and
  `pipeline_scheduler_test.cpp` uses one shared `Gate` with `GatedBackend`,
  which records the concurrency peak per component before parking. It never
  infers "this request is still queued" from a wait that timed out: it polls
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
- `cancellation_test.cpp`, `pipeline_device_test.cpp`,
  `pipeline_snapshot_test.cpp`, `pipeline_stream_test.cpp`,
  `pipeline_cancellation_test.cpp`, and `pipeline_scheduler_test.cpp` take no
  arguments and touch no filesystem: they build each `PipelinePackage` in
  memory from an embedded manifest string.

## Tests

These files are the tests. Build and run them with:

```console
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

The `dev` preset sets `ONNX_WORLD_MODEL_BUILD_TESTS=ON`; the default build does
not compile this directory.
