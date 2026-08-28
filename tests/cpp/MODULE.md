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

## Key Files

| File | CTest name | Coverage |
|---|---|---|
| `tensor_test.cpp` | `tensor_test` | `Tensor` construction, typed views, overflow checks, copy-on-write. |
| `model_test.cpp` | `model_test` | `Model` input and output validation through the `AddOneBackend` stub. |
| `pipeline_test.cpp` | `pipeline_test` | Manifest parsing and rejection, stage execution, guidance, schedulers, state lifecycle. |
| `pipeline_device_test.cpp` | `pipeline_device_test` | Device-tensor preservation across `PipelineSession`: rank adaptation, transform-free connections, reshape, public outputs, and CPU-transform materialization. |
| `pipeline_snapshot_test.cpp` | `pipeline_snapshot_test` | `PipelineSession::Snapshot`, `Restore`, `Fork`, and the named `Checkpoint`, `RestoreCheckpoint`, `DropCheckpoint`, and `HasCheckpoint`: state round trips, fork independence, package identity rejection, zero-copy device sharing, random-engine determinism, and checkpoint create/replace/rewind/drop plus empty and unknown name failures. |
| `pipeline_stream_test.cpp` | `pipeline_stream_test` | `PipelineSession::BeginStage` and `StageRun`: greedy, sampled, and per-lane early-stopping autoregressive parity with `RunStage`, the once-only token budget, iterative and single-pass parity, one terminal `StageEvent`, `Finish` after partial stepping, the active-run exclusions, cancellation and destruction, failure without rollback, moved handles, session move and destruction safety, and device-preserving event outputs. |

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
  access, and counts each `CopyToCpu`; `pipeline_snapshot_test.cpp` and
  `pipeline_stream_test.cpp` have their own `CountingDeviceBuffer` with the
  same shape.
- `pipeline_test.cpp` writes temporary package and scheduler directories under
  the filesystem temporary directory, and optionally accepts a real package
  directory and ORT library path as `argv[1]` and `argv[2]`.
- `pipeline_device_test.cpp`, `pipeline_snapshot_test.cpp`, and
  `pipeline_stream_test.cpp` take no arguments and touch no filesystem: they
  build each `PipelinePackage` in memory from an embedded manifest string.

## Tests

These files are the tests. Build and run them with:

```console
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

The `dev` preset sets `ONNX_WORLD_MODEL_BUILD_TESTS=ON`; the default build does
not compile this directory.
