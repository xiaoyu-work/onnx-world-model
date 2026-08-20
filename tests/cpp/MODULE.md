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

## Key Files

| File | CTest name | Coverage |
|---|---|---|
| `tensor_test.cpp` | `tensor_test` | `Tensor` construction, typed views, overflow checks, copy-on-write. |
| `model_test.cpp` | `model_test` | `Model` input and output validation through the `AddOneBackend` stub. |
| `pipeline_test.cpp` | `pipeline_test` | Manifest parsing and rejection, stage execution, guidance, schedulers, state lifecycle. |

Each file is a self-contained `main()` with local `Check`, `CheckThrows`, and
`CheckThrowsMessage` helpers, a file-local `failures` counter, and a non-zero
exit code on failure. There is no third-party test framework; a new test is a
new check inside an existing `main()`, or a new executable added to the
`foreach(test_name IN ITEMS ...)` list in the root `CMakeLists.txt`.

## Dependencies

- Links `onnx_world_model` and includes only public headers from
  `include/onnx_world_model`; it never includes `src/` internal headers.
- Components under test are stub implementations of
  `onnx_world_model::ModelBackend` defined inside each test file.
- `pipeline_test.cpp` writes temporary package and scheduler directories under
  the filesystem temporary directory, and optionally accepts a real package
  directory and ORT library path as `argv[1]` and `argv[2]`.

## Tests

These files are the tests. Build and run them with:

```console
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

The `dev` preset sets `ONNX_WORLD_MODEL_BUILD_TESTS=ON`; the default build does
not compile this directory.
