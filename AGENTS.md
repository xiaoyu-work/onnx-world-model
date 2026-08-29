# Agent Instructions

## Repository Documents

Read these sources in order:

1. [`ARCHITECTURE.md`](ARCHITECTURE.md) for system purpose, components,
   dependency rules, data flow, entry points, and cross-cutting constraints.
2. [`.agent/repo-map.json`](.agent/repo-map.json) for machine-readable module and
   file navigation. It is generated; never edit it by hand.
3. The nearest `MODULE.md` for directory-specific responsibilities, key files,
   dependencies, and tests:
   - [`include/onnx_world_model/MODULE.md`](include/onnx_world_model/MODULE.md)
     — installed public C++ API.
   - [`src/MODULE.md`](src/MODULE.md) — C++ implementation.
   - [`python/onnx_world_model/MODULE.md`](python/onnx_world_model/MODULE.md)
     — installed Python package.
   - [`tests/cpp/MODULE.md`](tests/cpp/MODULE.md) — CTest executables.
   - [`tests/python/MODULE.md`](tests/python/MODULE.md) — pytest suite.
4. Candidate source-file `@agent-*` headers before reading full files.

User-facing documentation lives in [`README.md`](README.md) and
[`docs/`](docs/): [Pipeline API](docs/pipeline-api.md),
[Low-level APIs](docs/low-level-apis.md), and
[Cosmos3 Edge validation](docs/cosmos3-edge-validation.md).

`bindings/`, `tools/`, `cmake/`, and `notebooks/` are small leaf directories
without their own guide; they inherit this file and `ARCHITECTURE.md`.

## Development

Prerequisites:

- CMake 3.24 or newer and a C++20 compiler. On Windows the repository is built
  with the Visual Studio 17 2022 generator; `cmake.exe` ships with the Visual
  Studio installation under
  `Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin`.
- Python 3.10 or newer. Environments are created with `uv`.
- Internet access during the first CMake configure, or local header paths.

Run from the repository root:

```console
uv venv --python 3.12 .venv
uv pip install -e ".[test]"
cmake --preset dev
cmake --build --preset dev
```

The `dev` preset builds `RelWithDebInfo` into `build/dev` with
`ONNX_WORLD_MODEL_BUILD_TESTS=ON`. `uv pip install` drives scikit-build-core,
which builds the `_native` extension separately from the `dev` preset. After
changing any C++ source, rebuild the extension before running Python tests:

```console
uv pip install -e ".[test]" --reinstall-package onnx-world-model
```

Environment variables:

- `ONNX_RUNTIME_LIBRARY_PATH`: absolute path to an ONNX Runtime shared library.
  Only needed when the installed `onnxruntime` wheel does not ship one.
- `ONNXRUNTIME_INCLUDE_DIR`, `NLOHMANN_JSON_INCLUDE_DIR`: CMake cache paths for
  offline builds that skip the verified header download.

## Testing

Use the narrowest existing command that covers the change. Expand validation
only when the result or change scope requires it.

```console
.venv\Scripts\python.exe -m pytest tests/python/test_preprocessing.py -q
.venv\Scripts\python.exe -m pytest tests/python -q
uvx ruff check python tools tests
ctest --preset dev
ctest --preset dev -R pipeline_test
```

Expected baseline: `ctest` reports 10 of 10 passing; `pytest` reports 203
passed and 20 skipped. The skips are tests whose fixtures require the optional
`mobius` exporter. `tests/python/test_guided_generation.py` needs the optional
`onnx_ir` package, which also gates the package fixtures in
`tests/python/test_pipeline_snapshot.py`,
`tests/python/test_pipeline_stream.py`,
`tests/python/test_cancellation.py`,
`tests/python/test_scheduling.py`,
`tests/python/test_placement.py`, and
`tests/python/test_telemetry.py`.

`pipeline_scheduler_test` and `pipeline_telemetry_test` are the concurrent
CTest executables. Run them repeatedly after changing
`src/pipeline_scheduler.cpp`, `src/pipeline_telemetry.cpp`,
`src/cancellation.cpp`, or any call site that takes an admission lease or
records telemetry:

```console
ctest --preset dev -R pipeline_scheduler_test --repeat until-fail:20
ctest --preset dev -R pipeline_telemetry_test --repeat until-fail:20
```

Run the placement tests repeatedly after changing component placement, the
per-port device metadata, or the transfer plan:

```console
ctest --preset dev -R pipeline_device_test
.venv\Scripts\python.exe -m pytest tests/python/test_placement.py -q
```

C++ changes require `cmake --build --preset dev` before `ctest`, and a
`--reinstall-package` rebuild before `pytest`.

## Conventions

- Unknown or unsupported semantics must fail loudly. Do not add a silent
  fallback for an unsupported capability, manifest field, stage kind, or
  execution provider.
- All C++ failures throw `onnx_world_model::Error` with an `ErrorCode`; do not
  introduce a second exception type. The Python binding maps that code onto
  `CancelledError`, `DeadlineExceededError`, or their base `WorldModelError`,
  so a new outcome is a new `ErrorCode`, never a message-text check.
- Cancellation is cooperative: poll `PipelineRunOptions::cancellation` at
  operation boundaries, never inside per-element loops, and never roll back
  what a cancelled call already applied. Deadlines are claimed by the one
  process-wide watchdog in `src/cancellation.cpp`; do not add a thread or a
  timer per request, and do not make work block on a token without going
  through `CancellationToken::WaitForCancellation`.
- Keep ONNX Runtime headers confined to `src/dynamic_library.cpp` and
  `src/ort_backend.cpp`, and keep them out of `include/onnx_world_model/`.
- Concurrency limits are admission scheduling only. Do not describe them as
  batching, and do not add continuous or dynamic batching, request merging or
  splitting, priority, or preemption under this name. Preserve the one-way
  lock order — `CancellationState` callback mutex, then scheduler mutex, then
  session mutex, then ONNX Runtime — by taking every admission lease before
  the session lock and never creating or destroying a cancellation
  registration while the scheduler mutex is held.
- Internal C++ helpers belong in `onnx_world_model::detail` in a non-installed
  `src/*.hpp`.
- Python targets 3.10, uses `from __future__ import annotations`, full type
  annotations, and frozen dataclasses for value types. Ruff is configured in
  `pyproject.toml` with `line-length = 95`.
- Keep `__all__` sorted and in sync in `python/onnx_world_model/__init__.py`.
- Do not change the installed public API, the `pipeline.json` contract, or the
  `WorldModelPipeline` and `LegacyWorldModel` aliases unless the task requires
  it. `include/onnx_world_model/` is installed, so a declaration change is an
  ABI change.
- Generated artifacts: `.agent/repo-map.json` is produced by the readability
  auditor, and `python/onnx_world_model/_native*.pyd` is produced by
  scikit-build-core. Never edit either by hand.
- `build/`, `dist/`, and `.venv/` are build outputs and are git-ignored.
- Source code is authoritative when documentation and implementation differ.
- Do not add dependencies or change public behavior unless the task requires it.

## Agent Workflow

Before editing:

1. Read this file, `ARCHITECTURE.md`, and `.agent/repo-map.json`.
2. Read the nearest applicable `MODULE.md` for every file in scope.
3. Read only the first 50 lines of candidate files and use their `@agent-*`
   headers to decide which files require full inspection.
4. Confirm relevant callers, dependencies, tests, and invariants in source.
5. Preserve unrelated user changes.

While editing:

1. Keep changes scoped to the requested behavior.
2. Follow the dependency rules in `ARCHITECTURE.md` and existing module
   boundaries.
3. Reuse repository helpers and preserve type safety.
4. Run or update the smallest relevant existing tests when behavior changes. A
   new manifest rule belongs in `tests/cpp/pipeline_test.cpp`.
5. Treat documentation as navigation, not a substitute for source inspection.

After editing:

1. Apply the documentation triggers below.
2. Regenerate the repository map after source, source-header, or module-guide
   changes:

   ```console
   node check-agent-readability.mjs . --generate-map
   ```

3. Run the readability auditor and the relevant project validation:

   ```console
   node check-agent-readability.mjs .
   ```

## Documentation Updates

| Change | Required update |
|---|---|
| Source-file purpose, API, invariant, or side effect changes | Update that file's `@agent-*` header, then regenerate the repository map. |
| Public or exported symbol changes | Update `@agent-public-api`, then regenerate the repository map. |
| File is added, deleted, renamed, moved, or split | Update the nearest applicable `MODULE.md` and, for C++ sources, the `add_library` list in `CMakeLists.txt`; then regenerate the repository map. |
| Module responsibility, dependencies, key files, or tests change | Update that module's `MODULE.md`. Update `ARCHITECTURE.md` only when the system-level design also changes. |
| Components, dependency rules, data flow, entry points, or cross-cutting constraints change | Update `ARCHITECTURE.md`. |
| Development commands, test commands, conventions, or agent workflow change | Update this `AGENTS.md`. |
| User-visible API, package contract, or supported scope changes | Update `README.md` and the affected page under `docs/`. |
| A file exceeds 2,000 lines and cannot be split safely | Add a narrow, path-scoped entry to `.agent-readability.json`. |

Documentation updates are part of the code change. Do not leave stale
architecture, paths, symbols, responsibilities, or commands for a later agent.
Never edit `.agent/repo-map.json` manually.
