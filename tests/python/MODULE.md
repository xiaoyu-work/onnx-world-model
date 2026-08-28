# tests/python

## Purpose

Holds the pytest suite that exercises the installed `onnx_world_model` Python
package, including the compiled `_native` extension, through its public API.

## Responsibilities

- Verify the typed wrappers in `_api.py`: metadata, provider selection and
  registration rejection, device-output forwarding, named-tensor validation,
  and latent-dynamics stepping.
- Verify pipeline loading and staged execution through `Pipeline` and
  `PipelineSession`.
- Verify the preprocessing and media layers against explicit reference
  implementations of the runtime tensor layouts.
- Verify classifier-free guidance and the image-to-video entry point.

## Key Files

| File | Coverage |
|---|---|
| `conftest.py` | Shared fixtures that build throwaway packages, tokenizers, and chat templates. |
| `test_api.py` | `_api.py` wrappers, providers, `LatentDynamicsModel`, `Rollout`. |
| `test_pipeline.py` | `Pipeline` and `PipelineSession` contract and stage execution. |
| `test_preprocessing.py` | `preprocessing.py` and `media.py`, including latent-token round trips. |
| `test_guided_generation.py` | Classifier-free guidance on graphs synthesized with `onnx_ir`. |
| `test_image_to_video_smoke.py` | The `tools/image_to_video_smoke.py` dry run and generation path. |

## Dependencies

- The installed `onnx_world_model` package and its compiled `_native` module;
  these tests import the package rather than the source tree.
- `pyproject.toml` sets `pythonpath = ["."]` and `testpaths =
  ["tests/python"]`, so `tools/` is importable from a test.
- Optional dependencies gate parts of the suite:
  - `mobius` — `world_model_path` and `pipeline_path` call
    `pytest.importorskip`, so `test_api.py` and `test_pipeline.py` skip
    without it;
  - `onnx_ir` — required by `test_guided_generation.py`;
  - an exported image-to-video package — guards the generation test in
    `test_image_to_video_smoke.py` with `skipif`.

Fixtures that build a package in `tmp_path` need no exporter and no network, so
`test_preprocessing.py` and `test_guided_generation.py` always run.

## Tests

```console
.venv\Scripts\python.exe -m pytest tests/python -q
.venv\Scripts\python.exe -m pytest tests/python/test_preprocessing.py -q
```

Reinstall the package after changing C++ sources so the tests exercise the
rebuilt extension:

```console
uv pip install -e ".[test]" --reinstall-package onnx-world-model
```
