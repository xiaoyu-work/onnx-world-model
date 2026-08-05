from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ModelMetadata:
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]


@dataclass(frozen=True)
class StepResult:
    next_state: NDArray[Any]
    observation_prediction: NDArray[Any]
    reward: NDArray[Any]
    continuation: NDArray[Any]


def _metadata_from_native(value: dict[str, Any]) -> ModelMetadata:
    def convert(spec: dict[str, Any]) -> TensorSpec:
        return TensorSpec(
            name=spec["name"],
            dtype=spec["dtype"],
            shape=tuple(spec["shape"]),
        )

    return ModelMetadata(
        inputs=tuple(convert(spec) for spec in value["inputs"]),
        outputs=tuple(convert(spec) for spec in value["outputs"]),
    )


def _step_result(values: tuple[NDArray[Any], ...]) -> StepResult:
    return StepResult(
        next_state=values[0],
        observation_prediction=values[1],
        reward=values[2],
        continuation=values[3],
    )


def _find_ort_library() -> Path:
    override = os.environ.get("ONNX_RUNTIME_LIBRARY_PATH")
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX_RUNTIME_LIBRARY_PATH does not exist: {path}")
        return path

    try:
        import onnxruntime
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is required unless ONNX_RUNTIME_LIBRARY_PATH is set"
        ) from error

    package_dir = Path(onnxruntime.__file__).resolve().parent
    patterns = (
        "onnxruntime.dll",
        "libonnxruntime.so",
        "libonnxruntime.so.*",
        "libonnxruntime.dylib",
    )
    for pattern in patterns:
        candidates = sorted(package_dir.rglob(pattern))
        if candidates:
            return candidates[0]
    raise RuntimeError(
        "The installed onnxruntime package does not contain a loadable C API "
        "library. Set ONNX_RUNTIME_LIBRARY_PATH to an ONNX Runtime shared library."
    )


class WorldModel:
    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        ort_library_path: str | os.PathLike[str] | None = None,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
        log_severity: int = 3,
    ) -> None:
        library_path = (
            Path(ort_library_path) if ort_library_path is not None else _find_ort_library()
        )
        self._core = _native.WorldModel(
            os.fspath(model_path),
            os.fspath(library_path),
            intra_op_threads,
            inter_op_threads,
            log_severity,
        )
        self._metadata = _metadata_from_native(self._core.metadata)

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def step(
        self,
        observation: ArrayLike,
        action: ArrayLike,
        state: ArrayLike,
    ) -> StepResult:
        values = self._core.step(
            np.asarray(observation),
            np.asarray(action),
            np.asarray(state),
        )
        return _step_result(values)

    def create_rollout(self) -> Rollout:
        return Rollout(self._core.create_rollout())


class Rollout:
    def __init__(self, core: _native.Rollout) -> None:
        self._core = core

    @property
    def state(self) -> NDArray[Any] | None:
        return self._core.state

    def reset(
        self,
        state: ArrayLike | None = None,
        *,
        batch_size: int | None = None,
    ) -> None:
        if state is not None and batch_size is not None:
            raise ValueError("Provide state or batch_size, not both")
        if state is not None:
            self._core.reset_state(np.asarray(state))
        elif batch_size is not None:
            self._core.reset_zeros(batch_size)
        else:
            self._core.reset()

    def step(self, observation: ArrayLike, action: ArrayLike) -> StepResult:
        values = self._core.step(
            np.asarray(observation),
            np.asarray(action),
        )
        return _step_result(values)
