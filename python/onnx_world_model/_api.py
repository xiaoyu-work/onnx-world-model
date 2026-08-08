from __future__ import annotations

import json
import os
from collections.abc import Mapping
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
class PipelineInputSpec:
    name: str
    port: str
    semantic: str | None
    required: bool
    presence: str | None
    dtype: str
    shape: tuple[int | str | None, ...]


@dataclass(frozen=True)
class PipelineOutputSpec:
    name: str
    port: str | None
    state: str | None
    dtype: str
    shape: tuple[int | str | None, ...]


@dataclass(frozen=True)
class PipelineStageSpec:
    name: str
    kind: str
    components: tuple[str, ...]
    run_on: str
    options: Mapping[str, Any]


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


class OnnxModel:
    """A generic named-tensor ONNX model session."""

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
        self._core = _native.Model(
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

    def run(
        self,
        inputs: Mapping[str, ArrayLike],
    ) -> dict[str, NDArray[Any]]:
        return self._core.run({name: np.asarray(value) for name, value in inputs.items()})


class Pipeline:
    """A validated Mobius pipeline package with shareable model sessions."""

    def __init__(
        self,
        package_path: str | os.PathLike[str],
        *,
        ort_library_path: str | os.PathLike[str] | None = None,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
        log_severity: int = 3,
    ) -> None:
        package = Path(package_path)
        with (package / "pipeline.json").open(encoding="utf-8") as file:
            self._document: dict[str, Any] = json.load(file)
        library_path = (
            Path(ort_library_path) if ort_library_path is not None else _find_ort_library()
        )
        self._core = _native.Pipeline(
            os.fspath(package),
            os.fspath(library_path),
            intra_op_threads,
            inter_op_threads,
            log_severity,
        )
        self._manifest: dict[str, Any] = self._document["manifest"]
        components = {
            component["name"]: component for component in self._manifest["components"]
        }

        self._inputs = tuple(
            PipelineInputSpec(
                name=entry.get("alias", entry["port"].split(".", 1)[1]),
                port=entry["port"],
                semantic=entry.get("semantic"),
                required=entry.get("required", True),
                presence=entry.get("presence"),
                dtype=_port_spec(components, entry["port"], "inputs")["dtype"],
                shape=tuple(_port_spec(components, entry["port"], "inputs")["shape"]),
            )
            for entry in self._manifest["inputs"]
            if entry["kind"] == "external"
        )
        states = {state["name"]: state for state in self._manifest.get("states", [])}
        output_specs: list[PipelineOutputSpec] = []
        for entry in self._manifest["outputs"]:
            port = entry.get("port")
            state_name = entry.get("state")
            if port is not None:
                spec = _port_spec(components, port, "outputs")
                default_name = port.split(".", 1)[1]
            else:
                state = states[state_name]
                spec = _port_spec(components, state["input"], "inputs")
                default_name = state_name
            output_specs.append(
                PipelineOutputSpec(
                    name=entry.get("alias", default_name),
                    port=port,
                    state=state_name,
                    dtype=spec["dtype"],
                    shape=tuple(spec["shape"]),
                )
            )
        self._outputs = tuple(output_specs)
        self._stages = tuple(
            PipelineStageSpec(
                name=stage["name"],
                kind=stage["kind"],
                components=tuple(stage["components"]),
                run_on=stage["run_on"],
                options=stage.get("options", {}),
            )
            for stage in self._manifest["stages"]
        )

    @property
    def profile(self) -> Mapping[str, Any] | None:
        return self._manifest.get("profile")

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._manifest.get("metadata", {})

    @property
    def inputs(self) -> tuple[PipelineInputSpec, ...]:
        return self._inputs

    @property
    def outputs(self) -> tuple[PipelineOutputSpec, ...]:
        return self._outputs

    @property
    def stages(self) -> tuple[PipelineStageSpec, ...]:
        return self._stages

    @property
    def required_capabilities(self) -> tuple[str, ...]:
        return tuple(self._manifest.get("required_capabilities", ()))

    def create_session(self) -> PipelineSession:
        return PipelineSession(self, self._core.create_session())


WorldModelPipeline = Pipeline


class PipelineSession:
    """Mutable per-request and per-trajectory state for a :class:`Pipeline`."""

    def __init__(self, pipeline: Pipeline, core: _native.PipelineSession) -> None:
        self._pipeline = pipeline
        self._core = core

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def outputs(self) -> dict[str, NDArray[Any]]:
        return self._core.outputs

    def state(self, name: str) -> NDArray[Any] | None:
        return self._core.state(name)

    def run_stage(
        self,
        stage: str,
        inputs: Mapping[str, ArrayLike] | None = None,
        *,
        overrides: Mapping[str, ArrayLike] | None = None,
        options: Mapping[str, bool | int | float | str] | None = None,
    ) -> dict[str, NDArray[Any]]:
        return self._core.run_stage(
            stage,
            _array_mapping(inputs),
            _array_mapping(overrides),
            dict(options or {}),
        )

    def step_stage(
        self,
        stage: str,
        inputs: Mapping[str, ArrayLike] | None = None,
        *,
        overrides: Mapping[str, ArrayLike] | None = None,
        options: Mapping[str, bool | int | float | str] | None = None,
    ) -> dict[str, NDArray[Any]]:
        return self._core.step_stage(
            stage,
            _array_mapping(inputs),
            _array_mapping(overrides),
            dict(options or {}),
        )

    def run(
        self,
        inputs: Mapping[str, ArrayLike],
        *,
        stages: tuple[str, ...] | list[str] | None = None,
        overrides: Mapping[str, ArrayLike] | None = None,
        options: Mapping[str, bool | int | float | str] | None = None,
    ) -> dict[str, NDArray[Any]]:
        if stages is None:
            selected = tuple(stage.name for stage in self._pipeline.stages)
        else:
            requested = tuple(stages)
            if len(requested) != len(set(requested)):
                raise ValueError("Pipeline stages must not be duplicated")
            known = {stage.name for stage in self._pipeline.stages}
            unknown = sorted(set(requested) - known)
            if unknown:
                raise ValueError(f"Unknown pipeline stages: {unknown}")
            selected = tuple(
                stage.name
                for stage in self._pipeline.stages
                if stage.name in set(requested)
            )
        result: dict[str, NDArray[Any]] = {}
        stage_inputs: Mapping[str, ArrayLike] | None = inputs
        for stage in selected:
            result.update(
                self.run_stage(
                    stage,
                    stage_inputs,
                    overrides=overrides,
                    options=options,
                )
            )
            self.release_stage(stage)
            stage_inputs = None
        return result

    def release_stage(self, stage: str) -> None:
        self._core.release_stage(stage)

    def reset(self) -> None:
        self._core.reset()


def _array_mapping(
    values: Mapping[str, ArrayLike] | None,
) -> dict[str, NDArray[Any]]:
    return {name: np.asarray(value) for name, value in (values or {}).items()}


def _port_spec(
    components: Mapping[str, Mapping[str, Any]],
    endpoint: str,
    direction: str,
) -> Mapping[str, Any]:
    component_name, port_name = endpoint.split(".", 1)
    return next(
        spec
        for spec in components[component_name][direction]
        if spec["name"] == port_name
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
