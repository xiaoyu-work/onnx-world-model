# @agent-file
# @agent-purpose: Wraps the `_native` extension in typed Python classes: it locates the ONNX Runtime library, maps manifest JSON to input, output, and stage specs, and exposes the generic model, pipeline, and latent-dynamics APIs.
# @agent-public-api: TensorSpec, ModelMetadata, PipelineInputSpec, PipelineOutputSpec, PipelineStageSpec, StepResult, ProviderOptionValue, ProviderOptions, available_execution_providers, register_execution_provider_library, supported_pipeline_capabilities, OnnxModel, Pipeline, WorldModelPipeline, PipelineSession, PipelineSessionSnapshot, LatentDynamicsModel, LegacyWorldModel, Rollout
# @agent-invariants: `ONNX_RUNTIME_LIBRARY_PATH` overrides library discovery and must point at an existing file; otherwise the library is found inside the installed `onnxruntime` wheel. Device outputs are opt-in and require the matching EP library to be registered first. All spec dataclasses are frozen. `WorldModelPipeline` and `LegacyWorldModel` are compatibility aliases that must keep pointing at `Pipeline` and `LatentDynamicsModel`. `PipelineSession.run` preserves manifest stage order, rejects duplicate or unknown stage names, and releases each stage after running it. A `PipelineSessionSnapshot` is only produced by `PipelineSession.snapshot`; native package identity is the sole authority for restore compatibility.
# @agent-side-effects: Reads `pipeline.json` from the package directory, loads ONNX Runtime and explicitly registered EP libraries, reads the `ONNX_RUNTIME_LIBRARY_PATH` environment variable, and preloads pip-installed CUDA libraries with `ctypes.CDLL` into the global namespace.

from __future__ import annotations

import ctypes
import json
import os
import sysconfig
from collections.abc import Mapping, Sequence
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
    execution_providers: tuple[str, ...]


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
        execution_providers=tuple(value["execution_providers"]),
    )


def _step_result(values: tuple[NDArray[Any], ...]) -> StepResult:
    return StepResult(
        next_state=values[0],
        observation_prediction=values[1],
        reward=values[2],
        continuation=values[3],
    )


_CUDA_LIBRARY_ORDER = (
    "libcudart.so",
    "libnvJitLink.so",
    "libcublasLt.so",
    "libcublas.so",
    "libcufft.so",
    "libcurand.so",
    "libcudnn.so",
)


def _preload_cuda_dependencies(package_dir: Path) -> None:
    """Resolve the CUDA libraries onnxruntime-gpu links by soname alone.

    A pip CUDA installation puts them under ``nvidia/*/lib`` rather than on the
    loader path, so opening the CUDA provider fails unless they are already
    mapped into the process. Missing pieces are ignored: the provider reports a
    clearer error than a preload failure would.
    """
    if not (package_dir / "libonnxruntime_providers_cuda.so").is_file():
        return
    roots = [
        Path(sysconfig.get_paths()["purelib"]) / "nvidia",
        Path(sysconfig.get_paths()["platlib"]) / "nvidia",
    ]
    directories = [
        directory
        for root in dict.fromkeys(roots)
        if root.is_dir()
        for directory in sorted(root.glob("*/lib"))
    ]
    if not directories:
        return
    for soname in _CUDA_LIBRARY_ORDER:
        for directory in directories:
            for candidate in sorted(directory.glob(f"{soname}*")):
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue
                break


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
            _preload_cuda_dependencies(candidates[0].parent)
            return candidates[0]
    raise RuntimeError(
        "The installed onnxruntime package does not contain a loadable C API "
        "library. Set ONNX_RUNTIME_LIBRARY_PATH to an ONNX Runtime shared library."
    )


ProviderOptionValue = str | bool | int | float
ProviderOptions = Mapping[str, Mapping[str, ProviderOptionValue]]


def _provider_arguments(
    providers: Sequence[str] | None,
    provider_options: ProviderOptions | None,
) -> tuple[list[str], dict[str, dict[str, ProviderOptionValue]]]:
    return (
        list(providers or ()),
        {
            provider: dict(options)
            for provider, options in (provider_options or {}).items()
        },
    )


def available_execution_providers(
    *,
    ort_library_path: str | os.PathLike[str] | None = None,
) -> tuple[str, ...]:
    library_path = (
        Path(ort_library_path) if ort_library_path is not None else _find_ort_library()
    )
    return tuple(_native.available_execution_providers(os.fspath(library_path)))


def register_execution_provider_library(
    registration_name: str,
    provider_library_path: str | os.PathLike[str],
    *,
    ort_library_path: str | os.PathLike[str] | None = None,
) -> None:
    """Register an EP library for device allocation and tensor transfers."""
    library_path = (
        Path(ort_library_path) if ort_library_path is not None else _find_ort_library()
    )
    _native.register_execution_provider_library(
        registration_name,
        os.fspath(provider_library_path),
        os.fspath(library_path),
    )


def supported_pipeline_capabilities() -> tuple[str, ...]:
    """Pipeline capability names this runtime implements.

    A package whose ``required_capabilities`` names anything outside this set
    is rejected while loading instead of failing later, or silently doing
    nothing.
    """
    return tuple(_native.supported_pipeline_capabilities())


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
        graph_optimization: str = "all",
        device_outputs: bool = False,
        providers: Sequence[str] | None = None,
        provider_options: ProviderOptions | None = None,
    ) -> None:
        library_path = (
            Path(ort_library_path) if ort_library_path is not None else _find_ort_library()
        )
        provider_names, options = _provider_arguments(providers, provider_options)
        self._core = _native.Model(
            os.fspath(model_path),
            os.fspath(library_path),
            intra_op_threads,
            inter_op_threads,
            log_severity,
            graph_optimization,
            device_outputs,
            provider_names,
            options,
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
        graph_optimization: str = "all",
        device_outputs: bool = False,
        providers: Sequence[str] | None = None,
        provider_options: ProviderOptions | None = None,
    ) -> None:
        package = Path(package_path)
        with (package / "pipeline.json").open(encoding="utf-8") as file:
            self._document: dict[str, Any] = json.load(file)
        library_path = (
            Path(ort_library_path) if ort_library_path is not None else _find_ort_library()
        )
        provider_names, options = _provider_arguments(providers, provider_options)
        self._core = _native.Pipeline(
            os.fspath(package),
            os.fspath(library_path),
            intra_op_threads,
            inter_op_threads,
            log_severity,
            graph_optimization,
            device_outputs,
            provider_names,
            options,
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

    @property
    def execution_providers(self) -> Mapping[str, tuple[str, ...]]:
        return {
            component: tuple(providers)
            for component, providers in self._core.execution_providers.items()
        }

    def create_session(self) -> PipelineSession:
        return PipelineSession(self, self._core.create_session())


WorldModelPipeline = Pipeline


class PipelineSessionSnapshot:
    """An immutable in-memory capture of one :class:`PipelineSession`'s state.

    Only :meth:`PipelineSession.snapshot` produces one. The native snapshot
    retains its originating package and rejects restore into an unrelated one.
    """

    def __init__(self, core: _native.PipelineSessionSnapshot) -> None:
        self._core = core

    @property
    def valid(self) -> bool:
        return bool(self._core.valid)


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

    def snapshot(self) -> PipelineSessionSnapshot:
        """Capture every mutable execution field of this session."""
        return PipelineSessionSnapshot(self._core.snapshot())

    def restore(self, snapshot: PipelineSessionSnapshot) -> None:
        """Replace every mutable execution field with the captured one."""
        self._core.restore(snapshot._core)

    def fork(self) -> PipelineSession:
        """Return an independent session started from a snapshot of this one."""
        return PipelineSession(self._pipeline, self._core.fork())


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


class LatentDynamicsModel:
    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        ort_library_path: str | os.PathLike[str] | None = None,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
        log_severity: int = 3,
        graph_optimization: str = "all",
        device_outputs: bool = False,
        providers: Sequence[str] | None = None,
        provider_options: ProviderOptions | None = None,
    ) -> None:
        library_path = (
            Path(ort_library_path) if ort_library_path is not None else _find_ort_library()
        )
        provider_names, options = _provider_arguments(providers, provider_options)
        self._core = _native.WorldModel(
            os.fspath(model_path),
            os.fspath(library_path),
            intra_op_threads,
            inter_op_threads,
            log_severity,
            graph_optimization,
            device_outputs,
            provider_names,
            options,
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


LegacyWorldModel = LatentDynamicsModel


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
