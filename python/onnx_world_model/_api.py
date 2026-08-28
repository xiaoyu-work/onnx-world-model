# @agent-file
# @agent-purpose: Wraps the `_native` extension in typed Python classes: it locates the ONNX Runtime library, maps manifest JSON to input, output, and stage specs, and exposes the generic model, pipeline, incremental stage run, cancellation, and latent-dynamics APIs.
# @agent-public-api: TensorSpec, ModelMetadata, PipelineInputSpec, PipelineOutputSpec, PipelineStageSpec, PipelineSchedulingStats, StepResult, StageEventKind, StageEvent, StageRun, CancellationReasonName, CancellationToken, CancellationSource, ProviderOptionValue, ProviderOptions, available_execution_providers, register_execution_provider_library, supported_pipeline_capabilities, OnnxModel, Pipeline, WorldModelPipeline, PipelineSession, PipelineSessionSnapshot, LatentDynamicsModel, LegacyWorldModel, Rollout
# @agent-invariants: `ONNX_RUNTIME_LIBRARY_PATH` overrides library discovery and must point at an existing file; otherwise the library is found inside the installed `onnxruntime` wheel. Device outputs are opt-in and require the matching EP library to be registered first. All spec dataclasses are frozen. `WorldModelPipeline` and `LegacyWorldModel` are compatibility aliases that must keep pointing at `Pipeline` and `LatentDynamicsModel`. `Pipeline`'s `max_concurrent_executions` and `max_concurrent_by_stage_kind` are admission scheduling only -- never batching -- and are forwarded to the native constructor unchanged, which is the sole authority on which stage-kind names are legal; the two read-only properties echo the accepted values and the per-kind mapping is exposed as an immutable view. `Pipeline.scheduling_stats` is the matching observability read: it converts one native dictionary into a frozen `PipelineSchedulingStats` whose two per-kind mappings are read-only views that always carry all six executable stage kinds, it counts permits rather than executions so an unlimited pipeline reports zeros, and it is a detached value that never updates itself. `PipelineSession.run` preserves manifest stage order, rejects duplicate or unknown stage names, and releases each stage after running it. A `PipelineSessionSnapshot` is only produced by `PipelineSession.snapshot`; native package identity is the sole authority for restore compatibility. The named-checkpoint methods forward names to the native session unchanged and hold no Python-side checkpoint state, so empty and unknown names surface as `WorldModelError` from the native layer. A `StageRun` is only produced by `PipelineSession.begin_stage`, holds a strong reference to its session, yields exactly one `StageEvent` with `finished` set and then stops iterating, and closes idempotently through `close`, the context manager, and a best-effort destructor; `iter_stage` starts its run eagerly and closes it in a `finally`, so an early `break` releases the session only when the generator is closed or collected. A `CancellationToken` is only produced by `CancellationSource.token`; `cancellation` and `timeout` are mutually exclusive on every call that accepts them, a `timeout` is validated as a finite non-negative number of seconds, and `PipelineSession.run` builds its timeout source once so one absolute deadline covers the whole stage sequence. `CancellationToken.wait` and `CancellationSource.wait` block without polling and release the GIL, so another Python thread can still cancel; a deadline releases them through the shared native watchdog rather than at the next boundary. `StageRun.request_cancellation` signals work already running and takes no session lock, while `close` waits for the lock and only releases the run slot; neither rolls anything back.
# @agent-side-effects: Reads `pipeline.json` from the package directory, loads ONNX Runtime and explicitly registered EP libraries, reads the `ONNX_RUNTIME_LIBRARY_PATH` environment variable, and preloads pip-installed CUDA libraries with `ctypes.CDLL` into the global namespace.

from __future__ import annotations

import contextlib
import ctypes
import json
import math
import os
import sysconfig
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, final

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
class PipelineSchedulingStats:
    """One instantaneous reading of a pipeline's admission scheduler.

    This is operational observability for admission only. It reports how many
    executions hold a permit and how many are waiting for one; it says nothing
    about batching, throughput, or latency, and it is not a profiler.

    ``active_by_stage_kind`` and ``queued_by_stage_kind`` always contain every
    stage kind the runtime executes -- ``single_pass``, ``autoregressive``,
    ``iterative``, ``state_transition``, ``composite``, and ``on_demand`` -- so
    a kind can be read without testing for its key, and a kind with nothing
    happening maps to ``0``. Both are read-only views.

    The counts are permits, not executions: a stage kind capped by neither the
    global limit nor its own is admitted without a permit, so an unlimited
    pipeline reports zeros however much work is inside it. The whole reading is
    taken at one moment under the scheduler's lock, so its fields agree with
    each other, and any of them may be stale as soon as it is returned.
    """

    active_executions: int
    queued_executions: int
    active_by_stage_kind: Mapping[str, int]
    queued_by_stage_kind: Mapping[str, int]


@dataclass(frozen=True)
class StepResult:
    next_state: NDArray[Any]
    observation_prediction: NDArray[Any]
    reward: NDArray[Any]
    continuation: NDArray[Any]


StageEventKind = Literal["token", "iteration", "transition", "completed"]


@dataclass(frozen=True)
class StageEvent:
    """One step of an incremental stage run.

    ``kind`` is ``"token"`` for autoregressive decoding, ``"iteration"`` for an
    iterative scheduler step, ``"transition"`` for the single pass of every
    other stage kind, and ``"completed"`` for the one terminal event that ends
    a run. ``token_ids`` is present only on a ``"token"`` event, ``finished``
    is true only on the ``"completed"`` event, and ``iteration`` counts step
    events within this run rather than the stage's lifetime cursor.

    ``outputs`` and ``token_ids`` are independent NumPy arrays, so a
    device-resident output is materialized at this boundary exactly as it is
    for :meth:`PipelineSession.run_stage`.
    """

    kind: StageEventKind
    stage: str
    iteration: int
    token_ids: NDArray[Any] | None
    outputs: dict[str, NDArray[Any]]
    finished: bool


CancellationReasonName = Literal["none", "cancelled", "deadline_exceeded"]


@final
class CancellationToken:
    """A copyable observer of one :class:`CancellationSource`.

    Only :meth:`CancellationSource.token` produces one. A token cannot cancel
    and cannot change its source's deadline; it is what a call reads at its own
    boundaries to decide whether to stop, or blocks on through :meth:`wait`.
    """

    def __init__(self, core: _native.CancellationToken) -> None:
        self._core = core

    @property
    def cancellable(self) -> bool:
        """False only for a token whose source can never cancel it."""
        return bool(self._core.cancellable)

    @property
    def cancelled(self) -> bool:
        """True once the source was cancelled or its deadline passed."""
        return bool(self._core.cancelled)

    @property
    def reason(self) -> CancellationReasonName:
        """``"none"``, ``"cancelled"``, or ``"deadline_exceeded"``."""
        return str(self._core.reason)  # type: ignore[return-value]

    def wait(self) -> CancellationReasonName:
        """Block until this token has a reason, and return it.

        This is for work that has no boundary of its own — a thread parked on
        an external event — and it never polls: an explicit
        :meth:`CancellationSource.cancel` or the shared deadline watchdog
        releases it. The GIL is released for the whole wait, so other Python
        threads keep running.

        It returns the reason rather than raising it; use the returned value,
        or read :attr:`reason`, to decide whether to unwind.
        """
        return str(self._core.wait())  # type: ignore[return-value]


@final
class CancellationSource:
    """The owning half of a cancellation state.

    ``timeout`` is a deadline in seconds measured from construction. A zero
    timeout is already exceeded, which is a meaningful request rather than an
    error; a negative, infinite, or NaN timeout is rejected.

    ```python
    source = CancellationSource()
    run = session.begin_stage("decode", prompt, cancellation=source.token())
    threading.Timer(5.0, source.cancel).start()
    ```

    Cancellation is cooperative and never rolls anything back: the interrupted
    call raises :class:`CancelledError` or :class:`DeadlineExceededError` and
    leaves everything it already applied in place.

    A deadline is claimed by one shared process-wide watchdog as well as by
    every poll, so :meth:`CancellationToken.wait` and work blocked inside a
    call are released at the deadline rather than at the next boundary.
    """

    def __init__(self, timeout: float | None = None) -> None:
        if timeout is not None:
            timeout = float(timeout)
            if not math.isfinite(timeout):
                raise ValueError("timeout must be a finite number of seconds")
            if timeout < 0:
                raise ValueError("timeout must not be negative")
        self._core = _native.CancellationSource(timeout)

    def token(self) -> CancellationToken:
        """An observer of this source, safe to pass to any cancellable call."""
        return CancellationToken(self._core.token())

    def cancel(self) -> None:
        """Request cancellation. Safe from any thread, and idempotent.

        Calling this while another thread is inside ``step``, ``finish``, or
        ``run_stage`` is the supported way to stop that call: it never takes
        the session lock.
        """
        self._core.cancel()

    @property
    def cancelled(self) -> bool:
        return bool(self._core.cancelled)

    @property
    def reason(self) -> CancellationReasonName:
        return str(self._core.reason)  # type: ignore[return-value]

    def wait(self) -> CancellationReasonName:
        """Block until this source has a reason, and return it.

        The same wait :meth:`CancellationToken.wait` performs, offered here so
        the owner does not have to make a token first.
        """
        return str(self._core.wait())  # type: ignore[return-value]


def _cancellation_token(
    cancellation: CancellationToken | None,
    timeout: float | None,
) -> tuple[_native.CancellationToken | None, CancellationSource | None]:
    """Resolve the two mutually exclusive ways to ask a call to stop.

    Returns the native token to forward and, for a ``timeout``, the source
    that owns it so a caller can keep it alive for the whole call.
    """
    if cancellation is not None and timeout is not None:
        raise ValueError(
            "Pass either cancellation or timeout, not both: a timeout creates "
            "its own token"
        )
    if timeout is not None:
        source = CancellationSource(timeout)
        return source._core.token(), source
    if cancellation is not None:
        return cancellation._core, None
    return None, None


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
        *,
        cancellation: CancellationToken | None = None,
        timeout: float | None = None,
    ) -> dict[str, NDArray[Any]]:
        """Run the graph, optionally under a token or a per-call timeout.

        ``cancellation`` and ``timeout`` are mutually exclusive. Cancellation is
        checked before the backend call and after the outputs are validated;
        ONNX Runtime itself can only stop between graph nodes, so a single long
        kernel still finishes before the call unwinds.
        """
        token, _source = _cancellation_token(cancellation, timeout)
        return self._core.run(
            {name: np.asarray(value) for name, value in inputs.items()},
            token,
        )


def _scheduling_stats(payload: Mapping[str, Any]) -> PipelineSchedulingStats:
    """Freezes one native admission reading into its typed value."""
    return PipelineSchedulingStats(
        active_executions=int(payload["active_executions"]),
        queued_executions=int(payload["queued_executions"]),
        active_by_stage_kind=MappingProxyType(
            dict(payload["active_by_stage_kind"])
        ),
        queued_by_stage_kind=MappingProxyType(
            dict(payload["queued_by_stage_kind"])
        ),
    )


class Pipeline:
    """A validated Mobius pipeline package with shareable model sessions.

    ``max_concurrent_executions`` and ``max_concurrent_by_stage_kind`` are
    admission scheduling only: they cap how many executions may be inside the
    runtime at once and queue the rest fairly. Nothing is merged, split,
    reordered, or preempted. ``0`` -- and any stage kind left out of the
    mapping -- means unlimited, which is the default. One execution is one
    ``run_stage``, one ``step_stage``, one ``StageRun.step``, or one
    ``StageRun.finish`` that still has work to drain; ``begin_stage`` and an
    idle handle hold nothing.

    Every key of ``max_concurrent_by_stage_kind`` must name a stage kind the
    runtime executes -- ``single_pass``, ``autoregressive``, ``iterative``,
    ``state_transition``, ``composite``, or ``on_demand`` -- and an unknown or
    empty key raises :class:`WorldModelError` here rather than being ignored.
    """

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
        max_concurrent_executions: int = 0,
        max_concurrent_by_stage_kind: Mapping[str, int] | None = None,
    ) -> None:
        package = Path(package_path)
        with (package / "pipeline.json").open(encoding="utf-8") as file:
            self._document: dict[str, Any] = json.load(file)
        library_path = (
            Path(ort_library_path) if ort_library_path is not None else _find_ort_library()
        )
        provider_names, options = _provider_arguments(providers, provider_options)
        stage_kind_limits = dict(max_concurrent_by_stage_kind or {})
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
            max_concurrent_executions,
            stage_kind_limits,
        )
        # Recorded only after the native constructor accepted them, so these
        # report the limits actually in force rather than what was requested.
        self._max_concurrent_executions = max_concurrent_executions
        self._max_concurrent_by_stage_kind: Mapping[str, int] = MappingProxyType(
            stage_kind_limits
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

    @property
    def max_concurrent_executions(self) -> int:
        """Executions admitted at once across every session, or 0 for no cap."""
        return self._max_concurrent_executions

    @property
    def max_concurrent_by_stage_kind(self) -> Mapping[str, int]:
        """Per-stage-kind admission caps; a missing kind has no cap."""
        return self._max_concurrent_by_stage_kind

    @property
    def scheduling_stats(self) -> PipelineSchedulingStats:
        """How many executions this pipeline has admitted and queued right now.

        Copies of one pipeline share an admission scheduler, so they report the
        same numbers, and every session and :class:`StageRun` created from any
        of them is counted. Reading admits, queues, and blocks nothing.
        """
        return _scheduling_stats(self._core.scheduling_stats)

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


def _stage_event(payload: Mapping[str, Any]) -> StageEvent:
    return StageEvent(
        kind=payload["kind"],
        stage=payload["stage"],
        iteration=int(payload["iteration"]),
        token_ids=payload["token_ids"],
        outputs=dict(payload["outputs"]),
        finished=bool(payload["finished"]),
    )


@final
class StageRun:
    """One incremental execution of a single stage.

    Only :meth:`PipelineSession.begin_stage` produces one, and the handle is
    not an extension point, so it is final. Each :meth:`step` advances the
    session by exactly one model or scheduler step and returns that step's
    :class:`StageEvent`; the run ends with exactly one event whose
    ``finished`` is true, and :meth:`finish` returns the same outputs
    :meth:`PipelineSession.run_stage` returns for the same arguments.

    Stepping is synchronous: it blocks until that step finishes. It can be
    stopped cooperatively -- :meth:`request_cancellation`, or cancelling the
    token passed to :meth:`PipelineSession.begin_stage`, makes the running
    step raise :class:`CancelledError` at its next boundary and never rolls
    anything back. A session runs one stage at a time,
    so while this run is unfinished the session raises
    :class:`WorldModelError` from ``begin_stage``, ``run_stage``,
    ``step_stage``, ``snapshot``, ``restore``, ``fork``, ``checkpoint``,
    ``restore_checkpoint``, ``drop_checkpoint``, ``reset``, and
    ``release_stage``. Reading ``outputs``, ``state``, and ``has_checkpoint``
    stays legal.

    Use the run as a context manager, or call :meth:`close`, to release the
    session when a caller stops early:

    ```python
    with session.begin_stage("decode", inputs) as run:
        for event in run:
            if event.kind == "token" and stop_now(event.token_ids):
                break
    ```
    """

    def __init__(self, session: PipelineSession, core: _native.StageRun) -> None:
        # The session reference is deliberately strong: a run keeps the
        # session it drives alive for as long as a caller holds the run.
        self._session = session
        self._core = core
        self._closed = False
        self._exhausted = False

    @property
    def session(self) -> PipelineSession:
        return self._session

    @property
    def stage(self) -> str:
        return str(self._core.stage)

    @property
    def done(self) -> bool:
        return self._closed or bool(self._core.done)

    @property
    def iteration(self) -> int:
        return int(self._core.iteration)

    def step(self) -> StageEvent:
        """Advance the run by one step and return that step's event."""
        if self._closed:
            raise _native.WorldModelError(
                f"Pipeline stage run for '{self.stage}' is closed"
            )
        return _stage_event(self._core.step())

    def finish(self) -> dict[str, NDArray[Any]]:
        """Drain the remaining steps and return the full ``run_stage`` result."""
        if self._closed:
            raise _native.WorldModelError(
                f"Pipeline stage run for '{self.stage}' is closed"
            )
        outputs = self._core.finish()
        self._exhausted = True
        return outputs

    def request_cancellation(self) -> None:
        """Ask the work this run is doing to stop, from any thread.

        Unlike :meth:`close` this never takes the session lock, so it is the
        one method that is safe to call while another thread is inside
        :meth:`step` or :meth:`finish`. That call then raises
        :class:`CancelledError` at its next boundary, releases the session,
        and keeps everything the run already applied.
        """
        self._core.request_cancellation()

    def close(self) -> None:
        """Abandon an unfinished run and release the session. Idempotent.

        This is the local close-and-release operation: it takes the session
        lock, so it waits for an in-flight step rather than interrupting it.
        Use :meth:`request_cancellation` to stop work already running.
        """
        if self._closed:
            return
        self._closed = True
        self._core.cancel()

    def __iter__(self) -> Iterator[StageEvent]:
        return self

    def __next__(self) -> StageEvent:
        if self._exhausted:
            raise StopIteration
        event = self.step()
        if event.finished:
            self._exhausted = True
        return event

    def __enter__(self) -> StageRun:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Best effort only: interpreter shutdown may already have torn down
        # the native module, and a destructor must not raise.
        with contextlib.suppress(Exception):
            self.close()


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
        cancellation: CancellationToken | None = None,
        timeout: float | None = None,
    ) -> dict[str, NDArray[Any]]:
        """Run one stage to completion, optionally under a token or timeout.

        ``cancellation`` and ``timeout`` are mutually exclusive; a ``timeout``
        creates one deadline that covers this call only. A stopped call raises
        :class:`CancelledError` or :class:`DeadlineExceededError` and keeps
        everything the stage already applied.
        """
        token, _source = _cancellation_token(cancellation, timeout)
        return self._core.run_stage(
            stage,
            _array_mapping(inputs),
            _array_mapping(overrides),
            dict(options or {}),
            token,
        )

    def step_stage(
        self,
        stage: str,
        inputs: Mapping[str, ArrayLike] | None = None,
        *,
        overrides: Mapping[str, ArrayLike] | None = None,
        options: Mapping[str, bool | int | float | str] | None = None,
        cancellation: CancellationToken | None = None,
        timeout: float | None = None,
    ) -> dict[str, NDArray[Any]]:
        token, _source = _cancellation_token(cancellation, timeout)
        return self._core.step_stage(
            stage,
            _array_mapping(inputs),
            _array_mapping(overrides),
            dict(options or {}),
            token,
        )

    def begin_stage(
        self,
        stage: str,
        inputs: Mapping[str, ArrayLike] | None = None,
        *,
        overrides: Mapping[str, ArrayLike] | None = None,
        options: Mapping[str, bool | int | float | str] | None = None,
        cancellation: CancellationToken | None = None,
        timeout: float | None = None,
    ) -> StageRun:
        """Start an incremental execution of ``stage``.

        The returned :class:`StageRun` owns this session's single run slot
        until it completes or is closed, so hold it in a ``with`` block when a
        caller may stop before the run finishes.

        ``cancellation`` and ``timeout`` are mutually exclusive. A ``timeout``
        is one absolute deadline for the whole run, checked at every step
        boundary. Whichever is supplied, the run also honors
        :meth:`StageRun.request_cancellation`.
        """
        token, _source = _cancellation_token(cancellation, timeout)
        return StageRun(
            self,
            self._core.begin_stage(
                stage,
                _array_mapping(inputs),
                _array_mapping(overrides),
                dict(options or {}),
                token,
            ),
        )

    def iter_stage(
        self,
        stage: str,
        inputs: Mapping[str, ArrayLike] | None = None,
        *,
        overrides: Mapping[str, ArrayLike] | None = None,
        options: Mapping[str, bool | int | float | str] | None = None,
        cancellation: CancellationToken | None = None,
        timeout: float | None = None,
    ) -> Iterator[StageEvent]:
        """Iterate the events of one incremental execution of ``stage``.

        The run starts before the first event is requested, so an invalid
        stage or option fails here rather than on the first ``next()``. The
        iterator closes the run when it is exhausted or closed, but a bare
        ``break`` only closes it once the generator is collected, which is not
        a guarantee every interpreter makes promptly. Use
        :meth:`begin_stage` with a ``with`` block, or call ``close()`` on this
        iterator, whenever a caller may stop early.
        """
        run = self.begin_stage(
            stage,
            inputs,
            overrides=overrides,
            options=options,
            cancellation=cancellation,
            timeout=timeout,
        )
        return _drain_stage_run(run)

    def run(
        self,
        inputs: Mapping[str, ArrayLike],
        *,
        stages: tuple[str, ...] | list[str] | None = None,
        overrides: Mapping[str, ArrayLike] | None = None,
        options: Mapping[str, bool | int | float | str] | None = None,
        cancellation: CancellationToken | None = None,
        timeout: float | None = None,
    ) -> dict[str, NDArray[Any]]:
        """Run the selected stages in declaration order, releasing each one.

        ``cancellation`` and ``timeout`` are mutually exclusive. A ``timeout``
        is one absolute deadline for the whole sequence rather than a fresh
        budget per stage, so a slow early stage eats into what the later ones
        have left.
        """
        # Resolved once, before the loop, so one absolute deadline covers the
        # whole sequence rather than restarting at every stage.
        _token, source = _cancellation_token(cancellation, timeout)
        stage_token = cancellation if source is None else source.token()
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
                    cancellation=stage_token,
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
        """Return an independent session started from a snapshot of this one.

        The fork inherits execution state but starts with an empty
        named-checkpoint namespace.
        """
        return PipelineSession(self._pipeline, self._core.fork())

    def checkpoint(self, name: str) -> None:
        """Store the current execution state under ``name``, replacing any prior one.

        Checkpoints are in-memory transaction markers held beside the session's
        execution state, so they survive stage execution and :meth:`restore`,
        never appear inside a snapshot, and are dropped by :meth:`reset`.
        """
        self._core.checkpoint(name)

    def restore_checkpoint(self, name: str) -> None:
        """Rewind the session to the checkpoint stored under ``name``.

        Raises :class:`WorldModelError` for an empty or unknown name, and the
        session is left unchanged. The checkpoint itself stays available.
        """
        self._core.restore_checkpoint(name)

    def drop_checkpoint(self, name: str) -> None:
        """Discard the checkpoint stored under ``name``.

        Dropping a name this session does not hold raises
        :class:`WorldModelError` rather than silently doing nothing.
        """
        self._core.drop_checkpoint(name)

    def has_checkpoint(self, name: str) -> bool:
        """Report whether this session currently holds a checkpoint ``name``."""
        return bool(self._core.has_checkpoint(name))


def _drain_stage_run(run: StageRun) -> Iterator[StageEvent]:
    try:
        yield from run
    finally:
        run.close()


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
