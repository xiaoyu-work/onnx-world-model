from __future__ import annotations

import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ._api import Pipeline, ProviderOptions
from .preprocessing import RawImage, WorldModelPreprocessor

OutputKind = Literal["text", "action", "video"]


@dataclass(frozen=True)
class TextGenerationConfig:
    max_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    enable_thinking: bool = False

    def options(self) -> dict[str, bool | int | float | str]:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        options: dict[str, bool | int | float | str] = {
            "max_tokens": self.max_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        return options


@dataclass(frozen=True)
class WorldGenerationConfig:
    frames: int = 5
    height: int = 256
    width: int = 256
    num_inference_steps: int | None = None
    action_steps: int = 0
    action_domain: str = "no_action"
    mode: str | None = None
    seed: int | None = None
    initial_vision_tokens: NDArray[np.floating[Any]] | None = field(
        default=None, repr=False
    )
    initial_action_tokens: NDArray[np.floating[Any]] | None = field(
        default=None, repr=False
    )
    generator_input_ids: NDArray[np.integer[Any]] | None = field(
        default=None, repr=False
    )


@dataclass(frozen=True)
class WorldModelResult:
    text: str | None = None
    token_ids: NDArray[np.int64] | None = None
    action: NDArray[Any] | None = None
    action_velocity: NDArray[Any] | None = None
    video: NDArray[Any] | None = None
    vision_velocity: NDArray[Any] | None = None
    timings: Mapping[str, float] = field(default_factory=dict)
    raw_outputs: Mapping[str, NDArray[Any]] = field(
        default_factory=dict, repr=False
    )


class WorldModel:
    """Foundry-style facade over a validated Mobius world-model package."""

    def __init__(
        self,
        package_path: str | os.PathLike[str],
        *,
        providers: Sequence[str] | None = None,
        provider_options: ProviderOptions | None = None,
        ort_library_path: str | os.PathLike[str] | None = None,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
        log_severity: int = 3,
    ) -> None:
        self.package_path = Path(package_path)
        started = time.perf_counter()
        self.pipeline = Pipeline(
            self.package_path,
            providers=providers,
            provider_options=provider_options,
            ort_library_path=ort_library_path,
            intra_op_threads=intra_op_threads,
            inter_op_threads=inter_op_threads,
            log_severity=log_severity,
        )
        self.load_seconds = time.perf_counter() - started
        self.preprocessor = WorldModelPreprocessor(self.package_path)
        self._reasoner_prompt_stage = self._find_stage(
            name="reasoner_prompt",
            kind="single_pass",
            run_on="prefill",
            required=False,
        )
        self._reasoner_decode_stage = self._find_stage(
            name="reasoner_decode", kind="autoregressive", required=False
        )
        self._world_stage = self._find_stage(
            name="world_generation", kind="iterative", required=False
        )
        self._video_stage = self._find_stage(
            name="decode_video", run_on="finalize", required=False
        )

    @classmethod
    def load(
        cls,
        package_path: str | os.PathLike[str],
        **options: Any,
    ) -> WorldModel:
        return cls(package_path, **options)

    @property
    def profile(self) -> Mapping[str, Any] | None:
        return self.pipeline.profile

    @property
    def execution_providers(self) -> Mapping[str, tuple[str, ...]]:
        return self.pipeline.execution_providers

    def respond(
        self,
        prompt: str,
        image: RawImage | None = None,
        *,
        system_prompt: str | None = None,
        config: TextGenerationConfig | None = None,
    ) -> WorldModelResult:
        session = self.pipeline.create_session()
        return self._respond(
            session,
            prompt,
            image,
            system_prompt=system_prompt,
            config=config or TextGenerationConfig(),
        )

    def generate_world(
        self,
        prompt: str,
        *,
        outputs: Iterable[OutputKind] = ("video",),
        config: WorldGenerationConfig | None = None,
    ) -> WorldModelResult:
        requested = _validate_outputs(outputs, allow_text=False)
        session = self.pipeline.create_session()
        return self._generate_world(
            session,
            prompt,
            requested,
            config=config or WorldGenerationConfig(),
        )

    def generate(
        self,
        prompt: str,
        image: RawImage | None = None,
        *,
        outputs: Iterable[OutputKind] = ("text",),
        system_prompt: str | None = None,
        text_config: TextGenerationConfig | None = None,
        world_config: WorldGenerationConfig | None = None,
    ) -> WorldModelResult:
        requested = _validate_outputs(outputs)
        if image is not None and requested - {"text"}:
            raise NotImplementedError(
                "Raw image conditioning for world/action generation is not "
                "implemented. image= is currently used only by the Reasoner."
            )
        session = self.pipeline.create_session()
        text_result = WorldModelResult()
        if "text" in requested:
            text_result = self._respond(
                session,
                prompt,
                image,
                system_prompt=system_prompt,
                config=text_config or TextGenerationConfig(),
            )
        world_result = WorldModelResult()
        world_outputs = requested - {"text"}
        if world_outputs:
            world_result = self._generate_world(
                session,
                prompt,
                world_outputs,
                config=world_config or WorldGenerationConfig(),
            )
        return _merge_results(text_result, world_result)

    def _respond(
        self,
        session: Any,
        prompt: str,
        image: RawImage | None,
        *,
        system_prompt: str | None,
        config: TextGenerationConfig,
    ) -> WorldModelResult:
        if self._reasoner_decode_stage is None:
            raise RuntimeError("This pipeline has no autoregressive Reasoner stage")
        timings: dict[str, float] = {}
        prepared = self.preprocessor.prepare_reasoner(
            prompt,
            image,
            system_prompt=system_prompt,
            enable_thinking=config.enable_thinking,
        )
        if image is not None:
            if self._reasoner_prompt_stage is None:
                raise RuntimeError("This pipeline has no Reasoner prefill stage")
            started = time.perf_counter()
            session.run_stage(
                self._reasoner_prompt_stage,
                prepared.pipeline_inputs(),
            )
            timings["reasoner_prefill"] = time.perf_counter() - started
            decode_inputs = None
            overrides = None
        else:
            decode_inputs = {"text.token_ids": prepared.input_ids}
            overrides = None
            empty_features = self.preprocessor.empty_image_features()
            if empty_features is not None:
                overrides = {
                    "reasoner_embedding.image_features": empty_features
                }
        started = time.perf_counter()
        output = session.run_stage(
            self._reasoner_decode_stage,
            decode_inputs,
            overrides=overrides,
            options=config.options(),
        )
        timings["decode"] = time.perf_counter() - started
        session.release_stage(self._reasoner_decode_stage)
        token_ids = np.asarray(output["generated_token_ids"], dtype=np.int64)
        text = self.preprocessor.decode(token_ids)
        return WorldModelResult(
            text=text,
            token_ids=token_ids,
            timings=timings,
            raw_outputs={
                "generated_token_ids": token_ids,
            },
        )

    def _generate_world(
        self,
        session: Any,
        prompt: str,
        outputs: set[str],
        *,
        config: WorldGenerationConfig,
    ) -> WorldModelResult:
        if self._world_stage is None:
            raise RuntimeError("This pipeline has no iterative world-generation stage")
        if "action" in outputs and config.action_domain == "no_action":
            raise ValueError(
                "An action_domain other than 'no_action' is required for action output"
            )
        prepared = self.preprocessor.prepare_world(
            prompt,
            frames=config.frames,
            height=config.height,
            width=config.width,
            action_steps=config.action_steps,
            action_domain=config.action_domain,
            include_action="action" in outputs,
            num_inference_steps=config.num_inference_steps,
            mode=config.mode,
            seed=config.seed,
            initial_vision_tokens=config.initial_vision_tokens,
            initial_action_tokens=config.initial_action_tokens,
            generator_input_ids=config.generator_input_ids,
        )
        timings: dict[str, float] = {}
        started = time.perf_counter()
        world = session.run_stage(
            self._world_stage,
            prepared.pipeline_inputs(),
            options=prepared.options,
        )
        timings["world_generation"] = time.perf_counter() - started
        action = None
        if "action" in outputs and "action" in world:
            width = self.preprocessor.action_dimension(prepared.action_domain)
            action = np.asarray(world["action"])[..., :width].copy()
        session.release_stage(self._world_stage)

        video = None
        decoded: dict[str, NDArray[Any]] = {}
        if "video" in outputs:
            if self._video_stage is None:
                raise RuntimeError("This pipeline has no video decoder stage")
            started = time.perf_counter()
            decoded = session.run_stage(
                self._video_stage,
                options=prepared.options,
            )
            timings["video_decode"] = time.perf_counter() - started
            if "video" not in decoded:
                raise RuntimeError("Video decoder stage produced no public video output")
            video = np.asarray(decoded["video"]).copy()
            session.release_stage(self._video_stage)

        raw: dict[str, NDArray[Any]] = {}
        for name in ("action", "action_velocity", "vision_velocity"):
            if name in world:
                raw[name] = np.asarray(world[name]).copy()
        if video is not None:
            raw["video"] = video
        return WorldModelResult(
            action=action,
            action_velocity=_copy_optional(world.get("action_velocity")),
            video=video,
            vision_velocity=_copy_optional(world.get("vision_velocity")),
            timings=timings,
            raw_outputs=raw,
        )

    def _find_stage(
        self,
        *,
        name: str,
        kind: str | None = None,
        run_on: str | None = None,
        required: bool = True,
    ) -> str | None:
        exact = next(
            (stage.name for stage in self.pipeline.stages if stage.name == name),
            None,
        )
        if exact is not None:
            return exact
        candidates = [
            stage.name
            for stage in self.pipeline.stages
            if (kind is None or stage.kind == kind)
            and (run_on is None or stage.run_on == run_on)
        ]
        if len(candidates) == 1:
            return candidates[0]
        if required:
            raise ValueError(
                f"Could not identify stage {name!r}; candidates: {candidates}"
            )
        return None


HighLevelWorldModel = WorldModel


def _validate_outputs(
    outputs: Iterable[OutputKind],
    *,
    allow_text: bool = True,
) -> set[str]:
    requested = set(outputs)
    allowed = {"text", "action", "video"} if allow_text else {"action", "video"}
    unknown = requested - allowed
    if unknown:
        raise ValueError(f"Unknown output kinds: {sorted(unknown)}")
    if not requested:
        raise ValueError("At least one output kind is required")
    return requested


def _copy_optional(value: Any) -> NDArray[Any] | None:
    return None if value is None else np.asarray(value).copy()


def _merge_results(
    left: WorldModelResult,
    right: WorldModelResult,
) -> WorldModelResult:
    return WorldModelResult(
        text=left.text or right.text,
        token_ids=left.token_ids if left.token_ids is not None else right.token_ids,
        action=left.action if left.action is not None else right.action,
        action_velocity=(
            left.action_velocity
            if left.action_velocity is not None
            else right.action_velocity
        ),
        video=left.video if left.video is not None else right.video,
        vision_velocity=(
            left.vision_velocity
            if left.vision_velocity is not None
            else right.vision_velocity
        ),
        timings={**left.timings, **right.timings},
        raw_outputs={**left.raw_outputs, **right.raw_outputs},
    )


__all__ = [
    "HighLevelWorldModel",
    "OutputKind",
    "TextGenerationConfig",
    "WorldGenerationConfig",
    "WorldModel",
    "WorldModelResult",
]
