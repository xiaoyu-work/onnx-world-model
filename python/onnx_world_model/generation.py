from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._api import Pipeline, ProviderOptions
from .preprocessing import RawImage, RawVideo, WorldModelPreprocessor


@dataclass(frozen=True)
class TextOutput:
    text: str
    token_ids: NDArray[np.int64]
    timings: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageOutput:
    images: NDArray[Any]
    timings: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoOutput:
    video: NDArray[Any]
    timings: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionOutput:
    actions: NDArray[Any]
    velocity: NDArray[Any] | None = None
    timings: Mapping[str, float] = field(default_factory=dict)


class WorldModel:
    """A modality-oriented API over one Mobius world-model package."""

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
        runtime = _GenerationRuntime(
            package_path,
            providers=providers,
            provider_options=provider_options,
            ort_library_path=ort_library_path,
            intra_op_threads=intra_op_threads,
            inter_op_threads=inter_op_threads,
            log_severity=log_severity,
        )
        self._runtime = runtime
        self.text = TextGenerator(runtime)
        self.image = ImageGenerator(runtime)
        self.video = VideoGenerator(runtime)
        self.action = ActionGenerator(runtime)

    @classmethod
    def from_pretrained(
        cls,
        package_path: str | os.PathLike[str],
        **options: Any,
    ) -> WorldModel:
        return cls(package_path, **options)

    load = from_pretrained

    @property
    def profile(self) -> Mapping[str, Any] | None:
        return self._runtime.pipeline.profile

    @property
    def execution_providers(self) -> Mapping[str, tuple[str, ...]]:
        return self._runtime.pipeline.execution_providers

    @property
    def capabilities(self) -> tuple[str, ...]:
        capabilities: list[str] = []
        if self._runtime.reasoner_decode_stage is not None:
            capabilities.append("text")
        if (
            self._runtime.world_stage is not None
            and self._runtime.video_stage is not None
        ):
            capabilities.extend(("image", "video"))
        if (
            self._runtime.world_stage is not None
            and self._runtime.preprocessor.metadata.get("action") is not None
        ):
            capabilities.append("action")
        return tuple(capabilities)


class TextGenerator:
    def __init__(self, runtime: _GenerationRuntime) -> None:
        self._runtime = runtime

    def generate(
        self,
        prompt: str,
        image: RawImage | None = None,
        *,
        video: RawVideo | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        thinking: bool = False,
        video_fps: float | None = None,
        video_num_frames: int | None = None,
        video_sample_fps: float | None = None,
    ) -> TextOutput:
        if image is not None and video is not None:
            raise ValueError("image and video are mutually exclusive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        options: dict[str, bool | int | float | str] = {
            "max_tokens": max_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        if seed is not None:
            options["seed"] = seed
        return self._runtime.generate_text(
            prompt,
            image,
            video,
            system_prompt=system_prompt,
            thinking=thinking,
            options=options,
            video_fps=video_fps,
            video_num_frames=video_num_frames,
            video_sample_fps=video_sample_fps,
        )


class ImageGenerator:
    def __init__(self, runtime: _GenerationRuntime) -> None:
        self._runtime = runtime

    def generate(
        self,
        prompt: str,
        *,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        initial_latents: NDArray[np.floating[Any]] | None = None,
        input_ids: NDArray[np.integer[Any]] | None = None,
    ) -> ImageOutput:
        result, timings = self._runtime.generate(
            prompt,
            output_video=True,
            output_action=False,
            frames=1,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=seed,
            initial_vision_tokens=initial_latents,
            generator_input_ids=input_ids,
        )
        video = result["video"]
        if video.ndim != 5 or video.shape[2] != 1:
            raise RuntimeError(
                "The image generation pipeline did not produce one video frame"
            )
        return ImageOutput(images=video[:, :, 0].copy(), timings=timings)


class VideoGenerator:
    def __init__(self, runtime: _GenerationRuntime) -> None:
        self._runtime = runtime

    def generate(
        self,
        prompt: str,
        *,
        frames: int = 5,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        initial_latents: NDArray[np.floating[Any]] | None = None,
        input_ids: NDArray[np.integer[Any]] | None = None,
    ) -> VideoOutput:
        result, timings = self._runtime.generate(
            prompt,
            output_video=True,
            output_action=False,
            frames=frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=seed,
            initial_vision_tokens=initial_latents,
            generator_input_ids=input_ids,
        )
        return VideoOutput(video=result["video"], timings=timings)


class ActionGenerator:
    def __init__(self, runtime: _GenerationRuntime) -> None:
        self._runtime = runtime

    def generate(
        self,
        prompt: str,
        *,
        domain: str,
        steps: int = 1,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        initial_actions: NDArray[np.floating[Any]] | None = None,
        initial_vision_latents: NDArray[np.floating[Any]] | None = None,
        video_frames: int = 5,
        video_height: int = 256,
        video_width: int = 256,
        input_ids: NDArray[np.integer[Any]] | None = None,
    ) -> ActionOutput:
        if domain == "no_action":
            raise ValueError("domain must select a real action domain")
        result, timings = self._runtime.generate(
            prompt,
            output_video=False,
            output_action=True,
            frames=video_frames,
            height=video_height,
            width=video_width,
            num_inference_steps=num_inference_steps,
            action_steps=steps,
            action_domain=domain,
            mode="action",
            seed=seed,
            initial_vision_tokens=initial_vision_latents,
            initial_action_tokens=initial_actions,
            generator_input_ids=input_ids,
        )
        return ActionOutput(
            actions=result["action"],
            velocity=result.get("action_velocity"),
            timings=timings,
        )


class _GenerationRuntime:
    def __init__(
        self,
        package_path: str | os.PathLike[str],
        **pipeline_options: Any,
    ) -> None:
        self.package_path = Path(package_path)
        started = time.perf_counter()
        self.pipeline = Pipeline(self.package_path, **pipeline_options)
        self.load_seconds = time.perf_counter() - started
        self.preprocessor = WorldModelPreprocessor(self.package_path)
        self.reasoner_prompt_stage = self._find_stage(
            name="reasoner_prompt",
            kind="single_pass",
            run_on="prefill",
        )
        self.reasoner_decode_stage = self._find_stage(
            name="reasoner_decode",
            kind="autoregressive",
        )
        self.world_stage = self._find_stage(
            name="world_generation",
            kind="iterative",
        )
        self.video_stage = self._find_stage(
            name="decode_video",
            run_on="finalize",
        )

    def generate_text(
        self,
        prompt: str,
        image: RawImage | None,
        video: RawVideo | None,
        *,
        system_prompt: str | None,
        thinking: bool,
        options: Mapping[str, bool | int | float | str],
        video_fps: float | None,
        video_num_frames: int | None,
        video_sample_fps: float | None,
    ) -> TextOutput:
        if self.reasoner_decode_stage is None:
            raise RuntimeError("This package has no text generation stage")
        session = self.pipeline.create_session()
        timings: dict[str, float] = {}
        if video is not None:
            if self.reasoner_prompt_stage is None:
                raise RuntimeError("This package has no visual text prefill stage")
            prepared_video = self.preprocessor.prepare_video_reasoner(
                prompt,
                video,
                system_prompt=system_prompt,
                enable_thinking=thinking,
                source_fps=video_fps,
                num_frames=video_num_frames,
                fps=video_sample_fps,
            )
            started = time.perf_counter()
            overrides: dict[str, NDArray[Any]] = {}
            empty_image = self.preprocessor.empty_image_features()
            if empty_image is not None:
                overrides["reasoner_embedding.image_features"] = empty_image
            session.run_stage(
                self.reasoner_prompt_stage,
                prepared_video.pipeline_inputs(),
                overrides=overrides,
                options={"vision_modality": "video"},
            )
            timings["prefill"] = time.perf_counter() - started
            decode_inputs = None
            overrides = {}
            empty_image = self.preprocessor.empty_image_features()
            empty_video = self.preprocessor.empty_video_features()
            if empty_image is not None:
                overrides["reasoner_embedding.image_features"] = empty_image
            if empty_video is not None:
                overrides["reasoner_embedding.video_features"] = empty_video
            if not overrides:
                overrides = None
        else:
            prepared = self.preprocessor.prepare_reasoner(
                prompt,
                image,
                system_prompt=system_prompt,
                enable_thinking=thinking,
            )
        if image is not None:
            if self.reasoner_prompt_stage is None:
                raise RuntimeError("This package has no visual text prefill stage")
            started = time.perf_counter()
            empty_video = self.preprocessor.empty_video_features()
            session.run_stage(
                self.reasoner_prompt_stage,
                prepared.pipeline_inputs(),
                overrides=(
                    None
                    if empty_video is None
                    else {
                        "reasoner_embedding.video_features": empty_video
                    }
                ),
            )
            timings["prefill"] = time.perf_counter() - started
            decode_inputs = None
            overrides = {}
            empty_image = self.preprocessor.empty_image_features()
            empty_video = self.preprocessor.empty_video_features()
            if empty_image is not None:
                overrides["reasoner_embedding.image_features"] = empty_image
            if empty_video is not None:
                overrides["reasoner_embedding.video_features"] = empty_video
            if not overrides:
                overrides = None
        elif video is None:
            decode_inputs = {"text.token_ids": prepared.input_ids}
            overrides = {}
            empty_image = self.preprocessor.empty_image_features()
            empty_video = self.preprocessor.empty_video_features()
            if empty_image is not None:
                overrides["reasoner_embedding.image_features"] = empty_image
            if empty_video is not None:
                overrides["reasoner_embedding.video_features"] = empty_video
            if not overrides:
                overrides = None
        started = time.perf_counter()
        generated = session.run_stage(
            self.reasoner_decode_stage,
            decode_inputs,
            overrides=overrides,
            options=options,
        )
        timings["generate"] = time.perf_counter() - started
        session.release_stage(self.reasoner_decode_stage)
        token_ids = np.asarray(generated["generated_token_ids"], dtype=np.int64)
        return TextOutput(
            text=self.preprocessor.decode(token_ids),
            token_ids=token_ids,
            timings=timings,
        )

    def generate(
        self,
        prompt: str,
        *,
        output_video: bool,
        output_action: bool,
        frames: int,
        height: int,
        width: int,
        num_inference_steps: int | None,
        seed: int | None,
        action_steps: int = 0,
        action_domain: str = "no_action",
        mode: str | None = None,
        initial_vision_tokens: NDArray[np.floating[Any]] | None = None,
        initial_action_tokens: NDArray[np.floating[Any]] | None = None,
        generator_input_ids: NDArray[np.integer[Any]] | None = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, float]]:
        if self.world_stage is None:
            raise RuntimeError("This package has no world generation stage")
        prepared = self.preprocessor.prepare_world(
            prompt,
            frames=frames,
            height=height,
            width=width,
            action_steps=action_steps,
            action_domain=action_domain,
            include_action=output_action,
            num_inference_steps=num_inference_steps,
            mode=mode,
            seed=seed,
            initial_vision_tokens=initial_vision_tokens,
            initial_action_tokens=initial_action_tokens,
            generator_input_ids=generator_input_ids,
        )
        session = self.pipeline.create_session()
        timings: dict[str, float] = {}
        started = time.perf_counter()
        generated = session.run_stage(
            self.world_stage,
            prepared.pipeline_inputs(),
            options=prepared.options,
        )
        timings["generate"] = time.perf_counter() - started
        result = {
            name: np.asarray(value).copy()
            for name, value in generated.items()
        }
        if output_action:
            if "action" not in result:
                raise RuntimeError("The package produced no action output")
            action_width = self.preprocessor.action_dimension(action_domain)
            result["action"] = result["action"][..., :action_width].copy()
        session.release_stage(self.world_stage)

        if output_video:
            if self.video_stage is None:
                raise RuntimeError("This package has no video generation stage")
            started = time.perf_counter()
            decoded = session.run_stage(
                self.video_stage,
                options=prepared.options,
            )
            timings["decode"] = time.perf_counter() - started
            if "video" not in decoded:
                raise RuntimeError("The package produced no video output")
            result["video"] = np.asarray(decoded["video"]).copy()
            session.release_stage(self.video_stage)
        return result, timings

    def _find_stage(
        self,
        *,
        name: str,
        kind: str | None = None,
        run_on: str | None = None,
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
        return candidates[0] if len(candidates) == 1 else None


__all__ = [
    "ActionGenerator",
    "ActionOutput",
    "ImageGenerator",
    "ImageOutput",
    "TextGenerator",
    "TextOutput",
    "VideoGenerator",
    "VideoOutput",
    "WorldModel",
]
