from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import ml_dtypes
import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment
from numpy.typing import NDArray
from PIL import Image
from tokenizers import Tokenizer

RawImage = str | os.PathLike[str] | NDArray[Any] | Image.Image
RawVideo = (
    str
    | os.PathLike[str]
    | NDArray[Any]
    | Sequence[RawImage]
)


@dataclass(frozen=True)
class PreparedReasonerInputs:
    input_ids: NDArray[np.int64]
    pixel_values: NDArray[Any] | None
    grid_thw: NDArray[np.int64] | None
    rendered_prompt: str
    image_token_count: int
    video_token_count: int = 0
    sampled_frames: int = 0
    video_features_target: str | None = None

    def pipeline_inputs(self) -> dict[str, NDArray[Any]]:
        values: dict[str, NDArray[Any]] = {"text.token_ids": self.input_ids}
        if self.pixel_values is not None:
            values["vision.pixel_values"] = self.pixel_values
        if self.grid_thw is not None:
            values["vision.grid_thw"] = self.grid_thw
        return values


@dataclass(frozen=True)
class PreparedConditioning:
    """Media conditioning that must be encoded before the diffusion stage."""

    encoder_stage: str
    encoder_input: str
    encoder_output: str
    pixel_values: NDArray[Any]
    latent_frames: tuple[int, ...]
    spatial_patch_size: int
    latent_grid: tuple[int, int]

    def pipeline_inputs(self) -> dict[str, NDArray[Any]]:
        return {self.encoder_input: self.pixel_values}


@dataclass(frozen=True)
class PreparedWorldInputs:
    input_ids: NDArray[np.int64]
    vision_tokens: NDArray[Any]
    action_tokens: NDArray[Any] | None
    sound_tokens: NDArray[Any] | None
    options: dict[str, bool | int | float | str]
    latent_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    action_domain: str
    unconditional_input_ids: NDArray[np.int64] | None = None
    unconditional_input_name: str | None = None
    conditioning: PreparedConditioning | None = None
    noisy_vision_token_indexes: NDArray[np.int64] | None = None
    noisy_vision_token_input: str | None = None

    def pipeline_inputs(self) -> dict[str, NDArray[Any]]:
        values: dict[str, NDArray[Any]] = {
            "text.token_ids": self.input_ids,
            "diffusion.initial_vision_latent": self.vision_tokens,
        }
        if self.action_tokens is not None:
            values["diffusion.initial_action_latent"] = self.action_tokens
        if self.sound_tokens is not None:
            values["diffusion.initial_sound_latent"] = self.sound_tokens
        if (
            self.unconditional_input_ids is not None
            and self.unconditional_input_name is not None
        ):
            values[self.unconditional_input_name] = self.unconditional_input_ids
        return values

    def pipeline_overrides(self) -> dict[str, NDArray[Any]]:
        """Generated inputs the host replaces for conditioned generation."""
        if (
            self.noisy_vision_token_indexes is None
            or self.noisy_vision_token_input is None
        ):
            return {}
        return {self.noisy_vision_token_input: self.noisy_vision_token_indexes}

    def with_conditioning_latent(
        self,
        latent: NDArray[Any],
    ) -> PreparedWorldInputs:
        """Anchor the conditioned latent frames to an encoded latent.

        ``latent`` is the ``[B, C, T, H, W]`` output of the manifest's
        conditioning encoder. Its latent frames are packed into generator token
        rows and written over the noise of the conditioned frames, leaving the
        remaining rows noisy exactly as the official recipe does.
        """
        conditioning = self.conditioning
        if conditioning is None:
            raise ValueError("These world inputs declare no media conditioning")
        packed = pack_latent_tokens(latent, conditioning.spatial_patch_size)
        grid_height, grid_width = conditioning.latent_grid
        frame_tokens = grid_height * grid_width
        encoded_frames = packed.shape[0] // frame_tokens
        if packed.shape[0] != encoded_frames * frame_tokens:
            raise ValueError(
                "The encoded conditioning latent does not match the requested "
                "latent grid"
            )
        if encoded_frames != len(conditioning.latent_frames):
            raise ValueError(
                f"The conditioning encoder produced {encoded_frames} latent "
                f"frames, expected {len(conditioning.latent_frames)}"
            )
        if packed.shape[1] != self.vision_tokens.shape[1]:
            raise ValueError(
                "The encoded conditioning latent has "
                f"{packed.shape[1]} channels, expected "
                f"{self.vision_tokens.shape[1]}"
            )
        vision_tokens = np.array(self.vision_tokens, copy=True)
        for position, frame in enumerate(conditioning.latent_frames):
            target = frame * frame_tokens
            source = position * frame_tokens
            vision_tokens[target : target + frame_tokens] = packed[
                source : source + frame_tokens
            ].astype(vision_tokens.dtype, copy=False)
        return replace(self, vision_tokens=vision_tokens)


def pack_latent_tokens(
    latent: NDArray[Any],
    spatial_patch_size: int,
) -> NDArray[Any]:
    """Pack a ``[B, C, T, H, W]`` latent into generator token rows.

    Rows are ordered ``(batch, frame, patch_row, patch_column)`` and each row
    concatenates its patch values as ``(patch_height, patch_width, channel)``,
    which is the inverse of the runtime's video unpatchify transform.
    """
    if spatial_patch_size <= 0:
        raise ValueError("spatial_patch_size must be positive")
    values = np.asarray(latent)
    if values.ndim != 5:
        raise ValueError("Latents must have shape [batch, channels, T, H, W]")
    batch, channels, frames, height, width = values.shape
    if height % spatial_patch_size or width % spatial_patch_size:
        raise ValueError(
            "Latent height and width must be divisible by spatial_patch_size"
        )
    grid_height = height // spatial_patch_size
    grid_width = width // spatial_patch_size
    # (B, C, T, H_p, p_h, W_p, p_w) -> (B, T, H_p, W_p, p_h, p_w, C)
    patches = values.reshape(
        batch,
        channels,
        frames,
        grid_height,
        spatial_patch_size,
        grid_width,
        spatial_patch_size,
    )
    patches = patches.transpose(0, 2, 3, 5, 4, 6, 1)
    return patches.reshape(
        batch * frames * grid_height * grid_width,
        spatial_patch_size * spatial_patch_size * channels,
    )


_CONDITIONING_RESAMPLING = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "nearest": Image.Resampling.NEAREST,
    "lanczos": Image.Resampling.LANCZOS,
}

#: The only conditioning resize strategy the runtime implements: frames are
#: scaled straight to the requested output geometry.
_CONDITIONING_RESIZE_STRATEGY = "stretch_to_target"


def conditioning_resample(preprocessing: Mapping[str, Any]) -> str:
    """The resampling filter a conditioning contract asks for.

    ``resample`` always names the filter. ``resize`` may name either the
    filter or the resize strategy, which keeps the block compatible with the
    vision-understanding preprocessing spelling.
    """
    declared = preprocessing.get("resample")
    if declared is None:
        declared = preprocessing.get("resize")
        if declared is not None and declared not in _CONDITIONING_RESAMPLING:
            if declared != _CONDITIONING_RESIZE_STRATEGY:
                raise ValueError(
                    f"Unsupported conditioning resize {declared!r}; expected a "
                    f"filter or {_CONDITIONING_RESIZE_STRATEGY!r}"
                )
            declared = None
    if declared is None:
        return "bilinear"
    if declared not in _CONDITIONING_RESAMPLING:
        raise ValueError(f"Unsupported conditioning resample {declared!r}")
    return str(declared)


class ConditioningFramePreprocessor:
    """Conditioning frames for a latent video encoder.

    Frames are resized to the requested output geometry and mapped to the
    encoder's signed range, producing ``[1, 3, frames, height, width]``.
    """

    def __init__(
        self,
        *,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        resample: str = "bilinear",
        dtype: Any = np.float32,
    ) -> None:
        if any(value <= 0 for value in std):
            raise ValueError("Conditioning standard deviations must be positive")
        if resample not in _CONDITIONING_RESAMPLING:
            raise ValueError(f"Unsupported conditioning resample {resample!r}")
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.resample = resample
        self.dtype = np.dtype(dtype)

    def __call__(
        self,
        image: RawImage,
        *,
        height: int,
        width: int,
        frames: int = 1,
    ) -> NDArray[Any]:
        if height <= 0 or width <= 0 or frames <= 0:
            raise ValueError("Conditioning geometry must be positive")
        resized = ImagePreprocessor._to_image(image).convert("RGB").resize(
            (width, height),
            _CONDITIONING_RESAMPLING[self.resample],
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        pixels = (pixels - self.mean) / self.std
        frame = np.transpose(pixels, (2, 0, 1))
        video = np.repeat(frame[:, None], frames, axis=1)
        return video[None].astype(self.dtype, copy=False)



class TextPreprocessor:
    """Tokenizer and chat-template processing from an exported package."""

    def __init__(self, package_path: str | os.PathLike[str]) -> None:
        self.package_path = Path(package_path)
        tokenizer_path = self._find_tokenizer()
        self._tokenizer = Tokenizer.from_file(os.fspath(tokenizer_path))
        self._template = self._load_template()

    def _find_tokenizer(self) -> Path:
        candidates = (
            self.package_path / "tokenizer.json",
            self.package_path / "text_tokenizer" / "tokenizer.json",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No tokenizer.json found in exported package {self.package_path}"
        )

    def _load_template(self) -> Any:
        template_path = self.package_path / "chat_template.jinja"
        if template_path.is_file():
            source = template_path.read_text(encoding="utf-8")
        else:
            config_path = self.package_path / "tokenizer_config.json"
            if not config_path.is_file():
                raise FileNotFoundError(
                    "The exported package has no chat_template.jinja or "
                    "tokenizer_config.json"
                )
            config = _read_json(config_path)
            source = config.get("chat_template")
            if not isinstance(source, str) or not source:
                raise ValueError("The exported tokenizer has no chat template")
        environment = ImmutableSandboxedEnvironment(
            autoescape=False,
            lstrip_blocks=True,
            trim_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )
        environment.filters["tojson"] = lambda value, **options: json.dumps(
            value, **options
        )
        return environment.from_string(source)

    def render_chat(
        self,
        prompt: str,
        *,
        image: bool = False,
        video: bool = False,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
        allow_empty: bool = False,
        add_generation_prompt: bool = True,
        add_vision_id: bool = False,
    ) -> str:
        if not prompt and not allow_empty:
            raise ValueError("prompt must be non-empty")
        messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        content: str | list[dict[str, str]]
        if image or video:
            content = []
            if image:
                content.append(
                    {"type": "image", "image": "<runtime-image>"}
                )
            if video:
                content.append(
                    {"type": "video", "video": "<runtime-video>"}
                )
            content.append({"type": "text", "text": prompt})
        else:
            content = prompt
        messages.append({"role": "user", "content": content})
        return self._template.render(
            messages=messages,
            tools=[],
            add_generation_prompt=add_generation_prompt,
            add_vision_id=add_vision_id,
            enable_thinking=enable_thinking,
        )

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def encode_chat(
        self,
        prompt: str,
        *,
        image: bool = False,
        video: bool = False,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
        allow_empty: bool = False,
        add_generation_prompt: bool = True,
        add_vision_id: bool = False,
    ) -> tuple[str, list[int]]:
        rendered = self.render_chat(
            prompt,
            image=image,
            video=video,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            allow_empty=allow_empty,
            add_generation_prompt=add_generation_prompt,
            add_vision_id=add_vision_id,
        )
        return rendered, self.encode(rendered)

    def decode(
        self,
        token_ids: list[int] | NDArray[np.integer[Any]],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return self._tokenizer.decode(
            [int(token) for token in np.asarray(token_ids).reshape(-1)],
            skip_special_tokens=skip_special_tokens,
        )

    def token_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)


class ImagePreprocessor:
    """Fixed-signature image preprocessing derived from a pipeline package."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        dtype: Any = np.float32,
    ) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("Image height and width must be positive")
        if any(value <= 0 for value in std):
            raise ValueError("Image standard deviations must be positive")
        self.height = height
        self.width = width
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.dtype = np.dtype(dtype)

    def __call__(self, image: RawImage) -> NDArray[Any]:
        pil_image = self._to_image(image)
        resized = pil_image.convert("RGB").resize(
            (self.width, self.height),
            Image.Resampling.BICUBIC,
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        pixels = (pixels - self.mean) / self.std
        return np.transpose(pixels, (2, 0, 1))[None].astype(
            self.dtype, copy=True
        )

    @staticmethod
    def _to_image(image: RawImage) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as opened:
                return opened.convert("RGB")
        array = np.asarray(image)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            raise ValueError("Image arrays must be HWC, CHW, or batch-one NCHW")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.shape[-1] not in (1, 3, 4):
            raise ValueError("Image arrays must have 1, 3, or 4 channels")
        if np.issubdtype(array.dtype, np.floating):
            values = array.astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Image arrays must contain only finite values")
            minimum = float(values.min()) if values.size else 0.0
            maximum = float(values.max()) if values.size else 0.0
            if -1.0 <= minimum < 0.0 and maximum <= 1.0:
                values = (values + 1.0) * 0.5
                values *= 255.0
            elif 0.0 <= minimum and maximum <= 1.0:
                values *= 255.0
            elif not (0.0 <= minimum and maximum <= 255.0):
                raise ValueError(
                    "Float image arrays must use [-1,1], [0,1], or [0,255]"
                )
            array = np.clip(values, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return Image.fromarray(array)


@dataclass(frozen=True)
class PackedImage:
    pixel_values: NDArray[Any]
    grid_thw: NDArray[np.int64]
    token_count: int


class PackedImagePreprocessor:
    """Cosmos3 Edge smart-resize and block-major patchification."""

    def __init__(
        self,
        *,
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
        min_pixels: int,
        max_pixels: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        dtype: Any,
    ) -> None:
        if min(patch_size, merge_size, temporal_patch_size) <= 0:
            raise ValueError("Patch and merge sizes must be positive")
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("Invalid image pixel-area limits")
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.dtype = np.dtype(dtype)

    def __call__(self, image: RawImage) -> PackedImage:
        pil_image = ImagePreprocessor._to_image(image).convert("RGB")
        resized_height, resized_width = self._smart_resize(
            pil_image.height,
            pil_image.width,
        )
        resized = pil_image.resize(
            (resized_width, resized_height),
            Image.Resampling.BICUBIC,
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        pixels = (pixels - self.mean) / self.std
        chw = np.transpose(pixels, (2, 0, 1))
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size
        channels = chw.shape[0]
        patches = chw.reshape(
            channels,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = np.transpose(patches, (1, 4, 2, 5, 3, 6, 0))
        flattened = patches.reshape(
            grid_h * grid_w,
            self.patch_size
            * self.patch_size
            * channels
            * self.temporal_patch_size,
        )
        if self.temporal_patch_size != 1:
            flattened = np.repeat(
                flattened[..., None],
                self.temporal_patch_size,
                axis=-1,
            ).reshape(flattened.shape[0], -1)
        return PackedImage(
            pixel_values=flattened.astype(self.dtype, copy=False),
            grid_thw=np.asarray([1, grid_h, grid_w], dtype=np.int64),
            token_count=grid_h * grid_w // (self.merge_size**2),
        )

    def _smart_resize(self, height: int, width: int) -> tuple[int, int]:
        factor = self.patch_size * self.merge_size
        if height < factor or width < factor:
            scale = max(factor / height, factor / width)
            height = int(height * scale)
            width = int(width * scale)
        if max(height, width) / min(height, width) > 200:
            raise ValueError("Image aspect ratio must be smaller than 200")
        resized_height = round(height / factor) * factor
        resized_width = round(width / factor) * factor
        pixels = resized_height * resized_width
        if pixels > self.max_pixels:
            beta = math.sqrt(height * width / self.max_pixels)
            resized_height = max(
                factor, math.floor(height / beta / factor) * factor
            )
            resized_width = max(
                factor, math.floor(width / beta / factor) * factor
            )
        elif pixels < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            resized_height = math.ceil(height * beta / factor) * factor
            resized_width = math.ceil(width * beta / factor) * factor
        return resized_height, resized_width


@dataclass(frozen=True)
class PreparedVideo:
    input_ids: NDArray[np.int64]
    pixel_values: NDArray[Any]
    grid_thw: NDArray[np.int64]
    rendered_prompt: str
    token_count: int
    sampled_frames: int
    features_target: str

    def pipeline_inputs(self) -> dict[str, NDArray[Any]]:
        return {
            "text.token_ids": self.input_ids,
            "vision.pixel_values": self.pixel_values,
            "vision.grid_thw": self.grid_thw,
        }


class PackedVideoPreprocessor(PackedImagePreprocessor):
    """Cosmos3 Edge video sampling and time-major patchification."""

    def __init__(
        self,
        *,
        fps: float,
        min_frames: int,
        max_frames: int,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        if fps <= 0 or min_frames <= 0 or max_frames < min_frames:
            raise ValueError("Invalid video sampling configuration")
        self.fps = fps
        self.min_frames = min_frames
        self.max_frames = max_frames

    def prepare(
        self,
        video: RawVideo,
        *,
        source_fps: float | None = None,
        num_frames: int | None = None,
        fps: float | None = None,
    ) -> tuple[PackedImage, NDArray[np.int64], float]:
        if isinstance(video, (str, os.PathLike)):
            path = Path(video)
            if not path.is_file():
                raise FileNotFoundError(path)
            metadata = iio.immeta(path)
            metadata_fps = float(metadata.get("fps") or 24.0)
            effective_fps = (
                source_fps if source_fps is not None else metadata_fps
            )
            if effective_fps <= 0:
                raise ValueError("source_fps must be positive")
            reported_frames = metadata.get("nframes")
            if (
                reported_frames is None
                or not np.isfinite(reported_frames)
            ):
                duration = metadata.get("duration")
                if duration is None:
                    raise ValueError(
                        "Video metadata has neither frame count nor duration"
                    )
                total_frames = max(
                    1, round(float(duration) * metadata_fps)
                )
            else:
                total_frames = int(reported_frames)
            indices = self._sample_indices(
                total_frames,
                source_fps=effective_fps,
                num_frames=num_frames,
                fps=fps,
            )
            selected = {int(index) for index in indices}
            decoded: dict[int, Image.Image] = {}
            last_index = max(selected)
            for index, frame in enumerate(
                iio.imiter(path, plugin="FFMPEG")
            ):
                if index in selected:
                    decoded[index] = Image.fromarray(np.asarray(frame))
                if index >= last_index:
                    break
            missing = selected - decoded.keys()
            if missing:
                raise ValueError(
                    f"Video ended before selected frames {sorted(missing)}"
                )
            sampled = [decoded[int(index)] for index in indices]
            return self._patchify_frames(sampled), indices, effective_fps

        frames, decoded_fps = self._decode(video)
        if not frames:
            raise ValueError("video must contain at least one frame")
        effective_fps = (
            source_fps
            if source_fps is not None
            else decoded_fps if decoded_fps is not None else 24.0
        )
        if effective_fps <= 0:
            raise ValueError("source_fps must be positive")
        indices = self._sample_indices(
            len(frames),
            source_fps=effective_fps,
            num_frames=num_frames,
            fps=fps,
        )
        sampled = [frames[int(index)] for index in indices]
        pixels = self._patchify_frames(sampled)
        return pixels, indices, effective_fps

    def _decode(
        self, video: RawVideo
    ) -> tuple[list[Image.Image], float | None]:
        if isinstance(video, np.ndarray):
            array = np.asarray(video)
            if array.ndim != 4:
                raise ValueError(
                    "Video arrays must have shape [T,H,W,C] or [T,C,H,W]"
                )
            return [
                ImagePreprocessor._to_image(frame)
                for frame in array
            ], None
        if isinstance(video, Sequence):
            return [
                ImagePreprocessor._to_image(frame)
                for frame in video
            ], None
        raise TypeError("Unsupported video input type")

    def _sample_indices(
        self,
        total_frames: int,
        *,
        source_fps: float,
        num_frames: int | None,
        fps: float | None,
    ) -> NDArray[np.int64]:
        if num_frames is not None and fps is not None:
            raise ValueError("num_frames and fps are mutually exclusive")
        if num_frames is not None:
            if num_frames <= 0:
                raise ValueError("num_frames must be positive")
            if num_frames > total_frames:
                raise ValueError(
                    f"num_frames={num_frames} exceeds source frame count "
                    f"{total_frames}"
                )
            count = num_frames
        else:
            target_fps = self.fps if fps is None else fps
            if target_fps <= 0:
                raise ValueError("fps must be positive")
            count = int(total_frames / source_fps * target_fps)
            count = min(
                max(count, self.min_frames),
                self.max_frames,
                total_frames,
            )
        return np.linspace(0, total_frames - 1, count).round().astype(np.int64)

    def _patchify_frames(self, frames: list[Image.Image]) -> PackedImage:
        first = frames[0].convert("RGB")
        resized_height, resized_width = self._smart_resize_video(
            len(frames), first.height, first.width
        )
        arrays: list[NDArray[np.float32]] = []
        for frame in frames:
            resized = frame.convert("RGB").resize(
                (resized_width, resized_height),
                Image.Resampling.BICUBIC,
            )
            values = np.asarray(resized, dtype=np.float32) / 255.0
            arrays.append((values - self.mean) / self.std)
        video = np.stack(arrays)
        pad = -video.shape[0] % self.temporal_patch_size
        if pad:
            video = np.concatenate(
                [video, np.repeat(video[-1:], pad, axis=0)],
                axis=0,
            )
        grid_t = video.shape[0] // self.temporal_patch_size
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size
        patches = np.transpose(video, (0, 3, 1, 2)).reshape(
            grid_t,
            self.temporal_patch_size,
            3,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = np.transpose(patches, (0, 3, 6, 4, 7, 5, 8, 2, 1))
        flattened = patches.reshape(
            grid_t * grid_h * grid_w,
            self.patch_size
            * self.patch_size
            * 3
            * self.temporal_patch_size,
        )
        return PackedImage(
            pixel_values=flattened.astype(self.dtype, copy=False),
            grid_thw=np.asarray(
                [grid_t, grid_h, grid_w], dtype=np.int64
            ),
            token_count=(
                grid_t * grid_h * grid_w // (self.merge_size**2)
            ),
        )

    def _smart_resize_video(
        self, frames: int, height: int, width: int
    ) -> tuple[int, int]:
        factor = self.patch_size * self.merge_size
        if frames < self.temporal_patch_size:
            raise ValueError("Video has fewer frames than temporal patch size")
        if height < factor or width < factor:
            scale = max(factor / height, factor / width)
            height = int(height * scale)
            width = int(width * scale)
        if max(height, width) / min(height, width) > 200:
            raise ValueError("Video aspect ratio must be smaller than 200")
        resized_height = round(height / factor) * factor
        resized_width = round(width / factor) * factor
        temporal = (
            round(frames / self.temporal_patch_size)
            * self.temporal_patch_size
        )
        pixels = temporal * resized_height * resized_width
        if pixels > self.max_pixels:
            beta = math.sqrt(frames * height * width / self.max_pixels)
            resized_height = max(
                factor, math.floor(height / beta / factor) * factor
            )
            resized_width = max(
                factor, math.floor(width / beta / factor) * factor
            )
        elif pixels < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (frames * height * width))
            resized_height = math.ceil(height * beta / factor) * factor
            resized_width = math.ceil(width * beta / factor) * factor
        return resized_height, resized_width


class GeneratorPromptPacker:
    """Prompt packing for a joint generator, described by the manifest.

    The generator consumes chat-formatted token IDs directly, so packaging
    declares the chat flags, optional per-modality system prompt, the metadata
    sentence templates, and the trailing special tokens that mark the start of
    generation. Everything is optional: an absent section means the runtime
    performs that step exactly as before.
    """

    def __init__(
        self,
        text: TextPreprocessor,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.config: dict[str, Any] = dict(config or {})
        chat = self.config.get("chat", {})
        if not isinstance(chat, Mapping):
            raise TypeError("generator_prompt.chat must be an object")
        unknown = set(chat) - {
            "add_generation_prompt",
            "add_vision_id",
            "enable_thinking",
        }
        if unknown:
            raise ValueError(
                "generator_prompt.chat has unsupported fields "
                f"{sorted(unknown)}"
            )
        self.add_generation_prompt = bool(chat.get("add_generation_prompt", True))
        self.add_vision_id = bool(chat.get("add_vision_id", False))
        self.enable_thinking = bool(chat.get("enable_thinking", False))
        self.system_prompts: dict[str, str] = {
            str(name): str(value)
            for name, value in (self.config.get("system_prompts") or {}).items()
        }
        self.templates: dict[str, str] = {
            str(name): str(value)
            for name, value in (self.config.get("templates") or {}).items()
        }
        self.suffix_token_ids = self._suffix_token_ids()

    #: Prompt-packing fields a generation recipe may restate for one mode.
    OVERRIDABLE = (
        "chat",
        "system_prompts",
        "templates",
        "suffix_token_ids",
        "suffix_tokens",
    )

    def merged(
        self,
        overrides: Mapping[str, Any] | None,
    ) -> GeneratorPromptPacker:
        """Apply a recipe's prompt-packing overrides to this packer."""
        selected = {
            name: value
            for name, value in (overrides or {}).items()
            if name in self.OVERRIDABLE
        }
        if not selected:
            return self
        return GeneratorPromptPacker(self.text, {**self.config, **selected})

    def _suffix_token_ids(self) -> tuple[int, ...]:
        declared = self.config.get("suffix_token_ids")
        if declared is not None:
            if not isinstance(declared, (list, tuple)) or not all(
                isinstance(value, int) for value in declared
            ):
                raise ValueError(
                    "generator_prompt.suffix_token_ids must be a list of integers"
                )
            return tuple(int(value) for value in declared)
        tokens = self.config.get("suffix_tokens")
        if tokens is None:
            return ()
        if not isinstance(tokens, (list, tuple)):
            raise TypeError(
                "generator_prompt.suffix_tokens must be a list of token strings"
            )
        resolved: list[int] = []
        for token in tokens:
            token_id = self.text.token_id(str(token))
            if token_id is None:
                raise ValueError(
                    f"Tokenizer does not define generator prompt token {token!r}"
                )
            resolved.append(token_id)
        return tuple(resolved)

    @property
    def has_system_prompts(self) -> bool:
        return bool(self.system_prompts)

    @property
    def has_templates(self) -> bool:
        return bool(self.templates)

    def system_prompt(self, *, is_image: bool) -> str:
        key = "image" if is_image else "video"
        if key not in self.system_prompts:
            known = ", ".join(sorted(self.system_prompts)) or "<none>"
            raise ValueError(
                "The package declares no generator system prompt for "
                f"{key!r}; declared: {known}"
            )
        return self.system_prompts[key]

    def _template(self, name: str) -> str:
        if name not in self.templates:
            known = ", ".join(sorted(self.templates)) or "<none>"
            raise ValueError(
                f"The package declares no generator prompt template {name!r}; "
                f"declared: {known}"
            )
        return self.templates[name]

    def apply_templates(
        self,
        text: str,
        *,
        negative: bool,
        is_image: bool,
        frames: int,
        height: int,
        width: int,
        fps: float,
        add_resolution_template: bool,
        add_duration_template: bool,
    ) -> str:
        if not (add_resolution_template or add_duration_template):
            return text
        if _is_structured_prompt(text):
            # Structured prompts already carry their metadata; appending
            # sentences would corrupt the document.
            return text
        prefix = "inverse_" if negative else ""
        if not is_image and add_duration_template:
            if fps <= 0:
                raise ValueError("fps must be positive to add a duration template")
            text = _append_sentence(
                text,
                self._template(f"{prefix}duration").format(
                    duration=frames / fps, fps=fps
                ),
            )
        if add_resolution_template:
            modality = "image" if is_image else "video"
            text = _append_sentence(
                text,
                self._template(f"{prefix}{modality}_resolution").format(
                    height=height, width=width
                ),
            )
        return text

    def encode(
        self,
        prompt: str,
        *,
        negative: bool = False,
        is_image: bool = False,
        frames: int = 1,
        height: int = 0,
        width: int = 0,
        fps: float = 0.0,
        add_resolution_template: bool = False,
        add_duration_template: bool = False,
        use_system_prompt: bool = False,
        system_prompt: str | None = None,
    ) -> NDArray[np.int64]:
        text = self.apply_templates(
            prompt,
            negative=negative,
            is_image=is_image,
            frames=frames,
            height=height,
            width=width,
            fps=fps,
            add_resolution_template=add_resolution_template,
            add_duration_template=add_duration_template,
        )
        if system_prompt is None and use_system_prompt:
            system_prompt = self.system_prompt(is_image=is_image)
        _, token_ids = self.text.encode_chat(
            text,
            image=False,
            system_prompt=system_prompt,
            enable_thinking=self.enable_thinking,
            allow_empty=True,
            add_generation_prompt=self.add_generation_prompt,
            add_vision_id=self.add_vision_id,
        )
        return np.asarray(
            list(token_ids) + list(self.suffix_token_ids), dtype=np.int64
        )


def _append_sentence(base: str, addition: str) -> str:
    base = base.rstrip(".")
    return f"{base}. {addition}" if base else addition


def _is_structured_prompt(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
    except ValueError:
        return False
    return True


class WorldModelPreprocessor:
    """High-level preprocessing compiled from a Mobius world-model package."""

    def __init__(self, package_path: str | os.PathLike[str]) -> None:
        self.package_path = Path(package_path)
        self.document = _read_json(self.package_path / "pipeline.json")
        self.manifest: dict[str, Any] = self.document["manifest"]
        self.metadata: dict[str, Any] = self.manifest.get("metadata", {})
        self.components = {
            component["name"]: component for component in self.manifest["components"]
        }
        self.inputs = {
            entry["port"]: entry for entry in self.manifest.get("inputs", [])
        }
        self.text = TextPreprocessor(self.package_path)
        self.root_config = _read_optional_json(self.package_path / "config.json")
        self.generator_prompt = GeneratorPromptPacker(
            self.text, self.metadata.get("generator_prompt")
        )
        self.image_token_id = -1
        self.image_feature_count = 0
        self.image_feature_width = 0
        self.image: ImagePreprocessor | PackedImagePreprocessor | None = None
        self._image_initialization_attempted = False
        self._embedding_feature_specs = self._find_embedding_feature_specs()
        self.video_token_id = -1
        self.video: PackedVideoPreprocessor | None = None
        self._video_initialization_attempted = False
        self._video_config = self._component_config("video_encoder")
        self._generator_config = self._component_config("generator")
        self.conditioning: ConditioningFramePreprocessor | None = None

    @property
    def profile(self) -> dict[str, Any] | None:
        return self.manifest.get("profile")

    @property
    def generation_recipes(self) -> dict[str, Any]:
        recipes = self.metadata.get("generation_recipes", {})
        return recipes if isinstance(recipes, dict) else {}

    def generation_recipe(self, mode: str | None) -> dict[str, Any]:
        if mode is None:
            return {}
        recipe = self.generation_recipes.get(mode)
        return dict(recipe) if isinstance(recipe, dict) else {}

    def diffusion_stage(self) -> tuple[str, dict[str, Any]]:
        """The iterative stage that denoises the packed generator sequence."""
        for stage in self.manifest.get("stages", []):
            if stage.get("kind") == "iterative":
                return stage["name"], dict(stage.get("options", {}))
        raise ValueError("This package has no iterative generation stage")

    def guidance_program(self) -> dict[str, Any]:
        try:
            _, options = self.diffusion_stage()
        except ValueError:
            return {}
        guidance = options.get("guidance")
        return dict(guidance) if isinstance(guidance, dict) else {}

    def conditioning_program(self, modality: str) -> dict[str, Any]:
        try:
            _, options = self.diffusion_stage()
        except ValueError:
            return {}
        conditioning = options.get("conditioning")
        if not isinstance(conditioning, dict):
            return {}
        program = conditioning.get(modality)
        return dict(program) if isinstance(program, dict) else {}

    def guidance_scale(
        self,
        *,
        mode: str | None,
        requested: float | None = None,
    ) -> float:
        """Resolve the guidance scale exactly as the runtime stage does."""
        guidance = self.guidance_program()
        if requested is not None:
            return float(requested)
        if not guidance:
            return 1.0
        option = guidance.get("scale_option", "guidance_scale")
        _, options = self.diffusion_stage()
        scheduler = options.get("scheduler", {})
        overrides = scheduler.get("mode_overrides", {}) if scheduler else {}
        if mode is not None and mode in overrides:
            override = overrides[mode].get(option)
            if isinstance(override, (int, float)):
                return float(override)
        return float(guidance.get("default_scale", 1.0))

    def unconditional_input_name(self) -> str | None:
        """Reserved input name carrying the unconditional guided value."""
        guidance = self.guidance_program()
        if not guidance:
            return None
        declared = guidance.get("unconditional_input")
        if isinstance(declared, str) and declared:
            return declared
        conditioning_input = guidance.get("conditioning_input")
        if not isinstance(conditioning_input, str) or not conditioning_input:
            return None
        return f"unconditional:{conditioning_input}"

    def default_negative_prompt(self, mode: str | None = None) -> str:
        """The negative prompt a package recommends for a generation mode.

        The asset may hold the prompt text directly, a wrapper object with a
        ``negative_prompt``/``default``/mode-named string, or the structured
        JSON document the checkpoint was trained on, which is passed through
        verbatim as a JSON string.
        """
        recipe = self.generation_recipe(mode)
        asset = recipe.get("prompt", {}).get("negative_asset")
        if not isinstance(asset, str) or not asset:
            return ""
        path = self.package_path / asset
        if not path.is_file():
            return ""
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in (mode, "negative_prompt", "default"):
                if key is not None and isinstance(value.get(key), str):
                    return str(value[key])
        return json.dumps(value)

    def _special_token_id(self, config_name: str, token: str) -> int:
        configured = self.root_config.get(config_name)
        if isinstance(configured, int):
            return configured
        token_id = self.text.token_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer does not define required token {token!r}")
        return token_id

    def _component_config(self, component_name: str) -> dict[str, Any]:
        component = self.components.get(component_name)
        return dict(component.get("config", {})) if component else {}

    def _image_input(self) -> tuple[str, dict[str, Any]]:
        for entry in self.manifest.get("inputs", []):
            if entry.get("semantic") == "vision.pixel_values":
                component_name, port_name = entry["port"].split(".", 1)
                component = self.components[component_name]
                spec = next(
                    value
                    for value in component["inputs"]
                    if value["name"] == port_name
                )
                return component_name, spec
        raise ValueError("Pipeline has no vision.pixel_values external input")

    def _find_embedding_feature_specs(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for component in self.components.values():
            for value in component.get("inputs", []):
                if value["name"] in {"image_features", "video_features"}:
                    result[value["name"]] = value
        return result

    def _image_feature_shape(self) -> tuple[int, int]:
        component_name, _ = self._image_input()
        component = self.components[component_name]
        output = next(
            (
                value
                for value in component["outputs"]
                if value["name"] == "image_features"
            ),
            component["outputs"][0],
        )
        shape = output["shape"]
        if (
            len(shape) != 2
            or not isinstance(shape[0], int)
            or not isinstance(shape[1], int)
        ):
            raise ValueError(
                "High-level image preprocessing requires a fixed rank-2 "
                "vision feature signature"
            )
        return shape[0], shape[1]

    def _create_image_preprocessor(
        self,
    ) -> ImagePreprocessor | PackedImagePreprocessor:
        _, spec = self._image_input()
        shape = spec["shape"]
        if len(shape) == 2:
            contract = self.metadata.get("vision_understanding", {})
            preprocessing = contract.get("preprocessing", {})
            patchify = preprocessing.get("patchify", {})
            if (
                preprocessing.get("resize")
                != "smart_resize_area_bounded_multiple_of_patch_times_merge"
                or patchify.get("layout") != "time_major_block_major"
                or patchify.get("patch_value_order")
                != "patch_height_patch_width_channel"
            ):
                raise ValueError(
                    "Packed vision input has no supported preprocessing contract"
                )
            asset = preprocessing.get(
                "image_processor_asset", "preprocessor_config.json"
            )
            config = _read_optional_json(self.package_path / asset)
            size = config.get("size", {})
            return PackedImagePreprocessor(
                patch_size=int(patchify["patch_size"]),
                merge_size=int(patchify["merge_size"]),
                temporal_patch_size=int(patchify["temporal_patch_size"]),
                min_pixels=int(size.get("shortest_edge", 256 * 256)),
                max_pixels=int(size.get("longest_edge", 4096 * 4096)),
                mean=_float_triplet(
                    preprocessing.get("normalize", {}).get(
                        "mean", config.get("image_mean", (0.5, 0.5, 0.5))
                    )
                ),
                std=_float_triplet(
                    preprocessing.get("normalize", {}).get(
                        "std", config.get("image_std", (0.5, 0.5, 0.5))
                    )
                ),
                dtype=_numpy_dtype(spec["dtype"]),
            )
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise ValueError(
                "High-level image preprocessing requires fixed NCHW or a "
                "supported packed vision contract"
            )
        if not isinstance(shape[2], int) or not isinstance(shape[3], int):
            raise TypeError("Vision input height and width must be static integers")
        config = _read_optional_json(self.package_path / "preprocessor_config.json")
        mean = _float_triplet(config.get("image_mean", (0.5, 0.5, 0.5)))
        std = _float_triplet(config.get("image_std", (0.5, 0.5, 0.5)))
        return ImagePreprocessor(
            height=shape[2],
            width=shape[3],
            mean=mean,
            std=std,
            dtype=_numpy_dtype(spec["dtype"]),
        )

    def _ensure_image_preprocessor(
        self,
    ) -> ImagePreprocessor | PackedImagePreprocessor:
        if self.image is not None:
            return self.image
        if self._image_initialization_attempted:
            raise ValueError(
                "This package does not expose a supported fixed-NCHW raw-image input"
            )
        self._image_initialization_attempted = True
        if self._external_input_by_semantic("vision.pixel_values") is None:
            raise ValueError("This package has no raw-image Reasoner input")
        self.image_token_id = self._special_token_id(
            "image_token_id", "<|image_pad|>"
        )
        self.image = self._create_image_preprocessor()
        if isinstance(self.image, ImagePreprocessor):
            self.image_feature_count, self.image_feature_width = (
                self._image_feature_shape()
            )
        elif self._embedding_feature_specs.get("image_features") is not None:
            width = self._embedding_feature_specs["image_features"]["shape"][-1]
            if isinstance(width, int):
                self.image_feature_width = width
        return self.image

    def empty_image_features(self) -> NDArray[Any] | None:
        return self.empty_features("image_features")

    def empty_video_features(self) -> NDArray[Any] | None:
        return self.empty_features("video_features")

    def empty_features(self, name: str) -> NDArray[Any] | None:
        spec = self._embedding_feature_specs.get(name)
        if spec is None:
            return None
        shape = spec["shape"]
        if len(shape) != 2 or not isinstance(shape[1], int):
            raise ValueError(
                "Embedding image feature width must be a static rank-2 dimension"
            )
        return np.zeros(
            (0, shape[1]),
            dtype=_numpy_dtype(spec["dtype"]),
        )

    def _ensure_video_preprocessor(self) -> PackedVideoPreprocessor:
        if self.video is not None:
            return self.video
        if self._video_initialization_attempted:
            raise ValueError("This package has no supported video understanding input")
        self._video_initialization_attempted = True
        contract = self.metadata.get("vision_understanding", {})
        preprocessing = contract.get("preprocessing", {})
        routing = contract.get("routing", {})
        if "video" not in routing:
            raise ValueError("This package has no video understanding route")
        image_processor = self._ensure_image_preprocessor()
        if not isinstance(image_processor, PackedImagePreprocessor):
            raise TypeError(
                "Video understanding requires a packed variable-resolution "
                "vision encoder"
            )
        asset = preprocessing.get(
            "video_processor_asset", "video_preprocessor_config.json"
        )
        config = _read_optional_json(self.package_path / asset)
        size = config.get("size", {})
        sampling = preprocessing.get("video_frame_sampling", {})
        self.video_token_id = self._special_token_id(
            "video_token_id", "<|video_pad|>"
        )
        self.video = PackedVideoPreprocessor(
            patch_size=image_processor.patch_size,
            merge_size=image_processor.merge_size,
            temporal_patch_size=image_processor.temporal_patch_size,
            min_pixels=int(size.get("shortest_edge", 64 * 64)),
            max_pixels=int(size.get("longest_edge", 24 * 1024 * 1024)),
            mean=tuple(float(value) for value in image_processor.mean),
            std=tuple(float(value) for value in image_processor.std),
            dtype=image_processor.dtype,
            fps=float(sampling.get("fps", 2)),
            min_frames=int(sampling.get("min_frames", 4)),
            max_frames=int(sampling.get("max_frames", 768)),
        )
        return self.video

    def _ensure_conditioning_preprocessor(
        self,
        program: Mapping[str, Any],
    ) -> ConditioningFramePreprocessor:
        if self.conditioning is not None:
            return self.conditioning
        encoder_input = str(program["encoder_input"])
        spec = self._port_spec(encoder_input)
        shape = spec["shape"]
        if len(shape) != 5:
            raise ValueError(
                "Media conditioning requires a rank-5 [B,C,T,H,W] encoder input"
            )
        preprocessing = dict(program.get("preprocessing", {}))
        if not preprocessing:
            handoff = self.metadata.get("conditioning_handoffs", {})
            video = handoff.get("video") if isinstance(handoff, dict) else None
            if isinstance(video, dict):
                preprocessing = dict(video.get("preprocessing", {}))
        normalize = preprocessing.get("normalize", {})
        self.conditioning = ConditioningFramePreprocessor(
            mean=_float_triplet(normalize.get("mean", (0.5, 0.5, 0.5))),
            std=_float_triplet(normalize.get("std", (0.5, 0.5, 0.5))),
            resample=conditioning_resample(preprocessing),
            dtype=_numpy_dtype(spec["dtype"]),
        )
        return self.conditioning

    def prepare_conditioning_frames(
        self,
        image: RawImage,
        *,
        height: int,
        width: int,
        frames: int = 1,
        modality: str = "vision",
    ) -> NDArray[Any]:
        """Normalized ``[1, 3, frames, height, width]`` conditioning frames."""
        program = self.conditioning_program(modality)
        if not program:
            raise ValueError(
                f"This package declares no {modality!r} media conditioning"
            )
        processor = self._ensure_conditioning_preprocessor(program)
        return processor(image, height=height, width=width, frames=frames)

    def prepare_reasoner(
        self,
        prompt: str,
        image: RawImage | None = None,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> PreparedReasonerInputs:
        rendered, token_ids = self.text.encode_chat(
            prompt,
            image=image is not None,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )
        pixels = None
        grid_thw = None
        if image is not None:
            image_processor = self._ensure_image_preprocessor()
            processed_image = image_processor(image)
            if isinstance(processed_image, PackedImage):
                pixels = processed_image.pixel_values
                grid_thw = processed_image.grid_thw
                image_token_count = processed_image.token_count
            else:
                pixels = processed_image
                image_token_count = self.image_feature_count
            positions = [
                index
                for index, value in enumerate(token_ids)
                if value == self.image_token_id
            ]
            if len(positions) != 1:
                raise ValueError(
                    "The chat template must emit exactly one image placeholder"
                )
            position = positions[0]
            token_ids = (
                token_ids[:position]
                + [self.image_token_id] * image_token_count
                + token_ids[position + 1 :]
            )
        return PreparedReasonerInputs(
            input_ids=np.asarray([token_ids], dtype=np.int64),
            pixel_values=pixels,
            grid_thw=grid_thw,
            rendered_prompt=rendered,
            image_token_count=image_token_count if image is not None else 0,
        )

    def prepare_video_reasoner(
        self,
        prompt: str,
        video: RawVideo,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
        source_fps: float | None = None,
        num_frames: int | None = None,
        fps: float | None = None,
    ) -> PreparedVideo:
        processor = self._ensure_video_preprocessor()
        processed, indices, effective_fps = processor.prepare(
            video,
            source_fps=source_fps,
            num_frames=num_frames,
            fps=fps,
        )
        rendered, token_ids = self.text.encode_chat(
            prompt,
            video=True,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )
        try:
            start = token_ids.index(self.video_token_id)
        except ValueError as error:
            raise ValueError(
                "The chat template did not emit a video placeholder"
            ) from error
        replacement: list[int] = []
        tokens_per_frame = (
            int(processed.grid_thw[1])
            * int(processed.grid_thw[2])
            // (processor.merge_size**2)
        )
        vision_start = self._special_token_id(
            "vision_start_token_id", "<|vision_start|>"
        )
        vision_end = self._special_token_id(
            "vision_end_token_id", "<|vision_end|>"
        )
        for frame_index in indices:
            timestamp = float(frame_index) / effective_fps
            replacement.extend(self.text.encode(f"<{timestamp:.1f} seconds>"))
            replacement.append(vision_start)
            replacement.extend([self.video_token_id] * tokens_per_frame)
            replacement.append(vision_end)

        # The template may wrap the single marker in a vision span. Replace
        # that complete wrapper, matching the checkpoint processor.
        begin = start
        end = start + 1
        if begin > 0 and token_ids[begin - 1] == vision_start:
            begin -= 1
        if end < len(token_ids) and token_ids[end] == vision_end:
            end += 1
        token_ids = token_ids[:begin] + replacement + token_ids[end:]
        target = self.metadata["vision_understanding"]["routing"]["video"]
        return PreparedVideo(
            input_ids=np.asarray([token_ids], dtype=np.int64),
            pixel_values=processed.pixel_values,
            grid_thw=processed.grid_thw,
            rendered_prompt=rendered,
            token_count=processed.token_count,
            sampled_frames=len(indices),
            features_target=target,
        )

    def prepare_generator_prompt(
        self,
        prompt: str,
        *,
        mode: str | None = None,
        negative: bool = False,
        frames: int = 1,
        height: int = 0,
        width: int = 0,
        fps: float = 0.0,
        add_resolution_template: bool = False,
        add_duration_template: bool = False,
        use_system_prompt: bool = False,
        system_prompt: str | None = None,
        allow_empty: bool = False,
    ) -> NDArray[np.int64]:
        if not prompt and not allow_empty:
            raise ValueError("prompt must be non-empty")
        packer = self.generator_prompt.merged(
            self.generation_recipe(mode).get("prompt")
        )
        return packer.encode(
            prompt,
            negative=negative,
            is_image=frames == 1,
            frames=frames,
            height=height,
            width=width,
            fps=fps,
            add_resolution_template=add_resolution_template,
            add_duration_template=add_duration_template,
            use_system_prompt=use_system_prompt,
            system_prompt=system_prompt,
        )

    def prepare_world(
        self,
        prompt: str,
        *,
        frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float | None = None,
        action_steps: int = 0,
        action_domain: str = "no_action",
        include_action: bool = False,
        num_inference_steps: int | None = None,
        mode: str | None = None,
        seed: int | None = None,
        image: RawImage | None = None,
        negative_prompt: str | None = None,
        guidance_scale: float | None = None,
        add_resolution_template: bool | None = None,
        add_duration_template: bool | None = None,
        use_system_prompt: bool | None = None,
        system_prompt: str | None = None,
        conditioned_latent_frames: Sequence[int] | None = None,
        initial_vision_tokens: NDArray[np.floating[Any]] | None = None,
        initial_action_tokens: NDArray[np.floating[Any]] | None = None,
        generator_input_ids: NDArray[np.integer[Any]] | None = None,
        unconditional_input_ids: NDArray[np.integer[Any]] | None = None,
    ) -> PreparedWorldInputs:
        if mode is None and image is not None:
            mode = self._image_conditioned_mode()
        recipe = self.generation_recipe(mode)
        prompt_recipe = recipe.get("prompt", {})
        frames = int(recipe.get("frames", 5) if frames is None else frames)
        height = int(recipe.get("height", 256) if height is None else height)
        width = int(recipe.get("width", 256) if width is None else width)
        fps = float(recipe.get("fps", 24.0) if fps is None else fps)
        if add_resolution_template is None:
            add_resolution_template = bool(
                prompt_recipe.get(
                    "add_resolution_template",
                    self.generator_prompt.has_templates,
                )
            )
        if add_duration_template is None:
            add_duration_template = bool(
                prompt_recipe.get(
                    "add_duration_template",
                    self.generator_prompt.has_templates,
                )
            )
        if use_system_prompt is None:
            use_system_prompt = bool(
                prompt_recipe.get(
                    "use_system_prompt",
                    self.generator_prompt.has_system_prompts,
                )
            )
        spatial_compression = int(
            self._video_config.get("spatial_compression", 16)
        )
        temporal_compression = int(
            self._video_config.get("temporal_compression", 4)
        )
        patch_size = int(
            self.metadata.get("packing", {}).get("latent_patch_size", 2)
        )
        patch_width = int(
            self.metadata.get("packing", {}).get(
                "patch_latent_dim",
                self._generator_config.get("patch_latent_dim", 192),
            )
        )
        if min(spatial_compression, temporal_compression, patch_size) <= 0:
            raise ValueError("Invalid video compression or patch size metadata")
        if height % spatial_compression or width % spatial_compression:
            raise ValueError(
                f"height and width must be divisible by {spatial_compression}"
            )
        if frames < 1 or (frames - 1) % temporal_compression:
            raise ValueError(
                "frames must satisfy frames = temporal_compression * k + 1"
            )
        latent_frames = (frames - 1) // temporal_compression + 1
        latent_height = height // spatial_compression
        latent_width = width // spatial_compression
        if latent_height % patch_size or latent_width % patch_size:
            raise ValueError(
                "Compressed latent height and width must be divisible by patch size"
            )
        vision_count = (
            latent_frames
            * (latent_height // patch_size)
            * (latent_width // patch_size)
        )
        vision_spec = self._external_spec_by_semantic(
            "diffusion.initial_vision_latent"
        )
        vision_dtype = _numpy_dtype(
            vision_spec["dtype"] if vision_spec is not None else "FLOAT"
        )
        generator = np.random.default_rng(seed)
        if initial_vision_tokens is None:
            vision_tokens = generator.standard_normal(
                (vision_count, patch_width), dtype=np.float32
            ).astype(vision_dtype)
        else:
            vision_tokens = np.asarray(initial_vision_tokens, dtype=vision_dtype)
            if vision_tokens.shape != (vision_count, patch_width):
                raise ValueError(
                    "initial_vision_tokens has shape "
                    f"{vision_tokens.shape}, expected {(vision_count, patch_width)}"
                )

        action_input = self._external_input_by_semantic(
            "diffusion.initial_action_latent"
        )
        action_tokens = None
        if action_input is not None:
            if action_steps < 0:
                raise ValueError("action_steps must be non-negative")
            action_width = int(
                self.metadata.get("action", {}).get("padded_dimension", 64)
            )
            action_spec = self._port_spec(action_input["port"])
            action_dtype = _numpy_dtype(action_spec["dtype"])
            if not include_action:
                action_steps = 0
                action_domain = "no_action"
            raw_width = self.action_dimension(action_domain)
            if initial_action_tokens is None:
                action_tokens = np.zeros(
                    (action_steps, action_width), dtype=action_dtype
                )
                if raw_width:
                    action_tokens[:, :raw_width] = generator.standard_normal(
                        (action_steps, raw_width), dtype=np.float32
                    ).astype(action_dtype)
            else:
                provided = np.asarray(initial_action_tokens, dtype=action_dtype)
                if provided.shape == (action_steps, raw_width):
                    action_tokens = np.zeros(
                        (action_steps, action_width), dtype=action_dtype
                    )
                    action_tokens[:, :raw_width] = provided
                elif provided.shape == (action_steps, action_width):
                    action_tokens = provided.copy()
                else:
                    raise ValueError(
                        "initial_action_tokens has shape "
                        f"{provided.shape}, expected {(action_steps, raw_width)} "
                        f"or {(action_steps, action_width)}"
                    )
                if np.any(action_tokens[:, raw_width:] != 0):
                    raise ValueError(
                        "Padded action channels must be zero for the selected domain"
                    )

        options: dict[str, bool | int | float | str] = {
            "video_batch": 1,
            "video_latent_frames": latent_frames,
            "video_latent_height": latent_height,
            "video_latent_width": latent_width,
        }
        if include_action:
            options["action_domain"] = action_domain
        if num_inference_steps is not None:
            if num_inference_steps <= 0:
                raise ValueError("num_inference_steps must be positive")
            options["num_inference_steps"] = num_inference_steps
        if mode is not None:
            options["mode"] = mode
        if seed is not None:
            options["seed"] = seed
        sound_tokens = None
        sound_input = self._external_input_by_semantic(
            "diffusion.initial_sound_latent"
        )
        if sound_input is not None:
            component_name, port_name = sound_input["port"].split(".", 1)
            spec = next(
                value
                for value in self.components[component_name]["inputs"]
                if value["name"] == port_name
            )
            sound_width = spec["shape"][-1]
            if not isinstance(sound_width, int):
                raise TypeError("Sound latent width must be static")
            sound_tokens = np.zeros(
                (0, sound_width), dtype=_numpy_dtype(spec["dtype"])
            )

        input_ids = (
            self.prepare_generator_prompt(
                prompt,
                mode=mode,
                frames=frames,
                height=height,
                width=width,
                fps=fps,
                add_resolution_template=add_resolution_template,
                add_duration_template=add_duration_template,
                use_system_prompt=use_system_prompt,
                system_prompt=system_prompt,
            )
            if generator_input_ids is None
            else np.asarray(generator_input_ids, dtype=np.int64).reshape(-1)
        )

        scale = self.guidance_scale(mode=mode, requested=guidance_scale)
        if guidance_scale is not None:
            options["guidance_scale"] = float(guidance_scale)
        unconditional_name = self.unconditional_input_name()
        unconditional_ids: NDArray[np.int64] | None = None
        if unconditional_input_ids is not None:
            if unconditional_name is None:
                raise ValueError(
                    "This package declares no classifier-free guidance input, "
                    "so unconditional_input_ids cannot be used"
                )
            if scale == 1.0:
                raise ValueError(
                    "unconditional_input_ids requires a guidance scale other "
                    "than 1"
                )
            if negative_prompt is not None:
                raise ValueError(
                    "negative_prompt and unconditional_input_ids are mutually "
                    "exclusive"
                )
            unconditional_ids = np.asarray(
                unconditional_input_ids, dtype=np.int64
            ).reshape(-1)
        elif scale != 1.0:
            if unconditional_name is None:
                raise ValueError(
                    "This package declares no classifier-free guidance input, "
                    "so guidance_scale must be 1"
                )
            unconditional_ids = self.prepare_generator_prompt(
                self._resolve_negative_prompt(mode, negative_prompt),
                mode=mode,
                negative=True,
                frames=frames,
                height=height,
                width=width,
                fps=fps,
                add_resolution_template=add_resolution_template,
                add_duration_template=add_duration_template,
                use_system_prompt=use_system_prompt,
                system_prompt=system_prompt,
                allow_empty=True,
            )
        elif negative_prompt:
            raise ValueError(
                "negative_prompt requires a guidance scale other than 1"
            )

        conditioning: PreparedConditioning | None = None
        noisy_indexes: NDArray[np.int64] | None = None
        noisy_indexes_input: str | None = None
        conditioned_frames = self._conditioned_latent_frames(
            recipe, conditioned_latent_frames
        )
        if image is not None:
            if not conditioned_frames:
                raise ValueError(
                    "Image conditioning requires at least one conditioned "
                    "latent frame"
                )
            program = self.conditioning_program("vision")
            if not program:
                raise ValueError(
                    "This package declares no vision media conditioning"
                )
            invalid = [
                frame
                for frame in conditioned_frames
                if not 0 <= frame < latent_frames
            ]
            if invalid:
                raise ValueError(
                    f"Conditioned latent frames {invalid} are outside the "
                    f"{latent_frames}-frame latent video"
                )
            if conditioned_frames != tuple(range(len(conditioned_frames))):
                raise ValueError(
                    "Media conditioning can only anchor a leading run of "
                    f"latent frames, got {list(conditioned_frames)}"
                )
            grid_height = latent_height // patch_size
            grid_width = latent_width // patch_size
            self._check_conditioning_packing(program, patch_size)
            conditioning = PreparedConditioning(
                encoder_stage=str(program["encoder_stage"]),
                encoder_input=str(program["encoder_input"]),
                encoder_output=self._public_output_alias(
                    str(program["encoder_output"])
                ),                # The latent video encoder is causal in time: latent frame k
                # only sees pixel frames up to k * temporal_compression, so
                # encoding that leading run reproduces the anchored latent
                # frames of the official recipe, which repeats the same frame
                # across the whole clip.
                pixel_values=self.prepare_conditioning_frames(
                    image,
                    height=height,
                    width=width,
                    frames=max(conditioned_frames) * temporal_compression + 1,
                ),
                latent_frames=conditioned_frames,
                spatial_patch_size=patch_size,
                latent_grid=(grid_height, grid_width),
            )
            frame_tokens = grid_height * grid_width
            conditioned_rows = {
                frame * frame_tokens + offset
                for frame in conditioned_frames
                for offset in range(frame_tokens)
            }
            noisy_indexes = np.asarray(
                [
                    index
                    for index in range(vision_count)
                    if index not in conditioned_rows
                ],
                dtype=np.int64,
            )
            if noisy_indexes.size == 0:
                raise ValueError(
                    "Every latent frame is conditioned; there is nothing to "
                    "denoise"
                )
            noisy_indexes_input = str(
                program.get(
                    "timestep_token_indexes_input",
                    f"{self._generator_component()}."
                    "vision_timestep_token_indexes",
                )
            )

        return PreparedWorldInputs(
            input_ids=input_ids,
            vision_tokens=vision_tokens,
            action_tokens=action_tokens,
            sound_tokens=sound_tokens,
            options=options,
            latent_shape=(latent_frames, latent_height, latent_width),
            output_shape=(frames, height, width),
            action_domain=action_domain,
            unconditional_input_ids=unconditional_ids,
            unconditional_input_name=(
                unconditional_name if unconditional_ids is not None else None
            ),
            conditioning=conditioning,
            noisy_vision_token_indexes=noisy_indexes,
            noisy_vision_token_input=noisy_indexes_input,
        )

    def _check_conditioning_packing(
        self,
        program: Mapping[str, Any],
        patch_size: int,
    ) -> None:
        """Fail loudly when the conditioning and packing contracts disagree."""
        packing = program.get("packing", {})
        declared = packing.get("spatial_patch_size")
        if declared is not None and int(declared) != patch_size:
            raise ValueError(
                "Conditioning packing declares spatial_patch_size "
                f"{int(declared)}, but the packed generator boundary uses "
                f"{patch_size}"
            )
        temporal = packing.get("temporal_patch_size", 1)
        if int(temporal) != 1:
            raise ValueError(
                "Conditioning packing requires temporal_patch_size 1; the "
                f"package declares {int(temporal)}"
            )
        layouts = (
            packing.get("input_layout", "BCTHW"),
            packing.get("output_layout", "NC"),
            packing.get("channel_order", "patch_height_patch_width_channel"),
        )
        if layouts != ("BCTHW", "NC", "patch_height_patch_width_channel"):
            raise ValueError(
                f"Unsupported conditioning packing layout {layouts}"
            )

    def _resolve_negative_prompt(        self,
        mode: str | None,
        requested: str | None,
    ) -> str:
        """The unconditional prompt text for a mode.

        An explicit value always wins. Otherwise the recipe decides between the
        empty prompt (the default, matching the reference pipeline) and the
        package's shipped negative-prompt asset.
        """
        if requested is not None:
            return requested
        recipe = self.generation_recipe(mode)
        default = recipe.get("prompt", {}).get("negative_default", "empty")
        if default == "asset":
            return self.default_negative_prompt(mode)
        if default != "empty":
            raise ValueError(
                f"Unknown negative_default {default!r} in the {mode!r} recipe; "
                "expected 'empty' or 'asset'"
            )
        return ""

    def _image_conditioned_mode(self) -> str:
        modes = [
            name
            for name, recipe in self.generation_recipes.items()
            if isinstance(recipe, dict)
            and recipe.get("conditioning", {}).get("modality") == "image"
        ]
        if len(modes) != 1:
            raise ValueError(
                "This package declares no single image-conditioned generation "
                "recipe; pass mode= explicitly"
            )
        return modes[0]

    def _conditioned_latent_frames(
        self,
        recipe: Mapping[str, Any],
        requested: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if requested is not None:
            frames = [int(frame) for frame in requested]
        else:
            declared = recipe.get("conditioning", {}).get(
                "conditioned_latent_frames"
            )
            if declared is None:
                declared = self.conditioning_program("vision").get(
                    "default_conditioned_latent_frames", []
                )
            frames = [int(frame) for frame in declared]
        if any(frame < 0 for frame in frames):
            raise ValueError("Conditioned latent frames must be non-negative")
        if len(set(frames)) != len(frames):
            raise ValueError("Conditioned latent frames must be unique")
        return tuple(sorted(frames))

    def _generator_component(self) -> str:
        guidance = self.guidance_program()
        conditioning_input = guidance.get("conditioning_input")
        if isinstance(conditioning_input, str) and "." in conditioning_input:
            return conditioning_input.split(".", 1)[0]
        entry = self._external_input_by_semantic(
            "diffusion.initial_vision_latent"
        )
        if entry is None:
            raise ValueError("This package has no packed diffusion generator")
        return str(entry["port"]).split(".", 1)[0]

    def _public_output_alias(self, port: str) -> str:
        for output in self.manifest.get("outputs", []):
            if output.get("port") == port:
                return str(output.get("alias") or port.split(".", 1)[1])
        raise ValueError(
            f"Port {port!r} is not a public pipeline output, so its value "
            "cannot be read back from a stage"
        )

    def decode(self, token_ids: NDArray[np.integer[Any]]) -> str:
        return self.text.decode(token_ids)

    def action_dimension(self, domain: str) -> int:
        dimensions = self.metadata.get("action", {}).get("raw_dimensions", {})
        if domain not in dimensions:
            known = ", ".join(sorted(dimensions)) or "<none>"
            raise ValueError(f"Unknown action domain {domain!r}; known: {known}")
        return int(dimensions[domain])

    def _external_input_by_semantic(self, semantic: str) -> dict[str, Any] | None:
        return next(
            (
                entry
                for entry in self.manifest.get("inputs", [])
                if entry.get("kind") == "external"
                and entry.get("semantic") == semantic
            ),
            None,
        )

    def _external_spec_by_semantic(self, semantic: str) -> dict[str, Any] | None:
        entry = self._external_input_by_semantic(semantic)
        return None if entry is None else self._port_spec(entry["port"])

    def _port_spec(self, endpoint: str) -> dict[str, Any]:
        component_name, port_name = endpoint.split(".", 1)
        return next(
            value
            for value in self.components[component_name]["inputs"]
            if value["name"] == port_name
        )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_optional_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _float_triplet(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Image normalization values must contain three numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _numpy_dtype(dtype: str) -> Any:
    types: dict[str, Any] = {
        "FLOAT": np.float32,
        "FLOAT16": np.float16,
        "BFLOAT16": ml_dtypes.bfloat16,
        "DOUBLE": np.float64,
        "INT64": np.int64,
        "INT32": np.int32,
        "INT16": np.int16,
        "INT8": np.int8,
        "UINT64": np.uint64,
        "UINT32": np.uint32,
        "UINT16": np.uint16,
        "UINT8": np.uint8,
        "BOOL": np.bool_,
    }
    try:
        return types[dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported manifest dtype {dtype!r}") from error


__all__ = [
    "ImagePreprocessor",
    "PackedImage",
    "PackedImagePreprocessor",
    "PackedVideoPreprocessor",
    "PreparedReasonerInputs",
    "PreparedVideo",
    "PreparedWorldInputs",
    "RawImage",
    "RawVideo",
    "TextPreprocessor",
    "WorldModelPreprocessor",
]
