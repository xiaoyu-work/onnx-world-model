# @agent-file
# @agent-purpose: Decodes and normalizes visual inputs: it loads images and videos, resizes them onto the model grid, and packs frames into the patch-token layout the vision encoder expects.
# @agent-public-api: RawImage, RawVideo, conditioning_resample, ConditioningFramePreprocessor, ImagePreprocessor, PackedImage, PackedImagePreprocessor, PreparedVideo, PackedVideoPreprocessor
# @agent-invariants: Resized extents are always whole multiples of the patch and merge factors, so packed token counts stay exact. Only the resampling filters and the single resize strategy declared by the manifest preprocessing block are accepted; anything else raises. PackedVideoPreprocessor extends PackedImagePreprocessor and keeps its packing layout. This module must not import from preprocessing.py, which imports from it.
# @agent-side-effects: Reads image and video files from disk and decodes video frames through imageio and its ffmpeg backend.

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray
from PIL import Image

RawImage = str | os.PathLike[str] | NDArray[Any] | Image.Image
RawVideo = (
    str
    | os.PathLike[str]
    | NDArray[Any]
    | Sequence[RawImage]
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


__all__ = [
    "ConditioningFramePreprocessor",
    "ImagePreprocessor",
    "PackedImage",
    "PackedImagePreprocessor",
    "PackedVideoPreprocessor",
    "PreparedVideo",
    "RawImage",
    "RawVideo",
    "conditioning_resample",
]
