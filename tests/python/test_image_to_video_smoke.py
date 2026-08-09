"""Image-to-video smoke coverage.

The dry run is exercised against the tiny fixture package on every test run.
The full run needs a complete exported package, so it is opt-in through
``ONNX_WORLD_MODEL_EDGE_PACKAGE`` (and optionally
``ONNX_WORLD_MODEL_EDGE_IMAGE``) and is skipped otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.image_to_video_smoke import dry_run, full_run


@pytest.fixture
def conditioning_image(tmp_path: Path) -> Path:
    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    pixels[:, :8] = 255
    path = tmp_path / "frame.png"
    Image.fromarray(pixels).save(path)
    return path


def test_dry_run_reports_the_conditioned_plan(
    conditioned_package: Path,
    conditioning_image: Path,
) -> None:
    report = dry_run(
        conditioned_package,
        conditioning_image,
        prompt="a robot",
        negative_prompt=None,
        frames=None,
        height=None,
        width=None,
        guidance_scale=None,
        steps=2,
        seed=1,
    )

    assert report["mode"] == "image_to_video"
    assert report["output_shape"] == [5, 32, 32]
    assert report["latent_shape"] == [3, 4, 4]
    assert report["vision_tokens"] == [12, 16]
    assert report["conditioning_stage"] == "encode_video"
    assert report["conditioning_input"] == "video_encoder.sample"
    assert report["conditioning_output"] == "encoded_video_latent"
    assert report["conditioning_pixels"] == [1, 3, 1, 32, 32]
    assert report["conditioned_latent_frames"] == [0]
    assert report["noisy_vision_tokens"] == 8
    assert report["guidance_scale"] == 5.0
    assert report["unconditional_input"] == "unconditional:generator.input_ids"
    assert report["unconditional_prompt_tokens"] is not None


@pytest.mark.skipif(
    "ONNX_WORLD_MODEL_EDGE_PACKAGE" not in os.environ,
    reason="Set ONNX_WORLD_MODEL_EDGE_PACKAGE to a full exported package",
)
def test_exported_package_generates_video(conditioning_image: Path) -> None:
    package = Path(os.environ["ONNX_WORLD_MODEL_EDGE_PACKAGE"])
    image = Path(
        os.environ.get("ONNX_WORLD_MODEL_EDGE_IMAGE", conditioning_image)
    )

    report = full_run(
        package,
        image,
        prompt="A robot moves forward.",
        negative_prompt=None,
        frames=int(os.environ.get("ONNX_WORLD_MODEL_EDGE_FRAMES", "5")),
        height=int(os.environ.get("ONNX_WORLD_MODEL_EDGE_HEIGHT", "256")),
        width=int(os.environ.get("ONNX_WORLD_MODEL_EDGE_WIDTH", "256")),
        guidance_scale=None,
        steps=int(os.environ.get("ONNX_WORLD_MODEL_EDGE_STEPS", "1")),
        seed=1234,
        providers=None,
    )

    assert report["finite"]
    assert report["video_shape"][0] == 1
    assert report["video_shape"][1] == 3
