"""Smoke-check image-to-video generation against an exported package.

The dry run only exercises host preprocessing, so it works with any exported
``pipeline.json`` and needs no ONNX sessions. Without ``--dry-run`` the script
loads every component and runs the full conditioning, guided denoising, and
decode path, which requires the complete package on disk.

```bash
python tools/image_to_video_smoke.py output/cosmos3-edge frame.png \
    --prompt "A robot picks up the red block." --steps 2 --frames 5
python tools/image_to_video_smoke.py output/cosmos3-edge frame.png --dry-run
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def dry_run(
    package: Path,
    image: Path,
    *,
    prompt: str,
    negative_prompt: str | None,
    frames: int | None,
    height: int | None,
    width: int | None,
    guidance_scale: float | None,
    steps: int | None,
    seed: int | None,
) -> dict[str, Any]:
    """Prepare every host-side tensor without loading ONNX components."""
    from onnx_world_model.preprocessing import WorldModelPreprocessor

    preprocessor = WorldModelPreprocessor(package)
    prepared = preprocessor.prepare_world(
        prompt,
        image=image,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        frames=frames,
        height=height,
        width=width,
        num_inference_steps=steps,
        seed=seed,
    )
    conditioning = prepared.conditioning
    if conditioning is None:
        raise SystemExit("The package prepared no media conditioning")
    return {
        "mode": prepared.options.get("mode"),
        "output_shape": list(prepared.output_shape),
        "latent_shape": list(prepared.latent_shape),
        "vision_tokens": list(prepared.vision_tokens.shape),
        "conditioning_stage": conditioning.encoder_stage,
        "conditioning_input": conditioning.encoder_input,
        "conditioning_output": conditioning.encoder_output,
        "conditioning_pixels": list(conditioning.pixel_values.shape),
        "conditioned_latent_frames": list(conditioning.latent_frames),
        "noisy_vision_tokens": int(
            0
            if prepared.noisy_vision_token_indexes is None
            else prepared.noisy_vision_token_indexes.size
        ),
        "prompt_tokens": int(prepared.input_ids.size),
        "unconditional_prompt_tokens": (
            None
            if prepared.unconditional_input_ids is None
            else int(prepared.unconditional_input_ids.size)
        ),
        "unconditional_input": prepared.unconditional_input_name,
        "guidance_scale": preprocessor.guidance_scale(
            mode=prepared.options.get("mode"),
            requested=guidance_scale,
        ),
    }


def full_run(
    package: Path,
    image: Path,
    *,
    prompt: str,
    negative_prompt: str | None,
    frames: int | None,
    height: int | None,
    width: int | None,
    guidance_scale: float | None,
    steps: int | None,
    seed: int | None,
    providers: list[str] | None,
) -> dict[str, Any]:
    from onnx_world_model import WorldModel

    model = WorldModel.from_pretrained(package, providers=providers)
    result = model.video.generate(
        prompt,
        image=image,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        frames=frames,
        height=height,
        width=width,
        num_inference_steps=steps,
        seed=seed,
    )
    video = np.asarray(result.video)
    return {
        "video_shape": list(video.shape),
        "video_min": float(video.min()),
        "video_max": float(video.max()),
        "finite": bool(np.isfinite(video).all()),
        "timings": {name: round(value, 3) for name, value in result.timings.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="Exported pipeline directory")
    parser.add_argument("image", type=Path, help="Conditioning frame")
    parser.add_argument("--prompt", default="A robot moves forward.")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--provider", action="append", dest="providers")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    common = {
        "prompt": arguments.prompt,
        "negative_prompt": arguments.negative_prompt,
        "frames": arguments.frames,
        "height": arguments.height,
        "width": arguments.width,
        "guidance_scale": arguments.guidance_scale,
        "steps": arguments.steps,
        "seed": arguments.seed,
    }
    if arguments.dry_run:
        report = dry_run(arguments.package, arguments.image, **common)
    else:
        report = full_run(
            arguments.package,
            arguments.image,
            providers=arguments.providers,
            **common,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
