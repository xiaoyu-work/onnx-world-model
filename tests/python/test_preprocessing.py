from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import imageio.v3 as iio
import ml_dtypes
import numpy as np
import pytest
from jinja2.exceptions import SecurityError
from onnx_world_model import (
    WorldModel,
    generation,
    preprocessing,
)
from onnx_world_model.preprocessing import (
    ImagePreprocessor,
    TextPreprocessor,
    WorldModelPreprocessor,
)
from tokenizers import Tokenizer


def _reference_packed_tokens(
    latent: np.ndarray,
    patch: int,
) -> np.ndarray:
    """Explicit transcription of the runtime's video unpatchify inverse."""
    batch, channels, frames, height, width = latent.shape
    grid_height = height // patch
    grid_width = width // patch
    tokens = np.zeros(
        (batch * frames * grid_height * grid_width, patch * patch * channels),
        dtype=latent.dtype,
    )
    for b in range(batch):
        for frame in range(frames):
            for gy in range(grid_height):
                for gx in range(grid_width):
                    token = (
                        (b * frames + frame) * grid_height + gy
                    ) * grid_width + gx
                    for py in range(patch):
                        for px in range(patch):
                            for channel in range(channels):
                                tokens[token, (py * patch + px) * channels + channel] = (
                                    latent[
                                        b,
                                        channel,
                                        frame,
                                        gy * patch + py,
                                        gx * patch + px,
                                    ]
                                )
    return tokens


@pytest.fixture
def edge_contract_package(preprocessing_package: Path) -> Path:
    """A package whose stage options mirror the exported Edge contract."""
    path = preprocessing_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    manifest = document["manifest"]
    for component in manifest["components"]:
        if component["name"] == "generator":
            component["config"] = {"patch_latent_dim": 192}
            for value in component["inputs"]:
                if value["name"] == "vision_tokens":
                    value["shape"] = ["vision", 192]
        if component["name"] == "video_encoder":
            component["config"] = {
                "spatial_compression": 16,
                "temporal_compression": 4,
            }
            component["inputs"] = [
                {
                    "name": "sample",
                    "dtype": "FLOAT",
                    "shape": [1, 3, "frames", "height", "width"],
                }
            ]
            component["outputs"] = [
                {
                    "name": "latent",
                    "dtype": "FLOAT",
                    "shape": [1, 48, "t", "h", "w"],
                }
            ]
    manifest["inputs"].append(
        {
            "port": "video_encoder.sample",
            "kind": "external",
            "semantic": "video.frames",
            "presence": "video_conditioning",
            "required": False,
        }
    )
    manifest["outputs"] = [
        {"port": "video_encoder.latent", "alias": "encoded_video_latent"}
    ]
    manifest["stages"] = [
        {
            "name": "world_generation",
            "kind": "iterative",
            "components": ["generator"],
            "run_on": "step",
            "capabilities": [
                "loop_carried_state",
                "classifier_free_guidance",
                "conditioned_diffusion",
            ],
            "options": {
                "scheduler": {
                    "kind": "UniPCMultistepScheduler",
                    "config_asset": "scheduler/scheduler_config.json",
                    "overrideable": [
                        "num_inference_steps",
                        "guidance_scale",
                        "flow_shift",
                        "use_karras_sigmas",
                    ],
                    "mode_overrides": {
                        "image_to_video": {
                            "flow_shift": 3.0,
                            "use_karras_sigmas": False,
                            "num_inference_steps": 50,
                            "guidance_scale": 5.0,
                        },
                        "action": {
                            "flow_shift": 10.0,
                            "use_karras_sigmas": False,
                            "num_inference_steps": 30,
                            "guidance_scale": 1.0,
                        },
                    },
                },
                "guidance": {
                    "kind": "classifier_free",
                    "conditioning_input": "generator.input_ids",
                    "scale_option": "guidance_scale",
                    "default_scale": 1.0,
                    "combine": (
                        "unconditional + scale * (conditional - unconditional)"
                    ),
                },
                "conditioning": {
                    "vision": {
                        "encoder_stage": "encode_video",
                        "encoder_input": "video_encoder.sample",
                        "encoder_output": "video_encoder.latent",
                        "state": "vision_state",
                        "conditioned_latent_frames_option": (
                            "vision_conditioned_latent_frames"
                        ),
                        "default_conditioned_latent_frames": [],
                        "packing": {
                            "spatial_patch_size": 2,
                            "temporal_patch_size": 1,
                            "input_layout": "BCTHW",
                            "output_layout": "NC",
                            "channel_order": "patch_height_patch_width_channel",
                        },
                    }
                },
                "default_steps": 50,
                "timestep": {"generator": "scheduler_timesteps", "scale": 1000},
                "prediction_type": "flow_prediction",
                "state_inputs": ["generator.vision_tokens"],
                "packed_modalities": True,
            },
        },
        {
            "name": "encode_video",
            "kind": "on_demand",
            "components": ["video_encoder"],
            "run_on": "on_demand",
            "options": {"presence": "video_conditioning"},
        },
    ]
    manifest["metadata"]["packing"] = {
        "generator_boundary": "packed_tokens",
        "latent_patch_size": 2,
        "patch_latent_dim": 192,
    }
    manifest["metadata"]["generation_recipes"] = {
        "image_to_video": {
            "conditioning": {
                "modality": "image",
                "encoder_stage": "encode_video",
                "conditioned_latent_frames": [0],
            },
            "prompt": {
                "positive": "json_or_text",
                "negative_asset": "assets/negative_prompt.json",
                "add_resolution_template": False,
                "add_duration_template": False,
                "use_system_prompt": False,
            },
            "height": 480,
            "width": 832,
            "frames": 121,
            "fps": 24.0,
        }
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    assets = preprocessing_package / "assets"
    assets.mkdir(exist_ok=True)
    # The shipped Edge asset is the structured negative prompt itself.
    (assets / "negative_prompt.json").write_text(
        json.dumps({"subjects": [{"description": "flickering"}]}),
        encoding="utf-8",
    )
    return preprocessing_package


def test_matches_the_exported_edge_generation_contract(
    edge_contract_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(edge_contract_package)
    image = np.zeros((90, 160, 3), dtype=np.uint8)

    prepared = preprocessor.prepare_world("a robot", image=image, seed=0)

    assert prepared.options["mode"] == "image_to_video"
    # 832x480 at 121 frames -> 31 latent frames of 30x52, packed 15x26.
    assert prepared.output_shape == (121, 480, 832)
    assert prepared.latent_shape == (31, 30, 52)
    assert prepared.vision_tokens.shape == (31 * 15 * 26, 192)
    conditioning = prepared.conditioning
    assert conditioning is not None
    assert conditioning.pixel_values.shape == (1, 3, 1, 480, 832)
    assert conditioning.latent_grid == (15, 26)
    assert prepared.noisy_vision_token_indexes is not None
    np.testing.assert_array_equal(
        prepared.noisy_vision_token_indexes,
        np.arange(15 * 26, 31 * 15 * 26),
    )
    assert preprocessor.guidance_scale(mode="image_to_video") == 5.0
    assert preprocessor.guidance_scale(mode="action") == 1.0
    assert prepared.unconditional_input_name == (
        "unconditional:generator.input_ids"
    )
    # The Edge recipe disables the metadata sentences and the system prompt.
    np.testing.assert_array_equal(
        prepared.input_ids,
        preprocessor.prepare_generator_prompt("a robot", mode="image_to_video"),
    )


def test_structured_negative_prompt_asset_is_passed_through(
    edge_contract_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(edge_contract_package)

    negative = preprocessor.default_negative_prompt("image_to_video")

    assert json.loads(negative) == {"subjects": [{"description": "flickering"}]}
    # A structured prompt survives template application untouched.
    assert (
        preprocessor.generator_prompt.apply_templates(
            negative,
            negative=True,
            is_image=False,
            frames=121,
            height=480,
            width=832,
            fps=24.0,
            add_resolution_template=True,
            add_duration_template=True,
        )
        == negative
    )


def test_recipe_can_default_the_negative_prompt_to_the_asset(
    edge_contract_package: Path,
) -> None:
    path = edge_contract_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    recipe = document["manifest"]["metadata"]["generation_recipes"]
    recipe["image_to_video"]["prompt"]["negative_default"] = "asset"
    path.write_text(json.dumps(document), encoding="utf-8")
    preprocessor = WorldModelPreprocessor(edge_contract_package)
    image = np.zeros((90, 160, 3), dtype=np.uint8)

    prepared = preprocessor.prepare_world("a robot", image=image)

    np.testing.assert_array_equal(
        prepared.unconditional_input_ids,
        preprocessor.prepare_generator_prompt(
            preprocessor.default_negative_prompt("image_to_video"),
            mode="image_to_video",
            negative=True,
        ),
    )


def test_recipe_can_declare_generator_prompt_suffix_tokens(
    edge_contract_package: Path,
) -> None:
    path = edge_contract_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    recipe = document["manifest"]["metadata"]["generation_recipes"]
    recipe["image_to_video"]["prompt"]["suffix_token_ids"] = [2, 3]
    path.write_text(json.dumps(document), encoding="utf-8")
    preprocessor = WorldModelPreprocessor(edge_contract_package)

    tokens = preprocessor.prepare_generator_prompt(
        "a robot", mode="image_to_video"
    )

    assert tokens[-2:].tolist() == [2, 3]
    assert preprocessor.prepare_generator_prompt("a robot")[-2:].tolist() != [
        2,
        3,
    ]


def test_rejects_disagreeing_conditioning_packing(
    edge_contract_package: Path,
) -> None:
    path = edge_contract_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    stage = document["manifest"]["stages"][0]
    stage["options"]["conditioning"]["vision"]["packing"][
        "spatial_patch_size"
    ] = 4
    path.write_text(json.dumps(document), encoding="utf-8")
    preprocessor = WorldModelPreprocessor(edge_contract_package)

    with pytest.raises(ValueError, match="spatial_patch_size"):
        preprocessor.prepare_world(
            "a robot", image=np.zeros((16, 16, 3), dtype=np.uint8)
        )


def test_conditioning_preprocessing_contract_is_honored(
    conditioned_package: Path,
) -> None:
    path = conditioned_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    vision = document["manifest"]["stages"][0]["options"]["conditioning"][
        "vision"
    ]
    vision["preprocessing"] = {
        "resize": "stretch_to_target",
        "resample": "nearest",
        "convert_rgb": True,
        "rescale_factor": 1 / 255,
        "normalize": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    preprocessor = WorldModelPreprocessor(conditioned_package)
    image = np.full((16, 16, 3), 128, dtype=np.uint8)

    frames = preprocessor.prepare_conditioning_frames(
        image, height=32, width=32
    )

    assert preprocessor.conditioning is not None
    assert preprocessor.conditioning.resample == "nearest"
    # mean 0 / std 1 keeps the frame in [0, 1] instead of the signed default.
    np.testing.assert_allclose(frames, 128 / 255.0, atol=1e-6)


def test_conditioning_preprocessing_defaults_to_the_signed_range(
    conditioned_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(conditioned_package)
    image = np.full((16, 16, 3), 128, dtype=np.uint8)

    frames = preprocessor.prepare_conditioning_frames(
        image, height=32, width=32
    )

    assert preprocessor.conditioning is not None
    assert preprocessor.conditioning.resample == "bilinear"
    np.testing.assert_allclose(
        frames, (128 / 255.0 - 0.5) / 0.5, atol=1e-6
    )


def test_rejects_unsupported_conditioning_resample(
    conditioned_package: Path,
) -> None:
    path = conditioned_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    vision = document["manifest"]["stages"][0]["options"]["conditioning"][
        "vision"
    ]
    vision["preprocessing"] = {"resample": "spline"}
    path.write_text(json.dumps(document), encoding="utf-8")
    preprocessor = WorldModelPreprocessor(conditioned_package)

    with pytest.raises(ValueError, match="resample"):
        preprocessor.prepare_conditioning_frames(
            np.zeros((8, 8, 3), dtype=np.uint8), height=32, width=32
        )


def test_rejects_unsupported_conditioning_resize_strategy(
    conditioned_package: Path,
) -> None:
    path = conditioned_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    vision = document["manifest"]["stages"][0]["options"]["conditioning"][
        "vision"
    ]
    vision["preprocessing"] = {"resize": "pad_to_target"}
    path.write_text(json.dumps(document), encoding="utf-8")
    preprocessor = WorldModelPreprocessor(conditioned_package)

    with pytest.raises(ValueError, match="stretch_to_target"):
        preprocessor.prepare_conditioning_frames(
            np.zeros((8, 8, 3), dtype=np.uint8), height=32, width=32
        )


def test_generator_prompt_chat_flags_reach_the_template(
    conditioned_package: Path,
) -> None:
    (conditioned_package / "chat_template.jinja").write_text(
        "{% if add_vision_id %}<|vision_start|>{% endif %}"
        "{{ messages[-1].content }}"
        "{% if add_generation_prompt %}<|im_end|>{% endif %}",
        encoding="utf-8",
    )
    path = conditioned_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    prompt = document["manifest"]["metadata"]["generator_prompt"]
    prompt["chat"] = {
        "add_generation_prompt": False,
        "add_vision_id": True,
        "enable_thinking": True,
    }
    prompt["suffix_token_ids"] = []
    path.write_text(json.dumps(document), encoding="utf-8")
    preprocessor = WorldModelPreprocessor(conditioned_package)

    token_ids = preprocessor.prepare_generator_prompt("cat")

    # <|vision_start|> is 3 and <|im_end|> is 2 in the fixture vocabulary.
    assert token_ids.tolist() == [3, 8]
    assert preprocessor.text.render_chat(
        "cat", add_generation_prompt=True, add_vision_id=False
    ) == "cat<|im_end|>"


def test_generator_prompt_enable_thinking_reaches_template(
    conditioned_package: Path,
) -> None:
    (conditioned_package / "chat_template.jinja").write_text(
        "{{ messages[-1].content }}"
        "{% if enable_thinking %}<|vision_start|>{% endif %}",
        encoding="utf-8",
    )
    path = conditioned_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    prompt = document["manifest"]["metadata"]["generator_prompt"]
    prompt["chat"] = {"enable_thinking": True}
    prompt["suffix_token_ids"] = []
    path.write_text(json.dumps(document), encoding="utf-8")
    preprocessor = WorldModelPreprocessor(conditioned_package)

    assert preprocessor.prepare_generator_prompt("cat").tolist() == [8, 3]


def test_rejects_unsupported_generator_prompt_chat_fields(
    conditioned_package: Path,
) -> None:
    path = conditioned_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    prompt = document["manifest"]["metadata"]["generator_prompt"]
    prompt["chat"] = {"add_thinking_prompt": True}
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="add_thinking_prompt"):
        WorldModelPreprocessor(conditioned_package)


def test_unconditional_ids_require_an_active_guidance_program(
    conditioned_package: Path,
) -> None:
    guided = WorldModelPreprocessor(conditioned_package)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    ids = np.asarray([1, 2, 3], dtype=np.int64)

    with pytest.raises(ValueError, match="guidance scale"):
        guided.prepare_world("cat", unconditional_input_ids=ids)
    with pytest.raises(ValueError, match="guidance scale"):
        guided.prepare_world(
            "cat",
            image=image,
            guidance_scale=1.0,
            unconditional_input_ids=ids,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        guided.prepare_world(
            "cat",
            image=image,
            negative_prompt="blurry",
            unconditional_input_ids=ids,
        )


def test_unconditional_ids_require_a_declared_guidance_input(
    preprocessing_package: Path,
) -> None:
    unguided = WorldModelPreprocessor(preprocessing_package)

    with pytest.raises(ValueError, match="classifier-free guidance"):
        unguided.prepare_world(
            "cat",
            unconditional_input_ids=np.asarray([1, 2, 3], dtype=np.int64),
        )


def test_unconditional_ids_are_used_when_guidance_is_active(
    conditioned_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(conditioned_package)
    ids = np.asarray([1, 2, 3], dtype=np.int64)

    prepared = preprocessor.prepare_world(
        "cat",
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        unconditional_input_ids=ids,
    )

    np.testing.assert_array_equal(prepared.unconditional_input_ids, ids)
    np.testing.assert_array_equal(
        prepared.pipeline_inputs()["unconditional:generator.input_ids"], ids
    )


def test_packs_latents_in_runtime_token_order() -> None:
    latent = np.arange(1 * 4 * 2 * 4 * 4, dtype=np.float32).reshape(
        1, 4, 2, 4, 4
    )

    packed = preprocessing.pack_latent_tokens(latent, 2)

    assert packed.shape == (8, 16)
    np.testing.assert_array_equal(packed, _reference_packed_tokens(latent, 2))


def test_rejects_latents_that_do_not_tile() -> None:
    with pytest.raises(ValueError, match="divisible"):
        preprocessing.pack_latent_tokens(
            np.zeros((1, 4, 1, 3, 4), dtype=np.float32), 2
        )


def test_prepares_image_to_video_inputs(conditioned_package: Path) -> None:
    preprocessor = WorldModelPreprocessor(conditioned_package)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :8] = 255

    prepared = preprocessor.prepare_world("cat", image=image, seed=5)

    assert prepared.options["mode"] == "image_to_video"
    assert prepared.output_shape == (5, 32, 32)
    assert prepared.latent_shape == (3, 4, 4)
    conditioning = prepared.conditioning
    assert conditioning is not None
    assert conditioning.encoder_stage == "encode_video"
    assert conditioning.encoder_output == "encoded_video_latent"
    assert conditioning.latent_frames == (0,)
    assert conditioning.pixel_values.shape == (1, 3, 1, 32, 32)
    assert conditioning.pixel_values.min() == pytest.approx(-1.0)
    assert conditioning.pixel_values.max() == pytest.approx(1.0)
    assert list(conditioning.pipeline_inputs()) == ["video_encoder.sample"]
    assert prepared.noisy_vision_token_indexes is not None
    np.testing.assert_array_equal(
        prepared.noisy_vision_token_indexes, np.arange(4, 12)
    )
    overrides = prepared.pipeline_overrides()
    assert list(overrides) == ["generator.vision_timestep_token_indexes"]
    np.testing.assert_array_equal(
        overrides["generator.vision_timestep_token_indexes"],
        prepared.noisy_vision_token_indexes,
    )
    assert prepared.unconditional_input_name == (
        "unconditional:generator.input_ids"
    )
    assert prepared.unconditional_input_ids is not None
    assert (
        "unconditional:generator.input_ids" in prepared.pipeline_inputs()
    )


def test_conditioned_latent_anchors_only_its_frames(
    conditioned_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(conditioned_package)
    prepared = preprocessor.prepare_world(
        "cat",
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        seed=5,
    )
    latent = np.arange(1 * 4 * 1 * 4 * 4, dtype=np.float32).reshape(
        1, 4, 1, 4, 4
    )

    conditioned = prepared.with_conditioning_latent(latent)

    expected = _reference_packed_tokens(latent, 2)
    np.testing.assert_array_equal(conditioned.vision_tokens[:4], expected)
    np.testing.assert_array_equal(
        conditioned.vision_tokens[4:], prepared.vision_tokens[4:]
    )
    # The noise of the conditioned frame is replaced, not blended.
    assert not np.array_equal(
        conditioned.vision_tokens[:4], prepared.vision_tokens[:4]
    )


def test_rejects_mismatched_conditioning_latent(
    conditioned_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(conditioned_package)
    prepared = preprocessor.prepare_world(
        "cat",
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        seed=5,
    )

    with pytest.raises(ValueError, match="latent frames"):
        prepared.with_conditioning_latent(
            np.zeros((1, 4, 2, 4, 4), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="channels"):
        prepared.with_conditioning_latent(
            np.zeros((1, 2, 1, 4, 4), dtype=np.float32)
        )


def test_conditioned_frames_must_be_a_leading_run(
    conditioned_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(conditioned_package)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="leading run"):
        preprocessor.prepare_world(
            "cat", image=image, conditioned_latent_frames=[1]
        )
    with pytest.raises(ValueError, match="outside"):
        preprocessor.prepare_world(
            "cat", image=image, conditioned_latent_frames=[0, 1, 2, 3]
        )
    with pytest.raises(ValueError, match="nothing to denoise"):
        preprocessor.prepare_world(
            "cat", image=image, conditioned_latent_frames=[0, 1, 2]
        )


def test_two_conditioned_frames_encode_the_matching_clip(
    conditioned_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(conditioned_package)

    prepared = preprocessor.prepare_world(
        "cat",
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        conditioned_latent_frames=[0, 1],
    )

    assert prepared.conditioning is not None
    # Two latent frames need temporal_compression * 1 + 1 pixel frames.
    assert prepared.conditioning.pixel_values.shape == (1, 3, 3, 32, 32)
    np.testing.assert_array_equal(
        prepared.noisy_vision_token_indexes, np.arange(8, 12)
    )


def test_guidance_scale_follows_the_manifest_mode(
    conditioned_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(conditioned_package)

    assert preprocessor.guidance_scale(mode="image_to_video") == 5.0
    assert preprocessor.guidance_scale(mode="action") == 1.0
    assert preprocessor.guidance_scale(mode=None) == 1.0
    assert (
        preprocessor.guidance_scale(mode="image_to_video", requested=2.5)
        == 2.5
    )
    assert preprocessor.default_negative_prompt("image_to_video") == "blurry"


def test_unconditional_prompt_defaults_to_empty_text(
    conditioned_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(conditioned_package)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    prepared = preprocessor.prepare_world("cat", image=image)
    empty = preprocessor.prepare_generator_prompt("", allow_empty=True)

    assert prepared.unconditional_input_ids is not None
    np.testing.assert_array_equal(prepared.unconditional_input_ids, empty)

    negative = preprocessor.prepare_world(
        "cat", image=image, negative_prompt="cat"
    )
    assert negative.unconditional_input_ids is not None
    assert not np.array_equal(negative.unconditional_input_ids, empty)


def test_unguided_generation_rejects_a_negative_prompt(
    conditioned_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(conditioned_package)

    with pytest.raises(ValueError, match="guidance scale"):
        preprocessor.prepare_world("cat", negative_prompt="blurry")


def test_guidance_requires_a_declared_program(
    preprocessing_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(preprocessing_package)

    assert preprocessor.unconditional_input_name() is None
    with pytest.raises(ValueError, match="classifier-free guidance"):
        preprocessor.prepare_world("cat", guidance_scale=5.0)


def test_image_conditioning_requires_a_declared_recipe(
    preprocessing_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(preprocessing_package)

    with pytest.raises(ValueError, match="image-conditioned"):
        preprocessor.prepare_world(
            "cat", image=np.zeros((8, 8, 3), dtype=np.uint8)
        )


def test_generator_prompt_appends_manifest_suffix_tokens(
    conditioned_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(conditioned_package)

    token_ids = preprocessor.prepare_generator_prompt("cat")

    assert token_ids[-2:].tolist() == [2, 3]
    assert preprocessor.generator_prompt.system_prompt(is_image=False) == (
        "system video"
    )


def test_undeclared_system_prompt_is_reported(
    preprocessing_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(preprocessing_package)

    assert preprocessor.prepare_generator_prompt("cat")[-1] != 3
    with pytest.raises(ValueError, match="system prompt"):
        preprocessor.prepare_generator_prompt("cat", use_system_prompt=True)


def test_metadata_templates_are_applied_per_polarity(
    conditioned_package: Path,
) -> None:
    packer = WorldModelPreprocessor(conditioned_package).generator_prompt

    positive = packer.apply_templates(
        "a robot",
        negative=False,
        is_image=False,
        frames=121,
        height=480,
        width=832,
        fps=24.0,
        add_resolution_template=True,
        add_duration_template=True,
    )
    negative = packer.apply_templates(
        "",
        negative=True,
        is_image=False,
        frames=121,
        height=480,
        width=832,
        fps=24.0,
        add_resolution_template=True,
        add_duration_template=True,
    )
    image = packer.apply_templates(
        "a robot",
        negative=False,
        is_image=True,
        frames=1,
        height=480,
        width=832,
        fps=24.0,
        add_resolution_template=True,
        add_duration_template=True,
    )

    assert positive == (
        "a robot. The video is 5.0 seconds long and is of 24 FPS. "
        "This video is of 480x832 resolution."
    )
    assert negative == (
        "The video is not 5.0 seconds long and is not of 24 FPS. "
        "This video is not of 480x832 resolution."
    )
    assert image == "a robot. This image is of 480x832 resolution."


def test_structured_prompts_are_never_rewritten(
    conditioned_package: Path,
) -> None:
    packer = WorldModelPreprocessor(conditioned_package).generator_prompt
    prompt = json.dumps({"actions": [{"description": "a robot moves."}]})

    assert (
        packer.apply_templates(
            prompt,
            negative=False,
            is_image=False,
            frames=121,
            height=480,
            width=832,
            fps=24.0,
            add_resolution_template=True,
            add_duration_template=True,
        )
        == prompt
    )


def test_recipe_disables_templates_by_default(
    conditioned_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(conditioned_package)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    prepared = preprocessor.prepare_world("cat", image=image)
    plain = preprocessor.prepare_generator_prompt("cat")
    templated = preprocessor.prepare_world(
        "cat",
        image=image,
        add_resolution_template=True,
    )

    np.testing.assert_array_equal(prepared.input_ids, plain)
    assert not np.array_equal(templated.input_ids, plain)


def test_missing_template_reports_the_expected_manifest_key(
    preprocessing_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(preprocessing_package)

    with pytest.raises(ValueError, match="video_resolution"):
        preprocessor.prepare_generator_prompt(
            "cat",
            frames=5,
            height=32,
            width=32,
            fps=24.0,
            add_resolution_template=True,
        )


def test_prepares_text_and_image(
    preprocessing_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(preprocessing_package)
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[:, :20, 0] = 255
    image[:, 20:, 2] = 255

    prepared = preprocessor.prepare_reasoner("cat", image)

    assert prepared.pixel_values is not None
    assert prepared.pixel_values.shape == (1, 3, 32, 32)
    assert prepared.pixel_values.dtype == np.float32
    assert prepared.image_token_count == 4
    assert np.count_nonzero(prepared.input_ids == 4) == 4
    assert prepared.pipeline_inputs()["text.token_ids"].ndim == 2


def test_prepares_deterministic_world_inputs(
    preprocessing_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(preprocessing_package)

    first = preprocessor.prepare_world(
        "cat",
        frames=5,
        height=32,
        width=32,
        action_steps=2,
        action_domain="robot",
        include_action=True,
        seed=42,
    )
    second = preprocessor.prepare_world(
        "cat",
        frames=5,
        height=32,
        width=32,
        action_steps=2,
        action_domain="robot",
        include_action=True,
        seed=42,
    )

    assert first.latent_shape == (3, 4, 4)
    assert first.vision_tokens.shape == (12, 16)
    assert first.action_tokens is not None
    assert first.action_tokens.shape == (2, 8)
    np.testing.assert_array_equal(first.vision_tokens, second.vision_tokens)
    np.testing.assert_array_equal(first.action_tokens, second.action_tokens)
    assert np.count_nonzero(first.action_tokens[:, 3:]) == 0
    assert preprocessor.action_dimension("robot") == 3
    assert first.input_ids[0] == 1


def test_rejects_invalid_world_geometry(
    preprocessing_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(preprocessing_package)

    with pytest.raises(ValueError, match="frames must satisfy"):
        preprocessor.prepare_world("cat", frames=4, height=32, width=32)


def test_video_only_world_input_uses_empty_no_action_state(
    preprocessing_package: Path,
) -> None:
    preprocessor = WorldModelPreprocessor(preprocessing_package)

    prepared = preprocessor.prepare_world(
        "cat",
        frames=5,
        height=32,
        width=32,
        seed=3,
    )

    assert prepared.action_tokens is not None
    assert prepared.action_tokens.shape == (0, 8)
    assert "action_domain" not in prepared.options


def test_float_image_ranges_are_distinguished() -> None:
    processor = ImagePreprocessor(
        height=1,
        width=1,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    byte_image = np.full((1, 1, 3), 128, dtype=np.uint8)
    float_byte_image = byte_image.astype(np.float32)

    np.testing.assert_allclose(
        processor(byte_image),
        processor(float_byte_image),
    )
    with pytest.raises(ValueError, match="finite"):
        processor(np.full((1, 1, 3), np.nan, dtype=np.float32))


def test_chat_template_is_sandboxed(preprocessing_package: Path) -> None:
    (preprocessing_package / "chat_template.jinja").write_text(
        "{{ ''.__class__.__mro__ }}",
        encoding="utf-8",
    )
    processor = TextPreprocessor(preprocessing_package)

    with pytest.raises(SecurityError):
        processor.render_chat("cat")


def test_text_only_package_does_not_require_vision_assets(
    preprocessing_package: Path,
) -> None:
    path = preprocessing_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    manifest = document["manifest"]
    manifest["components"] = [
        component
        for component in manifest["components"]
        if component["name"] != "reasoner_vision_encoder"
    ]
    manifest["inputs"] = [
        entry
        for entry in manifest["inputs"]
        if entry.get("semantic") != "vision.pixel_values"
    ]
    path.write_text(json.dumps(document), encoding="utf-8")

    preprocessor = WorldModelPreprocessor(preprocessing_package)
    prepared = preprocessor.prepare_reasoner("cat")

    assert prepared.pixel_values is None
    with pytest.raises(ValueError, match="no raw-image"):
        preprocessor.prepare_reasoner(
            "cat", np.zeros((4, 4, 3), dtype=np.uint8)
        )


def test_sound_capable_package_gets_empty_sound_latent(
    preprocessing_package: Path,
) -> None:
    path = preprocessing_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    generator = next(
        component
        for component in document["manifest"]["components"]
        if component["name"] == "generator"
    )
    generator["inputs"].append(
        {"name": "sound_tokens", "dtype": "FLOAT", "shape": ["sound", 6]}
    )
    document["manifest"]["inputs"].append(
        {
            "port": "generator.sound_tokens",
            "kind": "external",
            "semantic": "diffusion.initial_sound_latent",
        }
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    preprocessor = WorldModelPreprocessor(preprocessing_package)

    prepared = preprocessor.prepare_world(
        "cat", frames=5, height=32, width=32
    )

    assert prepared.sound_tokens is not None
    assert prepared.sound_tokens.shape == (0, 6)


@pytest.mark.parametrize(
    ("manifest_dtype", "expected_dtype"),
    [
        ("FLOAT16", np.dtype(np.float16)),
        ("BFLOAT16", np.dtype(ml_dtypes.bfloat16)),
    ],
)
def test_preprocessing_follows_manifest_float_dtype(
    preprocessing_package: Path,
    manifest_dtype: str,
    expected_dtype: np.dtype[Any],
) -> None:
    path = preprocessing_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    for component in document["manifest"]["components"]:
        for spec in [*component["inputs"], *component["outputs"]]:
            if spec["dtype"] == "FLOAT":
                spec["dtype"] = manifest_dtype
    path.write_text(json.dumps(document), encoding="utf-8")
    preprocessor = WorldModelPreprocessor(preprocessing_package)

    reasoner = preprocessor.prepare_reasoner(
        "cat", np.zeros((32, 32, 3), dtype=np.uint8)
    )
    world = preprocessor.prepare_world(
        "cat",
        frames=5,
        height=32,
        width=32,
        action_steps=1,
        action_domain="robot",
        include_action=True,
    )

    assert reasoner.pixel_values is not None
    assert reasoner.pixel_values.dtype == expected_dtype
    assert world.vision_tokens.dtype == expected_dtype
    assert world.action_tokens is not None
    assert world.action_tokens.dtype == expected_dtype
    empty_features = preprocessor.empty_image_features()
    assert empty_features is not None
    assert empty_features.dtype == expected_dtype


def test_packed_vision_signature_is_lazy(
    preprocessing_package: Path,
) -> None:
    path = preprocessing_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    vision = next(
        component
        for component in document["manifest"]["components"]
        if component["name"] == "reasoner_vision_encoder"
    )
    vision["inputs"][0]["shape"] = ["patches", 768]
    path.write_text(json.dumps(document), encoding="utf-8")

    preprocessor = WorldModelPreprocessor(preprocessing_package)
    assert preprocessor.prepare_reasoner("cat").pixel_values is None
    with pytest.raises(ValueError, match="no supported preprocessing"):
        preprocessor.prepare_reasoner(
            "cat", np.zeros((32, 32, 3), dtype=np.uint8)
        )


def test_prepares_supported_packed_vision_input(
    preprocessing_package: Path,
) -> None:
    path = preprocessing_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    vision = next(
        component
        for component in document["manifest"]["components"]
        if component["name"] == "reasoner_vision_encoder"
    )
    vision["inputs"][0]["shape"] = ["patches", 3 * 4 * 4]
    vision["outputs"][0]["shape"] = ["features", 8]
    document["manifest"]["inputs"].append(
        {
            "port": "reasoner_vision_encoder.grid_thw",
            "kind": "external",
            "semantic": "vision.grid_thw",
        }
    )
    vision["inputs"].append(
        {"name": "grid_thw", "dtype": "INT64", "shape": [3]}
    )
    document["manifest"]["metadata"]["vision_understanding"] = {
        "preprocessing": {
            "image_processor_asset": "preprocessor_config.json",
            "resize": (
                "smart_resize_area_bounded_multiple_of_patch_times_merge"
            ),
            "normalize": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "patchify": {
                "layout": "time_major_block_major",
                "patch_value_order": "patch_height_patch_width_channel",
                "patch_size": 4,
                "merge_size": 2,
                "temporal_patch_size": 1,
            },
        }
    }
    (preprocessing_package / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "size": {
                    "shortest_edge": 16 * 16,
                    "longest_edge": 64 * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    preprocessor = WorldModelPreprocessor(preprocessing_package)

    prepared = preprocessor.prepare_reasoner(
        "cat", np.zeros((16, 32, 3), dtype=np.uint8)
    )

    assert prepared.grid_thw is not None
    assert prepared.grid_thw.tolist() == [1, 4, 8]
    assert prepared.pixel_values is not None
    assert prepared.pixel_values.shape == (32, 48)
    assert prepared.image_token_count == 8
    assert np.count_nonzero(prepared.input_ids == 4) == 8
    assert "vision.grid_thw" in prepared.pipeline_inputs()


def test_prepares_video_understanding_input(
    preprocessing_package: Path,
) -> None:
    path = preprocessing_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    vision = next(
        component
        for component in document["manifest"]["components"]
        if component["name"] == "reasoner_vision_encoder"
    )
    vision["inputs"] = [
        {"name": "pixel_values", "dtype": "FLOAT", "shape": ["patches", 48]},
        {"name": "grid_thw", "dtype": "INT64", "shape": [3]},
    ]
    vision["outputs"][0]["shape"] = ["features", 8]
    document["manifest"]["inputs"].append(
        {
            "port": "reasoner_vision_encoder.grid_thw",
            "kind": "external",
            "semantic": "vision.grid_thw",
        }
    )
    document["manifest"]["metadata"]["vision_understanding"] = {
        "routing": {
            "image": "reasoner_embedding.image_features",
            "video": "reasoner_embedding.video_features",
        },
        "preprocessing": {
            "image_processor_asset": "preprocessor_config.json",
            "video_processor_asset": "video_preprocessor_config.json",
            "resize": (
                "smart_resize_area_bounded_multiple_of_patch_times_merge"
            ),
            "normalize": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "video_frame_sampling": {
                "fps": 2,
                "min_frames": 4,
                "max_frames": 8,
            },
            "patchify": {
                "layout": "time_major_block_major",
                "patch_value_order": "patch_height_patch_width_channel",
                "patch_size": 4,
                "merge_size": 2,
                "temporal_patch_size": 1,
            },
        },
    }
    for filename, shortest, longest in (
        ("preprocessor_config.json", 16 * 16, 64 * 64),
        ("video_preprocessor_config.json", 4 * 16 * 16, 8 * 64 * 64),
    ):
        (preprocessing_package / filename).write_text(
            json.dumps(
                {
                    "size": {
                        "shortest_edge": shortest,
                        "longest_edge": longest,
                    }
                }
            ),
            encoding="utf-8",
        )
    (preprocessing_package / "config.json").write_text(
        json.dumps(
            {
                "image_token_id": 4,
                "video_token_id": 10,
                "vision_start_token_id": 3,
                "vision_end_token_id": 5,
            }
        ),
        encoding="utf-8",
    )
    tokenizer = Tokenizer.from_file(
        str(preprocessing_package / "tokenizer.json")
    )
    tokenizer.add_tokens(["<|video_pad|>", "<0.0 seconds>"])
    tokenizer.save(str(preprocessing_package / "tokenizer.json"))
    template = (
        preprocessing_package / "chat_template.jinja"
    ).read_text(encoding="utf-8")
    template = template.replace(
        "{% if messages[-1].content is string %}",
        "{% if messages[-1].content is string %}",
    ).replace(
        "{% if item.type == 'image' %}<|vision_start|><|image_pad|>"
        "<|vision_end|>{% elif item.type == 'text' %}",
        "{% if item.type == 'image' %}<|vision_start|><|image_pad|>"
        "<|vision_end|>{% elif item.type == 'video' %}"
        "<|vision_start|><|video_pad|><|vision_end|>"
        "{% elif item.type == 'text' %}",
    )
    (preprocessing_package / "chat_template.jinja").write_text(
        template, encoding="utf-8"
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    preprocessor = WorldModelPreprocessor(preprocessing_package)
    frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)

    prepared = preprocessor.prepare_video_reasoner(
        "cat", frames, source_fps=2, num_frames=4
    )

    assert prepared.sampled_frames == 4
    assert prepared.grid_thw.tolist() == [4, 4, 4]
    assert prepared.pixel_values.shape == (64, 48)
    assert prepared.token_count == 16
    assert prepared.features_target == "reasoner_embedding.video_features"

    video_path = preprocessing_package / "test.mp4"
    iio.imwrite(video_path, frames, fps=2)
    from_path = preprocessor.prepare_video_reasoner(
        "cat", video_path, num_frames=4
    )
    assert from_path.sampled_frames == 4
    assert from_path.grid_thw.tolist() == [4, 4, 4]
    with pytest.raises(ValueError, match="exceeds source frame count"):
        preprocessor.prepare_video_reasoner(
            "cat", frames[:2], source_fps=2, num_frames=4
        )


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.released: list[str] = []

    def run_stage(
        self,
        stage: str,
        inputs: dict[str, np.ndarray] | None = None,
        *,
        overrides: dict[str, np.ndarray] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        self.calls.append(
            (
                stage,
                {
                    "inputs": inputs,
                    "overrides": overrides,
                    "options": options,
                },
            )
        )
        if stage == "reasoner_prompt":
            return {}
        if stage == "reasoner_decode":
            return {"generated_token_ids": np.asarray([[9]], dtype=np.int64)}
        if stage == "world_generation":
            return {
                "action": np.arange(8, dtype=np.float32)[None],
                "action_velocity": np.ones((1, 8), dtype=np.float32),
                "vision_velocity": np.ones((12, 16), dtype=np.float32),
            }
        if stage == "encode_video":
            return {
                "encoded_video_latent": np.full(
                    (1, 4, 1, 4, 4), 0.25, dtype=np.float32
                )
            }
        if stage == "decode_video":
            latent_frames = int((options or {}).get("video_latent_frames", 3))
            frames = (latent_frames - 1) * 2 + 1
            return {
                "video": np.zeros(
                    (1, 3, frames, 32, 32), dtype=np.float32
                )
            }
        raise AssertionError(f"Unexpected stage {stage}")

    def release_stage(self, stage: str) -> None:
        self.released.append(stage)


class _FakePipeline:
    last_instance: _FakePipeline | None = None

    def __init__(self, package_path: Path, **options: Any) -> None:
        self.package_path = package_path
        self.options = options
        self.profile = {"name": "test-world", "version": "1.0"}
        self.execution_providers = {"generator": ("CPUExecutionProvider",)}
        self.stages = (
            SimpleNamespace(
                name="reasoner_prompt",
                kind="single_pass",
                run_on="prefill",
            ),
            SimpleNamespace(
                name="reasoner_decode",
                kind="autoregressive",
                run_on="decode",
            ),
            SimpleNamespace(
                name="world_generation",
                kind="iterative",
                run_on="step",
            ),
            SimpleNamespace(
                name="encode_video",
                kind="on_demand",
                run_on="on_demand",
            ),
            SimpleNamespace(
                name="decode_video",
                kind="single_pass",
                run_on="finalize",
            ),
        )
        self.session = _FakeSession()
        _FakePipeline.last_instance = self

    def create_session(self) -> _FakeSession:
        return self.session


class _FakeWorldOnlyPipeline(_FakePipeline):
    def __init__(self, package_path: Path, **options: Any) -> None:
        super().__init__(package_path, **options)
        self.stages = (
            SimpleNamespace(
                name="world_generation",
                kind="iterative",
                run_on="step",
            ),
            SimpleNamespace(
                name="decode_video",
                kind="single_pass",
                run_on="finalize",
            ),
        )


@pytest.fixture
def world_model(
    preprocessing_package: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> WorldModel:
    monkeypatch.setattr(generation, "Pipeline", _FakePipeline)
    return WorldModel(
        preprocessing_package,
        providers=["cpu"],
        provider_options={"cpu": {"use_arena": False}},
    )


def test_llm_generate_preprocesses_and_decodes(
    world_model: WorldModel,
) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    result = world_model.text.generate(
        "cat",
        image,
        max_tokens=2,
    )

    assert result.text == "answer"
    assert result.token_ids.tolist() == [[9]]
    fake = _FakePipeline.last_instance
    assert fake is not None
    assert fake.options["providers"] == ["cpu"]
    assert fake.session.released == ["reasoner_decode"]
    reasoner_inputs = fake.session.calls[0][1]["inputs"]
    assert reasoner_inputs["vision.pixel_values"].shape == (1, 3, 32, 32)


def test_llm_generate_accepts_video(
    preprocessing_package: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reuse the packed video contract fixture setup.
    test_prepares_video_understanding_input(preprocessing_package)
    monkeypatch.setattr(generation, "Pipeline", _FakePipeline)
    model = WorldModel(preprocessing_package)
    frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)

    output = model.text.generate(
        "cat",
        video=frames,
        video_fps=2,
        video_num_frames=4,
        max_tokens=2,
    )

    assert output.text == "answer"
    fake = _FakePipeline.last_instance
    assert fake is not None
    prefill = fake.session.calls[0]
    assert prefill[0] == "reasoner_prompt"
    assert prefill[1]["inputs"]["vision.pixel_values"].shape == (64, 48)
    assert prefill[1]["inputs"]["vision.grid_thw"].tolist() == [4, 4, 4]
    assert (
        prefill[1]["overrides"]["reasoner_embedding.image_features"].shape
        == (0, 8)
    )
    assert prefill[1]["options"] == {"vision_modality": "video"}


def test_video_and_action_generators_return_modality_outputs(
    world_model: WorldModel,
) -> None:
    video = world_model.video.generate(
        "cat",
        frames=5,
        height=32,
        width=32,
        num_inference_steps=2,
        seed=7,
    )
    action = world_model.action.generate(
        "cat",
        domain="robot",
        steps=1,
        video_frames=5,
        video_height=32,
        video_width=32,
        num_inference_steps=2,
        seed=7,
    )

    assert video.video.shape == (1, 3, 5, 32, 32)
    assert action.actions.shape == (1, 3)
    fake = _FakePipeline.last_instance
    assert fake is not None
    assert fake.session.released == [
        "world_generation",
        "decode_video",
        "world_generation",
    ]


@pytest.fixture
def conditioned_world_model(
    conditioned_package: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> WorldModel:
    monkeypatch.setattr(generation, "Pipeline", _FakePipeline)
    return WorldModel(conditioned_package)


def test_video_generator_conditions_and_guides(
    conditioned_world_model: WorldModel,
) -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :8] = 255

    result = conditioned_world_model.video.generate(
        "cat",
        image=image,
        negative_prompt="cat",
        seed=3,
    )

    fake = _FakePipeline.last_instance
    assert fake is not None
    assert [call[0] for call in fake.session.calls] == [
        "encode_video",
        "world_generation",
        "decode_video",
    ]
    encode_inputs = fake.session.calls[0][1]["inputs"]
    assert encode_inputs["video_encoder.sample"].shape == (1, 3, 1, 32, 32)
    world = fake.session.calls[1][1]
    assert "unconditional:generator.input_ids" in world["inputs"]
    assert list(world["overrides"]) == [
        "generator.vision_timestep_token_indexes"
    ]
    np.testing.assert_array_equal(
        world["overrides"]["generator.vision_timestep_token_indexes"],
        np.arange(4, 12),
    )
    assert world["options"]["mode"] == "image_to_video"
    # Latent frame 0 carries the encoded conditioning frame.
    np.testing.assert_allclose(
        world["inputs"]["diffusion.initial_vision_latent"][:4],
        np.full((4, 16), 0.25, dtype=np.float32),
    )
    assert result.video.shape == (1, 3, 5, 32, 32)
    assert fake.session.released == [
        "encode_video",
        "world_generation",
        "decode_video",
    ]
    assert set(result.timings) == {"encode", "generate", "decode"}


def test_video_generator_can_disable_guidance(
    conditioned_world_model: WorldModel,
) -> None:
    result = conditioned_world_model.video.generate(
        "cat",
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        guidance_scale=1.0,
    )

    fake = _FakePipeline.last_instance
    assert fake is not None
    world = fake.session.calls[1][1]
    assert "unconditional:generator.input_ids" not in world["inputs"]
    assert world["options"]["guidance_scale"] == 1.0
    assert result.video.shape == (1, 3, 5, 32, 32)


def test_text_to_video_stays_unconditioned(
    conditioned_world_model: WorldModel,
) -> None:
    conditioned_world_model.video.generate(
        "cat",
        frames=5,
        height=32,
        width=32,
        num_inference_steps=2,
    )

    fake = _FakePipeline.last_instance
    assert fake is not None
    assert [call[0] for call in fake.session.calls] == [
        "world_generation",
        "decode_video",
    ]
    world = fake.session.calls[0][1]
    assert world["overrides"] is None
    assert "mode" not in world["options"]
    assert "unconditional:generator.input_ids" not in world["inputs"]


def test_image_generator_returns_one_frame(
    world_model: WorldModel,
) -> None:
    image = world_model.image.generate(
        "cat",
        height=32,
        width=32,
        num_inference_steps=1,
    )

    assert image.images.shape == (1, 3, 32, 32)


def test_public_api_uses_modality_names(world_model: WorldModel) -> None:
    assert world_model.capabilities == ("text", "image", "video", "action")
    assert not hasattr(world_model, "llm")
    assert not hasattr(world_model, "respond")
    assert not hasattr(world_model, "generate_world")


def test_world_only_pipeline_loads_without_reasoner_stages(
    preprocessing_package: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation, "Pipeline", _FakeWorldOnlyPipeline)

    model = WorldModel(preprocessing_package)
    result = model.video.generate(
        "cat",
        frames=5,
        height=32,
        width=32,
        num_inference_steps=1,
    )

    assert result.video.shape == (1, 3, 5, 32, 32)
