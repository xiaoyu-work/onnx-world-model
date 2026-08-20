# @agent-file
# @agent-purpose: Provides the shared pytest fixtures that build throwaway Mobius packages, tokenizers, chat templates, and conditioning configuration for the Python test suite.
# @agent-public-api: world_model_path, pipeline_path, preprocessing_package, conditioned_package
# @agent-invariants: `world_model_path` and `pipeline_path` are session scoped and call `pytest.importorskip("mobius")`, so the suite skips instead of failing when the exporter is absent. `preprocessing_package` and `conditioned_package` build a self-contained package under `tmp_path` with a WordLevel tokenizer whose special tokens match the chat template, so they need no network access and no exporter.
# @agent-side-effects: Writes tokenizer, chat-template, and package configuration files into pytest temporary directories, and imports `tools.export_mobius_test_model` when the exporter fixtures are used.

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


@pytest.fixture(scope="session")
def world_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("mobius")
    from tools.export_mobius_test_model import export_test_model

    return export_test_model(tmp_path_factory.mktemp("world-model"))


@pytest.fixture(scope="session")
def pipeline_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("mobius")
    from tools.export_mobius_test_model import export_test_pipeline

    return export_test_pipeline(tmp_path_factory.mktemp("pipeline"))


@pytest.fixture
def preprocessing_package(tmp_path: Path) -> Path:
    vocabulary = {
        "<unk>": 0,
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|vision_start|>": 3,
        "<|image_pad|>": 4,
        "<|vision_end|>": 5,
        "user": 6,
        "assistant": 7,
        "cat": 8,
        "answer": 9,
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(
        [
            "<unk>",
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
        ]
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "chat_template.jinja").write_text(
        """<|im_start|>user
{% if messages[-1].content is string %}{{ messages[-1].content }}{% else %}
{% for item in messages[-1].content %}{% if item.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% elif item.type == 'text' %}{{ item.text }}{% endif %}{% endfor %}
{% endif %}<|im_end|>
<|im_start|>assistant
""",
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"image_token_id": 4}),
        encoding="utf-8",
    )
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            }
        ),
        encoding="utf-8",
    )
    document = {
        "format": "mobius-pipeline",
        "schema_version": "1.1",
        "manifest": {
            "schema_version": "1.1",
            "components": [
                {
                    "name": "generator",
                    "role": "dynamics",
                    "run_on": "always",
                    "inputs": [
                        {
                            "name": "input_ids",
                            "dtype": "INT64",
                            "shape": ["text"],
                        },
                        {
                            "name": "vision_tokens",
                            "dtype": "FLOAT",
                            "shape": ["vision", 16],
                        },
                        {
                            "name": "action_tokens",
                            "dtype": "FLOAT",
                            "shape": ["action", 8],
                        },
                    ],
                    "outputs": [],
                    "config": {"patch_latent_dim": 16},
                },
                {
                    "name": "reasoner_embedding",
                    "role": "embedding",
                    "run_on": "always",
                    "inputs": [
                        {
                            "name": "input_ids",
                            "dtype": "INT64",
                            "shape": ["batch", "sequence"],
                        },
                        {
                            "name": "image_features",
                            "dtype": "FLOAT",
                            "shape": ["features", 8],
                        },
                        {
                            "name": "video_features",
                            "dtype": "FLOAT",
                            "shape": ["video_features", 8],
                        },
                    ],
                    "outputs": [],
                },
                {
                    "name": "reasoner_vision_encoder",
                    "role": "encoder",
                    "run_on": "always",
                    "inputs": [
                        {
                            "name": "pixel_values",
                            "dtype": "FLOAT",
                            "shape": [1, 3, 32, 32],
                        }
                    ],
                    "outputs": [
                        {
                            "name": "image_features",
                            "dtype": "FLOAT",
                            "shape": [4, 8],
                        }
                    ],
                },
                {
                    "name": "video_encoder",
                    "role": "encoder",
                    "run_on": "always",
                    "inputs": [],
                    "outputs": [],
                    "config": {
                        "spatial_compression": 8,
                        "temporal_compression": 2,
                    },
                },
            ],
            "connections": [],
            "stages": [],
            "inputs": [
                {
                    "port": "reasoner_vision_encoder.pixel_values",
                    "kind": "external",
                    "semantic": "vision.pixel_values",
                },
                {
                    "port": "generator.vision_tokens",
                    "kind": "external",
                    "semantic": "diffusion.initial_vision_latent",
                },
                {
                    "port": "generator.action_tokens",
                    "kind": "external",
                    "semantic": "diffusion.initial_action_latent",
                },
            ],
            "outputs": [],
            "metadata": {
                "packing": {
                    "latent_patch_size": 2,
                    "patch_latent_dim": 16,
                },
                "action": {
                    "padded_dimension": 8,
                    "raw_dimensions": {
                        "no_action": 0,
                        "robot": 3,
                    },
                },
            },
        },
        "component_files": {},
    }
    (tmp_path / "pipeline.json").write_text(
        json.dumps(document),
        encoding="utf-8",
    )
    return tmp_path


_GENERATOR_PROMPT_METADATA = {
    "chat": {"add_generation_prompt": True, "add_vision_id": False},
    "suffix_token_ids": [2, 3],
    "system_prompts": {
        "image": "system image",
        "video": "system video",
    },
    "templates": {
        "duration": "The video is {duration:.1f} seconds long "
        "and is of {fps:.0f} FPS.",
        "inverse_duration": "The video is not {duration:.1f} seconds long "
        "and is not of {fps:.0f} FPS.",
        "image_resolution": "This image is of {height}x{width} resolution.",
        "inverse_image_resolution": "This image is not of "
        "{height}x{width} resolution.",
        "video_resolution": "This video is of {height}x{width} resolution.",
        "inverse_video_resolution": "This video is not of "
        "{height}x{width} resolution.",
    },
}


@pytest.fixture
def conditioned_package(preprocessing_package: Path) -> Path:
    """A package that declares guided, image-conditioned video generation."""
    path = preprocessing_package / "pipeline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    manifest = document["manifest"]
    manifest["components"].append(
        {
            "name": "video_encoder",
            "role": "encoder",
            "run_on": "on_demand",
            "inputs": [
                {
                    "name": "sample",
                    "dtype": "FLOAT",
                    "shape": [1, 3, "frames", "height", "width"],
                }
            ],
            "outputs": [
                {
                    "name": "latent",
                    "dtype": "FLOAT",
                    "shape": [
                        1,
                        4,
                        "latent_frames",
                        "latent_height",
                        "latent_width",
                    ],
                }
            ],
            "config": {"spatial_compression": 8, "temporal_compression": 2},
        }
    )
    manifest["inputs"].append(
        {
            "port": "video_encoder.sample",
            "kind": "external",
            "semantic": "video.frames",
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
                    "kind": "FlowMatchEulerDiscreteScheduler",
                    "config_asset": "scheduler/scheduler_config.json",
                    "mode_overrides": {
                        "image_to_video": {
                            "flow_shift": 3.0,
                            "num_inference_steps": 4,
                            "guidance_scale": 5.0,
                        },
                        "action": {"guidance_scale": 1.0},
                    },
                },
                "guidance": {
                    "kind": "classifier_free",
                    "conditioning_input": "generator.input_ids",
                    "scale_option": "guidance_scale",
                    "default_scale": 1.0,
                    "combine": "unconditional + scale * "
                    "(conditional - unconditional)",
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
                "default_steps": 2,
            },
        },
        {
            "name": "encode_video",
            "kind": "on_demand",
            "components": ["video_encoder"],
            "run_on": "on_demand",
        },
    ]
    manifest["metadata"]["generator_prompt"] = _GENERATOR_PROMPT_METADATA
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
            "height": 32,
            "width": 32,
            "frames": 5,
            "fps": 24.0,
        }
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    assets = preprocessing_package / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "negative_prompt.json").write_text(
        json.dumps({"negative_prompt": "blurry"}),
        encoding="utf-8",
    )
    return preprocessing_package
