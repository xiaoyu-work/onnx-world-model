from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ml_dtypes
import numpy as np
import pytest
from jinja2.exceptions import SecurityError
from onnx_world_model import (
    HighLevelWorldModel,
    TextGenerationConfig,
    WorldGenerationConfig,
    high_level,
)
from onnx_world_model.preprocessing import (
    ImagePreprocessor,
    TextPreprocessor,
    WorldModelPreprocessor,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


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
        if stage == "decode_video":
            return {"video": np.zeros((1, 3, 5, 32, 32), dtype=np.float32)}
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
def high_level_model(
    preprocessing_package: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> HighLevelWorldModel:
    monkeypatch.setattr(high_level, "Pipeline", _FakePipeline)
    return HighLevelWorldModel(
        preprocessing_package,
        providers=["cpu"],
        provider_options={"cpu": {"use_arena": False}},
    )


def test_high_level_respond_preprocesses_and_decodes(
    high_level_model: HighLevelWorldModel,
) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    result = high_level_model.respond(
        "cat",
        image,
        config=TextGenerationConfig(max_tokens=2),
    )

    assert result.text == "answer"
    assert result.token_ids is not None
    assert result.token_ids.tolist() == [[9]]
    fake = _FakePipeline.last_instance
    assert fake is not None
    assert fake.options["providers"] == ["cpu"]
    assert fake.session.released == ["reasoner_decode"]
    reasoner_inputs = fake.session.calls[0][1]["inputs"]
    assert reasoner_inputs["vision.pixel_values"].shape == (1, 3, 32, 32)


def test_high_level_world_generation_returns_structured_outputs(
    high_level_model: HighLevelWorldModel,
) -> None:
    result = high_level_model.generate_world(
        "cat",
        outputs=("action", "video"),
        config=WorldGenerationConfig(
            frames=5,
            height=32,
            width=32,
            num_inference_steps=2,
            action_steps=1,
            action_domain="robot",
            seed=7,
        ),
    )

    assert result.action is not None
    assert result.action.shape == (1, 3)
    assert result.video is not None
    assert result.video.shape == (1, 3, 5, 32, 32)
    fake = _FakePipeline.last_instance
    assert fake is not None
    assert fake.session.released == ["world_generation", "decode_video"]


def test_high_level_rejects_implicit_image_world_conditioning(
    high_level_model: HighLevelWorldModel,
) -> None:
    with pytest.raises(NotImplementedError, match="image conditioning"):
        high_level_model.generate(
            "cat",
            np.zeros((32, 32, 3), dtype=np.uint8),
            outputs=("video",),
        )


def test_world_only_pipeline_loads_without_reasoner_stages(
    preprocessing_package: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(high_level, "Pipeline", _FakeWorldOnlyPipeline)

    model = HighLevelWorldModel(preprocessing_package)
    result = model.generate_world(
        "cat",
        config=WorldGenerationConfig(
            frames=5,
            height=32,
            width=32,
            num_inference_steps=1,
        ),
    )

    assert result.video is not None
