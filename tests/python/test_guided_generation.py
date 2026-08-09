"""End-to-end conditioned, guided generation on real ONNX Runtime sessions.

The components are tiny hand-built graphs, so the whole path — manifest
validation, conditioning encode, packed latent anchoring, the conditional and
unconditional generator passes, the guided combination, and the indexed
scheduler update — runs through the actual runtime instead of a stub.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
from onnx_world_model import Pipeline

_OPSET = 18


def _value(name: str, dtype: ir.DataType, shape: list[object]) -> ir.Value:
    return ir.Value(
        name=name,
        type=ir.TensorType(dtype),
        shape=ir.Shape(shape),
    )


def _model(graph: ir.Graph) -> ir.Model:
    return ir.Model(graph, ir_version=10, producer_name="onnx-world-model-test")


def _generator_graph() -> ir.Model:
    """``vision_pred`` rows encode ``max(input_ids) + und_len``.

    Both terms are per-pass: ``input_ids`` is the prompt the runtime binds for
    that pass, and ``und_len`` is a generated packed-layout input derived from
    it, so the value proves the unconditional pass rebuilt its own layout.
    """
    input_ids = _value("input_ids", ir.DataType.INT64, ["text"])
    und_len = _value("und_len", ir.DataType.INT64, [1])
    vision_tokens = _value("vision_tokens", ir.DataType.FLOAT, ["vision", 2])
    indexes = _value(
        "vision_timestep_token_indexes", ir.DataType.INT64, ["noisy"]
    )
    timesteps = _value("vision_timesteps", ir.DataType.FLOAT, ["noisy"])

    reduce_max = ir.Node(
        "",
        "ReduceMax",
        inputs=[input_ids],
        attributes=[ir.AttrInt64("keepdims", 0)],
        num_outputs=1,
    )
    reduce_und = ir.Node(
        "",
        "ReduceSum",
        inputs=[und_len],
        attributes=[ir.AttrInt64("keepdims", 0)],
        num_outputs=1,
    )
    prompt_length = ir.Node(
        "",
        "Add",
        inputs=[reduce_max.outputs[0], reduce_und.outputs[0]],
        num_outputs=1,
    )
    cast = ir.Node(
        "",
        "Cast",
        inputs=prompt_length.outputs,
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
        num_outputs=1,
    )
    gather = ir.Node(
        "",
        "Gather",
        inputs=[vision_tokens, indexes],
        attributes=[ir.AttrInt64("axis", 0)],
        num_outputs=1,
    )
    zero = ir.Node(
        "",
        "Constant",
        inputs=[],
        attributes=[
            ir.AttrTensor(
                "value",
                ir.tensor(np.zeros((1,), dtype=np.float32), name="zero"),
            )
        ],
        num_outputs=1,
    )
    scaled = ir.Node(
        "", "Mul", inputs=[gather.outputs[0], zero.outputs[0]], num_outputs=1
    )
    # Keep the timestep input live so ORT does not prune it from the signature.
    timestep_zero = ir.Node(
        "", "Mul", inputs=[timesteps, zero.outputs[0]], num_outputs=1
    )
    timestep_scalar = ir.Node(
        "",
        "ReduceSum",
        inputs=[timestep_zero.outputs[0]],
        attributes=[ir.AttrInt64("keepdims", 0)],
        num_outputs=1,
    )
    velocity = ir.Node(
        "", "Add", inputs=[cast.outputs[0], timestep_scalar.outputs[0]], num_outputs=1
    )
    prediction = ir.Node(
        "", "Add", inputs=[scaled.outputs[0], velocity.outputs[0]], num_outputs=1
    )
    prediction.outputs[0].name = "vision_pred"
    prediction.outputs[0].shape = ir.Shape(["noisy", 2])
    prediction.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    graph = ir.Graph(
        inputs=[input_ids, und_len, vision_tokens, indexes, timesteps],
        outputs=[prediction.outputs[0]],
        nodes=[
            reduce_max,
            reduce_und,
            prompt_length,
            cast,
            gather,
            zero,
            scaled,
            timestep_zero,
            timestep_scalar,
            velocity,
            prediction,
        ],
        name="generator",
        opset_imports={"": _OPSET},
    )
    return _model(graph)


def _encoder_graph() -> ir.Model:
    """A stand-in latent encoder that halves its conditioning frames."""
    sample = _value(
        "sample", ir.DataType.FLOAT, [1, 3, "frames", "height", "width"]
    )
    half = ir.Node(
        "",
        "Constant",
        inputs=[],
        attributes=[
            ir.AttrTensor(
                "value",
                ir.tensor(np.full((1,), 0.5, dtype=np.float32), name="half"),
            )
        ],
        num_outputs=1,
    )
    axes = ir.Node(
        "",
        "Constant",
        inputs=[],
        attributes=[
            ir.AttrTensor(
                "value",
                ir.tensor(
                    np.asarray([1, 2, 3, 4], dtype=np.int64), name="axes"
                ),
            )
        ],
        num_outputs=1,
    )
    reduced = ir.Node(
        "",
        "ReduceMean",
        inputs=[sample, axes.outputs[0]],
        attributes=[ir.AttrInt64("keepdims", 1)],
        num_outputs=1,
    )
    scaled = ir.Node(
        "", "Mul", inputs=[reduced.outputs[0], half.outputs[0]], num_outputs=1
    )
    shape = ir.Node(
        "",
        "Constant",
        inputs=[],
        attributes=[
            ir.AttrTensor(
                "value",
                ir.tensor(
                    np.asarray([1, 2, 1, 1, 1], dtype=np.int64), name="shape"
                ),
            )
        ],
        num_outputs=1,
    )
    expanded = ir.Node(
        "", "Expand", inputs=[scaled.outputs[0], shape.outputs[0]], num_outputs=1
    )
    expanded.outputs[0].name = "latent"
    expanded.outputs[0].shape = ir.Shape([1, 2, 1, 1, 1])
    expanded.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    graph = ir.Graph(
        inputs=[sample],
        outputs=[expanded.outputs[0]],
        nodes=[half, axes, reduced, scaled, shape, expanded],
        name="video_encoder",
        opset_imports={"": _OPSET},
    )
    return _model(graph)


_MANIFEST = {
    "format": "mobius-pipeline",
    "schema_version": "1.1",
    "manifest": {
        "schema_version": "1.1",
        "components": [
            {
                "name": "generator",
                "role": "dynamics",
                "run_on": "step",
                "inputs": [
                    {"name": "input_ids", "dtype": "INT64", "shape": ["text"]},
                    {"name": "und_len", "dtype": "INT64", "shape": [1]},
                    {
                        "name": "vision_tokens",
                        "dtype": "FLOAT",
                        "shape": ["vision", 2],
                    },
                    {
                        "name": "vision_timestep_token_indexes",
                        "dtype": "INT64",
                        "shape": ["noisy"],
                    },
                    {
                        "name": "vision_timesteps",
                        "dtype": "FLOAT",
                        "shape": ["noisy"],
                    },
                ],
                "outputs": [
                    {
                        "name": "vision_pred",
                        "dtype": "FLOAT",
                        "shape": ["noisy", 2],
                    }
                ],
                "preferred_execution_providers": ["cpu"],
                "parameter_dtype": "FLOAT",
            },
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
                        "shape": [1, 2, 1, 1, 1],
                    }
                ],
                "preferred_execution_providers": ["cpu"],
                "parameter_dtype": "FLOAT",
            },
        ],
        "connections": [
            {
                "source": "generator.vision_pred",
                "target": "generator.vision_tokens",
                "recurrent": True,
                "transform": "scheduler_step",
                "parameters": {
                    "scheduler_asset": "scheduler.json",
                    "stage": "world_generation",
                    "state": "vision_state",
                    "timestep_input": "generator.vision_timesteps",
                },
            }
        ],
        "stages": [
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
                        "config_asset": "scheduler.json",
                        "mode_overrides": {
                            "image_to_video": {
                                "num_inference_steps": 1,
                                "guidance_scale": 3.0,
                            }
                        },
                    },
                    "guidance": {
                        "kind": "classifier_free",
                        "conditioning_input": "generator.input_ids",
                        "scale_option": "guidance_scale",
                        "default_scale": 1.0,
                        "combine": (
                            "unconditional + scale * "
                            "(conditional - unconditional)"
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
                                "spatial_patch_size": 1,
                                "temporal_patch_size": 1,
                                "input_layout": "BCTHW",
                                "output_layout": "NC",
                                "channel_order": (
                                    "patch_height_patch_width_channel"
                                ),
                            },
                        }
                    },
                    "default_steps": 1,
                    "timestep": {},
                    "state_inputs": ["generator.vision_tokens"],
                },
            },
            {
                "name": "encode_video",
                "kind": "single_pass",
                "components": ["video_encoder"],
                "run_on": "on_demand",
            },
        ],
        "inputs": [
            {
                "port": "generator.input_ids",
                "kind": "external",
                "alias": "generator_input_ids",
                "semantic": "text.token_ids",
                "required": True,
            },
            {
                "port": "generator.vision_tokens",
                "kind": "external",
                "alias": "initial_vision_tokens",
                "semantic": "diffusion.initial_vision_latent",
                "required": True,
            },
            {
                "port": "generator.vision_timestep_token_indexes",
                "kind": "generated",
                "semantic": "packing.vision_timestep_token_indexes",
                "generator": {
                    "kind": "packed_sequence_layout",
                    "parameters": {
                        "modality": "vision",
                        "source": "generator.vision_tokens",
                        "index_kind": "vision_timestep_token_indexes",
                    },
                },
            },
            {
                "port": "generator.und_len",
                "kind": "generated",
                "semantic": "packing.und_len",
                "generator": {
                    "kind": "packed_sequence_layout",
                    "parameters": {
                        "modality": "text",
                        "source": "generator.input_ids",
                        "understanding_prefix": True,
                        "index_kind": "und_len",
                    },
                },
            },
            {
                "port": "generator.vision_timesteps",
                "kind": "generated",
                "semantic": "diffusion.vision.timesteps",
                "generator": {
                    "kind": "scheduler_timesteps",
                    "parameters": {
                        "stage": "world_generation",
                        "modality": "vision",
                    },
                },
            },
            {
                "port": "video_encoder.sample",
                "kind": "external",
                "alias": "video_encoder_sample",
                "semantic": "video.frames",
                "required": True,
            },
        ],
        "outputs": [
            {"state": "vision_state", "alias": "vision_latent"},
            {"port": "generator.vision_pred", "alias": "vision_velocity"},
            {"port": "video_encoder.latent", "alias": "encoded_video_latent"},
        ],
        "profile": {"name": "guided-world", "version": "1.0"},
        "states": [
            {
                "name": "vision_state",
                "kind": "diffusion_latent",
                "input": "generator.vision_tokens",
                "output": "generator.vision_pred",
                "lifetime": "request",
                "release_after": "world_generation",
                "sequence_axis": 0,
            }
        ],
        "assets": [{"path": "scheduler.json"}],
        "required_capabilities": [
            "iterative_scheduler",
            "loop_carried_state",
            "packed_sequence_program",
        ],
    },
    "component_files": {
        "generator": "generator/model.onnx",
        "video_encoder": "video_encoder/model.onnx",
    },
}


@pytest.fixture
def guided_package(tmp_path: Path) -> Path:
    (tmp_path / "generator").mkdir()
    (tmp_path / "video_encoder").mkdir()
    ir.save(_generator_graph(), tmp_path / "generator" / "model.onnx")
    ir.save(_encoder_graph(), tmp_path / "video_encoder" / "model.onnx")
    (tmp_path / "scheduler.json").write_text(
        json.dumps(
            {
                "_class_name": "FlowMatchEulerDiscreteScheduler",
                "num_train_timesteps": 1000,
                "shift": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "pipeline.json").write_text(
        json.dumps(_MANIFEST),
        encoding="utf-8",
    )
    return tmp_path


def test_guided_conditioned_stage_runs_on_onnxruntime(
    guided_package: Path,
) -> None:
    session = Pipeline(guided_package).create_session()

    encoded = session.run_stage(
        "encode_video",
        {"video.frames": np.ones((1, 3, 1, 2, 2), dtype=np.float32)},
    )
    latent = encoded["encoded_video_latent"]
    assert latent.shape == (1, 2, 1, 1, 1)

    # One conditioned latent row (frame 0) and two noisy rows.
    initial = np.array(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float32
    )
    initial[0] = latent.reshape(-1)
    # Unequal prompt lengths: the generated und_len must be rebuilt for the
    # unconditional pass instead of reusing the conditional layout.
    world = session.run_stage(
        "world_generation",
        {
            "text.token_ids": np.asarray([7, 7, 7], dtype=np.int64),
            "diffusion.initial_vision_latent": initial,
            "unconditional:generator.input_ids": np.asarray(
                [1], dtype=np.int64
            ),
        },
        overrides={
            "generator.vision_timestep_token_indexes": np.asarray(
                [1, 2], dtype=np.int64
            )
        },
        options={"mode": "image_to_video"},
    )

    # conditional = max(7) + und_len(3) = 10, unconditional = max(1) + 1 = 2,
    # so the guided velocity is 2 + 3 * (10 - 2) = 26.
    np.testing.assert_allclose(world["vision_velocity"], np.full((2, 2), 26.0))
    # A single Euler step over one inference step has delta = -1.
    np.testing.assert_allclose(
        world["vision_latent"],
        np.array([[0.5, 0.5], [-6.0, -6.0], [4.0, 4.0]], dtype=np.float32),
        atol=1e-5,
    )


def test_guided_stage_without_guidance_scale_runs_one_pass(
    guided_package: Path,
) -> None:
    session = Pipeline(guided_package).create_session()

    world = session.run_stage(
        "world_generation",
        {
            "text.token_ids": np.asarray([7], dtype=np.int64),
            "diffusion.initial_vision_latent": np.zeros(
                (3, 2), dtype=np.float32
            ),
        },
        overrides={
            "generator.vision_timestep_token_indexes": np.asarray(
                [1, 2], dtype=np.int64
            )
        },
        options={"mode": "image_to_video", "guidance_scale": 1.0},
    )

    # max(7) + und_len(1) = 8, applied to the noisy rows only.
    np.testing.assert_allclose(world["vision_velocity"], np.full((2, 2), 8.0))
    np.testing.assert_allclose(
        world["vision_latent"],
        np.array([[0.0, 0.0], [-8.0, -8.0], [-8.0, -8.0]], dtype=np.float32),
        atol=1e-5,
    )


def test_guided_passes_restore_the_conditional_layout(
    guided_package: Path,
) -> None:
    """A second guided step must start from the conditional pass's state."""
    session = Pipeline(guided_package).create_session()

    for _ in range(2):
        world = session.step_stage(
            "world_generation",
            {
                "text.token_ids": np.asarray([7, 7, 7], dtype=np.int64),
                "diffusion.initial_vision_latent": np.zeros(
                    (3, 2), dtype=np.float32
                ),
                "unconditional:generator.input_ids": np.asarray(
                    [1], dtype=np.int64
                ),
            },
            overrides={
                "generator.vision_timestep_token_indexes": np.asarray(
                    [1, 2], dtype=np.int64
                )
            },
            options={"mode": "image_to_video"},
        )

    # Both steps see the same guided velocity, and only the noisy rows move.
    np.testing.assert_allclose(world["vision_velocity"], np.full((2, 2), 26.0))
    np.testing.assert_allclose(
        world["vision_latent"],
        np.array([[0.0, 0.0], [-52.0, -52.0], [-52.0, -52.0]], dtype=np.float32),
        atol=1e-5,
    )
