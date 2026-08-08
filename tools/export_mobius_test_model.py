from __future__ import annotations

import argparse
from pathlib import Path

import onnx_ir as ir
import torch
from mobius import (
    MLPWorldModel,
    PipelineBuilder,
    WorldModelConfig,
    build_from_module,
)


def export_test_model(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = WorldModelConfig(
        observation_shape=(4,),
        action_shape=(2,),
        state_shape=(3,),
        hidden_size=8,
        num_hidden_layers=2,
    )
    module = MLPWorldModel(config)
    package = build_from_module(
        module,
        config,
        task="world-model",
        execution_provider="cpu",
    )

    weights: dict[str, torch.Tensor] = {}
    for name, value in package["model"].graph.initializers.items():
        if value.const_value is not None:
            continue
        tensor = torch.zeros(tuple(value.shape), dtype=torch.float32)
        if name == "state_head.bias":
            tensor = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        elif name == "observation_head.bias":
            tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        elif name == "reward_head.bias":
            tensor = torch.tensor([0.5], dtype=torch.float32)
        weights[name] = tensor

    package.apply_weights(weights)
    package.save(str(output_dir), progress_bar=False)
    return output_dir / "model.onnx"


def export_test_pipeline(output_dir: Path) -> Path:
    model = ir.load(export_test_model(output_dir / "source"))
    builder = PipelineBuilder()
    builder.add_model(
        "dynamics",
        model,
        role="dynamics",
        run_on="step",
        preferred_execution_providers=["cpu"],
        parameter_dtype="FLOAT",
    )
    for value in model.graph.inputs:
        builder.declare_external(
            f"dynamics.{value.name}",
            alias=value.name,
            semantic=f"latent_dynamics.{value.name}",
        )
    builder.add_stage("step", "single_pass", ["dynamics"], run_on="step")
    for value in model.graph.outputs:
        builder.add_public_output(f"dynamics.{value.name}", alias=value.name)
    builder.set_profile("latent-dynamics", "1.0")
    builder.set_metadata("profile", "world-model")
    builder.set_metadata("model_type", "test_latent_dynamics")

    package_path = output_dir / "package"
    builder.build().save(str(package_path), progress_bar=False)
    return package_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the deterministic Mobius world model used by runtime tests."
    )
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()
    print(export_test_model(arguments.output_dir))


if __name__ == "__main__":
    main()
