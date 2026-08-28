"""Per-component placement and the transfer plan it produces, through Python.

Nothing here needs an accelerator. The package is CPU-only, so the plan it
produces is entirely ``direct``; what this module protects is the Python
surface: the argument shapes the binding accepts and rejects, the component and
provider rules C++ owns, the device a CPU port reports, the immutability of the
plan, and the fact that placement reaches only the two APIs that load a
package.
"""

# @agent-file
# @agent-purpose: Tests the keyword-only component_placement and allow_unpreferred_providers arguments on Pipeline and WorldModel, the DeviceSpec a CPU port reports through ModelMetadata, and the frozen PipelineTransferPlan a loaded package exposes.
# @agent-public-api: none
# @agent-invariants: The package fixture builds its graph in-process with `pytest.importorskip("onnx_ir")` and never needs the Mobius exporter, so every test here runs by default. Every assertion is about configuration, error handling, or the shape of an inspection value, so nothing races and nothing depends on which execution providers the machine has beyond CPU. Argument shape is checked in the binding and reported as TypeError or ValueError, while component names, repeated providers, and options for a provider a component does not run on are checked in C++ and reported as WorldModelError; the tests are split that way on purpose so neither layer silently takes over the other's rule. A CPU-only package classifies every connection as `direct` with an empty reason, which is what makes `direct_bind_eligible` and the absence of a reason meaningful rather than accidental. `device_outputs_enabled` mirrors the constructor flag and never rewrites a kind. Placement is load-time only, so `OnnxModel` and `LatentDynamicsModel` are asserted not to accept it by signature rather than by calling them, and `WorldModel` is checked by signature too because it needs an exported Mobius package this suite does not build.
# @agent-side-effects: Writes ONNX models and package files into pytest temporary directories and runs ONNX Runtime session creation.

from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx_world_model import (
    ComponentPlacementSpec,
    DeviceSpec,
    LatentDynamicsModel,
    OnnxModel,
    Pipeline,
    WorldModel,
    WorldModelError,
)

_OPSET = 18

_MANIFEST: dict[str, Any] = {
    "format": "mobius-pipeline",
    "schema_version": "1.1",
    "manifest": {
        "schema_version": "1.1",
        "components": [
            {
                "name": "encoder",
                "role": "encoder",
                "run_on": "always",
                "inputs": [{"name": "x", "dtype": "FLOAT", "shape": [1]}],
                "outputs": [{"name": "y", "dtype": "FLOAT", "shape": [1]}],
                "preferred_execution_providers": ["cpu"],
            },
            {
                "name": "decoder",
                "role": "decoder",
                "run_on": "always",
                "inputs": [{"name": "z", "dtype": "FLOAT", "shape": [1]}],
                "outputs": [{"name": "w", "dtype": "FLOAT", "shape": [1]}],
                "preferred_execution_providers": ["cpu"],
            },
        ],
        "connections": [{"source": "encoder.y", "target": "decoder.z"}],
        "stages": [
            {
                "name": "run",
                "kind": "single_pass",
                "components": ["encoder", "decoder"],
                "run_on": "always",
            }
        ],
        "inputs": [{"port": "encoder.x", "kind": "external", "required": True}],
        "outputs": [{"port": "decoder.w", "alias": "value"}],
    },
    "component_files": {
        "encoder": "encoder/model.onnx",
        "decoder": "decoder/model.onnx",
    },
}


def _write_add_one(path: Path, input_name: str, output_name: str) -> None:
    """Writes a one-node graph that adds one to its single float input."""
    ir = pytest.importorskip("onnx_ir")
    value = ir.Value(
        name=input_name,
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([1]),
    )
    one = ir.Node(
        "",
        "Constant",
        inputs=[],
        attributes=[
            ir.AttrTensor(
                "value", ir.tensor(np.ones((1,), dtype=np.float32), name="one")
            )
        ],
        num_outputs=1,
    )
    add = ir.Node("", "Add", inputs=[value, one.outputs[0]], num_outputs=1)
    add.outputs[0].name = output_name
    add.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    add.outputs[0].shape = ir.Shape([1])
    graph = ir.Graph(
        inputs=[value],
        outputs=[add.outputs[0]],
        nodes=[one, add],
        name=path.stem,
        opset_imports={"": _OPSET},
    )
    ir.save(
        ir.Model(graph, ir_version=10, producer_name="onnx-world-model-test"),
        path,
    )


@pytest.fixture
def package(tmp_path: Path) -> Path:
    """A two-component CPU package whose one connection needs no transform."""
    for component, (input_name, output_name) in {
        "encoder": ("x", "y"),
        "decoder": ("z", "w"),
    }.items():
        directory = tmp_path / component
        directory.mkdir()
        _write_add_one(directory / "model.onnx", input_name, output_name)
    (tmp_path / "pipeline.json").write_text(
        json.dumps(_MANIFEST), encoding="utf-8"
    )
    return tmp_path


def test_a_cpu_port_reports_its_device(package: Path) -> None:
    model = OnnxModel(package / "encoder" / "model.onnx")

    assert model.metadata.inputs[0].device == DeviceSpec(type="cpu", id=0)
    assert model.metadata.outputs[0].device == DeviceSpec(type="cpu", id=0)


def test_an_all_cpu_package_plans_only_direct_transfers(package: Path) -> None:
    pipeline = Pipeline(package)
    plan = pipeline.transfer_plan

    assert len(plan.transfers) == 1
    transfer = plan.transfers[0]
    assert transfer.source == "encoder.y"
    assert transfer.target == "decoder.z"
    assert transfer.kind == "direct"
    assert transfer.direct_bind_eligible
    assert transfer.reason == ""
    assert transfer.transform is None
    assert not transfer.recurrent
    assert transfer.source_device == DeviceSpec(type="cpu", id=0)
    assert transfer.target_device == DeviceSpec(type="cpu", id=0)


def test_the_transfer_plan_is_frozen_and_stable(package: Path) -> None:
    pipeline = Pipeline(package)
    plan = pipeline.transfer_plan

    assert pipeline.transfer_plan is plan
    with pytest.raises(FrozenInstanceError):
        plan.device_outputs_enabled = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.transfers[0].kind = "upload"  # type: ignore[misc]


def test_device_outputs_are_reported_without_rewriting_the_plan(
    package: Path,
) -> None:
    default_plan = Pipeline(package).transfer_plan
    enabled_plan = Pipeline(package, device_outputs=True).transfer_plan

    assert not default_plan.device_outputs_enabled
    assert enabled_plan.device_outputs_enabled
    # The physical placement is the same either way, so the kinds must be too.
    assert [transfer.kind for transfer in enabled_plan.transfers] == [
        transfer.kind for transfer in default_plan.transfers
    ]


def test_component_placement_selects_providers(package: Path) -> None:
    pipeline = Pipeline(
        package,
        component_placement={
            "encoder": ComponentPlacementSpec(
                providers=["cpu"],
                provider_options={"cpu": {"use_arena": False}},
                graph_optimization="basic",
                intra_op_threads=1,
                inter_op_threads=1,
            )
        },
    )

    assert pipeline.execution_providers["encoder"] == ("CPUExecutionProvider",)
    assert pipeline.execution_providers["decoder"] == ("CPUExecutionProvider",)


def test_a_plain_mapping_is_accepted_like_a_spec(package: Path) -> None:
    pipeline = Pipeline(
        package,
        component_placement={"decoder": {"providers": ["cpu"]}},
    )

    assert pipeline.execution_providers["decoder"] == ("CPUExecutionProvider",)


def test_a_package_loads_unchanged_without_placement(package: Path) -> None:
    session = Pipeline(package).create_session()

    outputs = session.run_stage("run", {"encoder.x": np.zeros((1,), np.float32)})

    assert outputs["value"].tolist() == [2.0]


def test_an_unknown_component_is_rejected(package: Path) -> None:
    with pytest.raises(WorldModelError, match="unknown component"):
        Pipeline(package, component_placement={"missing": {"providers": ["cpu"]}})


def test_an_empty_component_name_is_rejected(package: Path) -> None:
    with pytest.raises(WorldModelError):
        Pipeline(package, component_placement={"": {"providers": ["cpu"]}})


def test_a_repeated_provider_is_rejected(package: Path) -> None:
    with pytest.raises(WorldModelError, match="more than once"):
        Pipeline(
            package,
            component_placement={
                "encoder": {"providers": ["cpu", "CPUExecutionProvider"]}
            },
        )


def test_options_for_an_unselected_provider_are_rejected(package: Path) -> None:
    with pytest.raises(WorldModelError, match="provider options"):
        Pipeline(
            package,
            component_placement={
                "encoder": {
                    "providers": ["cpu"],
                    "provider_options": {"cuda": {"device_id": 0}},
                }
            },
        )


def test_an_unpreferred_provider_still_needs_to_be_requested(
    package: Path,
) -> None:
    # allow_unpreferred_providers only unlocks a component's own list, so a
    # component that names nothing keeps following its manifest preferences.
    pipeline = Pipeline(package, allow_unpreferred_providers=True)

    assert pipeline.execution_providers["encoder"] == ("CPUExecutionProvider",)


def test_a_component_mapping_must_be_a_mapping(package: Path) -> None:
    with pytest.raises(TypeError):
        Pipeline(package, component_placement={"encoder": ["cpu"]})


def test_a_provider_list_must_not_be_a_bare_string(package: Path) -> None:
    # A bare string is a sequence of characters, so accepting one would quietly
    # request a provider per letter.
    with pytest.raises(TypeError):
        Pipeline(package, component_placement={"encoder": {"providers": "cpu"}})


def test_provider_names_must_be_strings(package: Path) -> None:
    with pytest.raises(TypeError):
        Pipeline(package, component_placement={"encoder": {"providers": [1]}})


def test_provider_options_must_be_mappings(package: Path) -> None:
    with pytest.raises(TypeError):
        Pipeline(
            package,
            component_placement={
                "encoder": {"providers": ["cpu"], "provider_options": {"cpu": 1}}
            },
        )


def test_a_provider_option_value_must_be_a_scalar(package: Path) -> None:
    with pytest.raises(TypeError):
        Pipeline(
            package,
            component_placement={
                "encoder": {
                    "providers": ["cpu"],
                    "provider_options": {"cpu": {"use_arena": [1]}},
                }
            },
        )


def test_an_unknown_placement_key_is_rejected(package: Path) -> None:
    with pytest.raises(ValueError, match="unknown key"):
        Pipeline(package, component_placement={"encoder": {"device_id": 0}})


def test_a_thread_count_must_not_be_a_bool(package: Path) -> None:
    with pytest.raises(TypeError, match="not a bool"):
        Pipeline(
            package, component_placement={"encoder": {"intra_op_threads": True}}
        )


def test_a_thread_count_must_be_an_int(package: Path) -> None:
    with pytest.raises(TypeError):
        Pipeline(
            package, component_placement={"encoder": {"inter_op_threads": 1.5}}
        )


def test_a_negative_thread_count_is_rejected(package: Path) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        Pipeline(
            package, component_placement={"encoder": {"intra_op_threads": -1}}
        )


def test_an_unknown_graph_optimization_is_rejected(package: Path) -> None:
    with pytest.raises(ValueError, match="graph_optimization"):
        Pipeline(
            package,
            component_placement={"encoder": {"graph_optimization": "maximum"}},
        )


def test_world_model_forwards_placement() -> None:
    parameters = inspect.signature(WorldModel.__init__).parameters

    assert parameters["component_placement"].default is None
    assert parameters["component_placement"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["allow_unpreferred_providers"].default is False
    assert (
        parameters["allow_unpreferred_providers"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_single_graph_apis_do_not_accept_placement() -> None:
    for api in (OnnxModel, LatentDynamicsModel):
        parameters = inspect.signature(api.__init__).parameters
        assert "component_placement" not in parameters
        assert "allow_unpreferred_providers" not in parameters
