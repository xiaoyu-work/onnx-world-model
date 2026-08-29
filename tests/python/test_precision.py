"""The precision report, through Python, against a real ONNX Runtime package.

Everything asserted here is either copied from the manifest and labelled as
declared or read from the loaded session. Nothing in this module claims that a
component's weights are quantized, because the runtime cannot see that: the
ONNX Runtime session API it uses exposes graph inputs and outputs, not ordinary
initializers and not nodes. What is protected here is the Python surface -- the
frozen dataclasses, the immutable tuple, the separation between the declared
parameter dtype and the real port dtypes, the provider reporting, and the
``WorldModel`` pass-through.
"""

# @agent-file
# @agent-purpose: Tests the Pipeline.precision_report property and its frozen PrecisionPort and ComponentPrecisionReport dataclasses against a package built in-process with onnx_ir, plus the read-only WorldModel pass-through.
# @agent-public-api: none
# @agent-invariants: The package fixture builds its graphs in-process with `pytest.importorskip("onnx_ir")` and never needs the Mobius exporter, so every test here runs by default. The manifest deliberately declares FLOAT16 parameters for a component whose ports are FLOAT and INT64 and INT8 parameters for a component with FLOAT ports, which is what makes "the declared dtype is never compared with a port dtype" and "a non-floating declaration is never rejected" observable rather than accidental. No test asserts that quantization was detected, because no field reports it. The report is read as an immutable tuple of frozen values and is asserted to be stable across reads, since the manifest and the loaded sessions cannot change after the package loads.
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
    ComponentPrecisionReport,
    Pipeline,
    PrecisionPort,
    WorldModel,
)

_OPSET = 18

# `reasoner` declares FLOAT16 parameters while taking INT64 token ids and
# producing a FLOAT tensor, and `head` declares INT8 parameters with FLOAT
# ports. Neither declaration agrees with any port, which is the point: they
# describe weight storage, and the ports describe activations and control
# tensors.
_MANIFEST: dict[str, Any] = {
    "format": "mobius-pipeline",
    "schema_version": "1.1",
    "manifest": {
        "schema_version": "1.1",
        "components": [
            {
                "name": "reasoner",
                "role": "dynamics",
                "run_on": "always",
                "inputs": [
                    {"name": "input_ids", "dtype": "INT64", "shape": [1]},
                ],
                "outputs": [{"name": "hidden", "dtype": "FLOAT", "shape": [1]}],
                "preferred_execution_providers": ["cpu"],
                "parameter_dtype": "FLOAT16",
            },
            {
                "name": "head",
                "role": "decoder",
                "run_on": "always",
                "inputs": [{"name": "hidden", "dtype": "FLOAT", "shape": [1]}],
                "outputs": [{"name": "value", "dtype": "FLOAT", "shape": [1]}],
                "preferred_execution_providers": ["cpu"],
                "parameter_dtype": "INT8",
            },
        ],
        "connections": [{"source": "reasoner.hidden", "target": "head.hidden"}],
        "stages": [
            {
                "name": "run",
                "kind": "single_pass",
                "components": ["reasoner", "head"],
                "run_on": "always",
            }
        ],
        "inputs": [
            {
                "port": "reasoner.input_ids",
                "kind": "external",
                "required": True,
                "semantic": "text.input_ids",
            }
        ],
        "outputs": [{"port": "head.value", "alias": "value"}],
    },
    "component_files": {
        "reasoner": "reasoner/model.onnx",
        "head": "head/model.onnx",
    },
}

_UNDECLARED_MANIFEST: dict[str, Any] = {
    "format": "mobius-pipeline",
    "schema_version": "1.1",
    "manifest": {
        "schema_version": "1.1",
        "components": [
            {
                "name": "head",
                "role": "decoder",
                "run_on": "always",
                "inputs": [{"name": "hidden", "dtype": "FLOAT", "shape": [1]}],
                "outputs": [{"name": "value", "dtype": "FLOAT", "shape": [1]}],
            }
        ],
        "connections": [],
        "stages": [
            {
                "name": "run",
                "kind": "single_pass",
                "components": ["head"],
                "run_on": "always",
            }
        ],
        "inputs": [
            {"port": "head.hidden", "kind": "external", "required": True}
        ],
        "outputs": [{"port": "head.value", "alias": "value"}],
    },
    "component_files": {"head": "model.onnx"},
}


def _write_cast_to_float(path: Path) -> None:
    """Writes a graph that casts one INT64 input to a FLOAT output."""
    ir = pytest.importorskip("onnx_ir")
    value = ir.Value(
        name="input_ids",
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape([1]),
    )
    cast = ir.Node(
        "",
        "Cast",
        inputs=[value],
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
        num_outputs=1,
    )
    cast.outputs[0].name = "hidden"
    cast.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    cast.outputs[0].shape = ir.Shape([1])
    graph = ir.Graph(
        inputs=[value],
        outputs=[cast.outputs[0]],
        nodes=[cast],
        name="reasoner",
        opset_imports={"": _OPSET},
    )
    ir.save(
        ir.Model(graph, ir_version=10, producer_name="onnx-world-model-test"),
        path,
    )


def _write_add_one(path: Path) -> None:
    """Writes a graph that adds one to its single FLOAT input."""
    ir = pytest.importorskip("onnx_ir")
    value = ir.Value(
        name="hidden",
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
    add.outputs[0].name = "value"
    add.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    add.outputs[0].shape = ir.Shape([1])
    graph = ir.Graph(
        inputs=[value],
        outputs=[add.outputs[0]],
        nodes=[one, add],
        name="head",
        opset_imports={"": _OPSET},
    )
    ir.save(
        ir.Model(graph, ir_version=10, producer_name="onnx-world-model-test"),
        path,
    )


@pytest.fixture
def package(tmp_path: Path) -> Path:
    """A two-component CPU package with mixed port dtypes."""
    (tmp_path / "reasoner").mkdir()
    _write_cast_to_float(tmp_path / "reasoner" / "model.onnx")
    (tmp_path / "head").mkdir()
    _write_add_one(tmp_path / "head" / "model.onnx")
    (tmp_path / "pipeline.json").write_text(
        json.dumps(_MANIFEST), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def undeclared_package(tmp_path: Path) -> Path:
    """A profile-less package whose one component declares no parameter dtype."""
    _write_add_one(tmp_path / "model.onnx")
    (tmp_path / "pipeline.json").write_text(
        json.dumps(_UNDECLARED_MANIFEST), encoding="utf-8"
    )
    return tmp_path


def test_the_report_follows_manifest_component_order(package: Path) -> None:
    report = Pipeline(package).precision_report

    assert [entry.component for entry in report] == ["reasoner", "head"]


def test_the_report_is_an_immutable_tuple_of_frozen_values(
    package: Path,
) -> None:
    pipeline = Pipeline(package)
    report = pipeline.precision_report

    assert isinstance(report, tuple)
    assert pipeline.precision_report is report
    assert all(isinstance(entry, ComponentPrecisionReport) for entry in report)
    assert all(
        isinstance(port, PrecisionPort) for port in report[0].graph_inputs
    )
    with pytest.raises(FrozenInstanceError):
        report[0].declared_parameter_dtype = "float32"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report[0].graph_inputs[0].dtype = "float32"  # type: ignore[misc]


def test_a_declared_parameter_dtype_is_reported_verbatim(package: Path) -> None:
    reasoner, head = Pipeline(package).precision_report

    assert reasoner.declared_parameter_dtype == "float16"
    # An integer declaration is a normal quantized-weight claim, so it is kept
    # exactly as declared rather than rejected for not being a float type.
    assert head.declared_parameter_dtype == "int8"


def test_an_undeclared_parameter_dtype_is_none(
    undeclared_package: Path,
) -> None:
    (entry,) = Pipeline(undeclared_package).precision_report

    assert entry.declared_parameter_dtype is None


def test_graph_ports_are_the_live_graphs_own_dtypes(package: Path) -> None:
    reasoner, head = Pipeline(package).precision_report

    assert [(port.name, port.dtype) for port in reasoner.graph_inputs] == [
        ("input_ids", "int64")
    ]
    assert [(port.name, port.dtype) for port in reasoner.graph_outputs] == [
        ("hidden", "float32")
    ]
    assert [(port.name, port.dtype) for port in head.graph_inputs] == [
        ("hidden", "float32")
    ]
    assert [(port.name, port.dtype) for port in head.graph_outputs] == [
        ("value", "float32")
    ]


def test_the_declared_dtype_is_independent_of_the_port_dtypes(
    package: Path,
) -> None:
    # The declaration describes weight storage and the ports describe
    # activations and control tensors, so a component may disagree with itself
    # here and still be a perfectly valid package.
    reasoner, head = Pipeline(package).precision_report

    assert reasoner.declared_parameter_dtype == "float16"
    assert {port.dtype for port in reasoner.graph_inputs} == {"int64"}
    assert {port.dtype for port in reasoner.graph_outputs} == {"float32"}
    assert head.declared_parameter_dtype == "int8"
    assert {port.dtype for port in head.graph_outputs} == {"float32"}


def test_a_stateless_package_reports_no_state_ports(package: Path) -> None:
    for entry in Pipeline(package).precision_report:
        assert entry.state_inputs == ()
        assert entry.state_outputs == ()


def test_the_report_names_the_selected_execution_providers(
    package: Path,
) -> None:
    pipeline = Pipeline(package)

    for entry in pipeline.precision_report:
        assert entry.execution_providers == pipeline.execution_providers[
            entry.component
        ]
        assert entry.execution_providers == ("CPUExecutionProvider",)


def test_the_report_says_nothing_about_quantized_weights(package: Path) -> None:
    # The declared dtype is the only precision claim in the report, and it is
    # the exporter's. There is deliberately no field that says whether
    # MatMulNBits, QDQ, QLinear, or a vendor format is present, because the
    # ONNX Runtime session API cannot see initializers or nodes.
    (entry, _) = Pipeline(package).precision_report
    fields = set(vars(entry))

    assert "declared_parameter_dtype" in fields
    assert not {name for name in fields if "quant" in name}
    assert not {name for name in fields if "verified" in name}


def test_world_model_forwards_the_report() -> None:
    # WorldModel needs an exported Mobius package this suite does not build, so
    # the pass-through is checked by shape: a read-only property that takes no
    # arguments.
    assert isinstance(
        inspect.getattr_static(WorldModel, "precision_report"), property
    )
    assert WorldModel.precision_report.fset is None
    assert list(
        inspect.signature(WorldModel.precision_report.fget).parameters
    ) == ["self"]
