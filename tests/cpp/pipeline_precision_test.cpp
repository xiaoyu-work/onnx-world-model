/**
 * @agent-file
 * @agent-purpose: Standalone test executable for the per-component precision report a PipelinePackage and a Pipeline answer with, covering manifest order, the declared parameter dtype it copies without checking, the live graph ports it lists, the state ports it associates, the providers it reports, and the detachment of every reading.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as pipeline_precision_test; it counts failures through a local Check helper, reports an unexpected onnx_world_model::Error as a failure instead of letting it abort the process, and returns a non-zero exit code when any check fails. Every component is an in-process StubBackend built from an explicit ModelMetadata, so the run needs no ONNX Runtime library and no real ONNX model, and every package is built in memory from an embedded manifest string. The checks are deliberately negative as well as positive: a declared INT8 parameter dtype survives beside FLOAT16 and INT64 ports, which is what proves the report neither restricts the declaration to a floating type nor compares it with a port type, and a backend that reports no execution providers reports an empty list rather than being credited with CPU. Nothing here asserts anything about weights or quantized operators, because the report cannot observe them.
 * @agent-side-effects: Writes failure descriptions to stderr.
 */

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_world_model/error.hpp"
#include "onnx_world_model/pipeline.hpp"
#include "onnx_world_model/tensor.hpp"

namespace {

int failures = 0;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    ++failures;
  }
}

// The one stub shape this file needs: a backend that reports exactly the
// metadata the test hands it, including an empty provider list, and that never
// runs. Nothing here executes a stage, because the report is computed from
// metadata alone.
class StubBackend final : public onnx_world_model::ModelBackend {
 public:
  explicit StubBackend(onnx_world_model::ModelMetadata metadata)
      : metadata_(std::move(metadata)) {}

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors&) const override {
    throw onnx_world_model::Error(
        onnx_world_model::ErrorCode::invalid_argument,
        "Precision report tests never execute a component");
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

onnx_world_model::TensorSpec Port(
    std::string name,
    onnx_world_model::DataType data_type,
    std::vector<std::int64_t> shape) {
  return {
      .name = std::move(name),
      .data_type = data_type,
      .shape = std::move(shape),
      .device = std::nullopt,
  };
}

onnx_world_model::Model StubModel(
    std::vector<onnx_world_model::TensorSpec> inputs,
    std::vector<onnx_world_model::TensorSpec> outputs,
    std::vector<std::string> execution_providers) {
  onnx_world_model::ModelMetadata metadata;
  metadata.inputs = std::move(inputs);
  metadata.outputs = std::move(outputs);
  metadata.execution_providers = std::move(execution_providers);
  return onnx_world_model::Model(
      std::make_shared<StubBackend>(std::move(metadata)));
}

// An executable profile requires a parameter dtype for every component, and
// the two declared here are deliberately FLOAT16 and INT8: the report must
// keep both, so a non-floating declaration is proven not to be rejected. The
// ports are deliberately mixed -- INT64 token ids, FLOAT16 activations, a
// FLOAT logits tensor -- and none of them agrees with either declaration.
constexpr std::string_view kMixedPrecisionManifest = R"json(
{
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
          {"name": "input_ids", "dtype": "INT64", "shape": ["batch", "sequence"]},
          {"name": "pixel_values", "dtype": "FLOAT16", "shape": ["batch", 3, 8, 8]}
        ],
        "outputs": [
          {"name": "hidden", "dtype": "FLOAT16", "shape": ["batch", "sequence", 4]},
          {"name": "logits", "dtype": "FLOAT", "shape": ["batch", "sequence", 8]}
        ],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT16"
      },
      {
        "name": "quantized_head",
        "role": "decoder",
        "run_on": "always",
        "inputs": [
          {"name": "hidden", "dtype": "FLOAT16", "shape": ["batch", "sequence", 4]}
        ],
        "outputs": [
          {"name": "token_ids", "dtype": "INT64", "shape": ["batch", "sequence"]}
        ],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "INT8"
      }
    ],
    "connections": [
      {"source": "reasoner.hidden", "target": "quantized_head.hidden"}
    ],
    "stages": [
      {
        "name": "run",
        "kind": "single_pass",
        "components": ["reasoner", "quantized_head"],
        "run_on": "always"
      }
    ],
    "inputs": [
      {
        "port": "reasoner.input_ids",
        "kind": "external",
        "required": true,
        "semantic": "text.input_ids"
      },
      {
        "port": "reasoner.pixel_values",
        "kind": "external",
        "required": true,
        "semantic": "vision.pixel_values"
      }
    ],
    "outputs": [
      {"port": "quantized_head.token_ids", "alias": "tokens"},
      {"port": "reasoner.logits", "alias": "logits"}
    ],
    "profile": {"name": "precision-probe", "version": "1.0"}
  },
  "component_files": {
    "reasoner": "reasoner/model.onnx",
    "quantized_head": "quantized_head/model.onnx"
  }
}
)json";

// No profile, so no component has to declare a parameter dtype, and this one
// does not.
constexpr std::string_view kUndeclaredManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "plain",
        "role": "encoder",
        "run_on": "always",
        "inputs": [{"name": "x", "dtype": "FLOAT", "shape": [1]}],
        "outputs": [{"name": "y", "dtype": "FLOAT", "shape": [1]}]
      }
    ],
    "connections": [],
    "stages": [
      {
        "name": "run",
        "kind": "single_pass",
        "components": ["plain"],
        "run_on": "always"
      }
    ],
    "inputs": [{"port": "plain.x", "kind": "external", "required": true}],
    "outputs": [{"port": "plain.y", "alias": "value"}]
  },
  "component_files": {"plain": "model.onnx"}
}
)json";

// Two states carried by one component. They share the output port they read
// from, which is exactly the case the report has to de-duplicate, and they
// write into two different input ports, which is the case whose manifest order
// it has to preserve. The declared parameter dtype is INT8 while every state
// port is FLOAT16, so nothing may be inferred from one to the other.
constexpr std::string_view kStateManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "carrier",
        "role": "dynamics",
        "run_on": "step",
        "inputs": [
          {"name": "state_a", "dtype": "FLOAT16", "shape": [1]},
          {"name": "state_b", "dtype": "FLOAT16", "shape": [1]},
          {"name": "x", "dtype": "INT64", "shape": [1]}
        ],
        "outputs": [
          {"name": "next", "dtype": "FLOAT16", "shape": [1]},
          {"name": "logits", "dtype": "FLOAT", "shape": [1]}
        ],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "INT8"
      }
    ],
    "connections": [
      {"source": "carrier.next", "target": "carrier.state_a", "recurrent": true},
      {"source": "carrier.next", "target": "carrier.state_b", "recurrent": true}
    ],
    "stages": [
      {
        "name": "transition",
        "kind": "state_transition",
        "components": ["carrier"],
        "run_on": "step",
        "options": {"state_names": ["cache_a", "cache_b"]},
        "capabilities": ["loop_carried_state"]
      }
    ],
    "inputs": [
      {
        "port": "carrier.state_a",
        "kind": "generated",
        "required": true,
        "semantic": "state.initial",
        "generator": {"kind": "zeros"}
      },
      {
        "port": "carrier.state_b",
        "kind": "generated",
        "required": true,
        "semantic": "state.initial",
        "generator": {"kind": "zeros"}
      },
      {
        "port": "carrier.x",
        "kind": "external",
        "required": true,
        "semantic": "text.input_ids"
      }
    ],
    "outputs": [{"port": "carrier.logits", "alias": "logits"}],
    "states": [
      {
        "name": "cache_a",
        "kind": "kv_cache",
        "input": "carrier.state_a",
        "output": "carrier.next",
        "lifetime": "request",
        "release_after": "transition"
      },
      {
        "name": "cache_b",
        "kind": "recurrent",
        "input": "carrier.state_b",
        "output": "carrier.next",
        "lifetime": "request",
        "release_after": "transition"
      }
    ],
    "required_capabilities": ["loop_carried_state"]
  },
  "component_files": {"carrier": "model.onnx"}
}
)json";

using onnx_world_model::DataType;
using onnx_world_model::Model;
using onnx_world_model::PipelineManifest;
using onnx_world_model::PipelinePackage;

PipelinePackage MixedPrecisionPackage(
    std::vector<std::string> reasoner_providers = {"CPUExecutionProvider"},
    std::vector<std::string> head_providers = {"CPUExecutionProvider"}) {
  std::unordered_map<std::string, Model> models;
  models.emplace(
      "reasoner",
      StubModel(
          {Port("input_ids", DataType::int64, {-1, -1}),
           Port("pixel_values", DataType::float16, {-1, 3, 8, 8})},
          {Port("hidden", DataType::float16, {-1, -1, 4}),
           Port("logits", DataType::float32, {-1, -1, 8})},
          std::move(reasoner_providers)));
  models.emplace(
      "quantized_head",
      StubModel(
          {Port("hidden", DataType::float16, {-1, -1, 4})},
          {Port("token_ids", DataType::int64, {-1, -1})},
          std::move(head_providers)));
  return PipelinePackage(
      {}, PipelineManifest::Parse(kMixedPrecisionManifest), std::move(models));
}

PipelinePackage StatePackage() {
  std::unordered_map<std::string, Model> models;
  models.emplace(
      "carrier",
      StubModel(
          {Port("state_a", DataType::float16, {1}),
           Port("state_b", DataType::float16, {1}),
           Port("x", DataType::int64, {1})},
          {Port("next", DataType::float16, {1}),
           Port("logits", DataType::float32, {1})},
          {"CPUExecutionProvider"}));
  return PipelinePackage(
      {}, PipelineManifest::Parse(kStateManifest), std::move(models));
}

std::vector<std::string> PortNames(
    const std::vector<onnx_world_model::PrecisionPort>& ports) {
  std::vector<std::string> names;
  names.reserve(ports.size());
  for (const auto& port : ports) {
    names.push_back(port.name);
  }
  return names;
}

std::vector<std::string> PortTypes(
    const std::vector<onnx_world_model::PrecisionPort>& ports) {
  std::vector<std::string> types;
  types.reserve(ports.size());
  for (const auto& port : ports) {
    types.emplace_back(onnx_world_model::ToString(port.data_type));
  }
  return types;
}

}  // namespace

int main() {
  using onnx_world_model::ComponentPrecisionReport;

  try {

  // Manifest order, not the order the component sessions happen to hash into.
  {
    const PipelinePackage package = MixedPrecisionPackage();
    const std::vector<ComponentPrecisionReport> report =
        package.precision_report();

    Check(report.size() == 2, "one report entry per manifest component");
    Check(
        report.at(0).component == "reasoner" &&
            report.at(1).component == "quantized_head",
        "precision report follows manifest component order");
  }

  // A declared parameter dtype is copied verbatim. INT8 survives, which is
  // what proves the declaration is not restricted to a floating type, and
  // neither declaration is compared with any port dtype.
  {
    const PipelinePackage package = MixedPrecisionPackage();
    const std::vector<ComponentPrecisionReport> report =
        package.precision_report();

    Check(
        report.at(0).declared_parameter_data_type == DataType::float16,
        "a FLOAT16 parameter declaration is reported unchanged");
    Check(
        report.at(1).declared_parameter_data_type == DataType::int8,
        "an INT8 parameter declaration is reported unchanged");
  }

  // No declaration at all on a profile-less manifest is nullopt, not a
  // guessed default and not a failure.
  {
    std::unordered_map<std::string, Model> models;
    models.emplace(
        "plain",
        StubModel(
            {Port("x", DataType::float32, {1})},
            {Port("y", DataType::float32, {1})},
            {"CPUExecutionProvider"}));
    const PipelinePackage package(
        {}, PipelineManifest::Parse(kUndeclaredManifest), std::move(models));
    const std::vector<ComponentPrecisionReport> report =
        package.precision_report();

    Check(report.size() == 1, "the profile-less package reports one component");
    Check(
        !report.at(0).declared_parameter_data_type.has_value(),
        "an undeclared parameter dtype is reported as nullopt");
  }

  // Graph ports are the loaded graph's own, in graph order, with their real
  // types. The reasoner declares FLOAT16 parameters while taking INT64 token
  // ids and producing a FLOAT logits tensor, and all three are reported side
  // by side without any of them being reconciled.
  {
    const PipelinePackage package = MixedPrecisionPackage();
    const std::vector<ComponentPrecisionReport> report =
        package.precision_report();
    const ComponentPrecisionReport& reasoner = report.at(0);
    const ComponentPrecisionReport& head = report.at(1);

    Check(
        PortNames(reasoner.graph_inputs) ==
            std::vector<std::string>{"input_ids", "pixel_values"},
        "graph inputs are listed in graph order");
    Check(
        PortTypes(reasoner.graph_inputs) ==
            std::vector<std::string>{"int64", "float16"},
        "graph input dtypes are the live graph's own");
    Check(
        PortNames(reasoner.graph_outputs) ==
            std::vector<std::string>{"hidden", "logits"},
        "graph outputs are listed in graph order");
    Check(
        PortTypes(reasoner.graph_outputs) ==
            std::vector<std::string>{"float16", "float32"},
        "graph output dtypes are the live graph's own");
    Check(
        PortTypes(head.graph_inputs) == std::vector<std::string>{"float16"} &&
            PortTypes(head.graph_outputs) ==
                std::vector<std::string>{"int64"},
        "an INT8-parameter component still reports its real port dtypes");
    // The point of the previous check, stated as the invariant it defends: a
    // component whose declared parameter dtype matches none of its ports is
    // reported, never rejected and never corrected.
    Check(
        head.declared_parameter_data_type == DataType::int8 &&
            head.graph_inputs.at(0).data_type == DataType::float16,
        "declared parameter dtype and port dtype are reported independently");
  }

  // A component with no state at all reports two empty state lists rather
  // than repeating its graph ports.
  {
    const PipelinePackage package = MixedPrecisionPackage();
    const std::vector<ComponentPrecisionReport> report =
        package.precision_report();

    Check(
        report.at(0).state_inputs.empty() &&
            report.at(0).state_outputs.empty(),
        "a stateless component reports no state ports");
  }

  // State association: the input ports two states write into, in manifest
  // state order, and the single output port both read from, de-duplicated.
  {
    const PipelinePackage package = StatePackage();
    const std::vector<ComponentPrecisionReport> report =
        package.precision_report();
    const ComponentPrecisionReport& carrier = report.at(0);

    Check(
        PortNames(carrier.state_inputs) ==
            std::vector<std::string>{"carrier.state_a", "carrier.state_b"},
        "state inputs are qualified and follow manifest state order");
    Check(
        PortNames(carrier.state_outputs) ==
            std::vector<std::string>{"carrier.next"},
        "a state output named by two states is listed once");
    Check(
        PortTypes(carrier.state_inputs) ==
            std::vector<std::string>{"float16", "float16"} &&
            PortTypes(carrier.state_outputs) ==
                std::vector<std::string>{"float16"},
        "state port dtypes are the live graph's own");
    Check(
        carrier.declared_parameter_data_type == DataType::int8,
        "a state-carrying component keeps its INT8 parameter declaration");
    // The control tensor is a graph input and takes part in no state, so it
    // must appear in one list and not the other.
    Check(
        PortNames(carrier.graph_inputs) ==
            std::vector<std::string>{"state_a", "state_b", "x"},
        "graph inputs still list every port, state or not");
  }

  // Providers are whatever the backend actually selected...
  {
    const PipelinePackage package = MixedPrecisionPackage(
        {"CUDAExecutionProvider", "CPUExecutionProvider"},
        {"CPUExecutionProvider"});
    const std::vector<ComponentPrecisionReport> report =
        package.precision_report();

    Check(
        report.at(0).execution_providers ==
            std::vector<std::string>{
                "CUDAExecutionProvider", "CPUExecutionProvider"},
        "selected providers are reported in preference order");
    Check(
        report.at(1).execution_providers ==
            std::vector<std::string>{"CPUExecutionProvider"},
        "each component reports its own providers");
  }

  // ...including nothing at all. A custom in-process backend that reports no
  // providers is never credited with CPU.
  {
    const PipelinePackage package = MixedPrecisionPackage({}, {});
    const std::vector<ComponentPrecisionReport> report =
        package.precision_report();

    Check(
        report.at(0).execution_providers.empty() &&
            report.at(1).execution_providers.empty(),
        "a backend with no providers reports an empty provider list");
  }

  // Every reading is a detached value: editing one cannot change the package,
  // the manifest, or the next reading.
  {
    const PipelinePackage package = MixedPrecisionPackage();
    std::vector<ComponentPrecisionReport> first = package.precision_report();
    first.at(0).component = "mutated";
    first.at(0).declared_parameter_data_type = DataType::float64;
    first.at(0).graph_inputs.clear();
    first.at(0).execution_providers.clear();
    first.pop_back();

    const std::vector<ComponentPrecisionReport> second =
        package.precision_report();

    Check(second.size() == 2, "a later reading is unaffected by an edited one");
    Check(
        second.at(0).component == "reasoner" &&
            second.at(0).declared_parameter_data_type == DataType::float16 &&
            second.at(0).graph_inputs.size() == 2 &&
            second.at(0).execution_providers.size() == 1,
        "the package is not mutated by editing a reading");
    Check(
        package.manifest().Component("reasoner").parameter_data_type ==
            DataType::float16,
        "the manifest keeps its own declared parameter dtype");
  }

  // A Pipeline reports exactly what its package does, and reporting needs no
  // session, no execution, and no ONNX Runtime library.
  {
    onnx_world_model::Pipeline pipeline(MixedPrecisionPackage());
    const std::vector<ComponentPrecisionReport> report =
        pipeline.precision_report();

    Check(
        report.size() == 2 && report.at(0).component == "reasoner" &&
            report.at(1).declared_parameter_data_type == DataType::int8,
        "Pipeline::precision_report forwards the package's report");
  }

  } catch (const onnx_world_model::Error& error) {
    std::cerr << "FAILED: unexpected error: " << error.what() << '\n';
    return 1;
  }

  if (failures != 0) {
    std::cerr << failures << " precision report check(s) failed\n";
    return 1;
  }
  std::cout << "pipeline precision report tests passed\n";
  return 0;
}
