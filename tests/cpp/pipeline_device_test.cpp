/**
 * @agent-file
 * @agent-purpose: Standalone test executable for device-tensor preservation across PipelineSession and for the conservative connection transfer plan a PipelinePackage computes, both driven by in-process stub backends.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as pipeline_device_test; it counts failures through local Check and CheckThrows helpers and returns a non-zero exit code when any check fails. FakeDeviceBuffer is a non-host-accessible TensorBuffer whose shared DeviceCopyCounter records every CPU materialization, which is how these checks assert zero copies for rank adaptation, transform-free connections, reshape, and public outputs, and exactly one copy per host transform. PlacedBackend is the second stub shape: it reports whatever TensorSpec::device the test asks for, including std::nullopt for a backend that reports no placement at all, which is how the transfer plan is exercised with no execution provider present. Every component is a stub ModelBackend, so the run needs no ONNX Runtime library and no real ONNX model, and every package is built in memory from an embedded manifest string. The static assertions at the top of main are the assertion that placement is load-time only: Pipeline is constructible from a package and scheduling options and is not constructible from those plus PipelinePlacementOptions, so an already-built package cannot be handed placement that would silently do nothing.
 * @agent-side-effects: Writes failure descriptions to stderr.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
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

template <typename Function>
void CheckThrows(Function&& function, const char* message) {
  try {
    function();
    Check(false, message);
  } catch (const onnx_world_model::Error&) {
  }
}

// Counts every CPU materialization performed by FakeDeviceBuffer, so a test
// can assert that a device tensor crossed the host boundary zero times or
// exactly once instead of once per element.
struct DeviceCopyCounter {
  int copies{0};
};

class FakeDeviceBuffer final : public onnx_world_model::TensorBuffer {
 public:
  FakeDeviceBuffer(
      std::vector<std::byte> storage,
      std::shared_ptr<DeviceCopyCounter> counter)
      : storage_(std::move(storage)),
        counter_(std::move(counter)),
        device_("fake", 0) {}

  [[nodiscard]] const onnx_world_model::TensorDevice& device()
      const noexcept override {
    return device_;
  }

  [[nodiscard]] std::size_t size_bytes() const noexcept override {
    return storage_.size();
  }

  [[nodiscard]] bool is_host_accessible() const noexcept override {
    return false;
  }

  [[nodiscard]] const void* data() const noexcept override {
    return storage_.data();
  }

  [[nodiscard]] std::span<const std::byte> bytes() const override {
    throw onnx_world_model::Error(
        onnx_world_model::ErrorCode::invalid_argument,
        "Fake device buffer is not host-accessible");
  }

  void CopyToCpu(std::span<std::byte> destination) const override {
    if (destination.size() != storage_.size()) {
      throw onnx_world_model::Error(
          onnx_world_model::ErrorCode::invalid_argument,
          "Fake device copy destination has the wrong byte size");
    }
    ++counter_->copies;
    std::copy(storage_.begin(), storage_.end(), destination.begin());
  }

 private:
  std::vector<std::byte> storage_;
  std::shared_ptr<DeviceCopyCounter> counter_;
  onnx_world_model::TensorDevice device_;
};

onnx_world_model::Tensor MakeDeviceTensor(
    std::vector<std::int64_t> shape,
    std::span<const float> values,
    const std::shared_ptr<DeviceCopyCounter>& counter) {
  const std::span<const std::byte> raw = std::as_bytes(values);
  return onnx_world_model::Tensor::FromBuffer(
      onnx_world_model::DataType::float32,
      std::move(shape),
      std::make_shared<FakeDeviceBuffer>(
          std::vector<std::byte>(raw.begin(), raw.end()), counter));
}

// Emits a device-backed output and records the storage identity of both the
// input it received and the output it produced.
class DeviceProducerBackend final : public onnx_world_model::ModelBackend {
 public:
  explicit DeviceProducerBackend(
      std::shared_ptr<DeviceCopyCounter> counter)
      : counter_(std::move(counter)) {
    metadata_.inputs.push_back({
        .name = "x",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {-1, 4},
    });
    metadata_.outputs.push_back({
        .name = "y",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {-1, 4},
    });
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    const onnx_world_model::Tensor& x = inputs.at("x");
    seen_input = x.buffer().get();
    seen_input_device = std::string(x.device().type());
    seen_input_shape = x.shape();
    const std::array<float, 4> values{1.5F, -2.5F, 3.5F, 4.5F};
    onnx_world_model::Tensor output =
        MakeDeviceTensor({1, 4}, std::span(values), counter_);
    produced = output.buffer().get();
    return {{"y", std::move(output)}};
  }

  mutable const onnx_world_model::TensorBuffer* seen_input{nullptr};
  mutable const onnx_world_model::TensorBuffer* produced{nullptr};
  mutable std::string seen_input_device;
  mutable std::vector<std::int64_t> seen_input_shape;

 private:
  onnx_world_model::ModelMetadata metadata_;
  std::shared_ptr<DeviceCopyCounter> counter_;
};

// Echoes its single input and records where that input's storage lives.
class RecordingBackend final : public onnx_world_model::ModelBackend {
 public:
  RecordingBackend(
      std::string input_name,
      std::string output_name,
      onnx_world_model::DataType data_type,
      std::vector<std::int64_t> shape)
      : input_name_(std::move(input_name)),
        output_name_(std::move(output_name)) {
    metadata_.inputs.push_back({
        .name = input_name_,
        .data_type = data_type,
        .shape = shape,
    });
    metadata_.outputs.push_back({
        .name = output_name_,
        .data_type = data_type,
        .shape = std::move(shape),
    });
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    const onnx_world_model::Tensor& value = inputs.at(input_name_);
    seen_input = value.buffer().get();
    seen_input_device = std::string(value.device().type());
    seen_input_shape = value.shape();
    return {{output_name_, value}};
  }

  mutable const onnx_world_model::TensorBuffer* seen_input{nullptr};
  mutable std::string seen_input_device;
  mutable std::vector<std::int64_t> seen_input_shape;

 private:
  std::string input_name_;
  std::string output_name_;
  onnx_world_model::ModelMetadata metadata_;
};

constexpr std::string_view kDeviceDirectManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "producer",
        "role": "encoder",
        "run_on": "always",
        "inputs": [{"name": "x", "dtype": "FLOAT", "shape": ["batch", 4]}],
        "outputs": [{"name": "y", "dtype": "FLOAT", "shape": ["batch", 4]}]
      },
      {
        "name": "consumer",
        "role": "decoder",
        "run_on": "always",
        "inputs": [{"name": "z", "dtype": "FLOAT", "shape": ["batch", 4]}],
        "outputs": [{"name": "w", "dtype": "FLOAT", "shape": ["batch", 4]}]
      }
    ],
    "connections": [{"source": "producer.y", "target": "consumer.z"}],
    "stages": [
      {
        "name": "run",
        "kind": "single_pass",
        "components": ["producer", "consumer"],
        "run_on": "always"
      }
    ],
    "inputs": [{"port": "producer.x", "kind": "external", "required": true}],
    "outputs": [
      {"port": "producer.y", "alias": "produced"},
      {"port": "consumer.w", "alias": "consumed"}
    ]
  },
  "component_files": {
    "producer": "producer/model.onnx",
    "consumer": "consumer/model.onnx"
  }
}
)json";

constexpr std::string_view kDeviceReshapeManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "producer",
        "role": "encoder",
        "run_on": "always",
        "inputs": [{"name": "x", "dtype": "FLOAT", "shape": ["batch", 4]}],
        "outputs": [{"name": "y", "dtype": "FLOAT", "shape": ["batch", 4]}]
      },
      {
        "name": "consumer",
        "role": "decoder",
        "run_on": "always",
        "inputs": [{"name": "z", "dtype": "FLOAT", "shape": [2, 2]}],
        "outputs": [{"name": "w", "dtype": "FLOAT", "shape": [2, 2]}]
      }
    ],
    "connections": [
      {
        "source": "producer.y",
        "target": "consumer.z",
        "transform": "reshape",
        "parameters": {"shape": [2, 2]}
      }
    ],
    "stages": [
      {
        "name": "run",
        "kind": "single_pass",
        "components": ["producer", "consumer"],
        "run_on": "always"
      }
    ],
    "inputs": [{"port": "producer.x", "kind": "external", "required": true}],
    "outputs": [{"port": "consumer.w", "alias": "consumed"}],
    "required_capabilities": ["tensor_reshape"]
  },
  "component_files": {
    "producer": "producer/model.onnx",
    "consumer": "consumer/model.onnx"
  }
}
)json";

constexpr std::string_view kDeviceCastManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "producer",
        "role": "encoder",
        "run_on": "always",
        "inputs": [{"name": "x", "dtype": "FLOAT", "shape": ["batch", 4]}],
        "outputs": [{"name": "y", "dtype": "FLOAT", "shape": ["batch", 4]}]
      },
      {
        "name": "consumer",
        "role": "decoder",
        "run_on": "always",
        "inputs": [{"name": "z", "dtype": "INT64", "shape": ["batch", 4]}],
        "outputs": [{"name": "w", "dtype": "INT64", "shape": ["batch", 4]}]
      }
    ],
    "connections": [
      {
        "source": "producer.y",
        "target": "consumer.z",
        "transform": "cast",
        "parameters": {"to": "INT64"}
      }
    ],
    "stages": [
      {
        "name": "run",
        "kind": "single_pass",
        "components": ["producer", "consumer"],
        "run_on": "always"
      }
    ],
    "inputs": [{"port": "producer.x", "kind": "external", "required": true}],
    "outputs": [{"port": "consumer.w", "alias": "consumed"}],
    "required_capabilities": ["tensor_cast"]
  },
  "component_files": {
    "producer": "producer/model.onnx",
    "consumer": "consumer/model.onnx"
  }
}
)json";

constexpr std::string_view kDeviceIdentityReshapeManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "producer",
        "role": "encoder",
        "run_on": "always",
        "inputs": [{"name": "x", "dtype": "FLOAT", "shape": [1, 4]}],
        "outputs": [{"name": "y", "dtype": "FLOAT", "shape": [1, 4]}]
      },
      {
        "name": "consumer",
        "role": "decoder",
        "run_on": "always",
        "inputs": [{"name": "z", "dtype": "FLOAT", "shape": [1, 4]}],
        "outputs": [{"name": "w", "dtype": "FLOAT", "shape": [1, 4]}]
      }
    ],
    "connections": [
      {
        "source": "producer.y",
        "target": "consumer.z",
        "transform": "reshape",
        "parameters": {"shape": [1, 4]}
      }
    ],
    "stages": [
      {
        "name": "run",
        "kind": "single_pass",
        "components": ["producer", "consumer"],
        "run_on": "always"
      }
    ],
    "inputs": [{"port": "producer.x", "kind": "external", "required": true}],
    "outputs": [{"port": "consumer.w", "alias": "consumed"}],
    "required_capabilities": ["tensor_reshape"]
  },
  "component_files": {
    "producer": "producer/model.onnx",
    "consumer": "consumer/model.onnx"
  }
}
)json";

constexpr std::string_view kDeviceRecurrentManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "producer",
        "role": "encoder",
        "run_on": "always",
        "inputs": [{"name": "x", "dtype": "FLOAT", "shape": ["batch", 4]}],
        "outputs": [{"name": "y", "dtype": "FLOAT", "shape": ["batch", 4]}]
      },
      {
        "name": "consumer",
        "role": "dynamics",
        "run_on": "step",
        "inputs": [
          {"name": "z", "dtype": "FLOAT", "shape": ["batch", 4]},
          {"name": "carry", "dtype": "FLOAT", "shape": ["batch", 4]}
        ],
        "outputs": [{"name": "w", "dtype": "FLOAT", "shape": ["batch", 4]}]
      }
    ],
    "connections": [
      {"source": "producer.y", "target": "consumer.z"},
      {"source": "consumer.w", "target": "consumer.carry", "recurrent": true}
    ],
    "stages": [
      {
        "name": "run",
        "kind": "state_transition",
        "components": ["producer", "consumer"],
        "run_on": "step",
        "options": {"state_names": ["carry"]},
        "capabilities": ["loop_carried_state"]
      }
    ],
    "inputs": [
      {"port": "producer.x", "kind": "external", "required": true},
      {
        "port": "consumer.carry",
        "kind": "generated",
        "required": true,
        "semantic": "state.initial",
        "generator": {"kind": "zeros"}
      }
    ],
    "outputs": [{"state": "carry", "alias": "result"}],
    "states": [
      {
        "name": "carry",
        "kind": "recurrent",
        "input": "consumer.carry",
        "output": "consumer.w",
        "lifetime": "session",
        "release_after": "run"
      }
    ],
    "required_capabilities": ["loop_carried_state"]
  },
  "component_files": {
    "producer": "producer/model.onnx",
    "consumer": "consumer/model.onnx"
  }
}
)json";

// A stub whose ports report whatever device the test asks for, which is how
// the transfer plan is exercised with no execution provider present. A
// disengaged device is exactly what a backend written before TensorSpec::device
// existed reports, so it stands in for "unknown".
class PlacedBackend final : public onnx_world_model::ModelBackend {
 public:
  PlacedBackend(
      std::vector<onnx_world_model::TensorSpec> inputs,
      std::vector<onnx_world_model::TensorSpec> outputs) {
    metadata_.inputs = std::move(inputs);
    metadata_.outputs = std::move(outputs);
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    (void)inputs;
    onnx_world_model::NamedTensors outputs;
    for (const auto& spec : metadata_.outputs) {
      std::vector<std::int64_t> shape = spec.shape;
      for (auto& extent : shape) {
        extent = extent < 0 ? 1 : extent;
      }
      outputs.emplace(
          spec.name,
          onnx_world_model::Tensor::Zeros(spec.data_type, std::move(shape)));
    }
    return outputs;
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

onnx_world_model::TensorSpec Port(
    std::string name,
    std::vector<std::int64_t> shape,
    std::optional<onnx_world_model::TensorDevice> device,
    onnx_world_model::DataType data_type =
        onnx_world_model::DataType::float32) {
  return {
      .name = std::move(name),
      .data_type = data_type,
      .shape = std::move(shape),
      .device = std::move(device),
  };
}

// The producer/consumer pair every two-component manifest above declares,
// with the two ports the transfer plan reads placed where the test wants them.
std::unordered_map<std::string, onnx_world_model::Model> PlacedModels(
    std::optional<onnx_world_model::TensorDevice> produced_device,
    std::optional<onnx_world_model::TensorDevice> consumed_device,
    onnx_world_model::DataType consumer_type =
        onnx_world_model::DataType::float32,
    std::vector<std::int64_t> consumer_shape = {-1, 4},
    std::vector<std::int64_t> producer_shape = {-1, 4}) {
  const onnx_world_model::TensorDevice host("cpu", 0);
  std::unordered_map<std::string, onnx_world_model::Model> models;
  models.emplace(
      "producer",
      onnx_world_model::Model(std::make_shared<PlacedBackend>(
          std::vector<onnx_world_model::TensorSpec>{
              Port("x", producer_shape, host)},
          std::vector<onnx_world_model::TensorSpec>{
              Port("y", producer_shape, produced_device)})));
  models.emplace(
      "consumer",
      onnx_world_model::Model(std::make_shared<PlacedBackend>(
          std::vector<onnx_world_model::TensorSpec>{
              Port("z", consumer_shape, consumed_device, consumer_type)},
          std::vector<onnx_world_model::TensorSpec>{
              Port("w", consumer_shape, consumed_device, consumer_type)})));
  return models;
}

onnx_world_model::PipelineTransferPlan PlanFor(
    std::string_view manifest,
    std::unordered_map<std::string, onnx_world_model::Model> models,
    bool device_outputs_enabled = false) {
  return onnx_world_model::PipelinePackage(
             {},
             onnx_world_model::PipelineManifest::Parse(manifest),
             std::move(models),
             device_outputs_enabled)
      .transfer_plan();
}

}  // namespace

int main() {
  using onnx_world_model::Model;
  using onnx_world_model::PipelineManifest;
  using onnx_world_model::PipelinePackage;

  // Placement is load-time only, and that is enforced by the API's shape
  // rather than by a runtime check: a Pipeline built over a package whose
  // sessions already exist cannot be handed placement at all, so there is no
  // call that looks like it configures placement and quietly does nothing.
  static_assert(
      std::is_constructible_v<
          onnx_world_model::Pipeline,
          PipelinePackage,
          onnx_world_model::PipelineSchedulingOptions>,
      "Pipeline still accepts a package and admission-scheduling options");
  static_assert(
      !std::is_constructible_v<
          onnx_world_model::Pipeline,
          PipelinePackage,
          onnx_world_model::PipelineSchedulingOptions,
          onnx_world_model::PipelinePlacementOptions>,
      "Placement must not be accepted for an already-built package");

  // Device tensors survive the session boundary. A component output that no
  // transform touches reaches the next component and the public outputs as
  // the very same TensorBuffer, and nothing materializes it.
  {
    auto counter = std::make_shared<DeviceCopyCounter>();
    auto producer = std::make_shared<DeviceProducerBackend>(counter);
    auto consumer = std::make_shared<RecordingBackend>(
        "z", "w", onnx_world_model::DataType::float32,
        std::vector<std::int64_t>{-1, 4});
    std::unordered_map<std::string, Model> device_models;
    device_models.emplace("producer", Model(producer));
    device_models.emplace("consumer", Model(consumer));
    onnx_world_model::Pipeline device_pipeline(PipelinePackage(
        {},
        PipelineManifest::Parse(kDeviceDirectManifest),
        std::move(device_models)));
    auto device_session = device_pipeline.CreateSession();

    const std::array<float, 4> external{9.0F, 8.0F, 7.0F, 6.0F};
    onnx_world_model::Tensor device_input =
        MakeDeviceTensor({4}, std::span(external), counter);
    const onnx_world_model::TensorBuffer* external_buffer =
        device_input.buffer().get();
    auto device_outputs =
        device_session.RunStage("run", {{"producer.x", device_input}});

    // Rank adaptation from {4} to {1, 4} is metadata only.
    Check(
        producer->seen_input == external_buffer,
        "external rank adaptation preserves the device buffer");
    Check(
        producer->seen_input_shape == std::vector<std::int64_t>{1, 4},
        "external rank adaptation inserts the batch axis");
    Check(
        producer->seen_input_device == "fake",
        "external device tensor stays on its device");
    Check(
        consumer->seen_input == producer->produced,
        "transform-free connection forwards the identical device buffer");
    Check(
        device_outputs.at("produced").buffer().get() == producer->produced,
        "public output keeps the component's device buffer");
    Check(
        device_outputs.at("consumed").buffer().get() == producer->produced,
        "downstream public output keeps the device buffer");
    Check(
        device_outputs.at("produced").device().type() == "fake",
        "public output reports its device");
    Check(counter->copies == 0, "device pass-through performs no CPU copy");
    CheckThrows(
        [&device_outputs] {
          (void)device_outputs.at("produced").values<float>();
        },
        "device-only public output rejects host access");
    Check(
        device_outputs.at("produced").CopyToCpu().values<float>()[1] == -2.5F,
        "explicit materialization returns the device values");
    Check(counter->copies == 1, "explicit materialization copies once");
  }

  // A reshape connection is a shape-only view over the same storage.
  {
    auto counter = std::make_shared<DeviceCopyCounter>();
    auto producer = std::make_shared<DeviceProducerBackend>(counter);
    auto consumer = std::make_shared<RecordingBackend>(
        "z", "w", onnx_world_model::DataType::float32,
        std::vector<std::int64_t>{2, 2});
    std::unordered_map<std::string, Model> reshape_models;
    reshape_models.emplace("producer", Model(producer));
    reshape_models.emplace("consumer", Model(consumer));
    onnx_world_model::Pipeline reshape_pipeline(PipelinePackage(
        {},
        PipelineManifest::Parse(kDeviceReshapeManifest),
        std::move(reshape_models)));
    auto reshape_session = reshape_pipeline.CreateSession();

    const std::array<float, 4> external{0.0F, 0.0F, 0.0F, 0.0F};
    auto reshape_outputs = reshape_session.RunStage(
        "run",
        {
            {
                "producer.x",
                MakeDeviceTensor({1, 4}, std::span(external), counter),
            },
        });
    Check(
        consumer->seen_input == producer->produced,
        "reshape connection preserves the identical device buffer");
    Check(
        consumer->seen_input_shape == std::vector<std::int64_t>{2, 2},
        "reshape connection applies the requested shape");
    Check(
        reshape_outputs.at("consumed").buffer().get() == producer->produced,
        "reshaped public output keeps the device buffer");
    Check(counter->copies == 0, "reshape performs no CPU copy");
  }

  // A CPU-only transform materializes its device source exactly once at the
  // transform boundary, not once per element.
  {
    auto counter = std::make_shared<DeviceCopyCounter>();
    auto producer = std::make_shared<DeviceProducerBackend>(counter);
    auto consumer = std::make_shared<RecordingBackend>(
        "z", "w", onnx_world_model::DataType::int64,
        std::vector<std::int64_t>{-1, 4});
    std::unordered_map<std::string, Model> cast_models;
    cast_models.emplace("producer", Model(producer));
    cast_models.emplace("consumer", Model(consumer));
    onnx_world_model::Pipeline cast_pipeline(PipelinePackage(
        {},
        PipelineManifest::Parse(kDeviceCastManifest),
        std::move(cast_models)));
    auto cast_session = cast_pipeline.CreateSession();

    const std::array<float, 4> external{0.0F, 0.0F, 0.0F, 0.0F};
    auto cast_outputs = cast_session.RunStage(
        "run",
        {
            {
                "producer.x",
                MakeDeviceTensor({1, 4}, std::span(external), counter),
            },
        });
    Check(
        counter->copies == 1,
        "a CPU transform materializes its device source exactly once");
    Check(
        consumer->seen_input_device == "cpu",
        "a CPU transform hands the component host storage");
    const auto cast_values = cast_outputs.at("consumed").values<std::int64_t>();
    Check(
        cast_values.size() == 4 && cast_values[0] == 1 &&
            cast_values[1] == -2 && cast_values[2] == 3 &&
            cast_values[3] == 4,
        "cast transform produces the correct host values");
  }

  // The transfer plan classifies every manifest connection from where the
  // ports actually are. Nothing below runs a stage: the plan is built while
  // the package is, so constructing one is the whole exercise.
  {
    using onnx_world_model::PipelineTransferKind;
    using onnx_world_model::PipelineTransferPlan;
    using onnx_world_model::TensorDevice;

    const TensorDevice host("cpu", 0);
    const TensorDevice fake_zero("fake", 0);
    const TensorDevice fake_one("fake", 1);
    const TensorDevice other("other", 0);

    // Same device, no transform: the producer's buffer can be handed over.
    const PipelineTransferPlan direct = PlanFor(
        kDeviceDirectManifest, PlacedModels(fake_zero, fake_zero));
    Check(direct.transfers.size() == 1, "one transfer per manifest connection");
    Check(
        direct.transfers[0].source.qualified() == "producer.y" &&
            direct.transfers[0].target.qualified() == "consumer.z",
        "a transfer names the connection it came from");
    Check(
        direct.transfers[0].kind == PipelineTransferKind::direct,
        "identical known devices with no transform are direct");
    Check(
        direct.transfers[0].direct_bind_eligible,
        "a direct transfer is bind eligible");
    Check(
        direct.transfers[0].reason.empty(),
        "a direct transfer needs no reason");
    Check(
        direct.transfers[0].source_device == fake_zero &&
            direct.transfers[0].target_device == fake_zero,
        "a transfer reports both endpoint devices");
    Check(
        !direct.transfers[0].recurrent &&
            !direct.transfers[0].transform.has_value(),
        "a forward transform-free connection reports neither");
    Check(
        !direct.device_outputs_enabled,
        "device outputs are off unless the package says otherwise");

    // Host to device and back.
    const PipelineTransferPlan upload =
        PlanFor(kDeviceDirectManifest, PlacedModels(host, fake_zero));
    Check(
        upload.transfers[0].kind == PipelineTransferKind::upload &&
            !upload.transfers[0].direct_bind_eligible &&
            !upload.transfers[0].reason.empty(),
        "cpu to a device is an upload with a reason");
    const PipelineTransferPlan download =
        PlanFor(kDeviceDirectManifest, PlacedModels(fake_zero, host));
    Check(
        download.transfers[0].kind == PipelineTransferKind::download &&
            !download.transfers[0].direct_bind_eligible &&
            !download.transfers[0].reason.empty(),
        "a device to cpu is a download with a reason");

    // Two different non-CPU devices have no peer-to-peer path here, whether
    // they differ by ordinal or by type.
    const PipelineTransferPlan staged_by_id =
        PlanFor(kDeviceDirectManifest, PlacedModels(fake_zero, fake_one));
    Check(
        staged_by_id.transfers[0].kind == PipelineTransferKind::host_staged &&
            !staged_by_id.transfers[0].direct_bind_eligible &&
            !staged_by_id.transfers[0].reason.empty(),
        "two device ordinals stage through the host");
    const PipelineTransferPlan staged_by_type =
        PlanFor(kDeviceDirectManifest, PlacedModels(fake_zero, other));
    Check(
        staged_by_type.transfers[0].kind == PipelineTransferKind::host_staged,
        "two device types stage through the host");

    // A backend that reports no placement at all leaves the plan unknown,
    // which outranks every other classification.
    const PipelineTransferPlan unknown = PlanFor(
        kDeviceDirectManifest, PlacedModels(std::nullopt, fake_zero));
    Check(
        unknown.transfers[0].kind == PipelineTransferKind::unknown &&
            !unknown.transfers[0].direct_bind_eligible &&
            !unknown.transfers[0].source_device.has_value() &&
            unknown.transfers[0].target_device == fake_zero,
        "an unreported endpoint device makes the transfer unknown");
    const PipelineTransferPlan unknown_transform = PlanFor(
        kDeviceCastManifest,
        PlacedModels(
            std::nullopt,
            std::nullopt,
            onnx_world_model::DataType::int64));
    Check(
        unknown_transform.transfers[0].kind == PipelineTransferKind::unknown,
        "unknown devices outrank a host transform");

    // Every transform this runtime evaluates on the host is host_transform.
    const PipelineTransferPlan cast = PlanFor(
        kDeviceCastManifest,
        PlacedModels(
            fake_zero, fake_zero, onnx_world_model::DataType::int64));
    Check(
        cast.transfers[0].kind == PipelineTransferKind::host_transform &&
            !cast.transfers[0].direct_bind_eligible &&
            cast.transfers[0].transform == std::optional<std::string>("cast"),
        "a cast connection is a host transform");

    // A reshape is treated the same way unless the declared shapes are
    // identical, because binding a device buffer requires the original shape
    // to equal the shape of the view wrapped around it.
    const PipelineTransferPlan reshaped = PlanFor(
        kDeviceReshapeManifest,
        PlacedModels(
            fake_zero,
            fake_zero,
            onnx_world_model::DataType::float32,
            {2, 2}));
    Check(
        reshaped.transfers[0].kind == PipelineTransferKind::host_transform &&
            !reshaped.transfers[0].direct_bind_eligible,
        "a shape-changing reshape is conservatively a host transform");
    Check(
        reshaped.transfers[0].reason.find("identical") != std::string::npos,
        "a conservative reshape explains what would make it direct");
    const PipelineTransferPlan identity_reshape = PlanFor(
        kDeviceIdentityReshapeManifest,
        PlacedModels(
            fake_zero,
            fake_zero,
            onnx_world_model::DataType::float32,
            {1, 4},
            {1, 4}));
    Check(
        identity_reshape.transfers[0].kind == PipelineTransferKind::direct &&
            identity_reshape.transfers[0].direct_bind_eligible,
        "a reshape between identical declared shapes stays direct");
    std::string misleading_reshape(kDeviceIdentityReshapeManifest);
    const std::string identity_shape =
        R"json("parameters": {"shape": [1, 4]})json";
    misleading_reshape.replace(
        misleading_reshape.find(identity_shape),
        identity_shape.size(),
        R"json("parameters": {"shape": [2, 2]})json");
    const PipelineTransferPlan explicit_shape_change = PlanFor(
        misleading_reshape,
        PlacedModels(
            fake_zero,
            fake_zero,
            onnx_world_model::DataType::float32,
            {1, 4},
            {1, 4}));
    Check(
        explicit_shape_change.transfers[0].kind ==
                PipelineTransferKind::host_transform &&
            !explicit_shape_change.transfers[0].direct_bind_eligible,
        "an explicit shape change overrides identical port declarations");

    // Recurrent edges are classified like any other and keep manifest order.
    std::unordered_map<std::string, onnx_world_model::Model> recurrent_models;
    recurrent_models.emplace(
        "producer",
        Model(std::make_shared<PlacedBackend>(
            std::vector<onnx_world_model::TensorSpec>{
                Port("x", {-1, 4}, host)},
            std::vector<onnx_world_model::TensorSpec>{
                Port("y", {-1, 4}, fake_zero)})));
    recurrent_models.emplace(
        "consumer",
        Model(std::make_shared<PlacedBackend>(
            std::vector<onnx_world_model::TensorSpec>{
                Port("z", {-1, 4}, fake_zero),
                Port("carry", {-1, 4}, host)},
            std::vector<onnx_world_model::TensorSpec>{
                Port("w", {-1, 4}, fake_zero)})));
    const PipelineTransferPlan recurrent = PlanFor(
        kDeviceRecurrentManifest, std::move(recurrent_models), true);
    Check(
        recurrent.transfers.size() == 2 && !recurrent.transfers[0].recurrent &&
            recurrent.transfers[1].recurrent,
        "the plan keeps manifest order and flags the recurrent edge");
    Check(
        recurrent.transfers[0].kind == PipelineTransferKind::direct &&
            recurrent.transfers[1].kind == PipelineTransferKind::download,
        "a recurrent edge is classified like any other connection");
    Check(
        recurrent.device_outputs_enabled,
        "the constructor flag reaches the plan");
  }

  if (failures == 0) {
    std::cout << "pipeline device tests passed\n";
  }
  return failures == 0 ? 0 : 1;
}
