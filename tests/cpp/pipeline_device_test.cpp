/**
 * @agent-file
 * @agent-purpose: Standalone test executable for device-tensor preservation across PipelineSession: which paths keep a producer's TensorBuffer untouched and which materialize it on the host, driven by in-process stub backends.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as pipeline_device_test; it counts failures through local Check and CheckThrows helpers and returns a non-zero exit code when any check fails. FakeDeviceBuffer is a non-host-accessible TensorBuffer whose shared DeviceCopyCounter records every CPU materialization, which is how these checks assert zero copies for rank adaptation, transform-free connections, reshape, and public outputs, and exactly one copy per host transform. Every component is a stub ModelBackend, so the run needs no ONNX Runtime library and no real ONNX model.
 * @agent-side-effects: Writes failure descriptions to stderr.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
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

}  // namespace

int main() {
  using onnx_world_model::Model;
  using onnx_world_model::PipelineManifest;
  using onnx_world_model::PipelinePackage;

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

  if (failures == 0) {
    std::cout << "pipeline device tests passed\n";
  }
  return failures == 0 ? 0 : 1;
}
