/**
 * @agent-file
 * @agent-purpose: Standalone test executable for opt-in runtime telemetry: the disabled default that collects nothing, per-component call counts, byte totals and durations, outcome classification by ErrorCode, stage execution versus step versus completion counting across RunStage, StepStage, and an incremental StageRun, admission wait outcomes for a constrained stage kind and their absence for an unlimited one, device-to-host materialization counts and component input residency, the epoch reset, and which Pipeline values share a collector.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as pipeline_telemetry_test; it counts failures through local Check and CheckThrowsCode helpers and returns a non-zero exit code when any check fails. It includes public headers only -- it never reaches into src/pipeline_telemetry.hpp -- so every claim is made through Pipeline, PipelineSession, StageRun, and Pipeline::telemetry_snapshot(). Every component is a stub ModelBackend and every PipelinePackage is built in memory from an embedded manifest, so the run needs no ONNX Runtime library, no ONNX model, and no filesystem access. No check asserts an absolute duration or byte rate: a duration is only ever compared against zero, and a maximum only against its own total, because wall-clock magnitudes are a property of the machine rather than of the runtime. Byte totals are exact and are asserted exactly, because they come from Tensor::size_bytes() rather than from timing. The admission checks are the only concurrent ones: they park a stub backend on a shared Gate -- a condition variable, never a sleep -- so a permit is provably held while a second thread queues, and "it is queued" is observed through Pipeline::scheduling_stats().queued_executions rather than inferred from a wait that timed out, bounded by kAdmissionBudget so a scheduler that never enqueues fails a check instead of hanging CTest. A worker publishes its outcome and its finished flag under one mutex hold, and every reader takes that mutex.
 * @agent-side-effects: Writes failure descriptions to stderr and starts short-lived worker threads that block inside a stub backend.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_world_model/cancellation.hpp"
#include "onnx_world_model/error.hpp"
#include "onnx_world_model/pipeline.hpp"
#include "onnx_world_model/tensor.hpp"

namespace {

using onnx_world_model::CancellationSource;
using onnx_world_model::CancellationToken;
using onnx_world_model::DataType;
using onnx_world_model::ErrorCode;
using onnx_world_model::Model;
using onnx_world_model::ModelMetadata;
using onnx_world_model::NamedTensors;
using onnx_world_model::Pipeline;
using onnx_world_model::PipelineAdmissionStats;
using onnx_world_model::PipelineComponentStats;
using onnx_world_model::PipelineManifest;
using onnx_world_model::PipelinePackage;
using onnx_world_model::PipelineRunOptions;
using onnx_world_model::PipelineSchedulingOptions;
using onnx_world_model::PipelineSession;
using onnx_world_model::PipelineStageStats;
using onnx_world_model::PipelineTelemetryOptions;
using onnx_world_model::PipelineTelemetrySnapshot;
using onnx_world_model::StageEvent;
using onnx_world_model::StageEventKind;
using onnx_world_model::StageRun;
using onnx_world_model::Tensor;
using onnx_world_model::TensorBuffer;
using onnx_world_model::TensorDevice;

int failures = 0;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    ++failures;
  }
}

template <typename Function>
void CheckThrowsCode(
    Function&& function,
    ErrorCode expected,
    const char* message) {
  try {
    function();
    Check(false, message);
  } catch (const onnx_world_model::Error& error) {
    if (error.code() != expected) {
      std::cerr << "FAILED: " << message << " (got: " << error.what() << ")\n";
      ++failures;
    }
  }
}

// Long enough that ordinary scheduling noise cannot exhaust it, short enough
// that a scheduler that never admits fails the run quickly instead of hanging
// CTest.
constexpr std::chrono::seconds kAdmissionBudget{5};
constexpr auto kAdmissionBudgetMs =
    std::chrono::duration_cast<std::chrono::milliseconds>(kAdmissionBudget);
// Long enough that a queue insertion is always observed before the deadline
// can fire, so a test that must see a request queued first is never racing
// its own timeout.
constexpr std::chrono::milliseconds kQueuedDeadline{1500};

// The six stage kinds every enabled reading carries.
constexpr std::array<std::string_view, 6> kStageKinds{
    "single_pass",
    "autoregressive",
    "iterative",
    "state_transition",
    "composite",
    "on_demand",
};

// -- Stub backends ----------------------------------------------------------

// Adds one to every element of its input. `failure` lets a test make the very
// next call throw a chosen ErrorCode, which is how outcome classification is
// asserted without depending on a message.
class CountingBackend final : public onnx_world_model::ModelBackend {
 public:
  CountingBackend(
      std::string input,
      std::string output,
      std::vector<std::int64_t> shape,
      std::shared_ptr<std::optional<ErrorCode>> failure = nullptr)
      : input_(std::move(input)),
        output_(std::move(output)),
        failure_(std::move(failure)) {
    metadata_.inputs.push_back({
        .name = input_,
        .data_type = DataType::float32,
        .shape = shape,
    });
    metadata_.outputs.push_back({
        .name = output_,
        .data_type = DataType::float32,
        .shape = std::move(shape),
    });
  }

  [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
    return metadata_;
  }

  [[nodiscard]] NamedTensors Run(const NamedTensors& inputs) const override {
    return Run(inputs, CancellationToken{});
  }

  [[nodiscard]] NamedTensors Run(
      const NamedTensors& inputs,
      const CancellationToken& cancellation) const override {
    cancellation.ThrowIfCancellationRequested();
    if (failure_ != nullptr && failure_->has_value()) {
      throw onnx_world_model::Error(
          **failure_, "Scripted component failure for the telemetry test");
    }
    Tensor result = inputs.at(input_);
    auto values = std::span(
        reinterpret_cast<float*>(result.mutable_bytes().data()),
        result.element_count());
    for (float& value : values) {
      value += 1.0F;
    }
    return {{output_, std::move(result)}};
  }

 private:
  ModelMetadata metadata_;
  std::string input_;
  std::string output_;
  std::shared_ptr<std::optional<ErrorCode>> failure_;
};

// Counts every CPU materialization, so a device tensor can be asserted to
// cross the host boundary exactly once rather than once per element.
struct DeviceCopyCounter {
  int copies{0};
};

class FakeDeviceBuffer final : public TensorBuffer {
 public:
  FakeDeviceBuffer(
      std::vector<std::byte> storage,
      std::shared_ptr<DeviceCopyCounter> counter)
      : storage_(std::move(storage)),
        counter_(std::move(counter)),
        device_("fake", 0) {}

  [[nodiscard]] const TensorDevice& device() const noexcept override {
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
        ErrorCode::invalid_argument,
        "Fake device buffer is not host-accessible");
  }

  void CopyToCpu(std::span<std::byte> destination) const override {
    if (destination.size() != storage_.size()) {
      throw onnx_world_model::Error(
          ErrorCode::invalid_argument,
          "Fake device copy destination has the wrong byte size");
    }
    ++counter_->copies;
    std::copy(storage_.begin(), storage_.end(), destination.begin());
  }

 private:
  std::vector<std::byte> storage_;
  std::shared_ptr<DeviceCopyCounter> counter_;
  TensorDevice device_;
};

[[nodiscard]] Tensor MakeDeviceTensor(
    std::vector<std::int64_t> shape,
    std::span<const float> values,
    const std::shared_ptr<DeviceCopyCounter>& counter) {
  const std::span<const std::byte> raw = std::as_bytes(values);
  return Tensor::FromBuffer(
      DataType::float32,
      std::move(shape),
      std::make_shared<FakeDeviceBuffer>(
          std::vector<std::byte>(raw.begin(), raw.end()), counter));
}

// Ignores whatever it is handed and returns a device-backed float32 output, so
// the connection that consumes it has to materialize it.
class DeviceProducerBackend final : public onnx_world_model::ModelBackend {
 public:
  explicit DeviceProducerBackend(std::shared_ptr<DeviceCopyCounter> counter)
      : counter_(std::move(counter)) {
    metadata_.inputs.push_back({
        .name = "x",
        .data_type = DataType::float32,
        .shape = {-1, 4},
    });
    metadata_.outputs.push_back({
        .name = "y",
        .data_type = DataType::float32,
        .shape = {-1, 4},
    });
  }

  [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
    return metadata_;
  }

  [[nodiscard]] NamedTensors Run(const NamedTensors& inputs) const override {
    (void)inputs;
    static constexpr std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
    NamedTensors outputs;
    outputs.emplace(
        "y", MakeDeviceTensor({1, 4}, std::span(values), counter_));
    return outputs;
  }

 private:
  ModelMetadata metadata_;
  std::shared_ptr<DeviceCopyCounter> counter_;
};

// The int64 consumer on the far side of the cast connection.
class Int64ConsumerBackend final : public onnx_world_model::ModelBackend {
 public:
  Int64ConsumerBackend() {
    metadata_.inputs.push_back({
        .name = "z",
        .data_type = DataType::int64,
        .shape = {-1, 4},
    });
    metadata_.outputs.push_back({
        .name = "w",
        .data_type = DataType::int64,
        .shape = {-1, 4},
    });
  }

  [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
    return metadata_;
  }

  [[nodiscard]] NamedTensors Run(const NamedTensors& inputs) const override {
    return {{"w", inputs.at("z")}};
  }

 private:
  ModelMetadata metadata_;
};

// The handshake the admission checks go through: a backend announces that it
// entered and then waits for the test to release it, so "a permit is held" is
// a recorded fact rather than an inference from timing.
struct Gate {
  std::mutex mutex;
  std::condition_variable condition;
  std::size_t entered{0};
  bool open{false};
};

class GatedBackend final : public onnx_world_model::ModelBackend {
 public:
  GatedBackend(std::shared_ptr<Gate> gate, std::vector<std::int64_t> shape)
      : gate_(std::move(gate)) {
    metadata_.inputs.push_back({
        .name = "x",
        .data_type = DataType::float32,
        .shape = shape,
    });
    metadata_.outputs.push_back({
        .name = "y",
        .data_type = DataType::float32,
        .shape = std::move(shape),
    });
  }

  [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
    return metadata_;
  }

  [[nodiscard]] NamedTensors Run(const NamedTensors& inputs) const override {
    {
      std::unique_lock lock(gate_->mutex);
      ++gate_->entered;
      gate_->condition.notify_all();
      gate_->condition.wait(lock, [this] { return gate_->open; });
    }
    return {{"y", inputs.at("x")}};
  }

 private:
  ModelMetadata metadata_;
  std::shared_ptr<Gate> gate_;
};

[[nodiscard]] bool WaitForEntries(Gate& gate, std::size_t count) {
  std::unique_lock lock(gate.mutex);
  return gate.condition.wait_for(
      lock, kAdmissionBudgetMs, [&gate, count] {
        return gate.entered >= count;
      });
}

void OpenGate(Gate& gate) {
  {
    std::scoped_lock lock(gate.mutex);
    gate.open = true;
  }
  gate.condition.notify_all();
}

// -- Manifests --------------------------------------------------------------

// Three independent one-component stages, one per stage kind this file
// measures: a single pass, a state transition, and a three-step iterative
// loop. Having three kinds is what makes "one execution, many steps, one
// completion" observable at all.
constexpr std::string_view kTelemetryManifest = R"json(
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
        "name": "counter",
        "role": "dynamics",
        "run_on": "step",
        "inputs": [{"name": "state", "dtype": "FLOAT", "shape": [1]}],
        "outputs": [{"name": "next_state", "dtype": "FLOAT", "shape": [1]}],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT"
      },
      {
        "name": "looper",
        "role": "dynamics",
        "run_on": "step",
        "inputs": [{"name": "state", "dtype": "FLOAT", "shape": [1]}],
        "outputs": [{"name": "next_state", "dtype": "FLOAT", "shape": [1]}],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT"
      }
    ],
    "connections": [
      {
        "source": "counter.next_state",
        "target": "counter.state",
        "recurrent": true
      },
      {
        "source": "looper.next_state",
        "target": "looper.state",
        "recurrent": true
      }
    ],
    "stages": [
      {
        "name": "pass",
        "kind": "single_pass",
        "components": ["producer"],
        "run_on": "always"
      },
      {
        "name": "transition",
        "kind": "state_transition",
        "components": ["counter"],
        "run_on": "step",
        "options": {"state_names": ["counter_state"]},
        "capabilities": ["loop_carried_state"]
      },
      {
        "name": "iterate",
        "kind": "iterative",
        "components": ["looper"],
        "run_on": "step",
        "options": {
          "scheduler": {
            "kind": "FlowMatchEulerDiscreteScheduler",
            "config_asset": "scheduler.json"
          },
          "default_steps": 3,
          "timestep": {},
          "state_inputs": ["looper.state"]
        },
        "capabilities": ["loop_carried_state"]
      }
    ],
    "inputs": [
      {"port": "producer.x", "kind": "external", "required": true},
      {
        "port": "counter.state",
        "kind": "generated",
        "required": true,
        "semantic": "state.initial",
        "generator": {"kind": "zeros"}
      },
      {
        "port": "looper.state",
        "kind": "generated",
        "required": true,
        "semantic": "state.initial",
        "generator": {"kind": "zeros"}
      }
    ],
    "outputs": [
      {"port": "producer.y", "alias": "produced"},
      {"state": "counter_state", "alias": "value"},
      {"state": "looper_state", "alias": "iterated"}
    ],
    "states": [
      {
        "name": "counter_state",
        "kind": "recurrent",
        "input": "counter.state",
        "output": "counter.next_state",
        "lifetime": "session",
        "release_after": "transition"
      },
      {
        "name": "looper_state",
        "kind": "recurrent",
        "input": "looper.state",
        "output": "looper.next_state",
        "lifetime": "request",
        "release_after": "iterate"
      }
    ],
    "assets": [{"path": "scheduler.json"}],
    "required_capabilities": ["loop_carried_state"]
  },
  "component_files": {
    "producer": "producer/model.onnx",
    "counter": "counter/model.onnx",
    "looper": "looper/model.onnx"
  }
}
)json";

// A device-backed producer feeding a cast connection, which is the shortest
// path to exactly one host materialization inside one stage.
constexpr std::string_view kDeviceManifest = R"json(
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

// One gated single-pass stage, which is all the admission checks need.
constexpr std::string_view kGatedManifest = R"json(
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
      }
    ],
    "connections": [],
    "stages": [
      {
        "name": "pass",
        "kind": "single_pass",
        "components": ["producer"],
        "run_on": "always"
      }
    ],
    "inputs": [{"port": "producer.x", "kind": "external", "required": true}],
    "outputs": [{"port": "producer.y", "alias": "produced"}]
  },
  "component_files": {"producer": "model.onnx"}
}
)json";

// -- Package and pipeline helpers -------------------------------------------

[[nodiscard]] PipelineTelemetryOptions Telemetry(bool enabled) {
  PipelineTelemetryOptions options;
  options.enabled = enabled;
  return options;
}

[[nodiscard]] PipelinePackage MakePackage(
    const std::shared_ptr<std::optional<ErrorCode>>& failure = nullptr) {
  std::unordered_map<std::string, Model> models;
  models.emplace(
      "producer",
      Model(std::make_shared<CountingBackend>(
          "x", "y", std::vector<std::int64_t>{1, 4}, failure)));
  models.emplace(
      "counter",
      Model(std::make_shared<CountingBackend>(
          "state", "next_state", std::vector<std::int64_t>{1})));
  models.emplace(
      "looper",
      Model(std::make_shared<CountingBackend>(
          "state", "next_state", std::vector<std::int64_t>{1})));
  return PipelinePackage(
      {}, PipelineManifest::Parse(kTelemetryManifest), std::move(models));
}

[[nodiscard]] Pipeline MakePipeline(
    bool telemetry_enabled = true,
    const std::shared_ptr<std::optional<ErrorCode>>& failure = nullptr) {
  return Pipeline(
      MakePackage(failure),
      PipelineSchedulingOptions{},
      Telemetry(telemetry_enabled));
}

[[nodiscard]] Pipeline MakeDevicePipeline(
    const std::shared_ptr<DeviceCopyCounter>& counter) {
  std::unordered_map<std::string, Model> models;
  models.emplace(
      "producer", Model(std::make_shared<DeviceProducerBackend>(counter)));
  models.emplace("consumer", Model(std::make_shared<Int64ConsumerBackend>()));
  return Pipeline(
      PipelinePackage(
          {}, PipelineManifest::Parse(kDeviceManifest), std::move(models)),
      PipelineSchedulingOptions{},
      Telemetry(true));
}

[[nodiscard]] Pipeline MakeGatedPipeline(
    const std::shared_ptr<Gate>& gate,
    PipelineSchedulingOptions scheduling) {
  std::unordered_map<std::string, Model> models;
  models.emplace(
      "producer",
      Model(std::make_shared<GatedBackend>(
          gate, std::vector<std::int64_t>{1, 4})));
  return Pipeline(
      PipelinePackage(
          {}, PipelineManifest::Parse(kGatedManifest), std::move(models)),
      std::move(scheduling),
      Telemetry(true));
}

[[nodiscard]] NamedTensors PassInputs() {
  static constexpr std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
  NamedTensors inputs;
  inputs.emplace(
      "producer.x", Tensor::FromValues<float>({1, 4}, std::span(values)));
  return inputs;
}

// 4 float32 values, which is the exact byte total every presentation check
// below is written against.
constexpr std::uint64_t kPassInputBytes = 16;

[[nodiscard]] NamedTensors DevicePassInputs(
    const std::shared_ptr<DeviceCopyCounter>& counter) {
  static constexpr std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
  NamedTensors inputs;
  inputs.emplace(
      "producer.x", MakeDeviceTensor({1, 4}, std::span(values), counter));
  return inputs;
}

[[nodiscard]] PipelineComponentStats ComponentStats(
    const Pipeline& pipeline,
    const std::string& component) {
  const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
  const auto found = snapshot.components.find(component);
  Check(
      found != snapshot.components.end(),
      "An enabled reading carries every manifest component");
  return found == snapshot.components.end() ? PipelineComponentStats{}
                                            : found->second;
}

[[nodiscard]] PipelineStageStats StageStats(
    const Pipeline& pipeline,
    const std::string& stage) {
  const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
  const auto found = snapshot.stages.find(stage);
  Check(
      found != snapshot.stages.end(),
      "An enabled reading carries every manifest stage");
  return found == snapshot.stages.end() ? PipelineStageStats{} : found->second;
}

[[nodiscard]] PipelineAdmissionStats AdmissionStats(
    const Pipeline& pipeline,
    const std::string& kind) {
  const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
  const auto found = snapshot.admission_by_stage_kind.find(kind);
  Check(
      found != snapshot.admission_by_stage_kind.end(),
      "An enabled reading carries every executable stage kind");
  return found == snapshot.admission_by_stage_kind.end()
             ? PipelineAdmissionStats{}
             : found->second;
}

// A duration is never asserted in absolute terms: it either advanced or it did
// not, and one call's maximum can never exceed the total it contributed to.
void CheckDuration(
    std::uint64_t total,
    std::uint64_t maximum,
    const char* message) {
  Check(total > 0, message);
  Check(maximum > 0 && maximum <= total, message);
}

// True once the pipeline's admission queue holds exactly `count` requests.
// This is the file's one synchronization on queue state, and it is the
// scheduler publishing its own depth rather than an inference from timing.
[[nodiscard]] bool WaitForQueued(const Pipeline& pipeline, std::size_t count) {
  const auto deadline = std::chrono::steady_clock::now() + kAdmissionBudget;
  for (;;) {
    if (pipeline.scheduling_stats().queued_executions == count) {
      return true;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
}

// A worker that runs one stage on its own session and records how it ended.
struct Worker {
  std::thread thread;
  std::mutex mutex;
  std::condition_variable condition;
  bool finished{false};
  bool completed{false};
  std::optional<ErrorCode> error;
};

void FinishWorker(
    Worker& worker,
    bool completed,
    std::optional<ErrorCode> error) {
  {
    std::scoped_lock lock(worker.mutex);
    worker.completed = completed;
    worker.error = error;
    worker.finished = true;
  }
  worker.condition.notify_all();
}

[[nodiscard]] bool WaitForWorker(Worker& worker) {
  std::unique_lock lock(worker.mutex);
  return worker.condition.wait_for(
      lock, kAdmissionBudgetMs, [&worker] { return worker.finished; });
}

[[nodiscard]] std::optional<ErrorCode> FailedWith(Worker& worker) {
  std::scoped_lock lock(worker.mutex);
  return worker.error;
}

// Runs one "pass" execution on its own session under `token`, publishing how
// it ended so the main thread can assert an outcome without owning the thread.
void StartPassWorker(
    Worker& worker,
    const Pipeline& pipeline,
    const CancellationToken& token) {
  worker.thread = std::thread([&worker, &pipeline, token] {
    try {
      PipelineSession session = pipeline.CreateSession();
      PipelineRunOptions options;
      options.cancellation = token;
      (void)session.RunStage("pass", PassInputs(), {}, options);
      FinishWorker(worker, true, std::nullopt);
    } catch (const onnx_world_model::Error& error) {
      FinishWorker(worker, false, error.code());
    } catch (...) {
      FinishWorker(worker, false, std::nullopt);
    }
  });
}

void Join(Worker& worker) {
  if (worker.thread.joinable()) {
    worker.thread.join();
  }
}

}  // namespace

// Every check lives here rather than in main, so an unexpected Error -- a
// manifest this file got wrong, say -- is reported as a failure instead of
// aborting the process with no message.
namespace {

void RunTelemetryChecks() {
  {
    // A pipeline that was never asked to collect anything reports the
    // disabled reading rather than a map of zeros, which is what tells a
    // caller that nothing was measured instead of that nothing happened.
    Pipeline pipeline = MakePipeline(false);
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("pass", PassInputs());

    PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
    Check(!snapshot.enabled, "a disabled pipeline reports enabled false");
    Check(snapshot.epoch == 0, "a disabled pipeline reports epoch zero");
    Check(snapshot.components.empty(), "a disabled reading has no components");
    Check(snapshot.stages.empty(), "a disabled reading has no stages");
    Check(
        snapshot.admission_by_stage_kind.empty(),
        "a disabled reading has no admission entries");
    Check(
        snapshot.transfers.device_to_host_copies == 0 &&
            snapshot.transfers.device_to_host_bytes == 0 &&
            snapshot.transfers.component_input_bytes_host == 0 &&
            snapshot.transfers.component_input_bytes_device_resident == 0,
        "a disabled reading has zero transfer counters");

    // Resetting something that collects nothing is a no-op rather than an
    // error, so a caller can reset unconditionally.
    pipeline.ResetTelemetry();
    snapshot = pipeline.telemetry_snapshot();
    Check(
        !snapshot.enabled && snapshot.epoch == 0,
        "resetting a disabled pipeline changes nothing");
  }

  {
    // An enabled pipeline is fully populated before anything runs, so a
    // caller reads a key rather than testing for it.
    const Pipeline pipeline = MakePipeline(true);
    const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
    Check(snapshot.enabled, "an enabled pipeline reports enabled true");
    Check(snapshot.epoch == 1, "an enabled pipeline starts at epoch one");
    Check(
        snapshot.components.size() == 3 &&
            snapshot.components.contains("producer") &&
            snapshot.components.contains("counter") &&
            snapshot.components.contains("looper"),
        "an idle reading already carries every manifest component");
    Check(
        snapshot.stages.size() == 3 && snapshot.stages.contains("pass") &&
            snapshot.stages.contains("transition") &&
            snapshot.stages.contains("iterate"),
        "an idle reading already carries every manifest stage");
    Check(
        snapshot.admission_by_stage_kind.size() == kStageKinds.size(),
        "an idle reading carries all six executable stage kinds");
    for (const std::string_view kind : kStageKinds) {
      Check(
          snapshot.admission_by_stage_kind.contains(std::string(kind)),
          "an idle reading carries every executable stage kind by name");
    }
    const PipelineComponentStats idle = ComponentStats(pipeline, "producer");
    Check(
        idle.successful_calls == 0 && idle.failed_calls == 0 &&
            idle.input_bytes == 0 && idle.output_bytes == 0 &&
            idle.total_duration_ns == 0 && idle.max_duration_ns == 0,
        "an idle component reads as all zeros");
  }

  {
    // One stage, one component call: the byte totals are exact because they
    // come from Tensor::size_bytes() rather than from timing.
    const Pipeline pipeline = MakePipeline(true);
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("pass", PassInputs());

    const PipelineComponentStats component =
        ComponentStats(pipeline, "producer");
    Check(
        component.successful_calls == 1,
        "a successful component call is counted once");
    Check(
        component.failed_calls == 0 && component.cancelled_calls == 0 &&
            component.deadline_exceeded_calls == 0,
        "a successful component call fills no failure bucket");
    Check(
        component.input_bytes == kPassInputBytes,
        "component input bytes are the exact presented byte total");
    Check(
        component.output_bytes == kPassInputBytes,
        "component output bytes are the exact returned byte total");
    CheckDuration(
        component.total_duration_ns,
        component.max_duration_ns,
        "a component call reports a positive duration whose maximum is "
        "within its total");

    const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
    Check(
        snapshot.transfers.component_input_bytes_host == kPassInputBytes &&
            snapshot.transfers.component_input_bytes_device_resident == 0,
        "a host input is presented entirely as host-resident bytes");
    Check(
        snapshot.transfers.device_to_host_copies == 0 &&
            snapshot.transfers.device_to_host_bytes == 0,
        "a CPU-only stage materializes nothing");

    // A component that this stage never runs is present and untouched.
    const PipelineComponentStats untouched =
        ComponentStats(pipeline, "counter");
    Check(
        untouched.successful_calls == 0 && untouched.input_bytes == 0,
        "a component the stage never ran stays at zero");
  }

  {
    // Outcome classification is by ErrorCode, so a cancellation and a
    // deadline are reported as themselves rather than as failures.
    struct Case {
      ErrorCode code;
      const char* message;
    };
    const std::array<Case, 3> cases{
        Case{ErrorCode::runtime_execution, "a failing component call"},
        Case{ErrorCode::cancelled, "a cancelled component call"},
        Case{ErrorCode::deadline_exceeded, "a component call past its "
                                           "deadline"},
    };
    for (const Case& scenario : cases) {
      auto failure = std::make_shared<std::optional<ErrorCode>>(scenario.code);
      const Pipeline pipeline = MakePipeline(true, failure);
      PipelineSession session = pipeline.CreateSession();
      CheckThrowsCode(
          [&session] { (void)session.RunStage("pass", PassInputs()); },
          scenario.code,
          scenario.message);

      const PipelineComponentStats component =
          ComponentStats(pipeline, "producer");
      const PipelineStageStats stage = StageStats(pipeline, "pass");
      const std::uint64_t component_bucket =
          scenario.code == ErrorCode::cancelled
              ? component.cancelled_calls
              : scenario.code == ErrorCode::deadline_exceeded
                    ? component.deadline_exceeded_calls
                    : component.failed_calls;
      const std::uint64_t stage_bucket =
          scenario.code == ErrorCode::cancelled
              ? stage.cancelled_executions
              : scenario.code == ErrorCode::deadline_exceeded
                    ? stage.deadline_exceeded_executions
                    : stage.failed_executions;
      Check(
          component_bucket == 1 && component.successful_calls == 0,
          "a component outcome lands in exactly the bucket its ErrorCode "
          "names");
      Check(
          stage_bucket == 1 && stage.successful_executions == 0,
          "a stage outcome lands in exactly the bucket its ErrorCode names");
      Check(
          component.input_bytes == kPassInputBytes,
          "a failed attempt still counts the bytes it was presented");
      Check(
          component.output_bytes == 0,
          "a failed attempt returns no bytes to count");
      CheckDuration(
          component.total_duration_ns,
          component.max_duration_ns,
          "a failed component call is still timed");
      Check(
          stage.steps == 0 && stage.completions == 0,
          "a failed execution completed no step and no stage");
    }
  }

  {
    // Execution, step, and completion are three different things, and this is
    // where they are told apart: one RunStage of a three-step iterative stage
    // is one execution, three steps, and exactly one completion.
    const Pipeline pipeline = MakePipeline(true);
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("pass", PassInputs());

    PipelineStageStats pass = StageStats(pipeline, "pass");
    Check(
        pass.successful_executions == 1 && pass.steps == 1 &&
            pass.completions == 1,
        "a one-pass RunStage is one execution, one step, and one completion");
    CheckDuration(
        pass.total_execution_duration_ns,
        pass.max_execution_duration_ns,
        "a stage execution reports a positive duration whose maximum is "
        "within its total");

    (void)session.RunStage("iterate");
    const PipelineStageStats iterate = StageStats(pipeline, "iterate");
    Check(
        iterate.successful_executions == 1 && iterate.steps == 3 &&
            iterate.completions == 1,
        "a drained three-step stage is one execution, three steps, and one "
        "completion");
    Check(
        ComponentStats(pipeline, "looper").successful_calls == 3,
        "each step of an iterative stage is one component call");

    // A second execution of the same stage adds to the same entry.
    (void)session.RunStage("pass", PassInputs());
    pass = StageStats(pipeline, "pass");
    Check(
        pass.successful_executions == 2 && pass.steps == 2 &&
            pass.completions == 2,
        "a second execution of a stage accumulates rather than replacing");
  }

  {
    // The same three-step stage driven incrementally: every Step is its own
    // execution, including the terminal one that produces no step at all.
    const Pipeline pipeline = MakePipeline(true);
    PipelineSession session = pipeline.CreateSession();
    StageRun run = session.BeginStage("iterate");

    Check(
        StageStats(pipeline, "iterate").successful_executions == 0,
        "BeginStage is not an execution and counts nothing");

    for (int index = 0; index < 3; ++index) {
      const StageEvent event = run.Step();
      Check(
          event.kind == StageEventKind::iteration,
          "an iterative run reports iteration events");
    }
    PipelineStageStats stage = StageStats(pipeline, "iterate");
    Check(
        stage.successful_executions == 3 && stage.steps == 3 &&
            stage.completions == 0,
        "three explicit steps are three executions, three steps, and no "
        "completion yet");

    const StageEvent terminal = run.Step();
    Check(terminal.finished, "the fourth step is the terminal event");
    stage = StageStats(pipeline, "iterate");
    Check(
        stage.successful_executions == 4 && stage.steps == 3 &&
            stage.completions == 1,
        "the terminal step is an execution that adds a completion rather "
        "than a step");

    // A Finish on a run that already completed returns its cached outputs
    // without executing, so it must change nothing at all.
    const PipelineTelemetrySnapshot before = pipeline.telemetry_snapshot();
    (void)run.Finish();
    const PipelineStageStats after = StageStats(pipeline, "iterate");
    Check(
        after.successful_executions ==
                before.stages.at("iterate").successful_executions &&
            after.steps == before.stages.at("iterate").steps &&
            after.completions == before.stages.at("iterate").completions &&
            after.total_execution_duration_ns ==
                before.stages.at("iterate").total_execution_duration_ns,
        "a cached Finish changes no counter");
  }

  {
    // A partially stepped run finished by Finish: the drain is one further
    // execution that carries the remaining steps and the completion.
    const Pipeline pipeline = MakePipeline(true);
    PipelineSession session = pipeline.CreateSession();
    StageRun run = session.BeginStage("iterate");
    (void)run.Step();
    (void)run.Finish();

    const PipelineStageStats stage = StageStats(pipeline, "iterate");
    Check(
        stage.successful_executions == 2,
        "one Step and one draining Finish are two executions");
    Check(
        stage.steps == 3 && stage.completions == 1,
        "a drained run reports every step once and exactly one completion");
  }

  {
    // A direct StepStage bypasses the StageEvent state machine, so it counts
    // one execution and one step and never a completion.
    const Pipeline pipeline = MakePipeline(true);
    PipelineSession session = pipeline.CreateSession();
    (void)session.StepStage("transition");
    (void)session.StepStage("transition");

    const PipelineStageStats stage = StageStats(pipeline, "transition");
    Check(
        stage.successful_executions == 2 && stage.steps == 2 &&
            stage.completions == 0,
        "two direct steps are two executions and two steps with no "
        "completion");
    Check(
        ComponentStats(pipeline, "counter").successful_calls == 2,
        "each direct step runs its component once");
  }

  {
    // An unlimited stage kind takes no permit, so admission has nothing to
    // report while the execution itself is still measured.
    const Pipeline pipeline = MakePipeline(true);
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("pass", PassInputs());

    const PipelineAdmissionStats admission =
        AdmissionStats(pipeline, "single_pass");
    Check(
        admission.queued_acquisitions == 0 &&
            admission.admitted_acquisitions == 0 &&
            admission.cancelled_while_queued == 0 &&
            admission.deadline_while_queued == 0 &&
            admission.total_wait_ns == 0 && admission.max_wait_ns == 0,
        "an unlimited stage kind records no admission at all");
    Check(
        StageStats(pipeline, "pass").successful_executions == 1,
        "an unlimited execution is still measured as a stage execution");
  }

  {
    // A constrained kind granted at once is admitted with no wait and is
    // deliberately not counted as queued.
    auto gate = std::make_shared<Gate>();
    OpenGate(*gate);
    PipelineSchedulingOptions scheduling;
    scheduling.max_concurrent_executions = 1;
    const Pipeline pipeline = MakeGatedPipeline(gate, scheduling);
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("pass", PassInputs());

    const PipelineAdmissionStats admission =
        AdmissionStats(pipeline, "single_pass");
    Check(
        admission.admitted_acquisitions == 1 &&
            admission.queued_acquisitions == 0,
        "an immediate grant is admitted without being queued");
    Check(
        admission.total_wait_ns == 0 && admission.max_wait_ns == 0,
        "an immediate grant waited no time at all");
  }

  {
    // One permit, two executions: the second one really does queue, and its
    // wait is reported as a wait rather than as part of the stage.
    auto gate = std::make_shared<Gate>();
    PipelineSchedulingOptions scheduling;
    scheduling.max_concurrent_executions = 1;
    const Pipeline pipeline = MakeGatedPipeline(gate, scheduling);

    Worker holder;
    Worker waiter;
    StartPassWorker(holder, pipeline, CancellationToken{});
    Check(
        WaitForEntries(*gate, 1),
        "the first execution reaches the gated backend");
    StartPassWorker(waiter, pipeline, CancellationToken{});
    Check(
        WaitForQueued(pipeline, 1),
        "the second execution is observed in the admission queue");

    OpenGate(*gate);
    Check(WaitForWorker(holder), "the permit holder finishes");
    Check(WaitForWorker(waiter), "the queued execution is admitted");
    Join(holder);
    Join(waiter);

    const PipelineAdmissionStats admission =
        AdmissionStats(pipeline, "single_pass");
    Check(
        admission.queued_acquisitions == 1,
        "only the execution that had to wait is counted as queued");
    Check(
        admission.admitted_acquisitions == 2,
        "both executions were admitted in the end");
    Check(
        admission.cancelled_while_queued == 0 &&
            admission.deadline_while_queued == 0,
        "a granted queue outcome fills no failure bucket");
    CheckDuration(
        admission.total_wait_ns,
        admission.max_wait_ns,
        "a queued execution reports a positive wait whose maximum is within "
        "its total");
    Check(
        StageStats(pipeline, "pass").successful_executions == 2,
        "both executions are measured as stage executions");
  }

  {
    // A queued execution stopped by its own token never becomes an execution:
    // it is an admission outcome and nothing else.
    auto gate = std::make_shared<Gate>();
    PipelineSchedulingOptions scheduling;
    scheduling.max_concurrent_executions = 1;
    const Pipeline pipeline = MakeGatedPipeline(gate, scheduling);

    Worker holder;
    Worker cancelled;
    CancellationSource source;
    StartPassWorker(holder, pipeline, CancellationToken{});
    Check(
        WaitForEntries(*gate, 1),
        "the permit holder reaches the gated backend");
    StartPassWorker(cancelled, pipeline, source.token());
    Check(
        WaitForQueued(pipeline, 1),
        "the cancelled execution is observed in the admission queue first");
    source.Cancel();
    Check(WaitForWorker(cancelled), "the queued execution is released");
    OpenGate(*gate);
    Check(WaitForWorker(holder), "the permit holder finishes");
    Join(holder);
    Join(cancelled);

    Check(
        FailedWith(cancelled) == ErrorCode::cancelled,
        "a queued execution released by a cancel fails with cancelled");
    const PipelineAdmissionStats admission =
        AdmissionStats(pipeline, "single_pass");
    Check(
        admission.queued_acquisitions == 1 &&
            admission.cancelled_while_queued == 1,
        "a queued cancellation is counted as one queued cancellation");
    Check(
        admission.admitted_acquisitions == 1,
        "the cancelled execution was never admitted");
    Check(
        admission.total_wait_ns > 0,
        "a queued cancellation still reports the time it waited");
    Check(
        StageStats(pipeline, "pass").successful_executions == 1 &&
            StageStats(pipeline, "pass").cancelled_executions == 0,
        "an execution stopped before admission is not a stage execution");
  }

  {
    // The same claim for a deadline, which is a different outcome rather than
    // a differently worded cancellation.
    auto gate = std::make_shared<Gate>();
    PipelineSchedulingOptions scheduling;
    scheduling.max_concurrent_executions = 1;
    const Pipeline pipeline = MakeGatedPipeline(gate, scheduling);

    Worker holder;
    Worker expired;
    CancellationSource source =
        CancellationSource::WithTimeout(kQueuedDeadline);
    StartPassWorker(holder, pipeline, CancellationToken{});
    Check(
        WaitForEntries(*gate, 1),
        "the permit holder reaches the gated backend");
    StartPassWorker(expired, pipeline, source.token());
    Check(
        WaitForQueued(pipeline, 1),
        "the expiring execution is observed in the admission queue first");
    Check(
        WaitForWorker(expired),
        "the queued execution is released by its deadline");
    OpenGate(*gate);
    Check(WaitForWorker(holder), "the permit holder finishes");
    Join(holder);
    Join(expired);

    Check(
        FailedWith(expired) == ErrorCode::deadline_exceeded,
        "a queued execution released by its deadline fails with "
        "deadline_exceeded");
    const PipelineAdmissionStats admission =
        AdmissionStats(pipeline, "single_pass");
    Check(
        admission.queued_acquisitions == 1 &&
            admission.deadline_while_queued == 1 &&
            admission.cancelled_while_queued == 0,
        "a queued deadline is counted as its own outcome");
    Check(
        admission.total_wait_ns > 0,
        "a queued deadline still reports the time it waited");
  }

  {
    // Exactly one host materialization per device source, with its exact byte
    // count, and no materialization at all for what stays on the host.
    auto counter = std::make_shared<DeviceCopyCounter>();
    const Pipeline pipeline = MakeDevicePipeline(counter);
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("run", PassInputs());

    const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
    Check(
        counter->copies == 1,
        "the cast connection materializes the device output exactly once");
    Check(
        snapshot.transfers.device_to_host_copies == 1,
        "one materialization is counted once");
    Check(
        snapshot.transfers.device_to_host_bytes == kPassInputBytes,
        "a materialization counts the bytes it moved");
    Check(
        snapshot.transfers.component_input_bytes_device_resident == 0,
        "a host input contributes nothing to device-resident bytes");
    // 16 bytes of float32 into the producer plus 32 bytes of int64 into the
    // consumer, which the cast produced on the host.
    Check(
        snapshot.transfers.component_input_bytes_host == 48,
        "presented host bytes are the exact total of both components");
    Check(
        ComponentStats(pipeline, "consumer").input_bytes == 32,
        "the cast target is presented as its own widened byte total");
  }

  {
    // The same package handed a device-resident external input: residency is
    // presentation rather than transfer, so nothing is copied for it.
    auto counter = std::make_shared<DeviceCopyCounter>();
    const Pipeline pipeline = MakeDevicePipeline(counter);
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("run", DevicePassInputs(counter));

    const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
    Check(
        snapshot.transfers.component_input_bytes_device_resident ==
            kPassInputBytes,
        "a device-resident input is presented as device-resident bytes");
    Check(
        snapshot.transfers.component_input_bytes_host == 32,
        "the host bytes are only what actually lived on the host");
    Check(
        snapshot.transfers.device_to_host_copies == 1 &&
            snapshot.transfers.device_to_host_bytes == kPassInputBytes,
        "presenting a device input copies nothing by itself, so the only "
        "materialization is still the cast");
    Check(
        ComponentStats(pipeline, "producer").input_bytes == kPassInputBytes,
        "a component's input bytes are the same whichever side they lived on");
  }

  {
    // Reset is a new epoch rather than an edit: the counters restart, the
    // maps stay populated, and collection continues.
    const Pipeline pipeline = MakePipeline(true);
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("pass", PassInputs());
    Check(
        StageStats(pipeline, "pass").successful_executions == 1,
        "the first epoch counted the execution");

    Pipeline resettable = pipeline;
    resettable.ResetTelemetry();

    const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
    Check(snapshot.epoch == 2, "a reset advances the epoch by one");
    Check(snapshot.enabled, "a reset leaves telemetry enabled");
    Check(
        snapshot.components.size() == 3 && snapshot.stages.size() == 3 &&
            snapshot.admission_by_stage_kind.size() == kStageKinds.size(),
        "a reset epoch is pre-populated exactly like the first one");
    const PipelineStageStats stage = snapshot.stages.at("pass");
    Check(
        stage.successful_executions == 0 && stage.steps == 0 &&
            stage.completions == 0 && stage.total_execution_duration_ns == 0,
        "a reset epoch starts at zero");
    Check(
        snapshot.components.at("producer").input_bytes == 0 &&
            snapshot.transfers.component_input_bytes_host == 0,
        "a reset epoch forgets the previous epoch's bytes");

    (void)session.RunStage("pass", PassInputs());
    const PipelineTelemetrySnapshot after = pipeline.telemetry_snapshot();
    Check(after.epoch == 2, "collecting again stays in the reset epoch");
    Check(
        after.stages.at("pass").successful_executions == 1 &&
            after.components.at("producer").input_bytes == kPassInputBytes,
        "a session created before the reset records into the new epoch");
  }

  {
    // Pipeline copies share one collector, and concurrent resets publish in
    // epoch order rather than letting a slower earlier build overwrite a
    // newer slab.
    Pipeline pipeline = MakePipeline(true);
    constexpr std::size_t kResetThreads = 16;
    constexpr std::size_t kResetsPerThread = 32;
    std::mutex start_mutex;
    std::condition_variable start_condition;
    bool start = false;
    std::mutex observations_mutex;
    std::vector<std::uint64_t> observed_epochs;
    observed_epochs.reserve(kResetThreads * kResetsPerThread);
    std::vector<std::thread> resets;
    resets.reserve(kResetThreads);
    for (std::size_t index = 0; index < kResetThreads; ++index) {
      resets.emplace_back([&] {
        {
          std::unique_lock lock(start_mutex);
          start_condition.wait(lock, [&start] { return start; });
        }
        for (std::size_t reset = 0; reset < kResetsPerThread; ++reset) {
          pipeline.ResetTelemetry();
          // Serialize observations, not resets. A thread delayed after its
          // reset reads the current live epoch rather than publishing a stale
          // local value into this sequence.
          std::scoped_lock lock(observations_mutex);
          observed_epochs.push_back(pipeline.telemetry_snapshot().epoch);
        }
      });
    }
    {
      std::scoped_lock lock(start_mutex);
      start = true;
    }
    start_condition.notify_all();
    for (std::thread& reset : resets) {
      reset.join();
    }
    Check(
        std::ranges::is_sorted(observed_epochs),
        "concurrent resets never make the visible epoch move backward");
    Check(
        pipeline.telemetry_snapshot().epoch ==
            1 + kResetThreads * kResetsPerThread,
        "concurrent resets publish every epoch exactly once");
  }

  {
    // One collector per Pipeline, shared by its copies, its sessions, and
    // their forks; a separately constructed Pipeline gets its own.
    const Pipeline pipeline = MakePipeline(true);
    const Pipeline copy = pipeline;
    PipelineSession from_copy = copy.CreateSession();
    (void)from_copy.RunStage("pass", PassInputs());
    Check(
        StageStats(pipeline, "pass").successful_executions == 1 &&
            StageStats(copy, "pass").successful_executions == 1,
        "a Pipeline copy shares the collector rather than starting a second");

    PipelineSession forked = from_copy.Fork();
    (void)forked.RunStage("pass", PassInputs());
    Check(
        StageStats(pipeline, "pass").successful_executions == 2,
        "a forked session records into the same collector");

    const Pipeline separate = MakePipeline(true);
    PipelineSession independent = separate.CreateSession();
    (void)independent.RunStage("pass", PassInputs());
    Check(
        StageStats(separate, "pass").successful_executions == 1,
        "a separately constructed Pipeline counts only its own work");
    Check(
        StageStats(pipeline, "pass").successful_executions == 2,
        "a separately constructed Pipeline never touches another's counters");
  }

  {
    // A moved-from Pipeline owns no collector, and reading or resetting it is
    // answered rather than punished.
    Pipeline source = MakePipeline(true);
    const Pipeline moved = std::move(source);
    const PipelineTelemetrySnapshot snapshot =
        source.telemetry_snapshot();  // NOLINT(bugprone-use-after-move)
    Check(
        !snapshot.enabled && snapshot.epoch == 0 &&
            snapshot.components.empty() && snapshot.stages.empty(),
        "a moved-from pipeline reports the disabled reading");
    source.ResetTelemetry();  // NOLINT(bugprone-use-after-move)
    Check(
        moved.telemetry_snapshot().enabled,
        "the moved-to pipeline still collects");
  }

  {
    // A reading is detached: it is a value copied out of the collector, so a
    // later execution cannot change one that was already returned.
    const Pipeline pipeline = MakePipeline(true);
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("pass", PassInputs());
    const PipelineTelemetrySnapshot first = pipeline.telemetry_snapshot();
    (void)session.RunStage("pass", PassInputs());

    Check(
        first.stages.at("pass").successful_executions == 1,
        "an already-returned reading never updates itself");
    Check(
        pipeline.telemetry_snapshot().stages.at("pass").successful_executions ==
            2,
        "the next reading sees the newer work");
  }

}

}  // namespace

int main() {
  try {
    RunTelemetryChecks();
  } catch (const onnx_world_model::Error& error) {
    std::cerr << "UNEXPECTED: " << error.what() << '\n';
    ++failures;
  }
  if (failures != 0) {
    std::cerr << failures << " telemetry check(s) failed\n";
    return 1;
  }
  std::cout << "pipeline telemetry tests passed\n";
  return 0;
}
