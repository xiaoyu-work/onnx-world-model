/**
 * @agent-file
 * @agent-purpose: Standalone test executable for pipeline cancellation and deadlines: it covers StageRun::RequestCancellation against a step that holds the session lock, externally supplied tokens, deadline_exceeded as a distinct outcome, run-slot ownership after a cancelled run, exception-safe restoration of the classifier-free guidance scratch state, and the promise that cancelling neither rolls back applied state nor materializes a device buffer.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as pipeline_cancellation_test; it counts failures through local Check, CheckThrowsCode, and CheckThrowsState helpers and returns a non-zero exit code when any check fails. It includes public headers only, every component is a stub ModelBackend, and every PipelinePackage is built in memory from an embedded manifest string, so the run needs no ONNX Runtime library, no ONNX model, and no filesystem access. Every concurrency assertion is driven by BlockingControl's condition variable rather than by a sleep: the blocking backend signals that it entered, waits to be released, and only then checks its token, so a cancellation requested while another thread holds the session lock is observed deterministically. CountingDeviceBuffer's shared copy counter asserts that cancelling never triggers a host materialization. GuidancePromptRecorder cancels its source on its second pass -- the unconditional half of a guided step -- and records every prompt it saw, so a later step run at guidance scale 1 with no replacement inputs proves the conditional conditioning tensor was restored rather than left holding the unconditional prompt.
 * @agent-side-effects: Writes failure descriptions to stderr.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
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

using onnx_world_model::CancellationReason;
using onnx_world_model::CancellationSource;
using onnx_world_model::CancellationToken;
using onnx_world_model::DataType;
using onnx_world_model::ErrorCode;
using onnx_world_model::Model;
using onnx_world_model::ModelMetadata;
using onnx_world_model::NamedTensors;
using onnx_world_model::Pipeline;
using onnx_world_model::PipelineManifest;
using onnx_world_model::PipelinePackage;
using onnx_world_model::PipelineRunOptions;
using onnx_world_model::PipelineSession;
using onnx_world_model::StageEvent;
using onnx_world_model::StageEventKind;
using onnx_world_model::StageRun;
using onnx_world_model::Tensor;

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

template <typename Function>
void CheckThrowsState(Function&& function, const char* message) {
  CheckThrowsCode(std::forward<Function>(function), ErrorCode::state, message);
}

// Adds one to its single float state, and counts how many times it was asked
// to, so "cancelling between steps ran no extra model pass" is a number.
class CountingIncrementBackend final : public onnx_world_model::ModelBackend {
 public:
  CountingIncrementBackend() {
    metadata_.inputs.push_back({
        .name = "state",
        .data_type = DataType::float32,
        .shape = {1},
    });
    metadata_.outputs.push_back({
        .name = "next_state",
        .data_type = DataType::float32,
        .shape = {1},
    });
  }

  [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
    return metadata_;
  }

  // Deliberately implements only the historical one-argument Run, so this
  // stub also exercises ModelBackend's default cancellable overload.
  [[nodiscard]] NamedTensors Run(const NamedTensors& inputs) const override {
    {
      std::scoped_lock lock(mutex_);
      ++calls_;
    }
    Tensor result = inputs.at("state");
    auto value = std::span(
        reinterpret_cast<float*>(result.mutable_bytes().data()),
        result.element_count());
    value[0] += 1.0F;
    return {{"next_state", std::move(result)}};
  }

  [[nodiscard]] std::size_t calls() const {
    std::scoped_lock lock(mutex_);
    return calls_;
  }

 private:
  ModelMetadata metadata_;
  mutable std::mutex mutex_;
  mutable std::size_t calls_{0};
};

struct BlockingControl {
  std::mutex mutex;
  std::condition_variable condition;
  std::size_t calls{0};
  bool entered{false};
  bool released{false};
};

void WaitForEntry(BlockingControl& control) {
  std::unique_lock lock(control.mutex);
  control.condition.wait(lock, [&control] { return control.entered; });
}

void Release(BlockingControl& control) {
  {
    std::scoped_lock lock(control.mutex);
    control.released = true;
  }
  control.condition.notify_all();
}

[[nodiscard]] std::size_t Calls(BlockingControl& control) {
  std::scoped_lock lock(control.mutex);
  return control.calls;
}

// Stands in for a long ONNX Runtime call: it announces that it entered, waits
// until the test releases it, and only then consults its token. That ordering
// is what makes "a cancellation requested while the session lock is held is
// observed by the work in flight" a deterministic assertion instead of a race
// against a sleep.
class BlockingCancellableBackend final
    : public onnx_world_model::ModelBackend {
 public:
  explicit BlockingCancellableBackend(
      std::shared_ptr<BlockingControl> control)
      : control_(std::move(control)) {
    metadata_.inputs.push_back({
        .name = "state",
        .data_type = DataType::float32,
        .shape = {1},
    });
    metadata_.outputs.push_back({
        .name = "next_state",
        .data_type = DataType::float32,
        .shape = {1},
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
    {
      std::unique_lock lock(control_->mutex);
      ++control_->calls;
      control_->entered = true;
      control_->condition.notify_all();
      control_->condition.wait(
          lock, [this] { return control_->released; });
    }
    cancellation.ThrowIfCancellationRequested();
    Tensor result = inputs.at("state");
    auto value = std::span(
        reinterpret_cast<float*>(result.mutable_bytes().data()),
        result.element_count());
    value[0] += 1.0F;
    return {{"next_state", std::move(result)}};
  }

 private:
  ModelMetadata metadata_;
  std::shared_ptr<BlockingControl> control_;
};

// Forwards its input unchanged, so an event can be asked whether it still
// holds the caller's exact buffer.
class PassthroughBackend final : public onnx_world_model::ModelBackend {
 public:
  PassthroughBackend() {
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
    return {{"y", inputs.at("x")}};
  }

 private:
  ModelMetadata metadata_;
};

struct DeviceCopyCounter {
  int copies{0};
};

class CountingDeviceBuffer final : public onnx_world_model::TensorBuffer {
 public:
  CountingDeviceBuffer(
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
        ErrorCode::invalid_argument,
        "Counting device buffer is not host-accessible");
  }

  void CopyToCpu(std::span<std::byte> destination) const override {
    if (destination.size() != storage_.size()) {
      throw onnx_world_model::Error(
          ErrorCode::invalid_argument,
          "Counting device copy destination has the wrong byte size");
    }
    ++counter_->copies;
    std::copy(storage_.begin(), storage_.end(), destination.begin());
  }

 private:
  std::vector<std::byte> storage_;
  std::shared_ptr<DeviceCopyCounter> counter_;
  onnx_world_model::TensorDevice device_;
};

[[nodiscard]] Tensor MakeDeviceTensor(
    std::vector<std::int64_t> shape,
    std::span<const float> values,
    const std::shared_ptr<DeviceCopyCounter>& counter) {
  const std::span<const std::byte> raw = std::as_bytes(values);
  return Tensor::FromBuffer(
      DataType::float32,
      std::move(shape),
      std::make_shared<CountingDeviceBuffer>(
          std::vector<std::byte>(raw.begin(), raw.end()), counter));
}

constexpr std::string_view kIterativeManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "counter",
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
      }
    ],
    "stages": [
      {
        "name": "iterate",
        "kind": "iterative",
        "components": ["counter"],
        "run_on": "step",
        "options": {
          "scheduler": {
            "kind": "FlowMatchEulerDiscreteScheduler",
            "config_asset": "scheduler.json"
          },
          "default_steps": 4,
          "timestep": {},
          "state_inputs": ["counter.state"]
        },
        "capabilities": ["loop_carried_state"]
      }
    ],
    "inputs": [
      {
        "port": "counter.state",
        "kind": "generated",
        "required": true,
        "semantic": "state.initial",
        "generator": {"kind": "zeros"}
      }
    ],
    "outputs": [{"state": "counter_state", "alias": "value"}],
    "profile": {"name": "counter-world", "version": "1.0"},
    "states": [
      {
        "name": "counter_state",
        "kind": "recurrent",
        "input": "counter.state",
        "output": "counter.next_state",
        "lifetime": "session",
        "release_after": "iterate"
      }
    ],
    "assets": [{"path": "scheduler.json"}],
    "required_capabilities": ["loop_carried_state"]
  },
  "component_files": {"counter": "model.onnx"}
}
)json";

constexpr std::string_view kPassthroughManifest = R"json(
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
      }
    ],
    "connections": [],
    "stages": [
      {
        "name": "run",
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

// The smallest manifest that reaches the classifier-free guidance path: one
// component, one external conditioning input, one guided output, and a
// recurrent connection so the stage can be iterative.
constexpr std::string_view kGuidedManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "denoiser",
        "role": "dynamics",
        "run_on": "step",
        "inputs": [
          {"name": "prompt", "dtype": "INT64", "shape": [2]},
          {"name": "latent", "dtype": "FLOAT", "shape": [2]}
        ],
        "outputs": [{"name": "pred", "dtype": "FLOAT", "shape": [2]}],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT"
      }
    ],
    "connections": [
      {
        "source": "denoiser.pred",
        "target": "denoiser.latent",
        "recurrent": true
      }
    ],
    "stages": [
      {
        "name": "denoise",
        "kind": "iterative",
        "components": ["denoiser"],
        "run_on": "step",
        "options": {
          "scheduler": {
            "kind": "FlowMatchEulerDiscreteScheduler",
            "config_asset": "scheduler.json"
          },
          "guidance": {
            "kind": "classifier_free",
            "conditioning_input": "denoiser.prompt",
            "scale_option": "guidance_scale",
            "default_scale": 1.0,
            "outputs": ["denoiser.pred"]
          },
          "default_steps": 4,
          "timestep": {},
          "state_inputs": ["denoiser.latent"]
        },
        "capabilities": ["loop_carried_state", "classifier_free_guidance"]
      }
    ],
    "inputs": [
      {
        "port": "denoiser.prompt",
        "kind": "external",
        "alias": "prompt",
        "required": true
      },
      {
        "port": "denoiser.latent",
        "kind": "generated",
        "required": true,
        "semantic": "state.initial",
        "generator": {"kind": "zeros"}
      }
    ],
    "outputs": [{"state": "latent_state", "alias": "value"}],
    "states": [
      {
        "name": "latent_state",
        "kind": "recurrent",
        "input": "denoiser.latent",
        "output": "denoiser.pred",
        "lifetime": "session",
        "release_after": "denoise"
      }
    ],
    "assets": [{"path": "scheduler.json"}],
    "required_capabilities": [
      "loop_carried_state",
      "classifier_free_guidance"
    ]
  },
  "component_files": {"denoiser": "model.onnx"}
}
)json";

// Records the conditioning prompt it was handed on every pass and cancels the
// source it was given on its second one, which is the unconditional pass of a
// guided step. That is exactly the window in which the session's conditioning
// tensor, endpoint map, and position cursors hold scratch values.
class GuidancePromptRecorder final : public onnx_world_model::ModelBackend {
 public:
  explicit GuidancePromptRecorder(CancellationSource* cancel_on_second_pass)
      : cancel_on_second_pass_(cancel_on_second_pass) {
    metadata_.inputs.push_back({
        .name = "prompt",
        .data_type = DataType::int64,
        .shape = {2},
    });
    metadata_.inputs.push_back({
        .name = "latent",
        .data_type = DataType::float32,
        .shape = {2},
    });
    metadata_.outputs.push_back({
        .name = "pred",
        .data_type = DataType::float32,
        .shape = {2},
    });
  }

  [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
    return metadata_;
  }

  [[nodiscard]] NamedTensors Run(const NamedTensors& inputs) const override {
    const std::span<const std::int64_t> prompt =
        inputs.at("prompt").values<std::int64_t>();
    {
      std::scoped_lock lock(mutex_);
      ++calls_;
      last_prompt_.assign(prompt.begin(), prompt.end());
      if (calls_ == 2 && cancel_on_second_pass_ != nullptr) {
        cancel_on_second_pass_->Cancel();
      }
    }
    Tensor result = inputs.at("latent");
    auto value = std::span(
        reinterpret_cast<float*>(result.mutable_bytes().data()),
        result.element_count());
    value[0] += 1.0F;
    value[1] += 1.0F;
    return {{"pred", std::move(result)}};
  }

  [[nodiscard]] std::vector<std::int64_t> last_prompt() const {
    std::scoped_lock lock(mutex_);
    return last_prompt_;
  }

  [[nodiscard]] std::size_t calls() const {
    std::scoped_lock lock(mutex_);
    return calls_;
  }

  void StopCancelling() {
    std::scoped_lock lock(mutex_);
    cancel_on_second_pass_ = nullptr;
  }

 private:
  ModelMetadata metadata_;
  mutable std::mutex mutex_;
  mutable std::size_t calls_{0};
  mutable std::vector<std::int64_t> last_prompt_;
  CancellationSource* cancel_on_second_pass_;
};

[[nodiscard]] Pipeline MakeGuidedPipeline(
    const std::shared_ptr<GuidancePromptRecorder>& backend) {
  std::unordered_map<std::string, Model> models;
  models.emplace("denoiser", Model(backend));
  return Pipeline(PipelinePackage(
      {}, PipelineManifest::Parse(kGuidedManifest), std::move(models)));
}

[[nodiscard]] Pipeline MakeCountingPipeline(
    const std::shared_ptr<CountingIncrementBackend>& backend) {
  std::unordered_map<std::string, Model> models;
  models.emplace("counter", Model(backend));
  return Pipeline(PipelinePackage(
      {}, PipelineManifest::Parse(kIterativeManifest), std::move(models)));
}

[[nodiscard]] Pipeline MakeBlockingPipeline(
    const std::shared_ptr<BlockingControl>& control) {
  std::unordered_map<std::string, Model> models;
  models.emplace(
      "counter",
      Model(std::make_shared<BlockingCancellableBackend>(control)));
  return Pipeline(PipelinePackage(
      {}, PipelineManifest::Parse(kIterativeManifest), std::move(models)));
}

[[nodiscard]] Pipeline MakePassthroughPipeline() {
  std::unordered_map<std::string, Model> models;
  models.emplace("producer", Model(std::make_shared<PassthroughBackend>()));
  return Pipeline(PipelinePackage(
      {}, PipelineManifest::Parse(kPassthroughManifest), std::move(models)));
}

[[nodiscard]] float CounterValue(const PipelineSession& session) {
  const std::optional<Tensor> value = session.state("counter_state");
  return value.has_value() ? value->CopyToCpu().values<float>()[0] : -1.0F;
}

[[nodiscard]] PipelineRunOptions OptionsWith(const CancellationToken& token) {
  PipelineRunOptions options;
  options.cancellation = token;
  return options;
}

void TestRequestCancellationInterruptsAStepHoldingTheSessionLock() {
  auto control = std::make_shared<BlockingControl>();
  const Pipeline pipeline = MakeBlockingPipeline(control);
  PipelineSession session = pipeline.CreateSession();
  StageRun run = session.BeginStage("iterate");

  std::optional<ErrorCode> observed;
  std::thread worker([&run, &observed] {
    try {
      // Finish holds the session mutex for the whole drain, which is exactly
      // the situation RequestCancellation exists for.
      (void)run.Finish();
    } catch (const onnx_world_model::Error& error) {
      observed = error.code();
    }
  });

  WaitForEntry(*control);
  // Takes no session lock, so it cannot deadlock against the drain above.
  run.RequestCancellation();
  Release(*control);
  worker.join();

  Check(
      observed.has_value() && *observed == ErrorCode::cancelled,
      "RequestCancellation makes an in-flight Finish throw cancelled");
  Check(run.done(), "A cancelled run reports itself done");
  Check(
      Calls(*control) == 1,
      "A cancelled drain runs no further component pass");
  // The slot was released exactly once, by the failing drain, so the session
  // accepts a new run.
  StageRun fresh = session.BeginStage("iterate");
  Check(
      !fresh.done(),
      "The session's run slot is free again after a cancelled run");
  fresh.Cancel();
}

void TestExternalSourceCancelsAnInFlightRun() {
  auto control = std::make_shared<BlockingControl>();
  const Pipeline pipeline = MakeBlockingPipeline(control);
  PipelineSession session = pipeline.CreateSession();
  CancellationSource source;
  StageRun run =
      session.BeginStage("iterate", {}, {}, OptionsWith(source.token()));

  std::optional<ErrorCode> observed;
  std::thread worker([&run, &observed] {
    try {
      (void)run.Finish();
    } catch (const onnx_world_model::Error& error) {
      observed = error.code();
    }
  });

  WaitForEntry(*control);
  source.Cancel();
  Release(*control);
  worker.join();

  Check(
      observed.has_value() && *observed == ErrorCode::cancelled,
      "A caller's own source cancels a run it started");
  Check(
      source.reason() == CancellationReason::cancelled,
      "The caller's source keeps reporting why it stopped");
}

void TestCancellationBetweenStepsRunsNoExtraComponentPass() {
  auto backend = std::make_shared<CountingIncrementBackend>();
  const Pipeline pipeline = MakeCountingPipeline(backend);
  PipelineSession session = pipeline.CreateSession();
  StageRun run = session.BeginStage("iterate");

  const StageEvent first = run.Step();
  Check(
      first.kind == StageEventKind::iteration,
      "The first step of an iterative run is an iteration event");
  Check(backend->calls() == 1, "One step is one component pass");

  run.RequestCancellation();
  CheckThrowsCode(
      [&run] { (void)run.Step(); },
      ErrorCode::cancelled,
      "A step after RequestCancellation throws cancelled");
  Check(
      backend->calls() == 1,
      "Cancelling between steps runs no extra component pass");
}

void TestPartialStateIsNotRolledBack() {
  auto backend = std::make_shared<CountingIncrementBackend>();
  const Pipeline pipeline = MakeCountingPipeline(backend);
  PipelineSession session = pipeline.CreateSession();
  StageRun run = session.BeginStage("iterate");

  (void)run.Step();
  (void)run.Step();
  Check(
      CounterValue(session) == 2.0F,
      "Two steps advanced the recurrent state twice");

  run.RequestCancellation();
  CheckThrowsCode(
      [&run] { (void)run.Step(); },
      ErrorCode::cancelled,
      "The third step is cancelled");

  Check(
      CounterValue(session) == 2.0F,
      "Cancelling keeps what the run already applied");
  Check(
      session.outputs().contains("value"),
      "The session still publishes the outputs the run produced");
}

void TestDeadlineIsDistinctFromCancellation() {
  auto backend = std::make_shared<CountingIncrementBackend>();
  const Pipeline pipeline = MakeCountingPipeline(backend);
  PipelineSession session = pipeline.CreateSession();

  const CancellationSource expired =
      CancellationSource::WithTimeout(std::chrono::milliseconds{0});
  CheckThrowsCode(
      [&session, &expired] {
        (void)session.RunStage("iterate", {}, {}, OptionsWith(expired.token()));
      },
      ErrorCode::deadline_exceeded,
      "An expired deadline fails a run with deadline_exceeded");
  Check(
      backend->calls() == 0,
      "An expired deadline runs no component pass at all");

  // A deadline that is still open lets the run start, and the run's own
  // cancellation then reports the other reason.
  const CancellationSource open = CancellationSource::WithDeadline(
      std::chrono::steady_clock::now() + std::chrono::hours{1});
  StageRun run =
      session.BeginStage("iterate", {}, {}, OptionsWith(open.token()));
  (void)run.Step();
  run.RequestCancellation();
  CheckThrowsCode(
      [&run] { (void)run.Step(); },
      ErrorCode::cancelled,
      "An explicit cancellation under an open deadline reports cancelled");
  Check(
      open.reason() == CancellationReason::none,
      "The caller's open-deadline source is not itself cancelled");
}

void TestExpiredDeadlineNeverClaimsTheRunSlot() {
  auto backend = std::make_shared<CountingIncrementBackend>();
  const Pipeline pipeline = MakeCountingPipeline(backend);
  PipelineSession session = pipeline.CreateSession();

  const CancellationSource source = CancellationSource::WithDeadline(
      std::chrono::steady_clock::now() + std::chrono::hours{1});
  StageRun run =
      session.BeginStage("iterate", {}, {}, OptionsWith(source.token()));
  (void)run.Step();

  // A second source with an already-passed deadline proves the run copies its
  // caller's deadline rather than ignoring it.
  const CancellationSource expired =
      CancellationSource::WithTimeout(std::chrono::milliseconds{0});
  run.Cancel();
  CheckThrowsCode(
      [&session, &expired] {
        (void)session.BeginStage(
            "iterate", {}, {}, OptionsWith(expired.token()));
      },
      ErrorCode::deadline_exceeded,
      "Beginning a run with an expired deadline fails immediately");
  StageRun replacement = session.BeginStage("iterate");
  Check(
      !replacement.done(),
      "A run rejected by its deadline never claimed the session's run slot");
  replacement.Cancel();
}

void TestStaleHandleCannotReleaseANewerRun() {
  auto backend = std::make_shared<CountingIncrementBackend>();
  const Pipeline pipeline = MakeCountingPipeline(backend);
  PipelineSession session = pipeline.CreateSession();

  StageRun stale = session.BeginStage("iterate");
  (void)stale.Step();
  stale.RequestCancellation();
  CheckThrowsCode(
      [&stale] { (void)stale.Step(); },
      ErrorCode::cancelled,
      "The stale run is cancelled");

  StageRun fresh = session.BeginStage("iterate");
  // Releasing the slot is keyed to the run identity, so a stale handle's
  // Cancel and destructor cannot hand a newer run's slot away.
  stale.Cancel();
  stale.RequestCancellation();
  CheckThrowsState(
      [&session] { (void)session.BeginStage("iterate"); },
      "A stale handle's Cancel does not release the newer run's slot");
  const StageEvent event = fresh.Step();
  Check(
      event.kind == StageEventKind::iteration,
      "The newer run still owns the session and can step");
  fresh.Cancel();
}

void TestSessionIsReusableWithAFreshTokenAfterCancellation() {
  auto backend = std::make_shared<CountingIncrementBackend>();
  const Pipeline pipeline = MakeCountingPipeline(backend);
  PipelineSession session = pipeline.CreateSession();

  CancellationSource cancelled;
  StageRun run =
      session.BeginStage("iterate", {}, {}, OptionsWith(cancelled.token()));
  (void)run.Step();
  cancelled.Cancel();
  CheckThrowsCode(
      [&run] { (void)run.Finish(); },
      ErrorCode::cancelled,
      "Finish reports the caller's cancellation");

  // Reusing the same cancelled token cancels immediately by design: a token
  // is a one-way latch, not a reusable flag.
  CheckThrowsCode(
      [&session, &cancelled] {
        (void)session.RunStage("iterate", {}, {}, OptionsWith(cancelled.token()));
      },
      ErrorCode::cancelled,
      "A reused cancelled token fails the next run immediately");

  const CancellationSource fresh;
  const NamedTensors outputs =
      session.RunStage("iterate", {}, {}, OptionsWith(fresh.token()));
  Check(
      outputs.contains("value"),
      "A fresh token lets the same session run to completion");
  Check(
      CounterValue(session) == 4.0F,
      "The resumed run finished the stage's four scheduler steps");
}

void TestDirectStepStageHonorsItsToken() {
  auto backend = std::make_shared<CountingIncrementBackend>();
  const Pipeline pipeline = MakeCountingPipeline(backend);
  PipelineSession session = pipeline.CreateSession();

  CancellationSource source;
  (void)session.StepStage("iterate", {}, {}, OptionsWith(source.token()));
  Check(backend->calls() == 1, "One direct step is one component pass");

  source.Cancel();
  CheckThrowsCode(
      [&session, &source] {
        (void)session.StepStage("iterate", {}, {}, OptionsWith(source.token()));
      },
      ErrorCode::cancelled,
      "Direct StepStage honors the token it was given");
  Check(
      backend->calls() == 1,
      "A cancelled direct step runs no component pass");
}

void TestCancellingNeverMaterializesADeviceBuffer() {
  const Pipeline pipeline = MakePassthroughPipeline();
  PipelineSession session = pipeline.CreateSession();
  auto counter = std::make_shared<DeviceCopyCounter>();
  const std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
  const Tensor input = MakeDeviceTensor({1, 4}, std::span(values), counter);

  CancellationSource source;
  StageRun run = session.BeginStage(
      "run", {{"x", input}}, {}, OptionsWith(source.token()));
  const StageEvent event = run.Step();
  Check(
      event.outputs.at("produced").buffer().get() == input.buffer().get(),
      "The device buffer reached the stage event untouched");
  Check(counter->copies == 0, "Running the stage copied nothing to CPU");

  source.Cancel();
  CheckThrowsCode(
      [&run] { (void)run.Step(); },
      ErrorCode::cancelled,
      "The next step observes the cancellation");
  Check(
      counter->copies == 0,
      "Cancelling never materializes a device buffer");
}

// A guided step temporarily replaces the session's conditioning tensor, its
// endpoint map, and its position cursors for the duration of the
// unconditional pass. Cancelling inside that window must not leave the
// unconditional prompt behind: the next step would silently condition on it.
void TestCancelledUnconditionalPassRestoresConditionalState() {
  CancellationSource source;
  auto backend = std::make_shared<GuidancePromptRecorder>(&source);
  const Pipeline pipeline = MakeGuidedPipeline(backend);
  PipelineSession session = pipeline.CreateSession();

  const std::array<std::int64_t, 2> conditional{11, 12};
  const std::array<std::int64_t, 2> unconditional{97, 98};
  NamedTensors inputs;
  inputs.emplace(
      "denoiser.prompt",
      Tensor::FromValues<std::int64_t>({2}, std::span(conditional)));
  inputs.emplace(
      "unconditional:denoiser.prompt",
      Tensor::FromValues<std::int64_t>({2}, std::span(unconditional)));

  PipelineRunOptions guided = OptionsWith(source.token());
  guided.numbers.emplace("guidance_scale", 3.0);
  CheckThrowsCode(
      [&session, &inputs, &guided] {
        (void)session.StepStage("denoise", inputs, {}, guided);
      },
      ErrorCode::cancelled,
      "Cancelling during the unconditional pass reports cancelled");
  Check(
      backend->calls() == 2,
      "The cancelled guided step ran exactly the two guidance passes");
  Check(
      backend->last_prompt() == std::vector<std::int64_t>({97, 98}),
      "The unconditional pass really did install its own prompt");

  // Guidance scale 1 is the conditional pass alone, and no inputs are
  // supplied, so whatever the backend now sees is whatever the cancelled step
  // left in the session's conditioning slot.
  backend->StopCancelling();
  const CancellationSource fresh;
  const NamedTensors outputs =
      session.StepStage("denoise", {}, {}, OptionsWith(fresh.token()));
  Check(
      backend->calls() == 3,
      "The session is still usable and ran one further pass");
  Check(
      backend->last_prompt() == std::vector<std::int64_t>({11, 12}),
      "The conditional prompt survived the cancelled unconditional pass");
  Check(
      outputs.contains("value"),
      "The reused session still publishes the stage's output");
}

void TestModelRunHonorsTheDefaultBackendOverload() {
  const Model model(std::make_shared<CountingIncrementBackend>());
  NamedTensors inputs;
  inputs.emplace("state", Tensor::Zeros(DataType::float32, {1}));

  const CancellationToken inert;
  const NamedTensors outputs = model.Run(inputs, inert);
  Check(
      outputs.contains("next_state"),
      "A default token lets Model::Run execute normally");

  CancellationSource source;
  source.Cancel();
  CheckThrowsCode(
      [&model, &inputs, &source] {
        (void)model.Run(inputs, source.token());
      },
      ErrorCode::cancelled,
      "Model::Run checks its token before reaching the backend");

  const CancellationSource expired =
      CancellationSource::WithTimeout(std::chrono::milliseconds{0});
  CheckThrowsCode(
      [&model, &inputs, &expired] {
        (void)model.Run(inputs, expired.token());
      },
      ErrorCode::deadline_exceeded,
      "Model::Run reports an expired deadline as deadline_exceeded");
}

}  // namespace

int main() {
  TestRequestCancellationInterruptsAStepHoldingTheSessionLock();
  TestExternalSourceCancelsAnInFlightRun();
  TestCancellationBetweenStepsRunsNoExtraComponentPass();
  TestPartialStateIsNotRolledBack();
  TestDeadlineIsDistinctFromCancellation();
  TestExpiredDeadlineNeverClaimsTheRunSlot();
  TestStaleHandleCannotReleaseANewerRun();
  TestSessionIsReusableWithAFreshTokenAfterCancellation();
  TestDirectStepStageHonorsItsToken();
  TestCancellingNeverMaterializesADeviceBuffer();
  TestCancelledUnconditionalPassRestoresConditionalState();
  TestModelRunHonorsTheDefaultBackendOverload();

  if (failures != 0) {
    std::cerr << failures << " pipeline cancellation checks failed\n";
    return 1;
  }
  std::cout << "pipeline cancellation tests passed\n";
  return 0;
}
