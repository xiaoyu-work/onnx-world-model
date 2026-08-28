/**
 * @agent-file
 * @agent-purpose: Standalone test executable for the incremental stage API: it holds PipelineSession::BeginStage, StageRun stepping, and StageEvent reporting to exact parity with RunStage across every stage kind, and covers the run slot's exclusion, cancellation, failure, and ownership rules.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as pipeline_stream_test; it counts failures through local Check, CheckThrowsCode, and CheckThrowsState helpers and returns a non-zero exit code when any check fails. Every component is a stub ModelBackend and every PipelinePackage is built in memory from an embedded manifest string, so the run needs no ONNX Runtime library, no ONNX model, and no filesystem access. Parity assertions always compare a RunStage session against a BeginStage session created from the same Pipeline, because equality of the two paths is the contract this file exists to protect. CountingDeviceBuffer is a non-host-accessible TensorBuffer whose shared copy counter asserts that a StageEvent carries device outputs without materializing them.
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
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_world_model/error.hpp"
#include "onnx_world_model/pipeline.hpp"
#include "onnx_world_model/tensor.hpp"

namespace {

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
using onnx_world_model::PipelineSessionSnapshot;
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

// Adds one to its single float state so a stage's progress is readable as a
// number, which is what makes step-by-step and all-at-once execution directly
// comparable.
class IncrementBackend final : public onnx_world_model::ModelBackend {
 public:
  IncrementBackend() {
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
    Tensor result = inputs.at("state");
    auto value = std::span(
        reinterpret_cast<float*>(result.mutable_bytes().data()),
        result.element_count());
    value[0] += 1.0F;
    return {{"next_state", std::move(result)}};
  }

 private:
  ModelMetadata metadata_;
};

struct BlockingControl {
  std::mutex mutex;
  std::condition_variable condition;
  bool first_call_entered{false};
  bool second_call_started{false};
  bool release_first_call{false};
  std::size_t calls{0};
};

// Holds the first model invocation until another thread has started a
// concurrent RunStage call. This makes whole-stage serialization observable
// without sleeping or depending on scheduler timing.
class BlockingIncrementBackend final
    : public onnx_world_model::ModelBackend {
 public:
  explicit BlockingIncrementBackend(
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
    {
      std::unique_lock lock(control_->mutex);
      ++control_->calls;
      if (!control_->first_call_entered) {
        control_->first_call_entered = true;
        control_->condition.notify_all();
        control_->condition.wait(lock, [this] {
          return control_->release_first_call;
        });
      }
    }
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

// Increments like IncrementBackend until its budget runs out and then throws,
// so a run can fail in flight without any I/O or a real model.
class FlakyIncrementBackend final : public onnx_world_model::ModelBackend {
 public:
  explicit FlakyIncrementBackend(int successes) : remaining_(successes) {
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
    if (remaining_ == 0) {
      throw onnx_world_model::Error(
          ErrorCode::runtime_execution,
          "Flaky component exhausted its budget");
    }
    --remaining_;
    Tensor result = inputs.at("state");
    auto value = std::span(
        reinterpret_cast<float*>(result.mutable_bytes().data()),
        result.element_count());
    value[0] += 1.0F;
    return {{"next_state", std::move(result)}};
  }

 private:
  ModelMetadata metadata_;
  mutable int remaining_;
};

// Predicts one past each lane's last token and saturates at the final
// vocabulary entry, so a greedy run walks a known path and lanes that start
// at different tokens reach the end-of-sequence token at different steps.
class NextTokenBackend final : public onnx_world_model::ModelBackend {
 public:
  explicit NextTokenBackend(std::int64_t vocabulary)
      : vocabulary_(vocabulary) {
    metadata_.inputs.push_back({
        .name = "tokens",
        .data_type = DataType::int64,
        .shape = {-1, -1},
    });
    metadata_.outputs.push_back({
        .name = "logits",
        .data_type = DataType::float32,
        .shape = {-1, -1, vocabulary},
    });
  }

  [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
    return metadata_;
  }

  [[nodiscard]] NamedTensors Run(const NamedTensors& inputs) const override {
    const Tensor tokens = inputs.at("tokens").CopyToCpu();
    const std::int64_t batch = tokens.shape()[0];
    const std::int64_t length = tokens.shape()[1];
    Tensor logits =
        Tensor::Zeros(DataType::float32, {batch, length, vocabulary_});
    auto scores = std::span(
        reinterpret_cast<float*>(logits.mutable_bytes().data()),
        logits.element_count());
    const auto ids = tokens.values<std::int64_t>();
    for (std::int64_t lane = 0; lane < batch; ++lane) {
      const auto last = static_cast<std::size_t>(
          (lane * length) + length - 1);
      const std::int64_t next =
          std::min(ids[last] + 1, vocabulary_ - 1);
      scores[static_cast<std::size_t>(
          (last * static_cast<std::size_t>(vocabulary_)) +
          static_cast<std::size_t>(next))] = 1.0F;
    }
    return {{"logits", std::move(logits)}};
  }

 private:
  ModelMetadata metadata_;
  std::int64_t vocabulary_;
};

// Spreads distinct scores over the vocabulary as a function of the last
// token, so a sampled run draws from a real distribution and a repetition
// penalty over the run's own history changes what it draws.
class SpreadLogitsBackend final : public onnx_world_model::ModelBackend {
 public:
  explicit SpreadLogitsBackend(std::int64_t vocabulary)
      : vocabulary_(vocabulary) {
    metadata_.inputs.push_back({
        .name = "tokens",
        .data_type = DataType::int64,
        .shape = {-1, -1},
    });
    metadata_.outputs.push_back({
        .name = "logits",
        .data_type = DataType::float32,
        .shape = {-1, -1, vocabulary},
    });
  }

  [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
    return metadata_;
  }

  [[nodiscard]] NamedTensors Run(const NamedTensors& inputs) const override {
    const Tensor tokens = inputs.at("tokens").CopyToCpu();
    const std::int64_t batch = tokens.shape()[0];
    const std::int64_t length = tokens.shape()[1];
    Tensor logits =
        Tensor::Zeros(DataType::float32, {batch, length, vocabulary_});
    auto scores = std::span(
        reinterpret_cast<float*>(logits.mutable_bytes().data()),
        logits.element_count());
    const auto ids = tokens.values<std::int64_t>();
    for (std::int64_t lane = 0; lane < batch; ++lane) {
      const auto last = static_cast<std::size_t>(
          (lane * length) + length - 1);
      for (std::int64_t token = 0; token < vocabulary_; ++token) {
        const std::int64_t spread =
            ((ids[last] * 3) + (token * 5)) % 7;
        scores[static_cast<std::size_t>(
            (last * static_cast<std::size_t>(vocabulary_)) +
            static_cast<std::size_t>(token))] =
            0.5F * static_cast<float>(spread);
      }
    }
    return {{"logits", std::move(logits)}};
  }

 private:
  ModelMetadata metadata_;
  std::int64_t vocabulary_;
};

// Forwards its input unchanged so a stage event can be asked whether it kept
// the caller's exact buffer.
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

constexpr std::string_view kCounterManifest = R"json(
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
        "name": "transition",
        "kind": "state_transition",
        "components": ["counter"],
        "run_on": "step",
        "options": {"state_names": ["counter_state"]},
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
        "release_after": "transition"
      }
    ],
    "required_capabilities": ["loop_carried_state"]
  },
  "component_files": {"counter": "model.onnx"}
}
)json";

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
          "default_steps": 3,
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
        "lifetime": "request",
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

// One decoder shape serves every autoregressive case; only the vocabulary and
// the stage options differ, so each case states exactly what it varies.
[[nodiscard]] std::string DecoderManifest(
    std::int64_t vocabulary,
    std::string_view stage_options) {
  std::ostringstream document;
  document << R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "decoder",
        "role": "decoder",
        "run_on": "decode",
        "inputs": [
          {"name": "tokens", "dtype": "INT64", "shape": ["batch", "sequence"]}
        ],
        "outputs": [
          {
            "name": "logits",
            "dtype": "FLOAT",
            "shape": ["batch", "sequence", )json"
           << vocabulary << R"json(]
          }
        ],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT"
      }
    ],
    "connections": [],
    "stages": [
      {
        "name": "decode",
        "kind": "autoregressive",
        "components": ["decoder"],
        "run_on": "decode",
        "options": )json"
           << stage_options << R"json(
      }
    ],
    "inputs": [
      {
        "port": "decoder.tokens",
        "kind": "external",
        "required": true,
        "semantic": "text.token_ids"
      }
    ],
    "outputs": [{"port": "decoder.logits"}],
    "profile": {"name": "decoder-world", "version": "1.0"},
    "assets": [{"path": "tokenizer.json"}]
  },
  "component_files": {"decoder": "model.onnx"}
}
)json";
  return document.str();
}

[[nodiscard]] Pipeline MakeCounterPipeline(std::string_view document) {
  std::unordered_map<std::string, Model> models;
  models.emplace("counter", Model(std::make_shared<IncrementBackend>()));
  return Pipeline(PipelinePackage(
      {}, PipelineManifest::Parse(document), std::move(models)));
}

[[nodiscard]] Pipeline MakeBlockingPipeline(
    const std::shared_ptr<BlockingControl>& control) {
  std::unordered_map<std::string, Model> models;
  models.emplace(
      "counter",
      Model(std::make_shared<BlockingIncrementBackend>(control)));
  return Pipeline(PipelinePackage(
      {},
      PipelineManifest::Parse(kIterativeManifest),
      std::move(models)));
}

[[nodiscard]] Pipeline MakeFlakyPipeline(int successes) {
  std::unordered_map<std::string, Model> models;
  models.emplace(
      "counter", Model(std::make_shared<FlakyIncrementBackend>(successes)));
  return Pipeline(PipelinePackage(
      {}, PipelineManifest::Parse(kIterativeManifest), std::move(models)));
}

[[nodiscard]] Pipeline MakePassthroughPipeline() {
  std::unordered_map<std::string, Model> models;
  models.emplace("producer", Model(std::make_shared<PassthroughBackend>()));
  return Pipeline(PipelinePackage(
      {}, PipelineManifest::Parse(kPassthroughManifest), std::move(models)));
}

[[nodiscard]] Pipeline MakeDecoderPipeline(
    const std::string& document,
    std::shared_ptr<onnx_world_model::ModelBackend> backend) {
  std::unordered_map<std::string, Model> models;
  models.emplace("decoder", Model(std::move(backend)));
  return Pipeline(PipelinePackage(
      {}, PipelineManifest::Parse(document), std::move(models)));
}

[[nodiscard]] std::vector<std::int64_t> Tokens(const NamedTensors& outputs) {
  const auto found = outputs.find("generated_token_ids");
  if (found == outputs.end()) {
    return {};
  }
  const auto span = found->second.values<std::int64_t>();
  return {span.begin(), span.end()};
}

[[nodiscard]] std::vector<std::int64_t> TokenShape(
    const NamedTensors& outputs) {
  const auto found = outputs.find("generated_token_ids");
  return found == outputs.end() ? std::vector<std::int64_t>{}
                                : found->second.shape();
}

[[nodiscard]] float CounterValue(const PipelineSession& session) {
  const std::optional<Tensor> value = session.state("counter_state");
  return value.has_value() ? value->CopyToCpu().values<float>()[0] : -1.0F;
}

[[nodiscard]] Tensor PromptTensor(
    std::vector<std::int64_t> shape,
    std::span<const std::int64_t> values) {
  return Tensor::FromValues<std::int64_t>(std::move(shape), values);
}

// Drives a run to completion and reports what it saw, which is what every
// parity check compares against the equivalent RunStage result.
struct RunTrace {
  std::vector<StageEventKind> kinds;
  std::vector<std::size_t> iterations;
  std::vector<std::vector<std::int64_t>> tokens;
  NamedTensors outputs;
};

[[nodiscard]] RunTrace Drain(StageRun& run) {
  RunTrace trace;
  while (!run.done()) {
    const StageEvent event = run.Step();
    trace.kinds.push_back(event.kind);
    trace.iterations.push_back(event.iteration);
    if (event.token_ids.has_value()) {
      const auto span = event.token_ids->values<std::int64_t>();
      trace.tokens.emplace_back(span.begin(), span.end());
    }
    if (event.finished) {
      trace.outputs = event.outputs;
    }
  }
  return trace;
}

[[nodiscard]] bool EndsWithSingleCompleted(const RunTrace& trace) {
  if (trace.kinds.empty() || trace.kinds.back() != StageEventKind::completed) {
    return false;
  }
  return std::ranges::count(trace.kinds, StageEventKind::completed) == 1;
}

}  // namespace

int main() {
  // Greedy autoregressive decoding: the streamed run visits exactly the
  // tokens the full run returns, in order, and closes with one completed
  // event that carries the same result.
  {
    const std::string document = DecoderManifest(
        4,
        R"json({
          "tokenizer_asset": "tokenizer.json",
          "sampling": {"do_sample": false},
          "stop": {"kind": "token_ids", "eos_token_ids": [3]},
          "max_tokens": {"default": 8, "limit": 8}
        })json");
    const Pipeline pipeline =
        MakeDecoderPipeline(document, std::make_shared<NextTokenBackend>(4));
    const std::array<std::int64_t, 1> prompt{0};
    const NamedTensors inputs{
        {"text.token_ids", PromptTensor({1, 1}, std::span(prompt))}};

    PipelineSession full = pipeline.CreateSession();
    const NamedTensors expected = full.RunStage("decode", inputs);

    PipelineSession streamed = pipeline.CreateSession();
    StageRun run = streamed.BeginStage("decode", inputs);
    Check(run.stage() == "decode", "a run reports the stage it drives");
    Check(!run.done(), "a fresh run is not done");
    Check(run.iteration() == 0, "a fresh run has emitted no step event");
    const RunTrace trace = Drain(run);

    Check(
        trace.kinds ==
            std::vector<StageEventKind>{
                StageEventKind::token,
                StageEventKind::token,
                StageEventKind::token,
                StageEventKind::completed,
            },
        "greedy decoding emits one token event per generated token");
    Check(
        trace.iterations == std::vector<std::size_t>{0, 1, 2, 3},
        "step events are indexed from zero and completed counts them");
    Check(
        trace.tokens ==
            std::vector<std::vector<std::int64_t>>{{1}, {2}, {3}},
        "token events carry the tokens in generation order");
    Check(
        Tokens(trace.outputs) == Tokens(expected),
        "streamed greedy decoding matches RunStage token for token");
    Check(
        TokenShape(trace.outputs) == std::vector<std::int64_t>{1, 3},
        "the packed token tensor stays batch-major");
    Check(run.done(), "a completed run reports done");
    Check(
        run.iteration() == 3,
        "a completed run reports how many steps it took");
    Check(
        expected.contains("logits") && trace.outputs.contains("logits"),
        "both paths publish the stage's manifest outputs");
  }

  // Lanes stop independently: a lane that emits the end-of-sequence token is
  // latched to it while the other lane keeps decoding, and the run ends only
  // when every lane has stopped.
  {
    const std::string document = DecoderManifest(
        4,
        R"json({
          "tokenizer_asset": "tokenizer.json",
          "sampling": {"do_sample": false},
          "stop": {"kind": "token_ids", "eos_token_ids": [3]},
          "max_tokens": {"default": 8, "limit": 8}
        })json");
    const Pipeline pipeline =
        MakeDecoderPipeline(document, std::make_shared<NextTokenBackend>(4));
    const std::array<std::int64_t, 4> prompt{0, 0, 0, 1};
    const NamedTensors inputs{
        {"text.token_ids", PromptTensor({2, 2}, std::span(prompt))}};

    PipelineSession full = pipeline.CreateSession();
    const NamedTensors expected = full.RunStage("decode", inputs);

    PipelineSession streamed = pipeline.CreateSession();
    StageRun run = streamed.BeginStage("decode", inputs);
    const RunTrace trace = Drain(run);

    Check(
        trace.tokens ==
            std::vector<std::vector<std::int64_t>>{{1, 2}, {2, 3}, {3, 3}},
        "a finished lane is held at its end-of-sequence token");
    Check(
        Tokens(expected) == std::vector<std::int64_t>{1, 2, 3, 2, 3, 3},
        "the full run packs each lane's history contiguously");
    Check(
        Tokens(trace.outputs) == Tokens(expected),
        "per-lane early stopping matches RunStage");
    Check(
        TokenShape(trace.outputs) == std::vector<std::int64_t>{2, 3},
        "the packed token tensor keeps both lanes");
  }

  // Sampling is a session-level random walk, so parity here means the
  // streamed run consumes the engine in exactly the same order.
  {
    const std::string document = DecoderManifest(
        6,
        R"json({
          "tokenizer_asset": "tokenizer.json",
          "sampling": {
            "do_sample": true,
            "temperature": 0.8,
            "top_k": 4,
            "repetition_penalty": 1.5
          },
          "stop": {"kind": "token_ids", "eos_token_ids": [99]},
          "max_tokens": {"default": 5, "limit": 5}
        })json");
    const Pipeline pipeline = MakeDecoderPipeline(
        document, std::make_shared<SpreadLogitsBackend>(6));
    const std::array<std::int64_t, 2> prompt{1, 2};
    const NamedTensors inputs{
        {"text.token_ids", PromptTensor({1, 2}, std::span(prompt))}};
    PipelineRunOptions options;
    options.integers.emplace("seed", 20260828);

    PipelineSession full = pipeline.CreateSession();
    const NamedTensors expected =
        full.RunStage("decode", inputs, {}, options);

    PipelineSession streamed = pipeline.CreateSession();
    StageRun run = streamed.BeginStage("decode", inputs, {}, options);
    const RunTrace trace = Drain(run);

    Check(
        Tokens(expected).size() == 5,
        "the sampled run generates its whole budget");
    Check(
        Tokens(trace.outputs) == Tokens(expected),
        "sampled decoding is identical for the same seed and history");
    Check(
        trace.tokens.size() == 5 &&
            trace.tokens.front().size() == 1,
        "every sampled step reports its own token");

    // Repeating the seeded run in the same session reproduces it, which is
    // what makes the comparison above a statement about the run rather than
    // about one engine position.
    PipelineSession repeated = pipeline.CreateSession();
    StageRun first = repeated.BeginStage("decode", inputs, {}, options);
    const std::vector<std::int64_t> first_tokens = Tokens(first.Finish());
    StageRun second = repeated.BeginStage("decode", inputs, {}, options);
    Check(
        Tokens(second.Finish()) == first_tokens,
        "re-seeding reproduces the sampled tokens");
  }

  // The token budget is measured from the prompt exactly once. Recomputing it
  // after the first step -- when the token input has shrunk to one column --
  // would silently grant extra tokens.
  {
    const std::string document = DecoderManifest(
        4,
        R"json({
          "tokenizer_asset": "tokenizer.json",
          "sampling": {"do_sample": false},
          "stop": {
            "kind": "token_ids",
            "eos_token_ids": [99],
            "max_sequence_length": 5
          },
          "max_tokens": {"default": 8, "limit": 8}
        })json");
    const Pipeline pipeline =
        MakeDecoderPipeline(document, std::make_shared<NextTokenBackend>(4));
    const std::array<std::int64_t, 2> prompt{0, 1};
    const NamedTensors inputs{
        {"text.token_ids", PromptTensor({1, 2}, std::span(prompt))}};

    PipelineSession full = pipeline.CreateSession();
    const NamedTensors expected = full.RunStage("decode", inputs);
    Check(
        Tokens(expected).size() == 3,
        "the prompt length caps the full run at three tokens");

    PipelineSession streamed = pipeline.CreateSession();
    StageRun run = streamed.BeginStage("decode", inputs);
    const RunTrace trace = Drain(run);
    Check(
        Tokens(trace.outputs) == Tokens(expected),
        "the streamed run honors the same prompt-derived budget");
    Check(
        trace.kinds.size() == 4,
        "the budget is not recomputed once the token input shrinks");
  }

  // Iterative stages report one event per scheduler step and then complete.
  {
    const Pipeline pipeline = MakeCounterPipeline(kIterativeManifest);
    PipelineSession full = pipeline.CreateSession();
    const NamedTensors expected = full.RunStage("iterate");

    PipelineSession streamed = pipeline.CreateSession();
    StageRun run = streamed.BeginStage("iterate");
    const RunTrace trace = Drain(run);

    Check(
        trace.kinds ==
            std::vector<StageEventKind>{
                StageEventKind::iteration,
                StageEventKind::iteration,
                StageEventKind::iteration,
                StageEventKind::completed,
            },
        "an iterative run emits one iteration event per declared step");
    Check(
        trace.tokens.empty(),
        "iteration events carry no tokens");
    Check(
        trace.outputs.at("value").values<float>()[0] ==
            expected.at("value").values<float>()[0],
        "streamed iteration matches RunStage");
    Check(
        CounterValue(streamed) == CounterValue(full),
        "both paths leave the same recurrent state");

    // A stage that already reached its target does no work at all, on either
    // path, and still reports exactly one completed event.
    const NamedTensors repeated = full.RunStage("iterate");
    StageRun exhausted = streamed.BeginStage("iterate");
    const RunTrace done = Drain(exhausted);
    Check(
        done.kinds == std::vector<StageEventKind>{StageEventKind::completed},
        "an already-complete iterative stage completes immediately");
    Check(
        done.outputs.at("value").values<float>()[0] ==
            repeated.at("value").values<float>()[0],
        "an immediate completion returns the same outputs RunStage returns");
    Check(
        CounterValue(streamed) == 3.0F,
        "an immediate completion runs no component");

    // An option that raises the target resumes from the stage cursor rather
    // than restarting, exactly as RunStage does.
    PipelineRunOptions more;
    more.integers.emplace("num_inference_steps", 5);
    StageRun resumed = streamed.BeginStage("iterate", {}, {}, more);
    const RunTrace tail = Drain(resumed);
    Check(
        tail.kinds ==
            std::vector<StageEventKind>{
                StageEventKind::iteration,
                StageEventKind::iteration,
                StageEventKind::completed,
            },
        "a resumed iterative run only performs the remaining steps");
    Check(CounterValue(streamed) == 5.0F, "the resumed run reaches the target");
  }

  // Single-step stage kinds run once and then complete.
  {
    const Pipeline pipeline = MakeCounterPipeline(kCounterManifest);
    PipelineSession full = pipeline.CreateSession();
    const NamedTensors expected = full.RunStage("transition");

    PipelineSession streamed = pipeline.CreateSession();
    StageRun run = streamed.BeginStage("transition");
    const RunTrace trace = Drain(run);
    Check(
        trace.kinds ==
            std::vector<StageEventKind>{
                StageEventKind::transition,
                StageEventKind::completed,
            },
        "a state transition emits one transition event and completes");
    Check(
        trace.outputs.at("value").values<float>()[0] ==
            expected.at("value").values<float>()[0],
        "streamed state transition matches RunStage");
  }
  {
    const Pipeline pipeline = MakePassthroughPipeline();
    const std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
    const NamedTensors inputs{
        {"x", Tensor::FromValues<float>({1, 4}, std::span(values))}};

    PipelineSession full = pipeline.CreateSession();
    const NamedTensors expected = full.RunStage("run", inputs);

    PipelineSession streamed = pipeline.CreateSession();
    StageRun run = streamed.BeginStage("run", inputs);
    const RunTrace trace = Drain(run);
    Check(
        trace.kinds ==
            std::vector<StageEventKind>{
                StageEventKind::transition,
                StageEventKind::completed,
            },
        "a single pass emits one transition event and completes");
    Check(
        trace.outputs.at("produced").values<float>()[3] ==
            expected.at("produced").values<float>()[3],
        "streamed single pass matches RunStage");
  }

  // The terminal event happens once. Stepping past it is a state error, and
  // Finish keeps handing back the same cached result.
  {
    const Pipeline pipeline = MakeCounterPipeline(kCounterManifest);
    PipelineSession session = pipeline.CreateSession();
    StageRun run = session.BeginStage("transition");
    const RunTrace trace = Drain(run);
    Check(EndsWithSingleCompleted(trace), "exactly one completed event");
    CheckThrowsState(
        [&run] { (void)run.Step(); },
        "stepping a completed run must fail");

    const NamedTensors finished = run.Finish();
    Check(
        finished.at("value").values<float>()[0] == 1.0F,
        "Finish after completion returns the cached result");
    Check(
        CounterValue(session) == 1.0F,
        "Finish after completion runs nothing again");
    Check(
        run.Finish().at("value").values<float>()[0] == 1.0F,
        "Finish stays idempotent");
    // The completed run released the slot, so the session is usable again.
    Check(
        session.RunStage("transition").at("value").values<float>()[0] == 2.0F,
        "a completed run frees the session");
  }

  // Finish picks up wherever stepping stopped and still returns the whole
  // RunStage result.
  {
    const std::string document = DecoderManifest(
        4,
        R"json({
          "tokenizer_asset": "tokenizer.json",
          "sampling": {"do_sample": false},
          "stop": {"kind": "token_ids", "eos_token_ids": [3]},
          "max_tokens": {"default": 8, "limit": 8}
        })json");
    const Pipeline pipeline =
        MakeDecoderPipeline(document, std::make_shared<NextTokenBackend>(4));
    const std::array<std::int64_t, 1> prompt{0};
    const NamedTensors inputs{
        {"text.token_ids", PromptTensor({1, 1}, std::span(prompt))}};

    PipelineSession full = pipeline.CreateSession();
    const NamedTensors expected = full.RunStage("decode", inputs);

    PipelineSession streamed = pipeline.CreateSession();
    StageRun run = streamed.BeginStage("decode", inputs);
    const StageEvent first = run.Step();
    Check(first.kind == StageEventKind::token, "the first step decodes");
    Check(!run.done(), "a partly stepped run is not done");
    const NamedTensors drained = run.Finish();
    Check(
        Tokens(drained) == Tokens(expected),
        "Finish after partial stepping matches RunStage");
    Check(run.done(), "Finish completes the run");
  }

  // While a run holds the session's slot, every conflicting operation fails
  // loudly rather than capturing or mutating a half-executed stage.
  {
    const Pipeline pipeline = MakeCounterPipeline(kCounterManifest);
    PipelineSession session = pipeline.CreateSession();
    session.Checkpoint("mark");
    const PipelineSessionSnapshot snapshot = session.Snapshot();

    StageRun run = session.BeginStage("transition");
    (void)run.Step();

    CheckThrowsState(
        [&session] { (void)session.BeginStage("transition"); },
        "a second run must be rejected");
    CheckThrowsState(
        [&session] { (void)session.RunStage("transition"); },
        "RunStage during an active run must be rejected");
    CheckThrowsState(
        [&session] { (void)session.StepStage("transition"); },
        "StepStage during an active run must be rejected");
    CheckThrowsState(
        [&session] { (void)session.Snapshot(); },
        "Snapshot during an active run must be rejected");
    CheckThrowsState(
        [&session, &snapshot] { session.Restore(snapshot); },
        "Restore during an active run must be rejected");
    CheckThrowsState(
        [&session] { (void)session.Fork(); },
        "Fork during an active run must be rejected");
    CheckThrowsState(
        [&session] { session.Checkpoint("other"); },
        "Checkpoint during an active run must be rejected");
    CheckThrowsState(
        [&session] { session.RestoreCheckpoint("mark"); },
        "RestoreCheckpoint during an active run must be rejected");
    CheckThrowsState(
        [&session] { session.DropCheckpoint("mark"); },
        "DropCheckpoint during an active run must be rejected");
    CheckThrowsState(
        [&session] { session.Reset(); },
        "Reset during an active run must be rejected");
    CheckThrowsState(
        [&session] { session.ReleaseStage("transition"); },
        "ReleaseStage during an active run must be rejected");

    // Reads stay legal, because they observe state rather than change it.
    Check(
        session.outputs().at("value").values<float>()[0] == 1.0F,
        "outputs() stays readable during an active run");
    Check(
        CounterValue(session) == 1.0F,
        "state() stays readable during an active run");
    Check(
        session.HasCheckpoint("mark"),
        "HasCheckpoint stays readable during an active run");

    (void)run.Finish();
    session.Checkpoint("after");
    Check(
        session.HasCheckpoint("after"),
        "a finished run releases the session's control operations");
  }

  // Cancelling and destroying release the slot where the run stopped, without
  // rewinding what it already applied.
  {
    const Pipeline pipeline = MakeCounterPipeline(kIterativeManifest);
    PipelineSession session = pipeline.CreateSession();
    StageRun run = session.BeginStage("iterate");
    (void)run.Step();
    run.Cancel();
    Check(run.done(), "a cancelled run reports done");
    run.Cancel();
    Check(run.done(), "cancelling twice is harmless");
    CheckThrowsState(
        [&run] { (void)run.Step(); },
        "stepping a cancelled run must fail");
    CheckThrowsState(
        [&run] { (void)run.Finish(); },
        "finishing a cancelled run must fail");
    Check(
        CounterValue(session) == 1.0F,
        "cancelling keeps the step the run already applied");
    Check(
        session.RunStage("iterate").at("value").values<float>()[0] == 3.0F,
        "a cancelled run frees the session and leaves its cursor in place");
  }
  {
    const Pipeline pipeline = MakeCounterPipeline(kIterativeManifest);
    PipelineSession session = pipeline.CreateSession();
    {
      StageRun abandoned = session.BeginStage("iterate");
      (void)abandoned.Step();
    }
    Check(
        session.RunStage("iterate").at("value").values<float>()[0] == 3.0F,
        "destroying a run frees the session");
  }

  // A failing step closes the run and frees the session, and everything the
  // run had already applied stays applied -- the same contract a failing
  // RunStage always had.
  {
    const Pipeline pipeline = MakeFlakyPipeline(1);
    PipelineSession session = pipeline.CreateSession();
    StageRun run = session.BeginStage("iterate");
    (void)run.Step();
    CheckThrowsCode(
        [&run] { (void)run.Step(); },
        ErrorCode::runtime_execution,
        "a component failure surfaces from Step");
    Check(run.done(), "a failed run reports done");
    CheckThrowsState(
        [&run] { (void)run.Step(); },
        "stepping a failed run must fail");
    Check(
        CounterValue(session) == 1.0F,
        "a failed run keeps the state it already applied");
    CheckThrowsCode(
        [&session] { (void)session.RunStage("iterate"); },
        ErrorCode::runtime_execution,
        "the session is usable after a failed run");
  }
  {
    // The same contract through Finish, which is also how a failing RunStage
    // now reaches the caller.
    const Pipeline pipeline = MakeFlakyPipeline(2);
    PipelineSession session = pipeline.CreateSession();
    StageRun run = session.BeginStage("iterate");
    CheckThrowsCode(
        [&run] { (void)run.Finish(); },
        ErrorCode::runtime_execution,
        "a component failure surfaces from Finish");
    Check(
        CounterValue(session) == 2.0F,
        "a failed Finish keeps the steps that succeeded");
    Check(session.HasCheckpoint("none") == false, "the session is not locked");
  }

  // Moving a run transfers the slot: the source owns nothing afterwards and
  // its destructor cancels nothing.
  {
    const Pipeline pipeline = MakeCounterPipeline(kCounterManifest);
    PipelineSession session = pipeline.CreateSession();
    StageRun moved = [&session] {
      StageRun started = session.BeginStage("transition");
      (void)started.Step();
      return std::move(started);
    }();
    Check(moved.stage() == "transition", "the moved-to handle owns the run");
    Check(!moved.done(), "the moved-to handle is still running");
    Check(moved.iteration() == 1, "the moved-to handle keeps the step count");
    const NamedTensors outputs = moved.Finish();
    Check(
        outputs.at("value").values<float>()[0] == 1.0F,
        "a moved run finishes normally");

    StageRun source = session.BeginStage("transition");
    StageRun destination = std::move(source);
    Check(source.done(), "a moved-from handle reports done");
    CheckThrowsState(
        [&source] { (void)source.Step(); },
        "stepping a moved-from handle must fail");
    CheckThrowsState(
        [&source] { (void)source.Finish(); },
        "finishing a moved-from handle must fail");
    CheckThrowsState(
        [&source] { (void)source.stage(); },
        "reading the stage of a moved-from handle must fail");
    CheckThrowsState(
        [&source] { (void)source.iteration(); },
        "reading the iteration of a moved-from handle must fail");
    source.Cancel();
    Check(
        !destination.done(),
        "cancelling a moved-from handle leaves the real run alone");
    (void)destination.Finish();
  }
  {
    // Move assignment abandons the run the target held and adopts the source.
    const Pipeline pipeline = MakeCounterPipeline(kCounterManifest);
    PipelineSession first = pipeline.CreateSession();
    PipelineSession second = pipeline.CreateSession();
    StageRun target = first.BeginStage("transition");
    (void)target.Step();
    StageRun source = second.BeginStage("transition");
    target = std::move(source);
    Check(
        first.RunStage("transition").at("value").values<float>()[0] == 2.0F,
        "assignment releases the abandoned run's slot");
    (void)target.Finish();
    Check(CounterValue(second) == 1.0F, "the adopted run still drives its own session");
  }

  // A run holds a share of the session's state, so a moved or destroyed
  // session wrapper cannot leave it dangling.
  {
    const Pipeline pipeline = MakeCounterPipeline(kIterativeManifest);
    StageRun orphan = [&pipeline] {
      PipelineSession session = pipeline.CreateSession();
      StageRun started = session.BeginStage("iterate");
      (void)started.Step();
      return started;
    }();
    const NamedTensors outputs = orphan.Finish();
    Check(
        outputs.at("value").values<float>()[0] == 3.0F,
        "a run outlives the session wrapper that started it");
  }
  {
    const Pipeline pipeline = MakeCounterPipeline(kIterativeManifest);
    PipelineSession session = pipeline.CreateSession();
    StageRun run = session.BeginStage("iterate");
    PipelineSession moved = std::move(session);
    (void)run.Step();
    CheckThrowsState(
        [&moved] { (void)moved.RunStage("iterate"); },
        "the moved session still sees its active run");
    (void)run.Finish();
    Check(
        moved.RunStage("iterate").at("value").values<float>()[0] == 3.0F,
        "the moved session is usable once the run ends");
  }

  // Device-backed outputs reach a StageEvent as the producer's buffer, with
  // no materialization anywhere along the way.
  {
    const Pipeline pipeline = MakePassthroughPipeline();
    const auto counter = std::make_shared<DeviceCopyCounter>();
    const std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
    const Tensor device_input =
        MakeDeviceTensor({1, 4}, std::span(values), counter);

    PipelineSession session = pipeline.CreateSession();
    StageRun run =
        session.BeginStage("run", {{"x", device_input}});
    const StageEvent step = run.Step();
    Check(
        step.outputs.at("produced").buffer().get() ==
            device_input.buffer().get(),
        "a step event forwards the device buffer it was given");
    const StageEvent completed = run.Step();
    Check(
        completed.outputs.at("produced").buffer().get() ==
            device_input.buffer().get(),
        "the completed event forwards the same device buffer");
    Check(
        counter->copies == 0,
        "reporting device outputs as events copies nothing to the host");
  }

  // Unknown stages fail where they always did, before any run slot is taken.
  {
    const Pipeline pipeline = MakeCounterPipeline(kCounterManifest);
    PipelineSession session = pipeline.CreateSession();
    CheckThrowsCode(
        [&session] { (void)session.BeginStage("missing"); },
        ErrorCode::invalid_argument,
        "beginning an unknown stage must fail");
    Check(
        session.RunStage("transition").at("value").values<float>()[0] == 1.0F,
        "a rejected BeginStage leaves the session free");
  }
  {
    // An autoregressive stage without a token budget is rejected by
    // BeginStage, because the budget is resolved when the run starts.
    const std::string document = DecoderManifest(
        4,
        R"json({
          "tokenizer_asset": "tokenizer.json",
          "sampling": {"do_sample": false},
          "stop": {"kind": "token_ids", "eos_token_ids": [3]}
        })json");
    const Pipeline pipeline =
        MakeDecoderPipeline(document, std::make_shared<NextTokenBackend>(4));
    const std::array<std::int64_t, 1> prompt{0};
    const NamedTensors inputs{
        {"text.token_ids", PromptTensor({1, 1}, std::span(prompt))}};
    PipelineSession session = pipeline.CreateSession();
    CheckThrowsCode(
        [&session, &inputs] { (void)session.BeginStage("decode", inputs); },
        ErrorCode::invalid_argument,
        "a missing token budget must fail when the run begins");
    PipelineRunOptions options;
    options.integers.emplace("max_tokens", 2);
    StageRun run = session.BeginStage("decode", inputs, {}, options);
    Check(
        Tokens(run.Finish()).size() == 2,
        "the session is still usable after a rejected BeginStage");
  }

  // Full RunStage retains its historical whole-stage lock: a concurrent full
  // run waits and succeeds rather than observing the internal active run slot.
  {
    auto control = std::make_shared<BlockingControl>();
    const Pipeline pipeline = MakeBlockingPipeline(control);
    PipelineSession session = pipeline.CreateSession();
    PipelineRunOptions options;
    options.integers.emplace("num_inference_steps", 1000);
    std::exception_ptr first_error;
    std::exception_ptr second_error;

    std::thread first([&] {
      try {
        (void)session.RunStage("iterate", {}, {}, options);
      } catch (...) {
        first_error = std::current_exception();
      }
    });
    {
      std::unique_lock lock(control->mutex);
      control->condition.wait(lock, [&control] {
        return control->first_call_entered;
      });
    }
    std::thread second([&] {
      {
        std::scoped_lock lock(control->mutex);
        control->second_call_started = true;
      }
      control->condition.notify_all();
      try {
        (void)session.StepStage("iterate", {}, {}, options);
      } catch (...) {
        second_error = std::current_exception();
      }
    });
    {
      std::unique_lock lock(control->mutex);
      control->condition.wait(lock, [&control] {
        return control->second_call_started;
      });
      // Give the second thread time to enter RunStage and block on the
      // session mutex while the first backend call still owns it.
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      lock.lock();
      control->release_first_call = true;
    }
    control->condition.notify_all();
    first.join();
    second.join();

    Check(first_error == nullptr, "the first full run succeeds");
    Check(
        second_error == nullptr,
        "a concurrent direct step waits instead of seeing an active run");
    Check(
        control->calls == 1001,
        "the direct step executes only after all 1000 full-run steps");
  }

  if (failures == 0) {
    std::cout << "pipeline_stream_test: all checks passed\n";
  }
  return failures == 0 ? 0 : 1;
}
