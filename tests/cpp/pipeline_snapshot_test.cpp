/**
 * @agent-file
 * @agent-purpose: Standalone test executable for the in-memory PipelineSession snapshot, restore, fork, and named-checkpoint contract: recurrent-state round trips, parent/child independence, package identity, device-buffer sharing, random-engine determinism, and checkpoint create/replace/restore/drop semantics.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as pipeline_snapshot_test; it counts failures through local Check, CheckThrowsCode, CheckThrowsState, and CheckThrowsInvalidArgument helpers and returns a non-zero exit code when any check fails. Every component is a stub ModelBackend, so the run needs no ONNX Runtime library, no real ONNX model, and no filesystem access: each PipelinePackage is built in memory from an embedded manifest string. CountingDeviceBuffer is a non-host-accessible TensorBuffer whose shared copy counter asserts that snapshot, fork, restore, and checkpoint save and restore never materialize a device tensor.
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
void CheckThrowsCode(
    Function&& function,
    onnx_world_model::ErrorCode expected,
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
  CheckThrowsCode(
      std::forward<Function>(function),
      onnx_world_model::ErrorCode::state,
      message);
}

template <typename Function>
void CheckThrowsInvalidArgument(Function&& function, const char* message) {
  CheckThrowsCode(
      std::forward<Function>(function),
      onnx_world_model::ErrorCode::invalid_argument,
      message);
}

// Adds one to its single float state, mutating the tensor it was handed. The
// session shares that tensor's storage with every live snapshot, so the write
// exercises Tensor copy-on-write rather than corrupting another branch.
class IncrementBackend final : public onnx_world_model::ModelBackend {
 public:
  IncrementBackend() {
    metadata_.inputs.push_back({
        .name = "state",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {1},
    });
    metadata_.outputs.push_back({
        .name = "next_state",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {1},
    });
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    onnx_world_model::Tensor result = inputs.at("state");
    auto value = std::span(
        reinterpret_cast<float*>(result.mutable_bytes().data()),
        result.element_count());
    value[0] += 1.0F;
    return {{"next_state", std::move(result)}};
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

// Emits uniform logits so every sampled token comes from the session's random
// engine alone, which is what makes the engine's snapshot and fork behavior
// observable through the public token output.
class UniformLogitsBackend final : public onnx_world_model::ModelBackend {
 public:
  UniformLogitsBackend() {
    metadata_.inputs = {
        {
            .name = "tokens",
            .data_type = onnx_world_model::DataType::int64,
            .shape = {1, -1},
        },
        {
            .name = "state",
            .data_type = onnx_world_model::DataType::float32,
            .shape = {1},
        },
    };
    metadata_.outputs = {
        {
            .name = "logits",
            .data_type = onnx_world_model::DataType::float32,
            .shape = {1, -1, 4},
        },
        {
            .name = "next_state",
            .data_type = onnx_world_model::DataType::float32,
            .shape = {1},
        },
    };
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    const auto& tokens = inputs.at("tokens");
    const float state = inputs.at("state").values<float>()[0];
    const std::array<float, 1> next_state{state + 1.0F};
    return {
        {
            "logits",
            onnx_world_model::Tensor::Zeros(
                onnx_world_model::DataType::float32,
                {1, tokens.shape()[1], 4}),
        },
        {
            "next_state",
            onnx_world_model::Tensor::FromValues<float>(
                {1}, std::span(next_state)),
        },
    };
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
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
        onnx_world_model::ErrorCode::invalid_argument,
        "Counting device buffer is not host-accessible");
  }

  void CopyToCpu(std::span<std::byte> destination) const override {
    if (destination.size() != storage_.size()) {
      throw onnx_world_model::Error(
          onnx_world_model::ErrorCode::invalid_argument,
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

onnx_world_model::Tensor MakeDeviceTensor(
    std::vector<std::int64_t> shape,
    std::span<const float> values,
    const std::shared_ptr<DeviceCopyCounter>& counter) {
  const std::span<const std::byte> raw = std::as_bytes(values);
  return onnx_world_model::Tensor::FromBuffer(
      onnx_world_model::DataType::float32,
      std::move(shape),
      std::make_shared<CountingDeviceBuffer>(
          std::vector<std::byte>(raw.begin(), raw.end()), counter));
}

// Forwards its device input unchanged so the session keeps the caller's
// buffer in both its endpoint values and its public outputs.
class PassthroughBackend final : public onnx_world_model::ModelBackend {
 public:
  PassthroughBackend() {
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
    return {{"y", inputs.at("x")}};
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

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

constexpr std::string_view kSamplingManifest = R"json(
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
          {"name": "tokens", "dtype": "INT64", "shape": [1, "sequence"]},
          {"name": "state", "dtype": "FLOAT", "shape": [1]}
        ],
        "outputs": [
          {"name": "logits", "dtype": "FLOAT", "shape": [1, "sequence", 4]},
          {"name": "next_state", "dtype": "FLOAT", "shape": [1]}
        ],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT"
      }
    ],
    "connections": [
      {
        "source": "decoder.next_state",
        "target": "decoder.state",
        "recurrent": true
      }
    ],
    "stages": [
      {
        "name": "decode",
        "kind": "autoregressive",
        "components": ["decoder"],
        "run_on": "decode",
        "options": {
          "tokenizer_asset": "tokenizer.json",
          "sampling": {"do_sample": true},
          "stop": {"kind": "token_ids", "eos_token_ids": [99]},
          "max_tokens": {"default": 8, "limit": 8},
          "state_names": ["cache"]
        },
        "capabilities": ["loop_carried_state"]
      }
    ],
    "inputs": [
      {
        "port": "decoder.state",
        "kind": "generated",
        "required": true,
        "semantic": "kv_cache.initial",
        "generator": {"kind": "zeros"}
      },
      {
        "port": "decoder.tokens",
        "kind": "external",
        "required": true,
        "semantic": "text.token_ids"
      }
    ],
    "outputs": [{"port": "decoder.logits"}],
    "profile": {"name": "decoder-world", "version": "1.0"},
    "states": [
      {
        "name": "cache",
        "kind": "kv_cache",
        "input": "decoder.state",
        "output": "decoder.next_state",
        "lifetime": "sequence",
        "release_after": "decode"
      }
    ],
    "assets": [{"path": "tokenizer.json"}],
    "required_capabilities": ["loop_carried_state"]
  },
  "component_files": {"decoder": "model.onnx"}
}
)json";

[[nodiscard]] onnx_world_model::Pipeline MakeCounterPipeline() {
  std::unordered_map<std::string, onnx_world_model::Model> models;
  models.emplace(
      "counter",
      onnx_world_model::Model(std::make_shared<IncrementBackend>()));
  return onnx_world_model::Pipeline(onnx_world_model::PipelinePackage(
      {},
      onnx_world_model::PipelineManifest::Parse(kCounterManifest),
      std::move(models)));
}

[[nodiscard]] float CounterValue(
    const onnx_world_model::PipelineSession& session) {
  const std::optional<onnx_world_model::Tensor> value =
      session.state("counter_state");
  if (!value.has_value()) {
    return -1.0F;
  }
  return value->CopyToCpu().values<float>()[0];
}

[[nodiscard]] const onnx_world_model::TensorBuffer* CounterBuffer(
    const onnx_world_model::PipelineSession& session) {
  const std::optional<onnx_world_model::Tensor> value =
      session.state("counter_state");
  return value.has_value() ? value->buffer().get() : nullptr;
}

[[nodiscard]] std::vector<std::int64_t> Tokens(
    const onnx_world_model::NamedTensors& outputs) {
  const auto span =
      outputs.at("generated_token_ids").values<std::int64_t>();
  return {span.begin(), span.end()};
}

}  // namespace

int main() {
  using onnx_world_model::DataType;
  using onnx_world_model::Model;
  using onnx_world_model::Pipeline;
  using onnx_world_model::PipelineManifest;
  using onnx_world_model::PipelinePackage;
  using onnx_world_model::PipelineRunOptions;
  using onnx_world_model::PipelineSession;
  using onnx_world_model::PipelineSessionSnapshot;
  using onnx_world_model::Tensor;

  // A snapshot reproduces the recurrent state and the public outputs the
  // session had when it was taken, however far the session has advanced.
  {
    const Pipeline pipeline = MakeCounterPipeline();
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("transition");
    const auto captured_outputs = session.RunStage("transition");
    Check(
        captured_outputs.at("value").values<float>()[0] == 2.0F,
        "the session advances before the snapshot");

    const PipelineSessionSnapshot snapshot = session.Snapshot();
    Check(snapshot.valid(), "a captured snapshot is valid");

    (void)session.RunStage("transition");
    (void)session.RunStage("transition");
    Check(CounterValue(session) == 4.0F, "the session advances past the snapshot");

    session.Restore(snapshot);
    Check(CounterValue(session) == 2.0F, "restore reproduces the recurrent state");
    Check(
        session.outputs().at("value").values<float>()[0] == 2.0F,
        "restore reproduces the public outputs");
    Check(
        session.RunStage("transition").at("value").values<float>()[0] == 3.0F,
        "a restored session resumes from the captured cursor");

    // The snapshot is immutable, so it survives every later restore.
    session.Restore(snapshot);
    Check(CounterValue(session) == 2.0F, "a snapshot restores more than once");
  }

  // A fork starts identical and then evolves on its own. Neither side sees
  // the other's runs, releases, or resets.
  {
    const Pipeline pipeline = MakeCounterPipeline();
    PipelineSession parent = pipeline.CreateSession();
    (void)parent.RunStage("transition");
    (void)parent.RunStage("transition");

    PipelineSession child = parent.Fork();
    Check(CounterValue(child) == 2.0F, "a fork begins at the parent's state");
    Check(
        CounterBuffer(child) == CounterBuffer(parent),
        "a fork shares the parent's tensor storage");

    (void)child.RunStage("transition");
    Check(CounterValue(child) == 3.0F, "the child advances");
    Check(CounterValue(parent) == 2.0F, "the child's run leaves the parent alone");
    Check(
        CounterBuffer(child) != CounterBuffer(parent),
        "mutating the child copies the shared tensor instead of writing it");

    (void)parent.RunStage("transition");
    (void)parent.RunStage("transition");
    Check(CounterValue(parent) == 4.0F, "the parent advances independently");
    Check(CounterValue(child) == 3.0F, "the parent's runs leave the child alone");

    child.Reset();
    Check(!child.state("counter_state").has_value(), "the child resets");
    Check(CounterValue(parent) == 4.0F, "resetting the child leaves the parent alone");

    parent.ReleaseStage("transition");
    Check(!parent.state("counter_state").has_value(), "the parent releases");
    Check(
        parent.Fork().state("counter_state").has_value() == false,
        "a fork of a released session is released too");

    // A fork of a fresh session is usable on its own.
    PipelineSession grandchild = child.Fork();
    Check(
        grandchild.RunStage("transition").at("value").values<float>()[0] == 1.0F,
        "a fork of a reset session starts from the generated initial state");
    Check(!child.state("counter_state").has_value(), "the grandchild's run is its own");
  }

  // Package identity, not manifest text, decides whether a snapshot fits.
  {
    const Pipeline pipeline = MakeCounterPipeline();
    const Pipeline twin = MakeCounterPipeline();
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("transition");
    const PipelineSessionSnapshot snapshot = session.Snapshot();

    PipelineSession stranger = twin.CreateSession();
    CheckThrowsState(
        [&stranger, &snapshot] { stranger.Restore(snapshot); },
        "a snapshot from a textually identical package must be rejected");
    Check(
        !stranger.state("counter_state").has_value(),
        "a rejected restore leaves the target session untouched");

    // A copied Pipeline shares the same immutable package, so its sessions
    // are interchangeable with the original's.
    const Pipeline copy = pipeline;
    PipelineSession sibling = copy.CreateSession();
    sibling.Restore(snapshot);
    Check(
        CounterValue(sibling) == 1.0F,
        "a session from a copied pipeline accepts the snapshot");

    PipelineSessionSnapshot moved = session.Snapshot();
    const PipelineSessionSnapshot taken = std::move(moved);
    Check(taken.valid(), "a moved-to snapshot is valid");
    Check(!moved.valid(), "a moved-from snapshot is not valid");
    CheckThrowsState(
        [&session, &moved] { session.Restore(moved); },
        "a moved-from snapshot must be rejected");
  }

  // Device-backed state is shared, never materialized, and a later run on one
  // branch replaces that branch's tensors without touching the other's.
  {
    auto counter = std::make_shared<DeviceCopyCounter>();
    std::unordered_map<std::string, Model> models;
    models.emplace("producer", Model(std::make_shared<PassthroughBackend>()));
    const Pipeline pipeline(PipelinePackage(
        {}, PipelineManifest::Parse(kPassthroughManifest), std::move(models)));
    PipelineSession parent = pipeline.CreateSession();

    const std::array<float, 4> first{1.0F, 2.0F, 3.0F, 4.0F};
    Tensor device_input = MakeDeviceTensor({1, 4}, std::span(first), counter);
    const onnx_world_model::TensorBuffer* first_buffer =
        device_input.buffer().get();
    (void)parent.RunStage("run", {{"producer.x", device_input}});
    Check(counter->copies == 0, "running on a device tensor performs no CPU copy");

    const PipelineSessionSnapshot snapshot = parent.Snapshot();
    PipelineSession child = parent.Fork();
    PipelineSession restored = pipeline.CreateSession();
    restored.Restore(snapshot);
    Check(
        counter->copies == 0,
        "snapshot, fork, and restore never materialize a device tensor");
    Check(
        parent.outputs().at("produced").buffer().get() == first_buffer &&
            child.outputs().at("produced").buffer().get() == first_buffer &&
            restored.outputs().at("produced").buffer().get() == first_buffer,
        "every branch shares the producing device buffer");

    // Replacing the parent's state must not reach into the other branches.
    const std::array<float, 4> second{9.0F, 9.0F, 9.0F, 9.0F};
    Tensor replacement = MakeDeviceTensor({1, 4}, std::span(second), counter);
    const onnx_world_model::TensorBuffer* second_buffer =
        replacement.buffer().get();
    (void)parent.RunStage("run", {{"producer.x", replacement}});
    Check(
        parent.outputs().at("produced").buffer().get() == second_buffer,
        "the parent picks up the replacement buffer");
    Check(
        child.outputs().at("produced").buffer().get() == first_buffer &&
            restored.outputs().at("produced").buffer().get() == first_buffer,
        "state replacement on one branch leaves the others on their buffer");
    Check(counter->copies == 0, "state replacement still performs no CPU copy");

    Check(
        child.outputs().at("produced").CopyToCpu().values<float>()[3] == 4.0F,
        "the untouched branch still holds the captured values");
    Check(
        parent.outputs().at("produced").CopyToCpu().values<float>()[3] == 9.0F,
        "the advanced branch holds the replacement values");
    Check(counter->copies == 2, "each explicit materialization copies once");
  }

  // The random engine is part of the captured state, so a restored or forked
  // session continues the parent's exact sampling sequence.
  {
    std::unordered_map<std::string, Model> models;
    models.emplace(
        "decoder", Model(std::make_shared<UniformLogitsBackend>()));
    const Pipeline pipeline(PipelinePackage(
        {}, PipelineManifest::Parse(kSamplingManifest), std::move(models)));
    PipelineSession parent = pipeline.CreateSession();

    PipelineRunOptions seeded;
    seeded.integers = {{"do_sample", 1}, {"seed", 11}};
    PipelineRunOptions unseeded;
    unseeded.integers = {{"do_sample", 1}};

    const std::array<std::int64_t, 2> prompt{5, 6};
    const auto first = Tokens(parent.RunStage(
        "decode",
        {{"text.token_ids", Tensor::FromValues<std::int64_t>(
                                {1, 2}, std::span(prompt))}},
        {},
        seeded));
    Check(first.size() == 8, "the sampling stage generates the requested tokens");

    const PipelineSessionSnapshot snapshot = parent.Snapshot();
    PipelineSession child = parent.Fork();
    PipelineSession restored = pipeline.CreateSession();
    restored.Restore(snapshot);

    // Continuing without a seed draws from wherever the engine was left.
    const auto continued = Tokens(parent.RunStage("decode", {}, {}, unseeded));
    Check(
        Tokens(child.RunStage("decode", {}, {}, unseeded)) == continued,
        "a fork continues the parent's random sequence");
    Check(
        Tokens(restored.RunStage("decode", {}, {}, unseeded)) == continued,
        "a restored session continues the captured random sequence");

    parent.Restore(snapshot);
    Check(
        Tokens(parent.RunStage("decode", {}, {}, unseeded)) == continued,
        "restoring the source session replays the same random sequence");

    std::vector<std::int64_t> sampled = first;
    sampled.insert(sampled.end(), continued.begin(), continued.end());
    Check(
        std::ranges::any_of(
            sampled,
            [&sampled](std::int64_t token) { return token != sampled.front(); }),
        "uniform logits actually sample more than one token value");
  }

  // Named checkpoints are in-memory transaction markers over the same capture
  // the snapshot API performs: create, query, replace, rewind, and drop.
  {
    const Pipeline pipeline = MakeCounterPipeline();
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("transition");
    (void)session.RunStage("transition");

    Check(
        !session.HasCheckpoint("before"),
        "a fresh session holds no named checkpoint");
    session.Checkpoint("before");
    Check(session.HasCheckpoint("before"), "a created checkpoint is visible");
    Check(
        !session.HasCheckpoint("other"),
        "an unrelated name is still absent");

    // Named checkpoints survive stage execution.
    (void)session.RunStage("transition");
    (void)session.RunStage("transition");
    Check(CounterValue(session) == 4.0F, "the session advances past the checkpoint");
    Check(
        session.HasCheckpoint("before"),
        "stage execution preserves named checkpoints");

    session.RestoreCheckpoint("before");
    Check(CounterValue(session) == 2.0F, "restoring a checkpoint rewinds the state");
    Check(
        session.outputs().at("value").values<float>()[0] == 2.0F,
        "restoring a checkpoint reproduces the public outputs");
    Check(
        session.HasCheckpoint("before"),
        "restoring a checkpoint does not consume it");
    Check(
        session.RunStage("transition").at("value").values<float>()[0] == 3.0F,
        "a checkpoint-restored session resumes from the captured cursor");

    // Replacing a name is atomic: the old capture is simply gone.
    session.Checkpoint("before");
    Check(
        session.HasCheckpoint("before"),
        "replacing a checkpoint keeps the name");
    (void)session.RunStage("transition");
    session.RestoreCheckpoint("before");
    Check(
        CounterValue(session) == 3.0F,
        "a replaced checkpoint restores the newer capture");

    session.DropCheckpoint("before");
    Check(!session.HasCheckpoint("before"), "a dropped checkpoint is gone");
    CheckThrowsState(
        [&session] { session.RestoreCheckpoint("before"); },
        "restoring a dropped checkpoint must fail");
    Check(
        CounterValue(session) == 3.0F,
        "a failed checkpoint restore leaves the session untouched");
  }

  // Unknown and empty names fail loudly instead of silently doing nothing.
  {
    const Pipeline pipeline = MakeCounterPipeline();
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("transition");

    CheckThrowsState(
        [&session] { session.RestoreCheckpoint("missing"); },
        "restoring an unknown checkpoint must throw ErrorCode::state");
    CheckThrowsState(
        [&session] { session.DropCheckpoint("missing"); },
        "dropping an unknown checkpoint is not a no-op");
    Check(
        CounterValue(session) == 1.0F,
        "a failed checkpoint operation changes nothing");

    CheckThrowsInvalidArgument(
        [&session] { session.Checkpoint(""); },
        "an empty checkpoint name must throw ErrorCode::invalid_argument");
    CheckThrowsInvalidArgument(
        [&session] { session.RestoreCheckpoint(""); },
        "an empty restore name must throw ErrorCode::invalid_argument");
    CheckThrowsInvalidArgument(
        [&session] { session.DropCheckpoint(""); },
        "an empty drop name must throw ErrorCode::invalid_argument");
    CheckThrowsInvalidArgument(
        [&session] { (void)session.HasCheckpoint(""); },
        "an empty query name must throw ErrorCode::invalid_argument");
  }

  // Checkpoints are control metadata, not execution state: Reset drops them,
  // a fork inherits none of them, and an ordinary Restore leaves the target
  // session's own checkpoint namespace alone.
  {
    const Pipeline pipeline = MakeCounterPipeline();
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("transition");
    session.Checkpoint("first");
    session.Checkpoint("second");

    PipelineSession child = session.Fork();
    Check(CounterValue(child) == 1.0F, "a fork begins at the parent's state");
    Check(
        !child.HasCheckpoint("first") && !child.HasCheckpoint("second"),
        "a fork starts with an empty named-checkpoint namespace");
    Check(
        session.HasCheckpoint("first") && session.HasCheckpoint("second"),
        "forking leaves the parent's checkpoints in place");
    CheckThrowsState(
        [&child] { child.RestoreCheckpoint("first"); },
        "a fork cannot restore a name it never declared");

    // A snapshot carries execution state only, so restoring one neither adds
    // nor removes names on the session that receives it.
    child.Checkpoint("child_only");
    const PipelineSessionSnapshot snapshot = session.Snapshot();
    child.Restore(snapshot);
    Check(
        CounterValue(child) == 1.0F,
        "an ordinary restore still replaces the execution state");
    Check(
        child.HasCheckpoint("child_only"),
        "an ordinary restore preserves the target session's checkpoints");
    Check(
        !child.HasCheckpoint("first"),
        "an ordinary restore does not import the source session's checkpoints");

    session.Reset();
    Check(
        !session.HasCheckpoint("first") && !session.HasCheckpoint("second"),
        "Reset clears every named checkpoint");
    CheckThrowsState(
        [&session] { session.RestoreCheckpoint("first"); },
        "a checkpoint cleared by Reset cannot be restored");
  }

  // Checkpointing and rewinding device-backed state shares buffers exactly
  // like Snapshot and Restore do, and never materializes one to the host.
  {
    auto counter = std::make_shared<DeviceCopyCounter>();
    std::unordered_map<std::string, Model> models;
    models.emplace("producer", Model(std::make_shared<PassthroughBackend>()));
    const Pipeline pipeline(PipelinePackage(
        {}, PipelineManifest::Parse(kPassthroughManifest), std::move(models)));
    PipelineSession session = pipeline.CreateSession();

    const std::array<float, 4> first{1.0F, 2.0F, 3.0F, 4.0F};
    Tensor device_input = MakeDeviceTensor({1, 4}, std::span(first), counter);
    const onnx_world_model::TensorBuffer* first_buffer =
        device_input.buffer().get();
    (void)session.RunStage("run", {{"producer.x", device_input}});

    session.Checkpoint("captured");
    Check(counter->copies == 0, "checkpointing never materializes a device tensor");

    const std::array<float, 4> second{9.0F, 9.0F, 9.0F, 9.0F};
    Tensor replacement = MakeDeviceTensor({1, 4}, std::span(second), counter);
    const onnx_world_model::TensorBuffer* second_buffer =
        replacement.buffer().get();
    (void)session.RunStage("run", {{"producer.x", replacement}});
    Check(
        session.outputs().at("produced").buffer().get() == second_buffer,
        "the session picks up the replacement buffer");

    session.RestoreCheckpoint("captured");
    Check(
        session.outputs().at("produced").buffer().get() == first_buffer,
        "restoring a checkpoint restores the captured device buffer");
    Check(
        counter->copies == 0,
        "checkpoint save and restore never materialize a device tensor");
  }

  if (failures == 0) {
    std::cout << "pipeline snapshot tests passed\n";
  }
  return failures == 0 ? 0 : 1;
}
