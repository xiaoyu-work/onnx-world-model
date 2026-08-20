/**
 * @agent-file
 * @agent-purpose: Implements the fixed latent-dynamics API: WorldModel enforces the three-input/four-output graph contract per step, and Rollout carries recurrent state across steps.
 * @agent-public-api: WorldModel::WorldModel, WorldModel::Load, WorldModel::metadata, WorldModel::Step, Rollout::Rollout, Rollout::Reset, Rollout::ResetZeros, Rollout::has_state, Rollout::state, Rollout::Step
 * @agent-invariants: Construction rejects any graph whose signature is not exactly observation, action, state to next_state, observation_prediction, reward, continuation, where next_state matches state, observation_prediction matches observation, and reward and continuation are [batch, 1]. Every step revalidates the returned shapes. Rollout guards all state under its mutex, lazily zero-initializes state from the observation batch, and adopts next_state only after a successful step; zero state requires all non-batch dimensions to be static.
 * @agent-side-effects: WorldModel::Load reads a model file and loads the ONNX Runtime shared library; Rollout mutates its own recurrent state.
 */

#include "onnx_world_model/world_model.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include "onnx_world_model/error.hpp"
#include "ort_backend.hpp"

namespace onnx_world_model {
namespace {

constexpr std::array<std::string_view, 3> kInputNames{
    "observation",
    "action",
    "state",
};
constexpr std::array<std::string_view, 4> kOutputNames{
    "next_state",
    "observation_prediction",
    "reward",
    "continuation",
};

[[nodiscard]] bool ShapesCompatible(
    const std::vector<std::int64_t>& left,
    const std::vector<std::int64_t>& right) {
  if (left.size() != right.size()) {
    return false;
  }
  return std::ranges::equal(
      left,
      right,
      [](std::int64_t left_dimension, std::int64_t right_dimension) {
        return left_dimension < 0 || right_dimension < 0 ||
               left_dimension == right_dimension;
      });
}

void ValidateBatch(
    const Tensor& observation,
    const Tensor& action,
    const Tensor& state) {
  if (observation.shape().empty() || action.shape().empty() || state.shape().empty()) {
    throw Error(ErrorCode::invalid_argument, "World-model tensors must include batch");
  }
  if (observation.shape()[0] != action.shape()[0] ||
      observation.shape()[0] != state.shape()[0]) {
    throw Error(
        ErrorCode::invalid_argument,
        "Observation, action, and state batch dimensions must match");
  }
}

void ValidateContract(const ModelMetadata& metadata) {
  if (metadata.inputs.size() != kInputNames.size()) {
    throw Error(
        ErrorCode::model_contract,
        "World model must have exactly three inputs");
  }
  if (metadata.outputs.size() != kOutputNames.size()) {
    throw Error(
        ErrorCode::model_contract,
        "World model must have exactly four outputs");
  }
  for (const auto name : kInputNames) {
    (void)metadata.Input(name);
  }
  for (const auto name : kOutputNames) {
    (void)metadata.Output(name);
  }

  const auto& observation = metadata.Input("observation");
  const auto& action = metadata.Input("action");
  const auto& state = metadata.Input("state");
  const auto& next_state = metadata.Output("next_state");
  const auto& observation_prediction =
      metadata.Output("observation_prediction");
  const auto& reward = metadata.Output("reward");
  const auto& continuation = metadata.Output("continuation");

  if (observation.shape.empty() || action.shape.empty() || state.shape.empty()) {
    throw Error(
        ErrorCode::model_contract,
        "World-model inputs must include a batch dimension");
  }
  if (state.data_type != next_state.data_type ||
      !ShapesCompatible(state.shape, next_state.shape)) {
    throw Error(
        ErrorCode::model_contract,
        "next_state must match the state data type and shape");
  }
  if (observation.data_type != observation_prediction.data_type ||
      !ShapesCompatible(observation.shape, observation_prediction.shape)) {
    throw Error(
        ErrorCode::model_contract,
        "observation_prediction must match the observation data type and shape");
  }
  for (const TensorSpec* scalar_output : {&reward, &continuation}) {
    if (scalar_output->shape.size() != 2 ||
        (scalar_output->shape[1] >= 0 && scalar_output->shape[1] != 1)) {
      throw Error(
          ErrorCode::model_contract,
          "reward and continuation must have shape [batch, 1]");
    }
  }
}

void ValidateOutput(
    const StepOutput& output,
    const StepInput& input,
    const ModelMetadata& metadata) {
  ValidateTensor(output.next_state, metadata.Output("next_state"));
  ValidateTensor(
      output.observation_prediction,
      metadata.Output("observation_prediction"));
  ValidateTensor(output.reward, metadata.Output("reward"));
  ValidateTensor(output.continuation, metadata.Output("continuation"));
  if (output.next_state.shape() != input.state.shape()) {
    throw Error(
        ErrorCode::runtime_execution,
        "next_state shape changed across a recurrent step");
  }
  if (output.observation_prediction.shape() != input.observation.shape()) {
    throw Error(
        ErrorCode::runtime_execution,
        "observation_prediction shape does not match observation");
  }
  const std::vector<std::int64_t> scalar_shape{
      input.observation.shape()[0],
      1,
  };
  if (output.reward.shape() != scalar_shape ||
      output.continuation.shape() != scalar_shape) {
    throw Error(
        ErrorCode::runtime_execution,
        "reward and continuation must have concrete shape [batch, 1]");
  }
}

class ModelBackendAdapter final : public Backend {
 public:
  explicit ModelBackendAdapter(Model model) : model_(std::move(model)) {}

  [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
    return model_.metadata();
  }

  [[nodiscard]] StepOutput Run(const StepInput& input) const override {
    NamedTensors outputs = model_.Run({
        {"observation", input.observation},
        {"action", input.action},
        {"state", input.state},
    });
    return {
        .next_state = std::move(outputs.at("next_state")),
        .observation_prediction =
            std::move(outputs.at("observation_prediction")),
        .reward = std::move(outputs.at("reward")),
        .continuation = std::move(outputs.at("continuation")),
    };
  }

 private:
  Model model_;
};

}  // namespace

WorldModel::WorldModel(BackendPtr backend) : backend_(std::move(backend)) {
  if (backend_ == nullptr) {
    throw Error(ErrorCode::invalid_argument, "Backend cannot be null");
  }
  ValidateContract(backend_->metadata());
}

WorldModel WorldModel::Load(
    const std::filesystem::path& model_path,
    const RuntimeOptions& options) {
  return WorldModel(
      std::make_shared<ModelBackendAdapter>(Model::Load(model_path, options)));
}

const ModelMetadata& WorldModel::metadata() const noexcept {
  return backend_->metadata();
}

StepOutput WorldModel::Step(
    const Tensor& observation,
    const Tensor& action,
    const Tensor& state) const {
  const auto& model_metadata = metadata();
  ValidateTensor(observation, model_metadata.Input("observation"));
  ValidateTensor(action, model_metadata.Input("action"));
  ValidateTensor(state, model_metadata.Input("state"));
  ValidateBatch(observation, action, state);

  StepInput input{
      .observation = observation,
      .action = action,
      .state = state,
  };
  StepOutput output = backend_->Run(input);
  ValidateOutput(output, input, model_metadata);
  return output;
}

Rollout::Rollout(WorldModel model) : model_(std::move(model)) {}

void Rollout::Reset() {
  std::scoped_lock lock(mutex_);
  state_.reset();
}

void Rollout::Reset(Tensor state) {
  ValidateTensor(state, model_.metadata().Input("state"));
  std::scoped_lock lock(mutex_);
  state_ = std::move(state);
}

void Rollout::ResetZeros(std::int64_t batch_size) {
  Tensor state = MakeZeroState(batch_size);
  std::scoped_lock lock(mutex_);
  state_ = std::move(state);
}

bool Rollout::has_state() const {
  std::scoped_lock lock(mutex_);
  return state_.has_value();
}

std::optional<Tensor> Rollout::state() const {
  std::scoped_lock lock(mutex_);
  return state_;
}

StepOutput Rollout::Step(const Tensor& observation, const Tensor& action) {
  std::scoped_lock lock(mutex_);
  if (observation.shape().empty()) {
    throw Error(
        ErrorCode::invalid_argument,
        "Observation must include a batch dimension");
  }
  if (!state_.has_value()) {
    state_ = MakeZeroState(observation.shape()[0]);
  }
  StepOutput output = model_.Step(observation, action, *state_);
  state_ = output.next_state;
  return output;
}

Tensor Rollout::MakeZeroState(std::int64_t batch_size) const {
  if (batch_size <= 0) {
    throw Error(ErrorCode::state, "Batch size must be positive");
  }
  const auto& state_spec = model_.metadata().Input("state");
  std::vector<std::int64_t> shape = state_spec.shape;
  shape[0] = batch_size;
  if (std::ranges::any_of(
          shape.begin() + 1,
          shape.end(),
          [](std::int64_t dimension) { return dimension < 0; })) {
    throw Error(
        ErrorCode::state,
        "Cannot create zero state when a non-batch state dimension is dynamic");
  }
  return Tensor::Zeros(state_spec.data_type, std::move(shape));
}

}  // namespace onnx_world_model
