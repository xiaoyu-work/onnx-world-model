#include "onnx_world_model/model.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

#include "onnx_world_model/error.hpp"
#include "ort_backend.hpp"

namespace onnx_world_model {

std::string NormalizeExecutionProviderName(std::string_view name) {
  std::string normalized;
  normalized.reserve(name.size());
  for (const unsigned char character : name) {
    if (std::isalnum(character) != 0) {
      normalized.push_back(
          static_cast<char>(std::tolower(character)));
    }
  }
  constexpr std::string_view suffix = "executionprovider";
  if (normalized.ends_with(suffix)) {
    normalized.resize(normalized.size() - suffix.size());
  }
  if (normalized == "directml") {
    normalized = "dml";
  } else if (normalized == "trtrtx" ||
             normalized == "nvtensorrtx") {
    normalized = "nvtensorrtrtx";
  } else if (normalized == "tensortrt") {
    normalized = "tensorrt";
  }
  if (normalized.empty()) {
    throw Error(
        ErrorCode::invalid_argument,
        "Execution provider name cannot be empty");
  }
  return normalized;
}

std::vector<std::string> AvailableExecutionProviders(
    const std::filesystem::path& ort_library_path) {
  return detail::GetAvailableOrtProviders(ort_library_path);
}

void ValidateTensor(const Tensor& tensor, const TensorSpec& spec) {
  if (tensor.data_type() != spec.data_type) {
    throw Error(
        ErrorCode::invalid_argument,
        "Tensor '" + spec.name + "' has data type " +
            std::string(ToString(tensor.data_type())) + ", expected " +
            std::string(ToString(spec.data_type)));
  }
  if (tensor.shape().size() != spec.shape.size()) {
    throw Error(
        ErrorCode::invalid_argument,
        "Tensor '" + spec.name + "' has rank " +
            std::to_string(tensor.shape().size()) + ", expected " +
            std::to_string(spec.shape.size()));
  }
  for (std::size_t axis = 0; axis < spec.shape.size(); ++axis) {
    if (spec.shape[axis] >= 0 && tensor.shape()[axis] != spec.shape[axis]) {
      throw Error(
          ErrorCode::invalid_argument,
          "Tensor '" + spec.name + "' has invalid dimension " +
              std::to_string(axis) + ": " +
              std::to_string(tensor.shape()[axis]) + ", expected " +
              std::to_string(spec.shape[axis]));
    }
  }
}

namespace {

void ValidateNames(
    const NamedTensors& tensors,
    const std::vector<TensorSpec>& specs,
    std::string_view kind) {
  for (const auto& spec : specs) {
    const auto found = tensors.find(spec.name);
    if (found == tensors.end()) {
      throw Error(
          ErrorCode::invalid_argument,
          "Model is missing " + std::string(kind) + " tensor '" + spec.name + "'");
    }
    ValidateTensor(found->second, spec);
  }
  for (const auto& [name, tensor] : tensors) {
    (void)tensor;
    if (std::ranges::find(specs, name, &TensorSpec::name) == specs.end()) {
      throw Error(
          ErrorCode::invalid_argument,
          "Model received unexpected " + std::string(kind) + " tensor '" + name + "'");
    }
  }
}

}  // namespace

const TensorSpec& ModelMetadata::Input(std::string_view name) const {
  const auto found = std::ranges::find(inputs, name, &TensorSpec::name);
  if (found == inputs.end()) {
    throw Error(
        ErrorCode::model_contract,
        "Model is missing required input '" + std::string(name) + "'");
  }
  return *found;
}

const TensorSpec& ModelMetadata::Output(std::string_view name) const {
  const auto found = std::ranges::find(outputs, name, &TensorSpec::name);
  if (found == outputs.end()) {
    throw Error(
        ErrorCode::model_contract,
        "Model is missing required output '" + std::string(name) + "'");
  }
  return *found;
}

Model::Model(ModelBackendPtr backend) : backend_(std::move(backend)) {
  if (backend_ == nullptr) {
    throw Error(ErrorCode::invalid_argument, "Backend cannot be null");
  }
}

Model Model::Load(
    const std::filesystem::path& model_path,
    const RuntimeOptions& options) {
  return Model(detail::CreateOrtBackend(model_path, options));
}

const ModelMetadata& Model::metadata() const noexcept {
  return backend_->metadata();
}

NamedTensors Model::Run(const NamedTensors& inputs) const {
  ValidateNames(inputs, metadata().inputs, "input");
  NamedTensors outputs = backend_->Run(inputs);
  ValidateNames(outputs, metadata().outputs, "output");
  return outputs;
}

}  // namespace onnx_world_model
