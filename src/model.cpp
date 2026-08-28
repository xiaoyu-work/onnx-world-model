/**
 * @agent-file
 * @agent-purpose: Implements execution-provider normalization and library registration, tensor-versus-signature validation, ModelMetadata lookups, and the validating Model facade.
 * @agent-public-api: NormalizeExecutionProviderName, AvailableExecutionProviders, RegisterExecutionProviderLibrary, ValidateTensor, ModelMetadata::Input, ModelMetadata::Output, ModelBackend::Run, Model::Model, Model::Load, Model::metadata, Model::Run
 * @agent-invariants: Model::Run rejects missing, unexpected, or mismatched tensors on both the input and the output side before and after the backend call; a negative spec dimension accepts any concrete extent; NormalizeExecutionProviderName strips non-alphanumeric characters and the ExecutionProvider suffix, then folds the directml, trtrtx, nvtensorrtx, and tensortrt aliases; a null backend throws ErrorCode::invalid_argument. The cancellable Model::Run overload checks its token before the backend call and after output validation, and ModelBackend's default cancellable overload does the same around the one-argument Run, so a backend that cannot interrupt itself still stops at those two boundaries.
 * @agent-side-effects: Model::Load and AvailableExecutionProviders load the ONNX Runtime shared library; RegisterExecutionProviderLibrary also loads an EP library into the process; Model::Load reads model files.
 */

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

void RegisterExecutionProviderLibrary(
    std::string_view registration_name,
    const std::filesystem::path& provider_library_path,
    const std::filesystem::path& ort_library_path) {
  detail::RegisterOrtExecutionProviderLibrary(
      registration_name,
      provider_library_path,
      ort_library_path);
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

// Every backend written before cancellation existed only implements the
// one-argument Run, so the cancellable overload is defined here rather than
// made pure: the token is honored at the two boundaries a non-interruptible
// backend can offer, and a backend that can interrupt itself overrides it.
NamedTensors ModelBackend::Run(
    const NamedTensors& inputs,
    const CancellationToken& cancellation) const {
  cancellation.ThrowIfCancellationRequested();
  NamedTensors outputs = Run(inputs);
  cancellation.ThrowIfCancellationRequested();
  return outputs;
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

NamedTensors Model::Run(
    const NamedTensors& inputs,
    const CancellationToken& cancellation) const {
  cancellation.ThrowIfCancellationRequested();
  ValidateNames(inputs, metadata().inputs, "input");
  NamedTensors outputs = backend_->Run(inputs, cancellation);
  ValidateNames(outputs, metadata().outputs, "output");
  cancellation.ThrowIfCancellationRequested();
  return outputs;
}

}  // namespace onnx_world_model
