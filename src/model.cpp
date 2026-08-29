/**
 * @agent-file
 * @agent-purpose: Implements execution-provider normalization and library registration, tensor-versus-signature validation, ModelMetadata lookups, and the validating Model facade with its three Run overloads.
 * @agent-public-api: NormalizeExecutionProviderName, AvailableExecutionProviders, RegisterExecutionProviderLibrary, ValidateTensor, ModelMetadata::Input, ModelMetadata::Output, ModelBackend::Run, Model::Model, Model::Load, Model::metadata, Model::Run
 * @agent-invariants: Model::Run rejects missing, unexpected, or mismatched tensors on both the input and the output side before and after the backend call; a negative spec dimension accepts any concrete extent; NormalizeExecutionProviderName strips non-alphanumeric characters and the ExecutionProvider suffix, then folds the directml, trtrtx, nvtensorrtx, and tensortrt aliases; a null backend throws ErrorCode::invalid_argument. The ModelRunOptions overload is the one Model::Run implementation: the one-argument and cancellable overloads build the options they imply and delegate to it, so validation and the two cancellation boundaries -- before the backend call and after output validation -- cannot drift between them. ModelBackend's default cancellable overload checks the token around the one-argument Run, and its default ModelRunOptions overload drops the profile-file prefix and forwards to that cancellable one, so a backend that cannot interrupt or profile itself still stops at those two boundaries and simply produces no trace file.
 * @agent-side-effects: Model::Load and AvailableExecutionProviders load the ONNX Runtime shared library; RegisterExecutionProviderLibrary also loads an EP library into the process; Model::Load reads model files. A Run whose options carry a profile-file prefix makes the ONNX Runtime backend write a trace file for that call.
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

// The same reasoning one layer up: a backend that cannot profile a single call
// keeps its existing behavior and simply drops the prefix. Silently doing
// nothing is correct here and only here, because the caller finds out by
// finding no trace file and records that as a profiling failure rather than
// as a successful trace it never got.
NamedTensors ModelBackend::Run(
    const NamedTensors& inputs,
    const ModelRunOptions& options) const {
  return Run(inputs, options.cancellation);
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
  return Run(inputs, ModelRunOptions{});
}

NamedTensors Model::Run(
    const NamedTensors& inputs,
    const CancellationToken& cancellation) const {
  return Run(inputs, ModelRunOptions{.cancellation = cancellation});
}

// The one implementation. The other two overloads build the options they
// imply and land here, so validation, the cancellation boundaries, and the
// outputs cannot drift between the three ways of calling a model.
NamedTensors Model::Run(
    const NamedTensors& inputs,
    const ModelRunOptions& options) const {
  options.cancellation.ThrowIfCancellationRequested();
  ValidateNames(inputs, metadata().inputs, "input");
  NamedTensors outputs = backend_->Run(inputs, options);
  ValidateNames(outputs, metadata().outputs, "output");
  options.cancellation.ThrowIfCancellationRequested();
  return outputs;
}

}  // namespace onnx_world_model
