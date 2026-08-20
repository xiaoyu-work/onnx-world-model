#pragma once

/**
 * @agent-file
 * @agent-purpose: Declares runtime session configuration (RuntimeOptions, provider naming helpers) and Model, the generic named-tensor ONNX graph session used by every higher-level API.
 * @agent-public-api: GraphOptimizationLevel, RuntimeOptions, NormalizeExecutionProviderName, AvailableExecutionProviders, NamedTensors, ModelBackend, ModelBackendPtr, Model
 * @agent-invariants: Model rejects a null backend; Model::Run validates every input and output tensor against metadata, so unknown or missing names throw instead of reaching ONNX Runtime; provider names are compared only after NormalizeExecutionProviderName folds case, separators, and the ExecutionProvider suffix.
 * @agent-side-effects: none in this header; the declared Model::Load and AvailableExecutionProviders load the ONNX Runtime shared library and read model files.
 */

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "onnx_world_model/backend.hpp"

namespace onnx_world_model {

enum class GraphOptimizationLevel {
  disabled,
  basic,
  extended,
  all,
};

struct RuntimeOptions {
  std::filesystem::path ort_library_path;
  int intra_op_threads{0};
  int inter_op_threads{0};
  int log_severity{3};
  GraphOptimizationLevel graph_optimization{GraphOptimizationLevel::all};
  std::vector<std::string> providers;
  std::unordered_map<
      std::string,
      std::unordered_map<std::string, std::string>>
      provider_options;
};

[[nodiscard]] std::string NormalizeExecutionProviderName(
    std::string_view name);
[[nodiscard]] std::vector<std::string> AvailableExecutionProviders(
    const std::filesystem::path& ort_library_path = {});

using NamedTensors = std::unordered_map<std::string, Tensor>;

class ModelBackend {
 public:
  virtual ~ModelBackend() = default;

  [[nodiscard]] virtual const ModelMetadata& metadata() const noexcept = 0;
  [[nodiscard]] virtual NamedTensors Run(const NamedTensors& inputs) const = 0;
};

using ModelBackendPtr = std::shared_ptr<ModelBackend>;

class Model {
 public:
  explicit Model(ModelBackendPtr backend);

  static Model Load(
      const std::filesystem::path& model_path,
      const RuntimeOptions& options = {});

  [[nodiscard]] const ModelMetadata& metadata() const noexcept;
  [[nodiscard]] NamedTensors Run(const NamedTensors& inputs) const;

 private:
  ModelBackendPtr backend_;
};

}  // namespace onnx_world_model
