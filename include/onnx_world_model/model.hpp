#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

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
};

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
