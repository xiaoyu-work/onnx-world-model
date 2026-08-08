#pragma once

#include <memory>
#include <string>
#include <vector>

#include "onnx_world_model/tensor.hpp"

namespace onnx_world_model {

struct TensorSpec {
  std::string name;
  DataType data_type;
  std::vector<std::int64_t> shape;
};

struct ModelMetadata {
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;

  [[nodiscard]] const TensorSpec& Input(std::string_view name) const;
  [[nodiscard]] const TensorSpec& Output(std::string_view name) const;
};

void ValidateTensor(const Tensor& tensor, const TensorSpec& spec);

struct StepInput {
  Tensor observation;
  Tensor action;
  Tensor state;
};

struct StepOutput {
  Tensor next_state;
  Tensor observation_prediction;
  Tensor reward;
  Tensor continuation;
};

class Backend {
 public:
  virtual ~Backend() = default;

  [[nodiscard]] virtual const ModelMetadata& metadata() const noexcept = 0;
  [[nodiscard]] virtual StepOutput Run(const StepInput& input) const = 0;
};

using BackendPtr = std::shared_ptr<Backend>;

}  // namespace onnx_world_model
