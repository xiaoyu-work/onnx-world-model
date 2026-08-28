#pragma once

/**
 * @agent-file
 * @agent-purpose: Declares tensor signature metadata (TensorSpec, ModelMetadata), its validation helper, and the fixed latent-dynamics Backend contract with its step input and output structs.
 * @agent-public-api: TensorSpec, ModelMetadata, ValidateTensor, StepInput, StepOutput, Backend, BackendPtr
 * @agent-invariants: A negative TensorSpec shape entry marks a dynamic dimension that ValidateTensor accepts for any concrete extent; ModelMetadata::Input and Output throw ErrorCode::model_contract for unknown names. TensorSpec::device is runtime placement rather than part of the graph signature, so ValidateTensor and the manifest-versus-model signature check both ignore it and an engaged value never makes a tensor invalid; it is the last member, so a backend written before it existed keeps compiling and reports nullopt.
 * @agent-side-effects: none
 */

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "onnx_world_model/tensor.hpp"

namespace onnx_world_model {

struct TensorSpec {
  std::string name;
  DataType data_type;
  std::vector<std::int64_t> shape;
  //: Where the backend actually places this port, as reported by graph
  //: partitioning rather than declared by the manifest. `std::nullopt` means
  //: unknown: either the backend does not report placement or it was written
  //: before this member existed. It is deliberately not part of the graph
  //: signature, so neither ValidateTensor nor the manifest signature check
  //: looks at it; only the pipeline transfer plan does.
  std::optional<TensorDevice> device;
};

struct ModelMetadata {
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
  std::vector<std::string> execution_providers;

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
