#pragma once

/**
 * @agent-file
 * @agent-purpose: Declares runtime session and device-output configuration, execution-provider helpers and registration, and Model, the generic named-tensor ONNX graph session used by every higher-level API.
 * @agent-public-api: GraphOptimizationLevel, RuntimeOptions, NormalizeExecutionProviderName, AvailableExecutionProviders, RegisterExecutionProviderLibrary, NamedTensors, ModelBackend, ModelBackendPtr, Model
 * @agent-invariants: Model rejects a null backend; Model::Run validates every input and output tensor against metadata, so unknown or missing names throw instead of reaching ONNX Runtime. Device outputs are opt-in and require the corresponding provider library to be registered with the process-wide ORT environment before materialization. Provider names are compared only after NormalizeExecutionProviderName folds case, separators, and the ExecutionProvider suffix. The cancellable ModelBackend::Run overload is virtual with a default that checks the token before and after the existing one-argument Run, so a backend written before cancellation existed keeps compiling and still honors boundary cancellation; only a backend that can interrupt work in flight, such as the ONNX Runtime one, overrides it.
 * @agent-side-effects: none in this header; the declared load, provider-query, and provider-registration functions load shared libraries or model files.
 */

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "onnx_world_model/backend.hpp"
#include "onnx_world_model/cancellation.hpp"

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
  bool device_outputs{false};
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
void RegisterExecutionProviderLibrary(
    std::string_view registration_name,
    const std::filesystem::path& provider_library_path,
    const std::filesystem::path& ort_library_path = {});

using NamedTensors = std::unordered_map<std::string, Tensor>;

class ModelBackend {
 public:
  virtual ~ModelBackend() = default;

  [[nodiscard]] virtual const ModelMetadata& metadata() const noexcept = 0;
  [[nodiscard]] virtual NamedTensors Run(const NamedTensors& inputs) const = 0;
  //: Runs the graph under `cancellation`. The default implementation checks
  //: the token before and after the one-argument Run, so every existing
  //: backend gains boundary cancellation without being modified. A backend
  //: that can interrupt work already in flight overrides this instead, either
  //: by registering its own interruption or by parking on
  //: CancellationToken::WaitForCancellation.
  [[nodiscard]] virtual NamedTensors Run(
      const NamedTensors& inputs,
      const CancellationToken& cancellation) const;
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
  //: Validates inputs, runs the backend under `cancellation`, and validates
  //: outputs. The token is checked before the backend call and after the
  //: outputs are validated, so a cancelled run reports ErrorCode::cancelled
  //: or ErrorCode::deadline_exceeded rather than a runtime failure.
  [[nodiscard]] NamedTensors Run(
      const NamedTensors& inputs,
      const CancellationToken& cancellation) const;

 private:
  ModelBackendPtr backend_;
};

}  // namespace onnx_world_model
