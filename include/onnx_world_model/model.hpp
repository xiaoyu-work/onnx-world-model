#pragma once

/**
 * @agent-file
 * @agent-purpose: Declares runtime session and device-output configuration, execution-provider helpers and registration, per-run options carrying cancellation and an optional ONNX Runtime profile-file prefix, and Model, the generic named-tensor ONNX graph session used by every higher-level API.
 * @agent-public-api: GraphOptimizationLevel, RuntimeOptions, NormalizeExecutionProviderName, AvailableExecutionProviders, RegisterExecutionProviderLibrary, NamedTensors, ModelRunOptions, ModelBackend, ModelBackendPtr, Model
 * @agent-invariants: Model rejects a null backend; Model::Run validates every input and output tensor against metadata, so unknown or missing names throw instead of reaching ONNX Runtime. Device outputs are opt-in and require the corresponding provider library to be registered with the process-wide ORT environment before materialization. Provider names are compared only after NormalizeExecutionProviderName folds case, separators, and the ExecutionProvider suffix. ModelRunOptions is the one per-call surface: it carries the cancellation token and an optional profile-file prefix, and the three Run overloads collapse onto it, so a caller that supplies neither behaves exactly as before. The ModelRunOptions ModelBackend::Run overload is virtual with a default that drops the prefix and forwards to the cancellable overload, whose own default checks the token before and after the one-argument Run, so a backend written before either existed keeps compiling; only a backend that can interrupt work in flight or profile it, such as the ONNX Runtime one, overrides them. Profiling is per run and never per session: a non-empty prefix enables profiling on that call's own Ort::RunOptions, so no session is rebuilt and a concurrent unprofiled call on the same session is unaffected. ONNX Runtime -- not this runtime -- names the file, appending its own local timestamp and the .json suffix to the prefix, so a caller discovers the file by prefix rather than by predicting a name.
 * @agent-side-effects: none in this header; the declared load, provider-query, and provider-registration functions load shared libraries or model files. A Run given a profile-file prefix makes ONNX Runtime write a trace file next to that prefix.
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

//: Everything one Model::Run call may be given beyond its inputs. It exists so
//: a per-call concern is added here rather than as another Run overload, and
//: a default-constructed value is exactly the historical behavior: no
//: cancellation and no profiling.
struct ModelRunOptions {
  //: Checked at this call's boundaries, and honored inside ONNX Runtime by a
  //: backend that can interrupt itself.
  CancellationToken cancellation;
  //: Where ONNX Runtime should write this call's node-level trace, or empty
  //: -- the default -- to run without profiling. It is a prefix rather than a
  //: file name: ONNX Runtime appends its own local timestamp and `.json`, so
  //: the caller finds the file by scanning for that prefix instead of
  //: predicting the name. Profiling is enabled on this call's own run options,
  //: so it never rebuilds the session and never affects a concurrent call.
  std::filesystem::path profile_file_prefix;
};

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
  //: Runs the graph under everything in `options`. The default implementation
  //: forwards to the cancellable overload and drops the profile-file prefix,
  //: because a backend that cannot profile must not pretend it did; the
  //: caller learns that no file appeared. A backend that can profile a single
  //: call overrides this.
  [[nodiscard]] virtual NamedTensors Run(
      const NamedTensors& inputs,
      const ModelRunOptions& options) const;
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
  //: The one implementation the other two delegate to. Validation, the
  //: cancellation boundaries, and the returned outputs are identical whatever
  //: overload was called; a non-empty ModelRunOptions::profile_file_prefix
  //: additionally asks the backend to write this call's ONNX Runtime trace,
  //: which never changes the outputs and never fails the call by itself.
  [[nodiscard]] NamedTensors Run(
      const NamedTensors& inputs,
      const ModelRunOptions& options) const;

 private:
  ModelBackendPtr backend_;
};

}  // namespace onnx_world_model
