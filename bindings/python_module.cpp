/**
 * @agent-file
 * @agent-purpose: Defines the pybind11 `_native` extension module that exposes the C++ runtime to Python and converts between NumPy arrays and onnx_world_model::Tensor.
 * @agent-public-api: _native module, WorldModelError, CancelledError, DeadlineExceededError, CancellationToken, CancellationSource, available_execution_providers, register_execution_provider_library, supported_pipeline_capabilities, Model, WorldModel, Pipeline, PipelineSession, PipelineSessionSnapshot, StageRun, Rollout
 * @agent-invariants: NumPy dtype names map one-to-one onto DataType; float16 and bfloat16 cross the boundary as raw 2-byte views. Every wrapper forwards the device_outputs policy unchanged. NumPy conversion explicitly materializes device buffers to CPU while the GIL is released. The GIL is released around every call that can block, which is every blocking ONNX Runtime or provider-library call and every session or run method that takes the session lock: run_stage, step_stage, begin_stage, outputs, state, release_stage, reset, snapshot, restore, fork, the named-checkpoint methods, and StageRun.step, finish, cancel, request_cancellation, done, and iteration. A C++ Error is translated by one custom translator that maps ErrorCode::cancelled to CancelledError and ErrorCode::deadline_exceeded to DeadlineExceededError, and every other code to their common base WorldModelError, so the code rather than the message decides the Python type and existing WorldModelError handlers still catch everything. A cancellation token crosses as its own argument rather than through the scalar options dictionary, because a token is not a bool, int, float, or str. PipelineSessionSnapshot, StageRun, and CancellationToken are exposed as opaque handles with no Python constructor, so they can only come from PipelineSession.snapshot(), PipelineSession.begin_stage(), and CancellationSource.token(); named checkpoints cross the boundary as plain strings and never expose a snapshot handle. A stage event crosses as a plain dictionary with a string kind, so the typed StageEvent lives in Python and the binding keeps no second value type in sync.
 * @agent-side-effects: Registers a Python module and exception type at import time; the wrapped constructors load the ONNX Runtime shared library and read model files, explicit provider registration loads an EP library, and output conversion may transfer tensors to CPU.
 */

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "onnx_world_model/onnx_world_model.hpp"

namespace py = pybind11;

namespace {

using onnx_world_model::DataType;
using onnx_world_model::AvailableExecutionProviders;
using onnx_world_model::CancellationSource;
using onnx_world_model::CancellationToken;
using onnx_world_model::Model;
using onnx_world_model::ModelMetadata;
using onnx_world_model::NamedTensors;
using onnx_world_model::Pipeline;
using onnx_world_model::PipelineRunOptions;
using onnx_world_model::PipelineSession;
using onnx_world_model::PipelineSessionSnapshot;
using onnx_world_model::RegisterExecutionProviderLibrary;
using onnx_world_model::Rollout;
using onnx_world_model::RuntimeOptions;
using onnx_world_model::StageEvent;
using onnx_world_model::StageEventKind;
using onnx_world_model::StageRun;
using onnx_world_model::StepOutput;
using onnx_world_model::Tensor;
using onnx_world_model::TensorSpec;
using onnx_world_model::WorldModel;

[[nodiscard]] DataType DataTypeFromNumpy(const py::dtype& dtype) {
  const std::string name = py::str(dtype).cast<std::string>();
  if (name == "float32") {
    return DataType::float32;
  }
  if (name == "float16") {
    return DataType::float16;
  }
  if (name == "bfloat16") {
    return DataType::bfloat16;
  }
  if (name == "float64") {
    return DataType::float64;
  }
  if (name == "int64") {
    return DataType::int64;
  }
  if (name == "int32") {
    return DataType::int32;
  }
  if (name == "int16") {
    return DataType::int16;
  }
  if (name == "int8") {
    return DataType::int8;
  }
  if (name == "uint64") {
    return DataType::uint64;
  }
  if (name == "uint32") {
    return DataType::uint32;
  }
  if (name == "uint16") {
    return DataType::uint16;
  }
  if (name == "uint8") {
    return DataType::uint8;
  }
  if (name == "bool") {
    return DataType::boolean;
  }
  throw py::type_error("Unsupported NumPy data type: " + name);
}

[[nodiscard]] py::dtype NumpyDataType(DataType data_type) {
  switch (data_type) {
    case DataType::float32:
      return py::dtype::of<float>();
    case DataType::float16:
      return py::dtype::from_args(py::str("float16"));
    case DataType::bfloat16: {
      const py::object type = py::module_::import("ml_dtypes").attr("bfloat16");
      return py::dtype::from_args(type);
    }
    case DataType::float64:
      return py::dtype::of<double>();
    case DataType::int64:
      return py::dtype::of<std::int64_t>();
    case DataType::int32:
      return py::dtype::of<std::int32_t>();
    case DataType::int16:
      return py::dtype::of<std::int16_t>();
    case DataType::int8:
      return py::dtype::of<std::int8_t>();
    case DataType::uint64:
      return py::dtype::of<std::uint64_t>();
    case DataType::uint32:
      return py::dtype::of<std::uint32_t>();
    case DataType::uint16:
      return py::dtype::of<std::uint16_t>();
    case DataType::uint8:
      return py::dtype::of<std::uint8_t>();
    case DataType::boolean:
      return py::dtype::of<bool>();
  }
  throw py::type_error("Unsupported world-model tensor data type");
}

[[nodiscard]] Tensor TensorFromNumpy(const py::array& array) {
  py::array contiguous = py::array::ensure(array, py::array::c_style);
  if (!contiguous) {
    throw py::type_error("Tensor must be convertible to a C-contiguous NumPy array");
  }
  std::vector<std::int64_t> shape;
  shape.reserve(static_cast<std::size_t>(contiguous.ndim()));
  for (py::ssize_t axis = 0; axis < contiguous.ndim(); ++axis) {
    shape.push_back(static_cast<std::int64_t>(contiguous.shape(axis)));
  }
  const auto bytes = std::span(
      reinterpret_cast<const std::byte*>(contiguous.data()),
      static_cast<std::size_t>(contiguous.nbytes()));
  return Tensor::FromBytes(
      DataTypeFromNumpy(contiguous.dtype()),
      std::move(shape),
      bytes);
}

[[nodiscard]] py::array TensorToNumpy(const Tensor& tensor) {
  Tensor cpu_tensor;
  {
    py::gil_scoped_release release;
    cpu_tensor = tensor.CopyToCpu();
  }
  std::vector<py::ssize_t> shape;
  shape.reserve(cpu_tensor.shape().size());
  for (const std::int64_t dimension : cpu_tensor.shape()) {
    shape.push_back(static_cast<py::ssize_t>(dimension));
  }
  py::array array(NumpyDataType(cpu_tensor.data_type()), shape);
  std::memcpy(
      array.mutable_data(),
      cpu_tensor.bytes().data(),
      cpu_tensor.size_bytes());
  return array;
}

[[nodiscard]] py::dict TensorSpecToDictionary(const TensorSpec& spec) {
  py::dict result;
  result["name"] = spec.name;
  result["dtype"] = std::string(onnx_world_model::ToString(spec.data_type));
  result["shape"] = spec.shape;
  return result;
}

[[nodiscard]] py::dict MetadataToDictionary(const ModelMetadata& metadata) {
  py::list inputs;
  for (const auto& spec : metadata.inputs) {
    inputs.append(TensorSpecToDictionary(spec));
  }
  py::list outputs;
  for (const auto& spec : metadata.outputs) {
    outputs.append(TensorSpecToDictionary(spec));
  }
  py::dict result;
  result["inputs"] = std::move(inputs);
  result["outputs"] = std::move(outputs);
  result["execution_providers"] = metadata.execution_providers;
  return result;
}

[[nodiscard]] NamedTensors NamedTensorsFromDictionary(const py::dict& values) {
  NamedTensors result;
  result.reserve(values.size());
  for (const auto& item : values) {
    if (!py::isinstance<py::str>(item.first)) {
      throw py::type_error("Model input names must be strings");
    }
    if (!py::isinstance<py::array>(item.second)) {
      throw py::type_error(
          "Model input '" + py::str(item.first).cast<std::string>() +
          "' must be a NumPy array");
    }
    result.emplace(
        py::str(item.first).cast<std::string>(),
        TensorFromNumpy(py::reinterpret_borrow<py::array>(item.second)));
  }
  return result;
}

[[nodiscard]] py::dict NamedTensorsToDictionary(const NamedTensors& values) {
  py::dict result;
  for (const auto& [name, tensor] : values) {
    result[py::str(name)] = TensorToNumpy(tensor);
  }
  return result;
}

[[nodiscard]] const char* StageEventKindName(StageEventKind kind) {
  switch (kind) {
    case StageEventKind::token:
      return "token";
    case StageEventKind::iteration:
      return "iteration";
    case StageEventKind::transition:
      return "transition";
    case StageEventKind::completed:
      return "completed";
  }
  throw py::value_error("Unsupported pipeline stage event kind");
}

// Events cross as plain dictionaries and the typed StageEvent is assembled in
// Python, which keeps the binding free of a second value type to keep in sync
// and materializes every device tensor exactly where the boundary requires
// an independent NumPy array.
[[nodiscard]] py::dict StageEventToDictionary(const StageEvent& event) {
  py::dict result;
  result["kind"] = StageEventKindName(event.kind);
  result["stage"] = event.stage;
  result["iteration"] = event.iteration;
  result["token_ids"] = event.token_ids.has_value()
                            ? py::object(TensorToNumpy(*event.token_ids))
                            : py::object(py::none());
  result["outputs"] = NamedTensorsToDictionary(event.outputs);
  result["finished"] = event.finished;
  return result;
}

[[nodiscard]] PipelineRunOptions PipelineOptionsFromDictionary(
    const py::dict& values) {
  PipelineRunOptions result;
  for (const auto& item : values) {
    if (!py::isinstance<py::str>(item.first)) {
      throw py::type_error("Pipeline option names must be strings");
    }
    const std::string name =
        py::str(item.first).cast<std::string>();
    if (py::isinstance<py::bool_>(item.second)) {
      result.integers.emplace(
          name, py::cast<bool>(item.second) ? 1 : 0);
    } else if (py::isinstance<py::int_>(item.second)) {
      result.integers.emplace(
          name, py::cast<std::int64_t>(item.second));
    } else if (py::isinstance<py::float_>(item.second)) {
      result.numbers.emplace(
          name, py::cast<double>(item.second));
    } else if (py::isinstance<py::str>(item.second)) {
      result.strings.emplace(
          name, py::str(item.second).cast<std::string>());
    } else {
      throw py::type_error(
          "Pipeline option '" + name +
          "' must be bool, int, float, or str");
    }
  }
  return result;
}

[[nodiscard]] onnx_world_model::GraphOptimizationLevel ParseGraphOptimization(
    const std::string& level) {
  using onnx_world_model::GraphOptimizationLevel;
  static const std::unordered_map<std::string, GraphOptimizationLevel> levels{
      {"disabled", GraphOptimizationLevel::disabled},
      {"basic", GraphOptimizationLevel::basic},
      {"extended", GraphOptimizationLevel::extended},
      {"all", GraphOptimizationLevel::all},
  };
  const auto found = levels.find(level);
  if (found == levels.end()) {
    throw py::value_error(
        "graph_optimization must be one of 'disabled', 'basic', 'extended', "
        "or 'all', not '" +
        level + "'");
  }
  return found->second;
}

[[nodiscard]] RuntimeOptions RuntimeOptionsFromPython(
    const std::string& ort_library_path,
    int intra_op_threads,
    int inter_op_threads,
    int log_severity,
    const std::string& graph_optimization,
    bool device_outputs,
    const std::vector<std::string>& providers,
    const py::dict& provider_options) {
  RuntimeOptions result{
      .ort_library_path = ort_library_path,
      .intra_op_threads = intra_op_threads,
      .inter_op_threads = inter_op_threads,
      .log_severity = log_severity,
      .graph_optimization = ParseGraphOptimization(graph_optimization),
      .device_outputs = device_outputs,
      .providers = providers,
  };
  for (const auto& provider : provider_options) {
    if (!py::isinstance<py::str>(provider.first) ||
        !py::isinstance<py::dict>(provider.second)) {
      throw py::type_error(
          "provider_options must map provider names to option dictionaries");
    }
    const std::string provider_name =
        py::str(provider.first).cast<std::string>();
    std::unordered_map<std::string, std::string> options;
    for (const auto& option :
         py::reinterpret_borrow<py::dict>(provider.second)) {
      if (!py::isinstance<py::str>(option.first) ||
          !(py::isinstance<py::str>(option.second) ||
            py::isinstance<py::bool_>(option.second) ||
            py::isinstance<py::int_>(option.second) ||
            py::isinstance<py::float_>(option.second))) {
        throw py::type_error(
            "Provider option names must be strings and values must be "
            "str, bool, int, or float");
      }
      std::string value;
      if (py::isinstance<py::bool_>(option.second)) {
        value = py::cast<bool>(option.second) ? "1" : "0";
      } else {
        value = py::str(option.second).cast<std::string>();
      }
      options.emplace(
          py::str(option.first).cast<std::string>(),
          std::move(value));
    }
    result.provider_options.emplace(
        provider_name, std::move(options));
  }
  return result;
}

[[nodiscard]] py::tuple StepOutputToTuple(const StepOutput& output) {
  return py::make_tuple(
      TensorToNumpy(output.next_state),
      TensorToNumpy(output.observation_prediction),
      TensorToNumpy(output.reward),
      TensorToNumpy(output.continuation));
}

// The three exception types are owned by the module dictionary, so a
// non-owning handle is enough here and avoids a py::object static that would
// outlive the interpreter.
py::handle& WorldModelErrorType() {
  static py::handle type;
  return type;
}

py::handle& CancelledErrorType() {
  static py::handle type;
  return type;
}

py::handle& DeadlineExceededErrorType() {
  static py::handle type;
  return type;
}

[[nodiscard]] py::handle ErrorTypeFor(onnx_world_model::ErrorCode code) {
  switch (code) {
    case onnx_world_model::ErrorCode::cancelled:
      return CancelledErrorType();
    case onnx_world_model::ErrorCode::deadline_exceeded:
      return DeadlineExceededErrorType();
    default:
      return WorldModelErrorType();
  }
}

// Replaces pybind11's generated single-type translator so the ErrorCode, not
// the message text, decides which Python exception a failure becomes. Any
// exception that is not an Error escapes this function, which is how pybind11
// is told to keep looking for a translator that handles it.
void TranslateWorldModelError(std::exception_ptr pointer) {
  if (!pointer) {
    return;
  }
  try {
    std::rethrow_exception(pointer);
  } catch (const onnx_world_model::Error& error) {
    py::set_error(ErrorTypeFor(error.code()), error.what());
  }
}

[[nodiscard]] const char* CancellationReasonName(
    onnx_world_model::CancellationReason reason) {
  switch (reason) {
    case onnx_world_model::CancellationReason::none:
      return "none";
    case onnx_world_model::CancellationReason::cancelled:
      return "cancelled";
    case onnx_world_model::CancellationReason::deadline_exceeded:
      return "deadline_exceeded";
  }
  throw py::value_error("Unsupported cancellation reason");
}

[[nodiscard]] CancellationSource MakeCancellationSource(
    std::optional<double> timeout_seconds) {
  if (!timeout_seconds.has_value()) {
    return CancellationSource();
  }
  const double seconds = *timeout_seconds;
  if (!std::isfinite(seconds)) {
    throw py::value_error("timeout must be a finite number of seconds");
  }
  if (seconds < 0.0) {
    throw py::value_error("timeout must not be negative");
  }
  // Milliseconds are the coarsest unit the deadline needs, but Python hands
  // out doubles with far more range than an int64 millisecond count, so both
  // the scaling and the cast have to saturate: `seconds * 1000.0` alone
  // overflows the representable range for a value such as 1e300, and casting
  // an out-of-range double to int64 is undefined. This limit is 2^63 exactly,
  // one past the largest representable count, so any double strictly below it
  // casts safely. Everything at or above it becomes milliseconds::max, which
  // CancellationSource::WithTimeout saturates to the clock's furthest instant
  // rather than wrapping into a deadline that has already passed.
  constexpr double kMillisecondLimit =
      static_cast<double>(std::numeric_limits<std::int64_t>::max());
  constexpr double kSecondLimit = kMillisecondLimit / 1000.0;
  if (seconds >= kSecondLimit) {
    return CancellationSource::WithTimeout(std::chrono::milliseconds::max());
  }
  const double milliseconds = seconds * 1000.0;
  if (milliseconds >= kMillisecondLimit) {
    return CancellationSource::WithTimeout(std::chrono::milliseconds::max());
  }
  return CancellationSource::WithTimeout(
      std::chrono::milliseconds(static_cast<std::int64_t>(milliseconds)));
}

// Scalar options and the cancellation token cross separately: a token is not
// a bool, int, float, or str, so folding it into the options dictionary would
// mean inventing a magic key the C++ contract does not have.
[[nodiscard]] PipelineRunOptions PipelineOptionsFromPython(
    const py::dict& values,
    const std::optional<CancellationToken>& cancellation) {
  PipelineRunOptions options = PipelineOptionsFromDictionary(values);
  if (cancellation.has_value()) {
    options.cancellation = *cancellation;
  }
  return options;
}

}  // namespace

PYBIND11_MODULE(_native, module) {
  module.doc() = "Native C++ runtime for Mobius world models";
  // WorldModelError stays the base of every failure this runtime raises, and
  // the two cancellation outcomes derive from it, so existing `except
  // WorldModelError` handlers keep catching everything they used to.
  const py::object world_model_error =
      py::exception<void>(module, "WorldModelError");
  WorldModelErrorType() = world_model_error;
  CancelledErrorType() = py::exception<void>(
      module, "CancelledError", world_model_error.ptr());
  DeadlineExceededErrorType() = py::exception<void>(
      module, "DeadlineExceededError", world_model_error.ptr());
  py::register_exception_translator(&TranslateWorldModelError);

  // Opaque: only a CancellationSource hands out a token, so Python cannot
  // fabricate one that claims to be cancellable.
  py::class_<CancellationToken>(module, "CancellationToken")
      .def_property_readonly(
          "cancellable",
          [](const CancellationToken& token) { return token.cancellable(); })
      .def_property_readonly(
          "cancelled",
          [](const CancellationToken& token) { return token.cancelled(); })
      .def_property_readonly(
          "reason",
          [](const CancellationToken& token) {
            return CancellationReasonName(token.reason());
          });

  py::class_<CancellationSource>(module, "CancellationSource")
      .def(
          py::init([](std::optional<double> timeout_seconds) {
            return MakeCancellationSource(timeout_seconds);
          }),
          py::arg("timeout_seconds") = py::none())
      .def(
          "token",
          [](const CancellationSource& source) { return source.token(); })
      .def(
          "cancel",
          [](CancellationSource& source) {
            // Never blocks on a session lock, so it must stay callable while
            // another thread runs a step with the GIL released.
            py::gil_scoped_release release;
            source.Cancel();
          })
      .def_property_readonly(
          "cancelled",
          [](const CancellationSource& source) { return source.cancelled(); })
      .def_property_readonly(
          "reason",
          [](const CancellationSource& source) {
            return CancellationReasonName(source.reason());
          });

  module.def(
      "available_execution_providers",
      [](const std::string& ort_library_path) {
        py::gil_scoped_release release;
        return AvailableExecutionProviders(ort_library_path);
      },
      py::arg("ort_library_path"));
  module.def(
      "register_execution_provider_library",
      [](const std::string& registration_name,
         const std::string& provider_library_path,
         const std::string& ort_library_path) {
        py::gil_scoped_release release;
        RegisterExecutionProviderLibrary(
            registration_name,
            provider_library_path,
            ort_library_path);
      },
      py::arg("registration_name"),
      py::arg("provider_library_path"),
      py::arg("ort_library_path"));
  module.def(
      "supported_pipeline_capabilities",
      [] { return onnx_world_model::PipelineManifest::SupportedCapabilities(); });

  py::class_<Model>(module, "Model")
      .def(
          py::init([](
                       const std::string& model_path,
                       const std::string& ort_library_path,
                       int intra_op_threads,
                       int inter_op_threads,
                       int log_severity,
                       const std::string& graph_optimization,
                       bool device_outputs,
                       const std::vector<std::string>& providers,
                       const py::dict& provider_options) {
            RuntimeOptions options = RuntimeOptionsFromPython(
                ort_library_path,
                intra_op_threads,
                inter_op_threads,
                log_severity,
                graph_optimization,
                device_outputs,
                providers,
                provider_options);
            py::gil_scoped_release release;
            return Model::Load(model_path, options);
          }),
          py::arg("model_path"),
          py::arg("ort_library_path"),
          py::arg("intra_op_threads") = 0,
          py::arg("inter_op_threads") = 0,
          py::arg("log_severity") = 3,
          py::arg("graph_optimization") = "all",
          py::arg("device_outputs") = false,
          py::arg("providers") = std::vector<std::string>{},
          py::arg("provider_options") = py::dict())
      .def_property_readonly(
          "metadata",
          [](const Model& model) {
            return MetadataToDictionary(model.metadata());
          })
      .def(
          "run",
          [](const Model& model,
             const py::dict& inputs,
             const std::optional<CancellationToken>& cancellation) {
            NamedTensors input_tensors = NamedTensorsFromDictionary(inputs);
            const CancellationToken token =
                cancellation.value_or(CancellationToken{});
            NamedTensors outputs;
            {
              py::gil_scoped_release release;
              outputs = model.Run(input_tensors, token);
            }
            return NamedTensorsToDictionary(outputs);
          },
          py::arg("inputs"),
          py::arg("cancellation") = py::none());

  py::class_<WorldModel>(module, "WorldModel")
      .def(
          py::init([](
                       const std::string& model_path,
                       const std::string& ort_library_path,
                       int intra_op_threads,
                       int inter_op_threads,
                       int log_severity,
                       const std::string& graph_optimization,
                       bool device_outputs,
                       const std::vector<std::string>& providers,
                       const py::dict& provider_options) {
            RuntimeOptions options = RuntimeOptionsFromPython(
                ort_library_path,
                intra_op_threads,
                inter_op_threads,
                log_severity,
                graph_optimization,
                device_outputs,
                providers,
                provider_options);
            py::gil_scoped_release release;
            return WorldModel::Load(model_path, options);
          }),
          py::arg("model_path"),
          py::arg("ort_library_path"),
          py::arg("intra_op_threads") = 0,
          py::arg("inter_op_threads") = 0,
          py::arg("log_severity") = 3,
          py::arg("graph_optimization") = "all",
          py::arg("device_outputs") = false,
          py::arg("providers") = std::vector<std::string>{},
          py::arg("provider_options") = py::dict())
      .def_property_readonly(
          "metadata",
          [](const WorldModel& model) {
            return MetadataToDictionary(model.metadata());
          })
      .def(
          "step",
          [](const WorldModel& model,
             const py::array& observation,
             const py::array& action,
             const py::array& state) {
            Tensor observation_tensor = TensorFromNumpy(observation);
            Tensor action_tensor = TensorFromNumpy(action);
            Tensor state_tensor = TensorFromNumpy(state);
            StepOutput output;
            {
              py::gil_scoped_release release;
              output = model.Step(
                  observation_tensor,
                  action_tensor,
                  state_tensor);
            }
            return StepOutputToTuple(output);
          },
          py::arg("observation"),
          py::arg("action"),
          py::arg("state"))
      .def(
          "create_rollout",
          [](const WorldModel& model) {
            return std::make_unique<Rollout>(model);
          });

  py::class_<Pipeline>(module, "Pipeline")
      .def(
          py::init([](
                       const std::string& package_path,
                       const std::string& ort_library_path,
                       int intra_op_threads,
                       int inter_op_threads,
                       int log_severity,
                       const std::string& graph_optimization,
                       bool device_outputs,
                       const std::vector<std::string>& providers,
                       const py::dict& provider_options) {
            RuntimeOptions options = RuntimeOptionsFromPython(
                ort_library_path,
                intra_op_threads,
                inter_op_threads,
                log_severity,
                graph_optimization,
                device_outputs,
                providers,
                provider_options);
            py::gil_scoped_release release;
            return Pipeline::Load(package_path, options);
          }),
          py::arg("package_path"),
          py::arg("ort_library_path"),
          py::arg("intra_op_threads") = 0,
          py::arg("inter_op_threads") = 0,
          py::arg("log_severity") = 3,
          py::arg("graph_optimization") = "all",
          py::arg("device_outputs") = false,
          py::arg("providers") = std::vector<std::string>{},
          py::arg("provider_options") = py::dict())
      .def_property_readonly(
          "execution_providers",
          &Pipeline::execution_providers)
      .def(
          "create_session",
          [](const Pipeline& pipeline) {
            return std::make_unique<PipelineSession>(
                pipeline.CreateSession());
          });

  py::class_<PipelineSessionSnapshot>(module, "PipelineSessionSnapshot")
      .def_property_readonly("valid", &PipelineSessionSnapshot::valid);

  // Opaque like PipelineSessionSnapshot: a run can only come from
  // PipelineSession.begin_stage, so Python cannot fabricate one that claims a
  // session's run slot.
  py::class_<StageRun>(module, "StageRun")
      .def_property_readonly(
          "stage",
          [](const StageRun& run) { return std::string(run.stage()); })
      .def_property_readonly(
          "done",
          [](const StageRun& run) {
            // Takes the session lock, so it must not block another thread's
            // step while holding the GIL.
            py::gil_scoped_release release;
            return run.done();
          })
      .def_property_readonly(
          "iteration",
          [](const StageRun& run) {
            py::gil_scoped_release release;
            return run.iteration();
          })
      .def(
          "step",
          [](StageRun& run) {
            StageEvent event;
            {
              py::gil_scoped_release release;
              event = run.Step();
            }
            return StageEventToDictionary(event);
          })
      .def(
          "finish",
          [](StageRun& run) {
            NamedTensors outputs;
            {
              py::gil_scoped_release release;
              outputs = run.Finish();
            }
            return NamedTensorsToDictionary(outputs);
          })
      .def(
          "request_cancellation",
          [](StageRun& run) {
            // Deliberately takes no session lock, so a second thread can call
            // this while step or finish is running with the GIL released.
            py::gil_scoped_release release;
            run.RequestCancellation();
          })
      .def(
          "cancel",
          [](StageRun& run) {
            py::gil_scoped_release release;
            run.Cancel();
          });

  py::class_<PipelineSession>(module, "PipelineSession")
      .def(
          "run_stage",
          [](PipelineSession& session,
             const std::string& stage,
             const py::dict& inputs,
             const py::dict& overrides,
             const py::dict& options,
             const std::optional<CancellationToken>& cancellation) {
            NamedTensors input_tensors =
                NamedTensorsFromDictionary(inputs);
            NamedTensors override_tensors =
                NamedTensorsFromDictionary(overrides);
            PipelineRunOptions run_options =
                PipelineOptionsFromPython(options, cancellation);
            NamedTensors outputs;
            {
              py::gil_scoped_release release;
              outputs = session.RunStage(
                  stage,
                  input_tensors,
                  override_tensors,
                  run_options);
            }
            return NamedTensorsToDictionary(outputs);
          },
          py::arg("stage"),
          py::arg("inputs") = py::dict(),
          py::arg("overrides") = py::dict(),
          py::arg("options") = py::dict(),
          py::arg("cancellation") = py::none())
      .def(
          "step_stage",
          [](PipelineSession& session,
             const std::string& stage,
             const py::dict& inputs,
             const py::dict& overrides,
             const py::dict& options,
             const std::optional<CancellationToken>& cancellation) {
            NamedTensors input_tensors =
                NamedTensorsFromDictionary(inputs);
            NamedTensors override_tensors =
                NamedTensorsFromDictionary(overrides);
            PipelineRunOptions run_options =
                PipelineOptionsFromPython(options, cancellation);
            NamedTensors outputs;
            {
              py::gil_scoped_release release;
              outputs = session.StepStage(
                  stage,
                  input_tensors,
                  override_tensors,
                  run_options);
            }
            return NamedTensorsToDictionary(outputs);
          },
          py::arg("stage"),
          py::arg("inputs") = py::dict(),
          py::arg("overrides") = py::dict(),
          py::arg("options") = py::dict(),
          py::arg("cancellation") = py::none())
      .def(
          "begin_stage",
          [](PipelineSession& session,
             const std::string& stage,
             const py::dict& inputs,
             const py::dict& overrides,
             const py::dict& options,
             const std::optional<CancellationToken>& cancellation) {
            NamedTensors input_tensors =
                NamedTensorsFromDictionary(inputs);
            NamedTensors override_tensors =
                NamedTensorsFromDictionary(overrides);
            PipelineRunOptions run_options =
                PipelineOptionsFromPython(options, cancellation);
            py::gil_scoped_release release;
            return std::make_unique<StageRun>(session.BeginStage(
                stage,
                input_tensors,
                override_tensors,
                run_options));
          },
          py::arg("stage"),
          py::arg("inputs") = py::dict(),
          py::arg("overrides") = py::dict(),
          py::arg("options") = py::dict(),
          py::arg("cancellation") = py::none())
      .def_property_readonly(
          "outputs",
          [](const PipelineSession& session) {
            NamedTensors outputs;
            {
              // Reading stays legal while a stage run is active, so it must
              // wait for that run's lock without holding the GIL.
              py::gil_scoped_release release;
              outputs = session.outputs();
            }
            return NamedTensorsToDictionary(outputs);
          })
      .def(
          "state",
          [](const PipelineSession& session,
             const std::string& name) -> py::object {
            std::optional<Tensor> value;
            {
              py::gil_scoped_release release;
              value = session.state(name);
            }
            if (!value.has_value()) {
              return py::none();
            }
            return TensorToNumpy(*value);
          },
          py::arg("name"))
      .def(
          "release_stage",
          [](PipelineSession& session, const std::string& stage) {
            py::gil_scoped_release release;
            session.ReleaseStage(stage);
          },
          py::arg("stage"))
      .def(
          "reset",
          [](PipelineSession& session) {
            py::gil_scoped_release release;
            session.Reset();
          })
      .def(
          "snapshot",
          [](const PipelineSession& session) {
            py::gil_scoped_release release;
            return session.Snapshot();
          })
      .def(
          "restore",
          [](PipelineSession& session,
             const PipelineSessionSnapshot& snapshot) {
            py::gil_scoped_release release;
            session.Restore(snapshot);
          },
          py::arg("snapshot"))
      .def(
          "fork",
          [](const PipelineSession& session) {
            py::gil_scoped_release release;
            return std::make_unique<PipelineSession>(session.Fork());
          })
      .def(
          "checkpoint",
          [](PipelineSession& session, const std::string& name) {
            py::gil_scoped_release release;
            session.Checkpoint(name);
          },
          py::arg("name"))
      .def(
          "restore_checkpoint",
          [](PipelineSession& session, const std::string& name) {
            py::gil_scoped_release release;
            session.RestoreCheckpoint(name);
          },
          py::arg("name"))
      .def(
          "drop_checkpoint",
          [](PipelineSession& session, const std::string& name) {
            py::gil_scoped_release release;
            session.DropCheckpoint(name);
          },
          py::arg("name"))
      .def(
          "has_checkpoint",
          [](const PipelineSession& session, const std::string& name) {
            py::gil_scoped_release release;
            return session.HasCheckpoint(name);
          },
          py::arg("name"));

  py::class_<Rollout>(module, "Rollout")
      .def("reset", py::overload_cast<>(&Rollout::Reset))
      .def(
          "reset_state",
          [](Rollout& rollout, const py::array& state) {
            rollout.Reset(TensorFromNumpy(state));
          },
          py::arg("state"))
      .def("reset_zeros", &Rollout::ResetZeros, py::arg("batch_size"))
      .def_property_readonly("has_state", &Rollout::has_state)
      .def_property_readonly(
          "state",
          [](const Rollout& rollout) -> py::object {
            const auto state = rollout.state();
            if (!state.has_value()) {
              return py::none();
            }
            return TensorToNumpy(*state);
          })
      .def(
          "step",
          [](Rollout& rollout,
             const py::array& observation,
             const py::array& action) {
            Tensor observation_tensor = TensorFromNumpy(observation);
            Tensor action_tensor = TensorFromNumpy(action);
            StepOutput output;
            {
              py::gil_scoped_release release;
              output = rollout.Step(observation_tensor, action_tensor);
            }
            return StepOutputToTuple(output);
          },
          py::arg("observation"),
          py::arg("action"));
}
