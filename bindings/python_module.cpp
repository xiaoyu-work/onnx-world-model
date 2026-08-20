/**
 * @agent-file
 * @agent-purpose: Defines the pybind11 `_native` extension module that exposes the C++ runtime to Python and converts between NumPy arrays and onnx_world_model::Tensor.
 * @agent-public-api: _native module, WorldModelError, available_execution_providers, supported_pipeline_capabilities, Model, WorldModel, Pipeline, PipelineSession, Rollout
 * @agent-invariants: NumPy dtype names map one-to-one onto DataType; float16 and bfloat16 cross the boundary as raw 2-byte views; the GIL is released around every blocking ONNX Runtime call; C++ Error is translated into the Python WorldModelError.
 * @agent-side-effects: Registers a Python module and exception type at import time; the wrapped constructors load the ONNX Runtime shared library and read model files from disk.
 */

#include <cstddef>
#include <cstring>
#include <memory>
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
using onnx_world_model::Model;
using onnx_world_model::ModelMetadata;
using onnx_world_model::NamedTensors;
using onnx_world_model::Pipeline;
using onnx_world_model::PipelineRunOptions;
using onnx_world_model::PipelineSession;
using onnx_world_model::Rollout;
using onnx_world_model::RuntimeOptions;
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
  std::vector<py::ssize_t> shape;
  shape.reserve(tensor.shape().size());
  for (const std::int64_t dimension : tensor.shape()) {
    shape.push_back(static_cast<py::ssize_t>(dimension));
  }
  py::array array(NumpyDataType(tensor.data_type()), shape);
  std::memcpy(array.mutable_data(), tensor.bytes().data(), tensor.size_bytes());
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
    const std::vector<std::string>& providers,
    const py::dict& provider_options) {
  RuntimeOptions result{
      .ort_library_path = ort_library_path,
      .intra_op_threads = intra_op_threads,
      .inter_op_threads = inter_op_threads,
      .log_severity = log_severity,
      .graph_optimization = ParseGraphOptimization(graph_optimization),
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

}  // namespace

PYBIND11_MODULE(_native, module) {
  module.doc() = "Native C++ runtime for Mobius world models";
  py::register_exception<onnx_world_model::Error>(
      module,
      "WorldModelError");
  module.def(
      "available_execution_providers",
      [](const std::string& ort_library_path) {
        py::gil_scoped_release release;
        return AvailableExecutionProviders(ort_library_path);
      },
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
                       const std::vector<std::string>& providers,
                       const py::dict& provider_options) {
            RuntimeOptions options = RuntimeOptionsFromPython(
                ort_library_path,
                intra_op_threads,
                inter_op_threads,
                log_severity,
                graph_optimization,
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
          py::arg("providers") = std::vector<std::string>{},
          py::arg("provider_options") = py::dict())
      .def_property_readonly(
          "metadata",
          [](const Model& model) {
            return MetadataToDictionary(model.metadata());
          })
      .def(
          "run",
          [](const Model& model, const py::dict& inputs) {
            NamedTensors input_tensors = NamedTensorsFromDictionary(inputs);
            NamedTensors outputs;
            {
              py::gil_scoped_release release;
              outputs = model.Run(input_tensors);
            }
            return NamedTensorsToDictionary(outputs);
          },
          py::arg("inputs"));

  py::class_<WorldModel>(module, "WorldModel")
      .def(
          py::init([](
                       const std::string& model_path,
                       const std::string& ort_library_path,
                       int intra_op_threads,
                       int inter_op_threads,
                       int log_severity,
                       const std::string& graph_optimization,
                       const std::vector<std::string>& providers,
                       const py::dict& provider_options) {
            RuntimeOptions options = RuntimeOptionsFromPython(
                ort_library_path,
                intra_op_threads,
                inter_op_threads,
                log_severity,
                graph_optimization,
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
                       const std::vector<std::string>& providers,
                       const py::dict& provider_options) {
            RuntimeOptions options = RuntimeOptionsFromPython(
                ort_library_path,
                intra_op_threads,
                inter_op_threads,
                log_severity,
                graph_optimization,
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

  py::class_<PipelineSession>(module, "PipelineSession")
      .def(
          "run_stage",
          [](PipelineSession& session,
             const std::string& stage,
             const py::dict& inputs,
             const py::dict& overrides,
             const py::dict& options) {
            NamedTensors input_tensors =
                NamedTensorsFromDictionary(inputs);
            NamedTensors override_tensors =
                NamedTensorsFromDictionary(overrides);
            PipelineRunOptions run_options =
                PipelineOptionsFromDictionary(options);
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
          py::arg("options") = py::dict())
      .def(
          "step_stage",
          [](PipelineSession& session,
             const std::string& stage,
             const py::dict& inputs,
             const py::dict& overrides,
             const py::dict& options) {
            NamedTensors input_tensors =
                NamedTensorsFromDictionary(inputs);
            NamedTensors override_tensors =
                NamedTensorsFromDictionary(overrides);
            PipelineRunOptions run_options =
                PipelineOptionsFromDictionary(options);
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
          py::arg("options") = py::dict())
      .def_property_readonly(
          "outputs",
          [](const PipelineSession& session) {
            return NamedTensorsToDictionary(session.outputs());
          })
      .def(
          "state",
          [](const PipelineSession& session,
             const std::string& name) -> py::object {
            const auto value = session.state(name);
            if (!value.has_value()) {
              return py::none();
            }
            return TensorToNumpy(*value);
          },
          py::arg("name"))
      .def(
          "release_stage",
          &PipelineSession::ReleaseStage,
          py::arg("stage"))
      .def("reset", &PipelineSession::Reset);

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
