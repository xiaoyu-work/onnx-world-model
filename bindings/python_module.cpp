#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "onnx_world_model/onnx_world_model.hpp"

namespace py = pybind11;

namespace {

using onnx_world_model::DataType;
using onnx_world_model::ModelMetadata;
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

  py::class_<WorldModel>(module, "WorldModel")
      .def(
          py::init([](
                       const std::string& model_path,
                       const std::string& ort_library_path,
                       int intra_op_threads,
                       int inter_op_threads,
                       int log_severity) {
            const RuntimeOptions options{
                .ort_library_path = ort_library_path,
                .intra_op_threads = intra_op_threads,
                .inter_op_threads = inter_op_threads,
                .log_severity = log_severity,
            };
            py::gil_scoped_release release;
            return WorldModel::Load(model_path, options);
          }),
          py::arg("model_path"),
          py::arg("ort_library_path"),
          py::arg("intra_op_threads") = 0,
          py::arg("inter_op_threads") = 0,
          py::arg("log_severity") = 3)
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
