#include "ort_backend.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dynamic_library.hpp"
#include "onnx_world_model/error.hpp"
#include "onnxruntime_cxx_api.h"

namespace onnx_world_model::detail {
namespace {

[[nodiscard]] DataType FromOrtDataType(ONNXTensorElementDataType data_type) {
  switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DataType::float32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DataType::float16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return DataType::bfloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return DataType::float64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DataType::int64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DataType::int32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return DataType::int16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return DataType::int8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return DataType::uint64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return DataType::uint32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return DataType::uint16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DataType::uint8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return DataType::boolean;
    default:
      throw Error(
          ErrorCode::model_contract,
          "Model tensor uses unsupported ONNX data type " +
              std::to_string(static_cast<int>(data_type)));
  }
}

[[nodiscard]] ONNXTensorElementDataType ToOrtDataType(DataType data_type) {
  switch (data_type) {
    case DataType::float32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DataType::float16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DataType::bfloat16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    case DataType::float64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case DataType::int64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DataType::int32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DataType::int16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case DataType::int8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case DataType::uint64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case DataType::uint32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case DataType::uint16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case DataType::uint8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DataType::boolean:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  }
  throw Error(ErrorCode::invalid_argument, "Unsupported tensor data type");
}

[[nodiscard]] ::GraphOptimizationLevel ToOrtGraphOptimizationLevel(
    GraphOptimizationLevel level) {
  switch (level) {
    case GraphOptimizationLevel::disabled:
      return ORT_DISABLE_ALL;
    case GraphOptimizationLevel::basic:
      return ORT_ENABLE_BASIC;
    case GraphOptimizationLevel::extended:
      return ORT_ENABLE_EXTENDED;
    case GraphOptimizationLevel::all:
      return ORT_ENABLE_ALL;
  }
  return ORT_ENABLE_ALL;
}

[[nodiscard]] TensorSpec ReadTensorSpec(
    const Ort::Session& session,
    std::size_t index,
    bool input,
    Ort::AllocatorWithDefaultOptions& allocator) {
  auto name = input ? session.GetInputNameAllocated(index, allocator)
                    : session.GetOutputNameAllocated(index, allocator);
  Ort::TypeInfo type_info =
      input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    throw Error(
        ErrorCode::model_contract,
        "Model value '" + std::string(name.get()) + "' must be a tensor");
  }
  const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  return TensorSpec{
      .name = name.get(),
      .data_type = FromOrtDataType(tensor_info.GetElementType()),
      .shape = tensor_info.GetShape(),
  };
}

[[nodiscard]] Ort::Value MakeOrtTensor(
    const Tensor& tensor,
    const Ort::MemoryInfo& memory_info) {
  const auto& shape = tensor.shape();
  return Ort::Value::CreateTensor(
      memory_info,
      const_cast<std::byte*>(tensor.bytes().data()),
      tensor.size_bytes(),
      shape.data(),
      shape.size(),
      ToOrtDataType(tensor.data_type()));
}

[[nodiscard]] Tensor CopyOrtTensor(const Ort::Value& value) {
  if (!value.IsTensor()) {
    throw Error(ErrorCode::runtime_execution, "ONNX Runtime returned a non-tensor value");
  }
  const auto info = value.GetTensorTypeAndShapeInfo();
  Tensor tensor(
      FromOrtDataType(info.GetElementType()),
      info.GetShape());
  if (tensor.size_bytes() != info.GetElementCount() * DataTypeSize(tensor.data_type())) {
    throw Error(
        ErrorCode::runtime_execution,
        "ONNX Runtime returned an inconsistent tensor byte size");
  }
  std::memcpy(
      tensor.mutable_bytes().data(),
      value.GetTensorRawData(),
      tensor.size_bytes());
  return tensor;
}

class OrtBackend final : public ModelBackend {
 public:
  OrtBackend(const std::filesystem::path& model_path, const RuntimeOptions& options)
      : env_(
            static_cast<OrtLoggingLevel>(options.log_severity),
            "onnx-world-model"),
        memory_info_(Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault)) {
    if (options.intra_op_threads < 0 || options.inter_op_threads < 0) {
      throw Error(
          ErrorCode::invalid_argument,
          "Thread counts must be zero (automatic) or positive");
    }
    if (options.intra_op_threads > 0) {
      session_options_.SetIntraOpNumThreads(options.intra_op_threads);
    }
    if (options.inter_op_threads > 0) {
      session_options_.SetInterOpNumThreads(options.inter_op_threads);
    }
    session_options_.SetGraphOptimizationLevel(
        ToOrtGraphOptimizationLevel(options.graph_optimization));
    session_ = Ort::Session(env_, model_path.c_str(), session_options_);

    Ort::AllocatorWithDefaultOptions allocator;
    metadata_.inputs.reserve(session_.GetInputCount());
    for (std::size_t index = 0; index < session_.GetInputCount(); ++index) {
      metadata_.inputs.push_back(ReadTensorSpec(session_, index, true, allocator));
    }
    metadata_.outputs.reserve(session_.GetOutputCount());
    for (std::size_t index = 0; index < session_.GetOutputCount(); ++index) {
      metadata_.outputs.push_back(ReadTensorSpec(session_, index, false, allocator));
    }
  }

  [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
    return metadata_;
  }

  [[nodiscard]] NamedTensors Run(const NamedTensors& inputs) const override {
    try {
      std::vector<Ort::Value> input_values;
      input_values.reserve(metadata_.inputs.size());
      std::vector<const char*> input_names;
      input_names.reserve(metadata_.inputs.size());
      for (const auto& spec : metadata_.inputs) {
        input_names.push_back(spec.name.c_str());
        input_values.push_back(MakeOrtTensor(inputs.at(spec.name), memory_info_));
      }

      std::vector<const char*> output_names;
      output_names.reserve(metadata_.outputs.size());
      for (const auto& spec : metadata_.outputs) {
        output_names.push_back(spec.name.c_str());
      }
      auto outputs = session_.Run(
          Ort::RunOptions{nullptr},
          input_names.data(),
          input_values.data(),
          input_values.size(),
          output_names.data(),
          output_names.size());
      if (outputs.size() != output_names.size()) {
        throw Error(
            ErrorCode::runtime_execution,
            "ONNX Runtime returned an unexpected number of outputs");
      }
      NamedTensors result;
      result.reserve(outputs.size());
      for (std::size_t index = 0; index < outputs.size(); ++index) {
        result.emplace(
            metadata_.outputs[index].name,
            CopyOrtTensor(outputs[index]));
      }
      return result;
    } catch (const Ort::Exception& exception) {
      throw Error(
          ErrorCode::runtime_execution,
          "ONNX Runtime inference failed: " + std::string(exception.what()));
    }
  }

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  mutable Ort::Session session_{nullptr};
  Ort::MemoryInfo memory_info_;
  ModelMetadata metadata_;
};

}  // namespace

ModelBackendPtr CreateOrtBackend(
    const std::filesystem::path& model_path,
    const RuntimeOptions& options) {
  if (model_path.empty()) {
    throw Error(ErrorCode::invalid_argument, "Model path cannot be empty");
  }
  if (!std::filesystem::is_regular_file(model_path)) {
    throw Error(
        ErrorCode::invalid_argument,
        "Model file does not exist: " + model_path.string());
  }
  if (options.log_severity < ORT_LOGGING_LEVEL_VERBOSE ||
      options.log_severity > ORT_LOGGING_LEVEL_FATAL) {
    throw Error(
        ErrorCode::invalid_argument,
        "log_severity must be between 0 (verbose) and 4 (fatal)");
  }

  InitializeOrtApi(options.ort_library_path);
  try {
    return std::make_shared<OrtBackend>(model_path, options);
  } catch (const Error&) {
    throw;
  } catch (const Ort::Exception& exception) {
    throw Error(
        ErrorCode::runtime_load,
        "Failed to create ONNX Runtime session: " +
            std::string(exception.what()));
  }
}

}  // namespace onnx_world_model::detail
