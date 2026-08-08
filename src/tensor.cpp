#include "onnx_world_model/tensor.hpp"

#include <algorithm>
#include <limits>
#include <string>

#include "onnx_world_model/error.hpp"

namespace onnx_world_model {
namespace {

[[nodiscard]] std::size_t CheckedElementCount(
    const std::vector<std::int64_t>& shape) {
  std::size_t count = 1;
  for (const std::int64_t dimension : shape) {
    if (dimension < 0) {
      throw Error(
          ErrorCode::invalid_argument,
          "Concrete tensor dimensions cannot be negative");
    }
    const auto unsigned_dimension = static_cast<std::size_t>(dimension);
    if (unsigned_dimension != 0 &&
        count > std::numeric_limits<std::size_t>::max() / unsigned_dimension) {
      throw Error(ErrorCode::invalid_argument, "Tensor element count overflows size_t");
    }
    count *= unsigned_dimension;
  }
  return count;
}

}  // namespace

Tensor::Tensor() : Tensor(DataType::float32, {}) {}

std::string_view ToString(DataType data_type) noexcept {
  switch (data_type) {
    case DataType::float32:
      return "float32";
    case DataType::float16:
      return "float16";
    case DataType::bfloat16:
      return "bfloat16";
    case DataType::float64:
      return "float64";
    case DataType::int64:
      return "int64";
    case DataType::int32:
      return "int32";
    case DataType::int16:
      return "int16";
    case DataType::int8:
      return "int8";
    case DataType::uint64:
      return "uint64";
    case DataType::uint32:
      return "uint32";
    case DataType::uint16:
      return "uint16";
    case DataType::uint8:
      return "uint8";
    case DataType::boolean:
      return "bool";
  }
  return "unknown";
}

std::size_t DataTypeSize(DataType data_type) {
  switch (data_type) {
    case DataType::float32:
    case DataType::int32:
    case DataType::uint32:
      return 4;
    case DataType::float16:
    case DataType::bfloat16:
    case DataType::int16:
    case DataType::uint16:
      return 2;
    case DataType::float64:
    case DataType::int64:
    case DataType::uint64:
      return 8;
    case DataType::int8:
    case DataType::uint8:
    case DataType::boolean:
      return 1;
  }
  throw Error(ErrorCode::invalid_argument, "Unsupported tensor data type");
}

Tensor::Tensor(DataType data_type, std::vector<std::int64_t> shape)
    : data_type_(data_type),
      shape_(std::move(shape)),
      element_count_(CheckedElementCount(shape_)),
      data_(std::make_shared<std::vector<std::byte>>()) {
  const std::size_t element_size = DataTypeSize(data_type_);
  if (element_count_ > std::numeric_limits<std::size_t>::max() / element_size) {
    throw Error(ErrorCode::invalid_argument, "Tensor byte size overflows size_t");
  }
  data_->resize(element_count_ * element_size);
}

Tensor Tensor::FromBytes(
    DataType data_type,
    std::vector<std::int64_t> shape,
    std::span<const std::byte> bytes) {
  Tensor tensor(data_type, std::move(shape));
  if (tensor.size_bytes() != bytes.size()) {
    throw Error(
        ErrorCode::invalid_argument,
        "Tensor byte count does not match its data type and shape");
  }
  std::copy(bytes.begin(), bytes.end(), tensor.data_->begin());
  return tensor;
}

Tensor Tensor::Zeros(DataType data_type, std::vector<std::int64_t> shape) {
  return Tensor(data_type, std::move(shape));
}

std::span<std::byte> Tensor::mutable_bytes() {
  if (data_.use_count() != 1) {
    data_ = std::make_shared<std::vector<std::byte>>(*data_);
  }
  return *data_;
}

}  // namespace onnx_world_model
