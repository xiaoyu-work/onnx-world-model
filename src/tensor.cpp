/**
 * @agent-file
 * @agent-purpose: Implements canonical tensor devices, owned CPU buffers, checked tensor layout construction, host-access guards, CPU materialization, and copy-on-write mutation.
 * @agent-public-api: TensorDevice::TensorDevice, ToString, DataTypeSize, Tensor::Tensor, Tensor::FromBytes, Tensor::FromBuffer, Tensor::Zeros, Tensor::size_bytes, Tensor::device, Tensor::is_host_accessible, Tensor::bytes, Tensor::mutable_bytes, Tensor::CopyToCpu
 * @agent-invariants: Device names are canonical lowercase tokens, CPU uses ID zero, and every buffer exactly matches its tensor layout. Host-visible data is aligned for its data type. Device-only bytes and mutable access throw instead of causing an implicit transfer; CopyToCpu is the explicit materialization boundary. Mutable CPU access clones shared or externally owned storage before writing.
 * @agent-side-effects: CopyToCpu invokes the source TensorBuffer and may synchronize and transfer accelerator memory.
 */

#include "onnx_world_model/tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "onnx_world_model/error.hpp"

namespace onnx_world_model {
namespace {

class OwnedCpuBuffer final : public TensorBuffer {
 public:
  explicit OwnedCpuBuffer(std::size_t size_bytes) : bytes_(size_bytes) {}

  [[nodiscard]] const TensorDevice& device() const noexcept override {
    static const TensorDevice cpu;
    return cpu;
  }

  [[nodiscard]] std::size_t size_bytes() const noexcept override {
    return bytes_.size();
  }

  [[nodiscard]] bool is_host_accessible() const noexcept override {
    return true;
  }

  [[nodiscard]] const void* data() const noexcept override {
    return bytes_.data();
  }

  [[nodiscard]] std::span<const std::byte> bytes() const override {
    return bytes_;
  }

  void CopyToCpu(std::span<std::byte> destination) const override {
    if (destination.size() != bytes_.size()) {
      throw Error(
          ErrorCode::invalid_argument,
          "CPU tensor copy destination has the wrong byte size");
    }
    std::copy(bytes_.begin(), bytes_.end(), destination.begin());
  }

  [[nodiscard]] std::span<std::byte> mutable_bytes() noexcept {
    return bytes_;
  }

 private:
  std::vector<std::byte> bytes_;
};

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

[[nodiscard]] std::size_t CheckedByteSize(
    DataType data_type,
    std::size_t element_count) {
  const std::size_t element_size = DataTypeSize(data_type);
  if (element_count > std::numeric_limits<std::size_t>::max() / element_size) {
    throw Error(ErrorCode::invalid_argument, "Tensor byte size overflows size_t");
  }
  return element_count * element_size;
}

[[nodiscard]] bool IsCanonicalDeviceCharacter(char character) {
  return (character >= 'a' && character <= 'z') ||
         (character >= '0' && character <= '9') ||
         character == '_';
}

[[nodiscard]] std::string DeviceLabel(const TensorDevice& device) {
  return std::string(device.type()) + ":" + std::to_string(device.id());
}

[[nodiscard]] const TensorBuffer& RequireBuffer(
    const std::shared_ptr<TensorBuffer>& buffer) {
  if (buffer == nullptr) {
    throw Error(
        ErrorCode::invalid_argument,
        "Tensor has no storage, possibly because it was moved from");
  }
  return *buffer;
}

void ValidateBuffer(
    DataType data_type,
    std::size_t element_count,
    const std::shared_ptr<TensorBuffer>& buffer) {
  const TensorBuffer& value = RequireBuffer(buffer);
  const std::size_t expected = CheckedByteSize(data_type, element_count);
  if (value.size_bytes() != expected) {
    throw Error(
        ErrorCode::invalid_argument,
        "Tensor buffer byte count does not match its data type and shape");
  }
  if (expected != 0 && value.data() == nullptr) {
    throw Error(
        ErrorCode::invalid_argument,
        "Non-empty tensor buffer must expose a data address");
  }
  if (value.device().type() == "cpu" && value.device().id() != 0) {
    throw Error(ErrorCode::invalid_argument, "CPU tensor device ID must be zero");
  }
  if (value.device().type() == "cpu" && !value.is_host_accessible()) {
    throw Error(
        ErrorCode::invalid_argument,
        "CPU tensor buffers must be host-accessible");
  }
  if (!value.is_host_accessible()) {
    return;
  }
  const std::span<const std::byte> bytes = value.bytes();
  if (bytes.size() != expected) {
    throw Error(
        ErrorCode::invalid_argument,
        "Host-accessible tensor buffer returned the wrong byte count");
  }
  if (!bytes.empty() && bytes.data() != value.data()) {
    throw Error(
        ErrorCode::invalid_argument,
        "Host-accessible tensor buffer data and byte view disagree");
  }
  const std::size_t alignment =
      std::min(DataTypeSize(data_type), alignof(std::max_align_t));
  if (!bytes.empty() &&
      reinterpret_cast<std::uintptr_t>(bytes.data()) % alignment != 0) {
    throw Error(
        ErrorCode::invalid_argument,
        "Host-accessible tensor buffer is not aligned for its data type");
  }
}

}  // namespace

Tensor::Tensor() : Tensor(DataType::float32, {}) {}

TensorDevice::TensorDevice(std::string type, std::int32_t id)
    : type_(std::move(type)), id_(id) {
  if (type_.empty() || type_.front() < 'a' || type_.front() > 'z' ||
      !std::ranges::all_of(type_, IsCanonicalDeviceCharacter)) {
    throw Error(
        ErrorCode::invalid_argument,
        "Tensor device type must be a canonical lowercase token");
  }
  if (id_ < 0) {
    throw Error(
        ErrorCode::invalid_argument,
        "Tensor device ID must be non-negative");
  }
  if (type_ == "cpu" && id_ != 0) {
    throw Error(ErrorCode::invalid_argument, "CPU tensor device ID must be zero");
  }
}

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
      buffer_(std::make_shared<OwnedCpuBuffer>(
          CheckedByteSize(data_type_, element_count_))) {}

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
  std::copy(bytes.begin(), bytes.end(), tensor.mutable_bytes().begin());
  return tensor;
}

Tensor Tensor::FromBuffer(
    DataType data_type,
    std::vector<std::int64_t> shape,
    std::shared_ptr<TensorBuffer> buffer) {
  Tensor tensor;
  tensor.data_type_ = data_type;
  tensor.shape_ = std::move(shape);
  tensor.element_count_ = CheckedElementCount(tensor.shape_);
  ValidateBuffer(data_type, tensor.element_count_, buffer);
  tensor.buffer_ = std::move(buffer);
  return tensor;
}

Tensor Tensor::Zeros(DataType data_type, std::vector<std::int64_t> shape) {
  return Tensor(data_type, std::move(shape));
}

std::size_t Tensor::size_bytes() const {
  return RequireBuffer(buffer_).size_bytes();
}

const TensorDevice& Tensor::device() const {
  return RequireBuffer(buffer_).device();
}

bool Tensor::is_host_accessible() const noexcept {
  return buffer_ != nullptr && buffer_->is_host_accessible();
}

std::span<const std::byte> Tensor::bytes() const {
  const TensorBuffer& storage = RequireBuffer(buffer_);
  if (!storage.is_host_accessible()) {
    throw Error(
        ErrorCode::invalid_argument,
        "Tensor on device '" + DeviceLabel(storage.device()) +
            "' is not host-accessible; call CopyToCpu() first");
  }
  const std::span<const std::byte> result = storage.bytes();
  if (result.size() != storage.size_bytes()) {
    throw Error(
        ErrorCode::runtime_execution,
        "Host-accessible tensor buffer changed its byte count");
  }
  return result;
}

std::span<std::byte> Tensor::mutable_bytes() {
  const TensorBuffer& storage = RequireBuffer(buffer_);
  if (storage.device() != TensorDevice{} || !storage.is_host_accessible()) {
    throw Error(
        ErrorCode::invalid_argument,
        "Tensor on device '" + DeviceLabel(storage.device()) +
            "' must be copied to CPU before mutation");
  }
  auto* owned = dynamic_cast<OwnedCpuBuffer*>(buffer_.get());
  if (owned == nullptr || buffer_.use_count() != 1) {
    auto copy = std::make_shared<OwnedCpuBuffer>(storage.size_bytes());
    storage.CopyToCpu(copy->mutable_bytes());
    buffer_ = std::move(copy);
    owned = static_cast<OwnedCpuBuffer*>(buffer_.get());
  }
  return owned->mutable_bytes();
}

Tensor Tensor::CopyToCpu() const {
  const TensorBuffer& storage = RequireBuffer(buffer_);
  if (storage.device() == TensorDevice{} && storage.is_host_accessible()) {
    return *this;
  }
  Tensor result(data_type_, shape_);
  storage.CopyToCpu(result.mutable_bytes());
  return result;
}

}  // namespace onnx_world_model
