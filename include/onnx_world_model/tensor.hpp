#pragma once

/**
 * @agent-file
 * @agent-purpose: Declares dense tensor data types, canonical device identities, the device-buffer contract, and the Tensor value type shared by every runtime layer.
 * @agent-public-api: DataType, ToString, DataTypeSize, DataTypeOf, TensorDevice, TensorBuffer, Tensor
 * @agent-invariants: TensorDevice type names use canonical lowercase tokens and CPU always has device ID zero. TensorBuffer size and device are immutable, CopyToCpu completes before returning, and a host-accessible buffer is aligned for its tensor data type. Tensor has value semantics with copy-on-write CPU storage; shape entries must be non-negative, size_bytes() equals element_count() * DataTypeSize(data_type()), and host access to a device-only buffer fails until CopyToCpu() is called.
 * @agent-side-effects: Tensor::CopyToCpu may synchronize and copy data from an accelerator through the supplied TensorBuffer implementation.
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "onnx_world_model/error.hpp"

namespace onnx_world_model {

enum class DataType {
  float32,
  float16,
  bfloat16,
  float64,
  int64,
  int32,
  int16,
  int8,
  uint64,
  uint32,
  uint16,
  uint8,
  boolean,
};

[[nodiscard]] std::string_view ToString(DataType data_type) noexcept;
[[nodiscard]] std::size_t DataTypeSize(DataType data_type);

class TensorDevice {
 public:
  TensorDevice() = default;
  explicit TensorDevice(std::string type, std::int32_t id = 0);

  [[nodiscard]] std::string_view type() const noexcept { return type_; }
  [[nodiscard]] std::int32_t id() const noexcept { return id_; }

  bool operator==(const TensorDevice&) const = default;

 private:
  std::string type_{"cpu"};
  std::int32_t id_{0};
};

class TensorBuffer {
 public:
  virtual ~TensorBuffer() = default;

  [[nodiscard]] virtual const TensorDevice& device() const noexcept = 0;
  [[nodiscard]] virtual std::size_t size_bytes() const noexcept = 0;
  [[nodiscard]] virtual bool is_host_accessible() const noexcept = 0;
  [[nodiscard]] virtual const void* data() const noexcept = 0;
  [[nodiscard]] virtual std::span<const std::byte> bytes() const = 0;
  virtual void CopyToCpu(std::span<std::byte> destination) const = 0;
};

template <typename T>
struct DataTypeOf;

template <>
struct DataTypeOf<float> {
  static constexpr DataType value = DataType::float32;
};

template <>
struct DataTypeOf<double> {
  static constexpr DataType value = DataType::float64;
};

template <>
struct DataTypeOf<std::int64_t> {
  static constexpr DataType value = DataType::int64;
};

template <>
struct DataTypeOf<std::int32_t> {
  static constexpr DataType value = DataType::int32;
};

template <>
struct DataTypeOf<std::int16_t> {
  static constexpr DataType value = DataType::int16;
};

template <>
struct DataTypeOf<std::int8_t> {
  static constexpr DataType value = DataType::int8;
};

template <>
struct DataTypeOf<std::uint64_t> {
  static constexpr DataType value = DataType::uint64;
};

template <>
struct DataTypeOf<std::uint32_t> {
  static constexpr DataType value = DataType::uint32;
};

template <>
struct DataTypeOf<std::uint16_t> {
  static constexpr DataType value = DataType::uint16;
};

template <>
struct DataTypeOf<std::uint8_t> {
  static constexpr DataType value = DataType::uint8;
};

class Tensor {
 public:
  Tensor();
  Tensor(DataType data_type, std::vector<std::int64_t> shape);

  static Tensor FromBytes(
      DataType data_type,
      std::vector<std::int64_t> shape,
      std::span<const std::byte> bytes);
  static Tensor FromBuffer(
      DataType data_type,
      std::vector<std::int64_t> shape,
      std::shared_ptr<TensorBuffer> buffer);

  template <typename T>
  static Tensor FromValues(
      std::vector<std::int64_t> shape,
      std::span<const T> values) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto bytes = std::as_bytes(values);
    return FromBytes(DataTypeOf<std::remove_cv_t<T>>::value, std::move(shape), bytes);
  }

  static Tensor Zeros(DataType data_type, std::vector<std::int64_t> shape);

  [[nodiscard]] DataType data_type() const noexcept { return data_type_; }
  [[nodiscard]] const std::vector<std::int64_t>& shape() const noexcept { return shape_; }
  [[nodiscard]] std::size_t element_count() const noexcept { return element_count_; }
  [[nodiscard]] std::size_t size_bytes() const;
  [[nodiscard]] const TensorDevice& device() const;
  [[nodiscard]] bool is_host_accessible() const noexcept;
  [[nodiscard]] const std::shared_ptr<TensorBuffer>& buffer() const noexcept {
    return buffer_;
  }
  [[nodiscard]] std::span<const std::byte> bytes() const;
  [[nodiscard]] std::span<std::byte> mutable_bytes();
  [[nodiscard]] Tensor CopyToCpu() const;

  template <typename T>
  [[nodiscard]] std::span<const T> values() const {
    if (DataTypeOf<std::remove_cv_t<T>>::value != data_type_) {
      throw Error(
          ErrorCode::invalid_argument,
          "Requested tensor values do not match tensor data type " +
              std::string(ToString(data_type_)));
    }
    const std::span<const std::byte> raw = bytes();
    return {
        reinterpret_cast<const T*>(raw.data()),
        element_count_,
    };
  }

 private:
  DataType data_type_{DataType::float32};
  std::vector<std::int64_t> shape_;
  std::size_t element_count_{0};
  std::shared_ptr<TensorBuffer> buffer_;
};

}  // namespace onnx_world_model
