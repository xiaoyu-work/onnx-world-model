#pragma once

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
  Tensor() = default;
  Tensor(DataType data_type, std::vector<std::int64_t> shape);

  static Tensor FromBytes(
      DataType data_type,
      std::vector<std::int64_t> shape,
      std::span<const std::byte> bytes);

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
  [[nodiscard]] std::size_t size_bytes() const noexcept { return data_->size(); }
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return *data_; }
  [[nodiscard]] std::span<std::byte> mutable_bytes();

  template <typename T>
  [[nodiscard]] std::span<const T> values() const {
    if (DataTypeOf<std::remove_cv_t<T>>::value != data_type_) {
      throw Error(
          ErrorCode::invalid_argument,
          "Requested tensor values do not match tensor data type " +
              std::string(ToString(data_type_)));
    }
    return {
        reinterpret_cast<const T*>(data_->data()),
        element_count_,
    };
  }

 private:
  DataType data_type_{DataType::float32};
  std::vector<std::int64_t> shape_;
  std::size_t element_count_{0};
  std::shared_ptr<std::vector<std::byte>> data_{
      std::make_shared<std::vector<std::byte>>()};
};

}  // namespace onnx_world_model
