#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

#include "onnx_world_model/error.hpp"
#include "onnx_world_model/tensor.hpp"

namespace {

int failures = 0;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    ++failures;
  }
}

template <typename Function>
void CheckThrows(Function&& function, const char* message) {
  try {
    function();
    Check(false, message);
  } catch (const onnx_world_model::Error&) {
  }
}

}  // namespace

int main() {
  using onnx_world_model::DataType;
  using onnx_world_model::Tensor;

  Tensor zeros = Tensor::Zeros(DataType::float32, {2, 3});
  Check(zeros.element_count() == 6, "element count");
  Check(zeros.size_bytes() == 6 * sizeof(float), "byte count");
  for (const std::byte value : zeros.bytes()) {
    Check(value == std::byte{0}, "zero initialization");
  }

  const std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
  Tensor tensor = Tensor::FromValues<float>({2, 2}, std::span(values));
  Check(tensor.values<float>()[2] == 3.0F, "typed values");

  Tensor shared = tensor;
  auto mutable_values = std::span(
      reinterpret_cast<float*>(shared.mutable_bytes().data()),
      shared.element_count());
  mutable_values[0] = 9.0F;
  Check(shared.values<float>()[0] == 9.0F, "mutable copy");
  Check(tensor.values<float>()[0] == 1.0F, "copy-on-write");

  CheckThrows(
      [] { (void)Tensor::Zeros(DataType::float32, {-1, 2}); },
      "negative dimensions must fail");
  CheckThrows(
      [] {
        const std::array<std::byte, 3> bytes{};
        (void)Tensor::FromBytes(DataType::float32, {1}, bytes);
      },
      "incorrect byte count must fail");

  if (failures == 0) {
    std::cout << "tensor tests passed\n";
  }
  return failures == 0 ? 0 : 1;
}
