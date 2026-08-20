/**
 * @agent-file
 * @agent-purpose: Standalone test executable for the Tensor value type: shape and byte-size arithmetic, typed construction and views, zero initialization, and copy-on-write semantics.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as tensor_test; it links no test framework, counts failures through local Check and CheckThrows helpers, and returns a non-zero exit code when any check fails. It exercises only tensor.hpp and error.hpp, so it needs no ONNX Runtime library.
 * @agent-side-effects: Writes failure descriptions to stderr and returns a process exit code.
 */

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

  Tensor scalar;
  Check(scalar.element_count() == 1, "default scalar element count");
  Check(scalar.size_bytes() == sizeof(float), "default scalar byte count");

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
