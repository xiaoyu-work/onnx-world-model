/**
 * @agent-file
 * @agent-purpose: Standalone test executable for Tensor: checked layout arithmetic, CPU and device-buffer construction, explicit CPU materialization, typed access, zero initialization, and copy-on-write semantics.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as tensor_test; it links no test framework, counts failures through local Check and CheckThrows helpers, and returns a non-zero exit code when any check fails. Its fake device buffer implements the installed TensorBuffer contract without ONNX Runtime, proving that device-only host access fails and CopyToCpu materializes independent CPU storage.
 * @agent-side-effects: Writes failure descriptions to stderr and returns a process exit code.
 */

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <utility>
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

class FakeDeviceBuffer final : public onnx_world_model::TensorBuffer {
 public:
  explicit FakeDeviceBuffer(std::span<const std::byte> bytes)
      : bytes_(bytes.begin(), bytes.end()) {}

  [[nodiscard]] const onnx_world_model::TensorDevice& device()
      const noexcept override {
    return device_;
  }

  [[nodiscard]] std::size_t size_bytes() const noexcept override {
    return bytes_.size();
  }

  [[nodiscard]] bool is_host_accessible() const noexcept override {
    return false;
  }

  [[nodiscard]] const void* data() const noexcept override {
    return bytes_.data();
  }

  [[nodiscard]] std::span<const std::byte> bytes() const override {
    return {};
  }

  void CopyToCpu(std::span<std::byte> destination) const override {
    if (destination.size() != bytes_.size()) {
      throw onnx_world_model::Error(
          onnx_world_model::ErrorCode::invalid_argument,
          "Fake device destination size mismatch");
    }
    std::copy(bytes_.begin(), bytes_.end(), destination.begin());
  }

 private:
  onnx_world_model::TensorDevice device_{"test", 2};
  std::vector<std::byte> bytes_;
};

}  // namespace

int main() {
  using onnx_world_model::DataType;
  using onnx_world_model::Tensor;
  using onnx_world_model::TensorDevice;

  Tensor scalar;
  Check(scalar.element_count() == 1, "default scalar element count");
  Check(scalar.size_bytes() == sizeof(float), "default scalar byte count");
  Check(scalar.device() == TensorDevice{}, "default tensor is on CPU");
  Check(scalar.is_host_accessible(), "default tensor is host-accessible");

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

  const std::array<float, 2> device_values{5.0F, 7.0F};
  const auto device_bytes = std::as_bytes(std::span(device_values));
  Tensor device_tensor = Tensor::FromBuffer(
      DataType::float32,
      {2},
      std::make_shared<FakeDeviceBuffer>(device_bytes));
  Check(
      device_tensor.device() == TensorDevice("test", 2),
      "device tensor identity");
  Check(!device_tensor.is_host_accessible(), "device tensor host access");
  CheckThrows(
      [&device_tensor] { (void)device_tensor.bytes(); },
      "device bytes must require CPU materialization");
  CheckThrows(
      [&device_tensor] { (void)device_tensor.values<float>(); },
      "device typed values must require CPU materialization");
  CheckThrows(
      [&device_tensor] { (void)device_tensor.mutable_bytes(); },
      "device mutation must require CPU materialization");
  Tensor materialized = device_tensor.CopyToCpu();
  Check(materialized.device() == TensorDevice{}, "materialized tensor device");
  Check(materialized.is_host_accessible(), "materialized tensor host access");
  Check(materialized.values<float>()[0] == 5.0F, "materialized first value");
  Check(materialized.values<float>()[1] == 7.0F, "materialized second value");

  CheckThrows(
      [] { (void)Tensor::Zeros(DataType::float32, {-1, 2}); },
      "negative dimensions must fail");
  CheckThrows(
      [] {
        const std::array<std::byte, 3> bytes{};
        (void)Tensor::FromBytes(DataType::float32, {1}, bytes);
      },
      "incorrect byte count must fail");
  CheckThrows(
      [&device_bytes] {
        (void)Tensor::FromBuffer(
            DataType::float32,
            {3},
            std::make_shared<FakeDeviceBuffer>(device_bytes));
      },
      "device buffer byte count must match layout");
  CheckThrows(
      [] {
        (void)Tensor::FromBuffer(
            DataType::float32,
            {1},
            nullptr);
      },
      "null tensor buffer must fail");
  CheckThrows(
      [] { (void)TensorDevice("CUDA"); },
      "device type must be canonical");
  CheckThrows(
      [] { (void)TensorDevice("cuda", -1); },
      "device ID must be non-negative");
  CheckThrows(
      [] { (void)TensorDevice("cpu", 1); },
      "CPU device ID must be zero");

  if (failures == 0) {
    std::cout << "tensor tests passed\n";
  }
  return failures == 0 ? 0 : 1;
}
