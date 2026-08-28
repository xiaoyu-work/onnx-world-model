/**
 * @agent-file
 * @agent-purpose: Standalone test executable for Model, runtime option defaults, provider helpers, and the named-tensor metadata contract, using an in-process AddOneBackend stub.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as model_test; it counts failures through local Check and CheckThrows helpers and returns a non-zero exit code when any check fails. It covers input and output validation, runtime defaults, provider-name normalization, and registration argument rejection through public APIs; invalid provider-library paths fail before ONNX Runtime is loaded.
 * @agent-side-effects: Writes failure descriptions to stderr and returns a process exit code.
 */

#include <array>
#include <cstddef>
#include <iostream>
#include <memory>
#include <span>
#include <utility>

#include "onnx_world_model/error.hpp"
#include "onnx_world_model/model.hpp"

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

class AddOneBackend final : public onnx_world_model::ModelBackend {
 public:
  AddOneBackend() {
    metadata_.inputs.push_back({
        .name = "input",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {-1, 2},
    });
    metadata_.outputs.push_back({
        .name = "output",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {-1, 2},
    });
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    onnx_world_model::Tensor output = inputs.at("input");
    auto bytes = output.mutable_bytes();
    auto values = std::span(
        reinterpret_cast<float*>(bytes.data()),
        output.element_count());
    for (float& value : values) {
      value += 1.0F;
    }
    return {{"output", std::move(output)}};
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

}  // namespace

int main() {
  using onnx_world_model::Model;
  using onnx_world_model::NamedTensors;
  using onnx_world_model::RuntimeOptions;
  using onnx_world_model::Tensor;

  Check(
      onnx_world_model::NormalizeExecutionProviderName("trt-rtx") ==
          "nvtensorrtrtx",
      "Mobius TRT-RTX provider alias");
  Check(
      onnx_world_model::NormalizeExecutionProviderName(
          "CUDAExecutionProvider") == "cuda",
      "ORT provider suffix normalization");
  Check(!RuntimeOptions{}.device_outputs, "device outputs are opt-in");
  CheckThrows(
      [] {
        onnx_world_model::RegisterExecutionProviderLibrary(
            "missing-test-provider",
            "missing-provider-library");
      },
      "missing provider library must fail before loading ORT");

  Model model(std::make_shared<AddOneBackend>());
  const std::array<float, 2> values{2.0F, 4.0F};
  NamedTensors outputs = model.Run({
      {"input", Tensor::FromValues<float>({1, 2}, std::span(values))},
  });
  Check(outputs.at("output").values<float>()[0] == 3.0F, "named model output");
  Check(outputs.at("output").values<float>()[1] == 5.0F, "backend execution");

  CheckThrows(
      [&model] { (void)model.Run({}); },
      "missing input must fail");
  CheckThrows(
      [&model, &values] {
        (void)model.Run({
            {"wrong", Tensor::FromValues<float>({1, 2}, std::span(values))},
        });
      },
      "wrong input name must fail");
  CheckThrows(
      [&model, &values] {
        (void)model.Run({
            {"input", Tensor::FromValues<float>({2}, std::span(values))},
        });
      },
      "wrong input rank must fail");

  if (failures == 0) {
    std::cout << "model tests passed\n";
  }
  return failures == 0 ? 0 : 1;
}
