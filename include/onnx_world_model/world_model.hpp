#pragma once

#include <cstdint>
#include <mutex>
#include <optional>

#include "onnx_world_model/model.hpp"

namespace onnx_world_model {

class WorldModel {
 public:
  explicit WorldModel(BackendPtr backend);

  static WorldModel Load(
      const std::filesystem::path& model_path,
      const RuntimeOptions& options = {});

  [[nodiscard]] const ModelMetadata& metadata() const noexcept;
  [[nodiscard]] StepOutput Step(
      const Tensor& observation,
      const Tensor& action,
      const Tensor& state) const;

 private:
  BackendPtr backend_;
};

class Rollout {
 public:
  explicit Rollout(WorldModel model);

  void Reset();
  void Reset(Tensor state);
  void ResetZeros(std::int64_t batch_size);

  [[nodiscard]] bool has_state() const;
  [[nodiscard]] std::optional<Tensor> state() const;
  [[nodiscard]] StepOutput Step(const Tensor& observation, const Tensor& action);

 private:
  [[nodiscard]] Tensor MakeZeroState(std::int64_t batch_size) const;

  WorldModel model_;
  mutable std::mutex mutex_;
  std::optional<Tensor> state_;
};

}  // namespace onnx_world_model
