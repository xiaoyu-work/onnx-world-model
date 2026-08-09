#pragma once

#include <filesystem>

#include "onnx_world_model/backend.hpp"
#include "onnx_world_model/model.hpp"

namespace onnx_world_model::detail {

[[nodiscard]] ModelBackendPtr CreateOrtBackend(
    const std::filesystem::path& model_path,
    const RuntimeOptions& options);
[[nodiscard]] std::vector<std::string> GetAvailableOrtProviders(
    const std::filesystem::path& library_path);

}  // namespace onnx_world_model::detail
