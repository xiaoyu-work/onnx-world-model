#pragma once

#include <filesystem>

#include "onnx_world_model/backend.hpp"
#include "onnx_world_model/model.hpp"

namespace onnx_world_model::detail {

[[nodiscard]] ModelBackendPtr CreateOrtBackend(
    const std::filesystem::path& model_path,
    const RuntimeOptions& options);

}  // namespace onnx_world_model::detail
