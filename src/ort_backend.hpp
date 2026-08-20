/**
 * @agent-file
 * @agent-purpose: Declares the factory that builds an ONNX Runtime backed ModelBackend and the query for execution providers a given ORT library offers.
 * @agent-public-api: onnx_world_model::detail::CreateOrtBackend, onnx_world_model::detail::GetAvailableOrtProviders
 * @agent-invariants: Internal header that is not installed; it is the only seam through which the rest of src/ reaches ONNX Runtime, so no other file includes ORT headers except dynamic_library.cpp.
 * @agent-side-effects: none in this header; the declared functions load the ORT library and read model files.
 */

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
