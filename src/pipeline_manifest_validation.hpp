/**
 * @agent-file
 * @agent-purpose: Declares the semantic validation pass that checks an already-parsed PipelineManifest, plus the component and endpoint lookups its callers reuse.
 * @agent-public-api: onnx_world_model::detail::FindComponent, RequireEndpoint, SupportedCapabilityNames, ValidateManifest
 * @agent-invariants: Internal header that is not installed; ValidateManifest runs after PipelineManifest::Parse has populated the manifest and reports every violation as ErrorCode::pipeline_manifest. SupportedCapabilityNames is the single source for PipelineManifest::SupportedCapabilities().
 * @agent-side-effects: none
 */

#pragma once

#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "onnx_world_model/pipeline.hpp"

namespace onnx_world_model::detail {

[[nodiscard]] const PipelineComponent* FindComponent(
    const std::vector<PipelineComponent>& components,
    std::string_view name);

[[nodiscard]] const TensorSpec& RequireEndpoint(
    const std::vector<PipelineComponent>& components,
    const Endpoint& endpoint,
    bool input,
    std::string_view context);

[[nodiscard]] const std::set<std::string>& SupportedCapabilityNames();

void ValidateManifest(const PipelineManifest& manifest);

}  // namespace onnx_world_model::detail
