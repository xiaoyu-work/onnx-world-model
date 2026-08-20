/**
 * @agent-file
 * @agent-purpose: Declares the JSON field, token, and portable-name checks shared by pipeline manifest parsing and manifest semantic validation.
 * @agent-public-api: onnx_world_model::detail::Json, Fail, RequireObject, ValidateToken, ValidateComponentName, Lower, CheckUnique, ParseStringArray
 * @agent-invariants: Internal header that is not installed; every check here reports failure by calling Fail, which always throws ErrorCode::pipeline_manifest and never returns.
 * @agent-side-effects: none
 */

#pragma once

#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace onnx_world_model::detail {

using Json = nlohmann::json;

[[noreturn]] void Fail(std::string message);

void RequireObject(const Json& value, std::string_view context);

void ValidateToken(std::string_view value, std::string_view context);

void ValidateComponentName(std::string_view value);

[[nodiscard]] std::string Lower(std::string_view value);

void CheckUnique(
    const std::vector<std::string>& values,
    std::string_view context,
    bool case_insensitive = false);

[[nodiscard]] std::vector<std::string> ParseStringArray(
    const Json& value,
    std::string_view context);

}  // namespace onnx_world_model::detail
