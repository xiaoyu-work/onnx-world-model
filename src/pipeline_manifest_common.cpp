/**
 * @agent-file
 * @agent-purpose: Implements the JSON field, token, and portable-name checks shared by pipeline manifest parsing and manifest semantic validation.
 * @agent-public-api: Fail, RequireObject, ValidateToken, ValidateComponentName, Lower, CheckUnique, ParseStringArray
 * @agent-invariants: Fail always throws ErrorCode::pipeline_manifest and never returns, so every other check here either succeeds or throws. ValidateComponentName additionally rejects names that are unsafe as a portable path segment, including Windows reserved device stems. ParseStringArray requires non-empty unique string entries.
 * @agent-side-effects: none
 */

#include "pipeline_manifest_common.hpp"

#include <algorithm>
#include <cctype>
#include <unordered_set>
#include <utility>

#include "onnx_world_model/error.hpp"

namespace onnx_world_model::detail {

[[noreturn]] void Fail(std::string message) {
  throw Error(ErrorCode::pipeline_manifest, std::move(message));
}

void RequireObject(const Json& value, std::string_view context) {
  if (!value.is_object()) {
    Fail(std::string(context) + " must be a JSON object");
  }
}

void ValidateToken(std::string_view value, std::string_view context) {
  if (value.empty()) {
    Fail(std::string(context) + " must be non-empty");
  }
  if (std::ranges::any_of(value, [](unsigned char character) {
        return std::iscntrl(character) != 0;
      })) {
    Fail(std::string(context) + " contains a control character");
  }
}

void ValidateComponentName(std::string_view value) {
  ValidateToken(value, "Component name");
  if (value.front() == ' ' || value.back() == ' ' ||
      value == "." || value == ".." || value.find("..") != std::string_view::npos ||
      value.find_first_of("./\\:*?\"<>|") != std::string_view::npos) {
    Fail(
        "Component name '" + std::string(value) +
        "' is not a safe portable path segment");
  }
  const std::string stem = Lower(value.substr(0, value.find('.')));
  static const std::unordered_set<std::string> reserved{
      "con",  "prn",  "aux",  "nul",  "com1", "com2", "com3",
      "com4", "com5", "com6", "com7", "com8", "com9", "lpt1",
      "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8",
      "lpt9",
  };
  if (reserved.contains(stem)) {
    Fail("Component name '" + std::string(value) + "' is reserved");
  }
}

std::string Lower(std::string_view value) {
  std::string result(value);
  std::ranges::transform(result, result.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return result;
}

void CheckUnique(
    const std::vector<std::string>& values,
    std::string_view context,
    bool case_insensitive) {
  std::unordered_set<std::string> seen;
  for (const auto& value : values) {
    const std::string key = case_insensitive ? Lower(value) : value;
    if (!seen.insert(key).second) {
      Fail(std::string(context) + " '" + value + "' is duplicated");
    }
  }
}

std::vector<std::string> ParseStringArray(
    const Json& value,
    std::string_view context) {
  if (!value.is_array()) {
    Fail(std::string(context) + " must be an array");
  }
  std::vector<std::string> result;
  result.reserve(value.size());
  for (const auto& item : value) {
    if (!item.is_string() || item.get_ref<const std::string&>().empty()) {
      Fail(std::string(context) + " entries must be non-empty strings");
    }
    result.push_back(item.get<std::string>());
  }
  CheckUnique(result, context);
  return result;
}

}  // namespace onnx_world_model::detail
