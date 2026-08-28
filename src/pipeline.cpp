/**
 * @agent-file
 * @agent-purpose: Parses pipeline.json into a PipelineManifest and loads a Mobius package: it resolves component files inside the package root and opens their ONNX sessions as a PipelinePackage.
 * @agent-public-api: Endpoint::Parse, Endpoint::qualified, PipelineManifest::Parse, PipelineManifest::Load, PipelineManifest::SupportedCapabilities, PipelineManifest accessors, PipelineManifest::Component, PipelineManifest::Input, PipelineManifest::Output, PipelinePackage::PipelinePackage, PipelinePackage::Load, PipelinePackage accessors
 * @agent-invariants: Only Mobius pipeline schema major version 1 is accepted, and unknown JSON fields are rejected rather than ignored. Every declared path stays inside the package root, and each component ONNX signature must agree with the manifest. Parsing populates the manifest and then hands it to detail::ValidateManifest, so a PipelineManifest that exists has already passed semantic validation. Pipeline is defined in pipeline_session.cpp, not here; it owns component sessions through a shared_ptr so copies share one set of ONNX sessions and one admission scheduler.
 * @agent-side-effects: Reads pipeline.json, component ONNX files, and declared assets from disk; loads the ONNX Runtime shared library and creates one ORT session per component.
 */

#include "onnx_world_model/pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

#include "onnx_world_model/error.hpp"
#include "pipeline_manifest_common.hpp"
#include "pipeline_manifest_validation.hpp"

namespace onnx_world_model {

using detail::CheckUnique;
using detail::Fail;
using detail::FindComponent;
using detail::Json;
using detail::Lower;
using detail::ParseStringArray;
using detail::RequireEndpoint;
using detail::RequireObject;
using detail::SupportedCapabilityNames;
using detail::ValidateComponentName;
using detail::ValidateManifest;
using detail::ValidateToken;

namespace {

void ValidateKeys(
    const Json& value,
    std::initializer_list<std::string_view> allowed,
    std::initializer_list<std::string_view> required,
    std::string_view context) {
  RequireObject(value, context);
  for (const auto& [key, child] : value.items()) {
    (void)child;
    if (std::ranges::find(allowed, key) == allowed.end()) {
      Fail(std::string(context) + " has unknown field '" + key + "'");
    }
  }
  for (const std::string_view key : required) {
    if (!value.contains(key)) {
      Fail(std::string(context) + " is missing required field '" +
           std::string(key) + "'");
    }
  }
}

std::string RequireString(
    const Json& value,
    std::string_view field,
    std::string_view context) {
  const auto found = value.find(field);
  if (found == value.end() || !found->is_string() ||
      found->get_ref<const std::string&>().empty()) {
    Fail(std::string(context) + " field '" + std::string(field) +
         "' must be a non-empty string");
  }
  return found->get<std::string>();
}

int VersionMajor(std::string_view version, std::string_view context) {
  const std::size_t separator = version.find('.');
  if (separator == std::string_view::npos || separator == 0 ||
      separator + 1 == version.size() ||
      version.find('.', separator + 1) != std::string_view::npos) {
    Fail(std::string(context) + " must have the form '<major>.<minor>'");
  }
  const std::string_view major_text = version.substr(0, separator);
  const std::string_view minor_text = version.substr(separator + 1);
  if (!std::ranges::all_of(major_text, [](unsigned char character) {
        return std::isdigit(character) != 0;
      }) ||
      !std::ranges::all_of(minor_text, [](unsigned char character) {
        return std::isdigit(character) != 0;
      })) {
    Fail(std::string(context) + " must have the form '<major>.<minor>'");
  }
  return std::stoi(std::string(major_text));
}

DataType ParseDataType(std::string_view value, std::string_view context) {
  static const std::unordered_map<std::string, DataType> data_types{
      {"FLOAT", DataType::float32},
      {"FLOAT16", DataType::float16},
      {"BFLOAT16", DataType::bfloat16},
      {"DOUBLE", DataType::float64},
      {"INT64", DataType::int64},
      {"INT32", DataType::int32},
      {"INT16", DataType::int16},
      {"INT8", DataType::int8},
      {"UINT64", DataType::uint64},
      {"UINT32", DataType::uint32},
      {"UINT16", DataType::uint16},
      {"UINT8", DataType::uint8},
      {"BOOL", DataType::boolean},
  };
  const auto found = data_types.find(std::string(value));
  if (found == data_types.end()) {
    Fail(std::string(context) + " has unsupported dtype '" +
         std::string(value) + "'");
  }
  return found->second;
}

struct ParsedTensorSpec {
  TensorSpec spec;
  std::vector<std::string> dimension_symbols;
};

ParsedTensorSpec ParseTensorSpec(
    const Json& value,
    std::string_view context) {
  ValidateKeys(
      value,
      {"name", "dtype", "shape"},
      {"name", "dtype", "shape"},
      context);
  const std::string name = RequireString(value, "name", context);
  const std::string dtype = RequireString(value, "dtype", context);
  const Json& shape = value.at("shape");
  if (!shape.is_array()) {
    Fail(std::string(context) + " field 'shape' must be an array");
  }
  std::vector<std::int64_t> dimensions;
  std::vector<std::string> dimension_symbols;
  dimensions.reserve(shape.size());
  dimension_symbols.reserve(shape.size());
  for (const auto& dimension : shape) {
    if (dimension.is_number_integer() && !dimension.is_boolean()) {
      const auto parsed = dimension.get<std::int64_t>();
      if (parsed < 0) {
        Fail(std::string(context) + " has a negative static dimension");
      }
      dimensions.push_back(parsed);
      dimension_symbols.emplace_back();
    } else if (dimension.is_string()) {
      dimensions.push_back(-1);
      dimension_symbols.push_back(dimension.get<std::string>());
    } else if (dimension.is_null()) {
      dimensions.push_back(-1);
      dimension_symbols.emplace_back();
    } else {
      Fail(
          std::string(context) +
          " dimensions must be non-negative integers, strings, or null");
    }
  }
  return {
      .spec =
          {
              .name = name,
              .data_type = ParseDataType(dtype, context),
              .shape = std::move(dimensions),
          },
      .dimension_symbols = std::move(dimension_symbols),
  };
}

struct ParsedTensorSpecs {
  std::vector<TensorSpec> specs;
  std::unordered_map<std::string, std::vector<std::string>> dimension_symbols;
};

ParsedTensorSpecs ParseTensorSpecs(
    const Json& values,
    std::string_view context) {
  if (!values.is_array()) {
    Fail(std::string(context) + " must be an array");
  }
  ParsedTensorSpecs result;
  result.specs.reserve(values.size());
  std::vector<std::string> names;
  names.reserve(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    ParsedTensorSpec parsed = ParseTensorSpec(
        values[index],
        std::string(context) + "[" + std::to_string(index) + "]");
    names.push_back(parsed.spec.name);
    result.dimension_symbols.emplace(
        parsed.spec.name, std::move(parsed.dimension_symbols));
    result.specs.push_back(std::move(parsed.spec));
  }
  CheckUnique(names, context);
  return result;
}

std::filesystem::path ParseSafePath(
    const std::string& value,
    std::string_view context) {
  if (value.empty() || value.find('\\') != std::string::npos) {
    Fail(
        std::string(context) +
        " must be a non-empty '/'-separated relative path");
  }
  const std::filesystem::path path(value);
  if (path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
    Fail(std::string(context) + " must be relative");
  }
  for (const auto& part : path) {
    if (part.empty() || part == "." || part == "..") {
      Fail(std::string(context) + " contains an unsafe path segment");
    }
    const std::string segment = part.string();
    if (segment.front() == ' ' || segment.back() == ' ' ||
        segment.back() == '.' ||
        segment.find_first_of(":*?\"<>|") != std::string::npos ||
        std::ranges::any_of(segment, [](unsigned char character) {
          return std::iscntrl(character) != 0;
        })) {
      Fail(std::string(context) + " contains a non-portable path segment");
    }
    const std::string stem = Lower(segment.substr(0, segment.find('.')));
    static const std::unordered_set<std::string> reserved{
        "con",  "prn",  "aux",  "nul",  "com1", "com2", "com3",
        "com4", "com5", "com6", "com7", "com8", "com9", "lpt1",
        "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8",
        "lpt9",
    };
    if (reserved.contains(stem)) {
      Fail(std::string(context) + " contains a reserved path segment");
    }
  }
  return path;
}

PipelineInputKind ParseInputKind(std::string_view value) {
  if (value == "external") {
    return PipelineInputKind::external;
  }
  if (value == "generated") {
    return PipelineInputKind::generated;
  }
  if (value == "stateful") {
    return PipelineInputKind::stateful;
  }
  if (value == "defaulted") {
    return PipelineInputKind::defaulted;
  }
  Fail("Unknown pipeline input kind '" + std::string(value) + "'");
}

std::filesystem::path ResolveInside(
    const std::filesystem::path& root,
    const std::filesystem::path& relative,
    std::string_view context) {
  try {
    const auto resolved = std::filesystem::canonical(root / relative);
    const auto within = resolved.lexically_relative(root);
    if (within.empty() || within.is_absolute() ||
        *within.begin() == std::filesystem::path("..")) {
      Fail(std::string(context) + " resolves outside the package directory");
    }
    if (!std::filesystem::is_regular_file(resolved)) {
      Fail(std::string(context) + " is not a regular file");
    }
    return resolved;
  } catch (const Error&) {
    throw;
  } catch (const std::filesystem::filesystem_error& error) {
    throw Error(
        ErrorCode::pipeline_manifest,
        std::string(context) + " could not be resolved: " + error.what());
  }
}

void ValidateModelMetadata(
    const PipelineComponent& component,
    const ModelMetadata& actual) {
  const auto validate = [&component](
                            const std::vector<TensorSpec>& expected,
                            const std::vector<TensorSpec>& found,
                            std::string_view kind) {
    if (expected.size() != found.size()) {
      Fail(
          "Component '" + component.name + "' " + std::string(kind) +
          " count does not match pipeline.json");
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
      if (expected[index].name != found[index].name ||
          expected[index].data_type != found[index].data_type ||
          expected[index].shape.size() != found[index].shape.size()) {
        Fail(
            "Component '" + component.name + "' " + std::string(kind) +
            " signature does not match pipeline.json");
      }
      for (std::size_t axis = 0; axis < expected[index].shape.size(); ++axis) {
        const bool expected_dynamic = expected[index].shape[axis] < 0;
        const bool found_dynamic = found[index].shape[axis] < 0;
        if (expected_dynamic != found_dynamic ||
            (!expected_dynamic &&
             expected[index].shape[axis] != found[index].shape[axis])) {
          Fail(
              "Component '" + component.name + "' " + std::string(kind) +
              " shape does not match pipeline.json");
        }
      }
    }
  };
  validate(component.metadata.inputs, actual.inputs, "input");
  validate(component.metadata.outputs, actual.outputs, "output");
}

RuntimeOptions ComponentRuntimeOptions(
    const RuntimeOptions& requested,
    const PipelineComponent& component) {
  RuntimeOptions resolved = requested;
  std::unordered_set<std::string> allowed;
  for (const auto& provider :
       component.preferred_execution_providers) {
    allowed.insert(NormalizeExecutionProviderName(provider));
  }

  const std::vector<std::string>& candidates =
      requested.providers.empty()
          ? component.preferred_execution_providers
          : requested.providers;
  resolved.providers.clear();
  for (const auto& provider : candidates) {
    const std::string normalized =
        NormalizeExecutionProviderName(provider);
    if (allowed.empty() || allowed.contains(normalized)) {
      resolved.providers.push_back(provider);
    }
  }
  // A component's manifest preferences say where it runs best; they are not an
  // allowlist. Keep an explicitly requested CPU provider even when they omit
  // it, because dropping the CPU fallback makes ONNX Runtime refuse to build a
  // session that contains any node the preferred provider does not implement.
  const auto names_cpu = [](const std::vector<std::string>& providers) {
    return std::any_of(
        providers.begin(),
        providers.end(),
        [](const std::string& provider) {
          return NormalizeExecutionProviderName(provider) == "cpu";
        });
  };
  if (names_cpu(requested.providers) && !names_cpu(resolved.providers)) {
    resolved.providers.emplace_back("cpu");
  }
  if (resolved.providers.empty()) {
    if (requested.providers.empty() && allowed.empty()) {
      resolved.providers.emplace_back("cpu");
    } else {
      std::ostringstream message;
      message << "Component '" << component.name
              << "' has no execution provider compatible with ";
      if (requested.providers.empty()) {
        message << "its manifest preferences";
      } else {
        message << "the requested provider order";
      }
      throw Error(ErrorCode::runtime_load, message.str());
    }
  }

  std::unordered_set<std::string> selected;
  for (const auto& provider : resolved.providers) {
    selected.insert(NormalizeExecutionProviderName(provider));
  }
  resolved.provider_options.clear();
  for (const auto& [provider, values] : requested.provider_options) {
    if (selected.contains(NormalizeExecutionProviderName(provider))) {
      resolved.provider_options.emplace(provider, values);
    }
  }
  return resolved;
}

}  // namespace

Endpoint Endpoint::Parse(std::string_view value) {
  const std::size_t separator = value.find('.');
  if (separator == std::string_view::npos || separator == 0 ||
      separator + 1 == value.size()) {
    Fail(
        "Endpoint '" + std::string(value) +
        "' must have the form '<component>.<port>'");
  }
  Endpoint endpoint{
      .component = std::string(value.substr(0, separator)),
      .port = std::string(value.substr(separator + 1)),
  };
  ValidateComponentName(endpoint.component);
  ValidateToken(endpoint.port, "Endpoint port");
  return endpoint;
}

std::string Endpoint::qualified() const {
  return component + "." + port;
}

PipelineManifest PipelineManifest::Parse(std::string_view document) {
  try {
    const Json root = Json::parse(document);
    ValidateKeys(
        root,
        {"format", "schema_version", "manifest", "component_files"},
        {"format", "schema_version", "manifest", "component_files"},
        "pipeline.json");
    if (RequireString(root, "format", "pipeline.json") !=
        "mobius-pipeline") {
      Fail("pipeline.json field 'format' must be 'mobius-pipeline'");
    }

    PipelineManifest result;
    result.schema_version_ =
        RequireString(root, "schema_version", "pipeline.json");
    if (VersionMajor(result.schema_version_, "Pipeline schema version") != 1) {
      Fail(
          "Unsupported pipeline schema version '" + result.schema_version_ +
          "'");
    }

    const Json& manifest = root.at("manifest");
    ValidateKeys(
        manifest,
        {
            "schema_version",
            "components",
            "connections",
            "stages",
            "inputs",
            "outputs",
            "assets",
            "states",
            "profile",
            "required_capabilities",
            "metadata",
        },
        {
            "schema_version",
            "components",
            "connections",
            "stages",
            "inputs",
            "outputs",
        },
        "manifest");
    if (RequireString(manifest, "schema_version", "manifest") !=
        result.schema_version_) {
      Fail("Top-level and manifest schema versions must match");
    }

    if (manifest.contains("profile")) {
      const Json& profile = manifest.at("profile");
      ValidateKeys(
          profile,
          {"name", "version"},
          {"name", "version"},
          "pipeline profile");
      result.profile_ = RequireString(profile, "name", "pipeline profile");
      result.profile_version_ =
          RequireString(profile, "version", "pipeline profile");
      (void)VersionMajor(result.profile_version_, "Profile version");
    }
    if (manifest.contains("metadata")) {
      const Json& metadata = manifest.at("metadata");
      RequireObject(metadata, "manifest metadata");
      if (metadata.contains("model_type")) {
        result.model_type_ =
            RequireString(metadata, "model_type", "manifest metadata");
      }
      result.metadata_json_ = metadata.dump();
    } else {
      result.metadata_json_ = "{}";
    }

    const Json& components = manifest.at("components");
    if (!components.is_array()) {
      Fail("manifest field 'components' must be an array");
    }
    result.components_.reserve(components.size());
    for (std::size_t index = 0; index < components.size(); ++index) {
      const Json& component = components[index];
      const std::string context =
          "component[" + std::to_string(index) + "]";
      ValidateKeys(
          component,
          {
              "name",
              "role",
              "run_on",
              "inputs",
              "outputs",
              "presence",
              "capabilities",
              "preferred_execution_providers",
              "parameter_dtype",
              "source",
              "config",
              "metadata",
          },
          {"name", "role", "run_on", "inputs", "outputs"},
          context);
      ParsedTensorSpecs parsed_inputs =
          ParseTensorSpecs(component.at("inputs"), context + " inputs");
      ParsedTensorSpecs parsed_outputs =
          ParseTensorSpecs(component.at("outputs"), context + " outputs");
      PipelineComponent parsed{
          .name = RequireString(component, "name", context),
          .role = RequireString(component, "role", context),
          .run_on = RequireString(component, "run_on", context),
          .metadata =
              {
                  .inputs = std::move(parsed_inputs.specs),
                  .outputs = std::move(parsed_outputs.specs),
              },
          .input_dimension_symbols =
              std::move(parsed_inputs.dimension_symbols),
          .output_dimension_symbols =
              std::move(parsed_outputs.dimension_symbols),
      };
      if (component.contains("presence")) {
        parsed.presence =
            RequireString(component, "presence", context);
      }
      if (component.contains("capabilities")) {
        parsed.capabilities = ParseStringArray(
            component.at("capabilities"), context + " capabilities");
      }
      if (component.contains("preferred_execution_providers")) {
        parsed.preferred_execution_providers = ParseStringArray(
            component.at("preferred_execution_providers"),
            context + " preferred_execution_providers");
      }
      if (component.contains("parameter_dtype")) {
        parsed.parameter_data_type = ParseDataType(
            RequireString(component, "parameter_dtype", context),
            context);
      }
      if (component.contains("source")) {
        parsed.source = RequireString(component, "source", context);
      }
      if (component.contains("config")) {
        RequireObject(component.at("config"), context + " config");
        parsed.config_json = component.at("config").dump();
      }
      if (component.contains("metadata")) {
        RequireObject(component.at("metadata"), context + " metadata");
        parsed.metadata_json = component.at("metadata").dump();
      }
      result.components_.push_back(std::move(parsed));
    }

    const Json& connections = manifest.at("connections");
    if (!connections.is_array()) {
      Fail("manifest field 'connections' must be an array");
    }
    result.connections_.reserve(connections.size());
    for (std::size_t index = 0; index < connections.size(); ++index) {
      const Json& connection = connections[index];
      const std::string context =
          "connection[" + std::to_string(index) + "]";
      ValidateKeys(
          connection,
          {
              "source",
              "target",
              "recurrent",
              "transform",
              "context",
              "parameters",
          },
          {"source", "target"},
          context);
      PipelineConnection parsed{
          .source = Endpoint::Parse(
              RequireString(connection, "source", context)),
          .target = Endpoint::Parse(
              RequireString(connection, "target", context)),
          .recurrent = connection.value("recurrent", false),
      };
      if (connection.contains("transform")) {
        parsed.transform =
            RequireString(connection, "transform", context);
      }
      if (connection.contains("context")) {
        const auto endpoints =
            ParseStringArray(connection.at("context"), context + " context");
        for (const auto& endpoint : endpoints) {
          parsed.context.push_back(Endpoint::Parse(endpoint));
        }
      }
      if (connection.contains("parameters")) {
        RequireObject(connection.at("parameters"), context + " parameters");
        parsed.parameters_json = connection.at("parameters").dump();
      }
      result.connections_.push_back(std::move(parsed));
    }

    const Json& inputs = manifest.at("inputs");
    if (!inputs.is_array()) {
      Fail("manifest field 'inputs' must be an array");
    }
    result.inputs_.reserve(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      const Json& input = inputs[index];
      const std::string context =
          "input[" + std::to_string(index) + "]";
      ValidateKeys(
          input,
          {
              "port",
              "kind",
              "value",
              "alias",
              "semantic",
              "required",
              "presence",
              "generator",
          },
          {"port", "kind"},
          context);
      PipelineInput parsed{
          .port =
              Endpoint::Parse(RequireString(input, "port", context)),
          .kind = ParseInputKind(RequireString(input, "kind", context)),
          .required = input.value("required", true),
      };
      if (input.contains("alias")) {
        if (parsed.kind != PipelineInputKind::external) {
          Fail(context + " field 'alias' is only valid for external inputs");
        }
        parsed.name = RequireString(input, "alias", context);
      } else {
        parsed.name = parsed.port.port;
      }
      if (input.contains("value")) {
        parsed.value_json = input.at("value").dump();
      }
      if (input.contains("semantic")) {
        parsed.semantic = RequireString(input, "semantic", context);
      }
      if (input.contains("presence")) {
        parsed.presence = RequireString(input, "presence", context);
      }
      if (input.contains("generator")) {
        const Json& generator = input.at("generator");
        ValidateKeys(
            generator,
            {"kind", "parameters"},
            {"kind"},
            context + " generator");
        parsed.generator_kind =
            RequireString(generator, "kind", context + " generator");
        if (generator.contains("parameters")) {
          RequireObject(
              generator.at("parameters"), context + " generator parameters");
          parsed.generator_json = generator.at("parameters").dump();
        } else {
          parsed.generator_json = "{}";
        }
      }
      result.inputs_.push_back(std::move(parsed));
    }

    const Json& outputs = manifest.at("outputs");
    if (!outputs.is_array()) {
      Fail("manifest field 'outputs' must be an array");
    }
    result.outputs_.reserve(outputs.size());
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const Json& output = outputs[index];
      const std::string context =
          "output[" + std::to_string(index) + "]";
      ValidateKeys(
          output,
          {"port", "state", "alias"},
          {},
          context);
      const bool has_port = output.contains("port");
      const bool has_state = output.contains("state");
      if (has_port == has_state) {
        Fail(context + " must reference exactly one component port or state");
      }
      PipelineOutput parsed;
      if (has_port) {
        parsed.port =
            Endpoint::Parse(RequireString(output, "port", context));
        parsed.name = parsed.port->port;
      } else {
        parsed.state = RequireString(output, "state", context);
        parsed.name = *parsed.state;
      }
      if (output.contains("alias")) {
        parsed.name = RequireString(output, "alias", context);
      }
      result.outputs_.push_back(std::move(parsed));
    }

    const Json& stages = manifest.at("stages");
    if (!stages.is_array()) {
      Fail("manifest field 'stages' must be an array");
    }
    result.stages_.reserve(stages.size());
    for (std::size_t index = 0; index < stages.size(); ++index) {
      const Json& stage = stages[index];
      const std::string context =
          "stage[" + std::to_string(index) + "]";
      ValidateKeys(
          stage,
          {
              "name",
              "kind",
              "components",
              "run_on",
              "options",
              "capabilities",
              "metadata",
          },
          {"name", "kind", "components", "run_on"},
          context);
      PipelineStage parsed{
          .name = RequireString(stage, "name", context),
          .kind = RequireString(stage, "kind", context),
          .components = ParseStringArray(
              stage.at("components"), context + " components"),
          .run_on = RequireString(stage, "run_on", context),
      };
      if (stage.contains("options")) {
        RequireObject(stage.at("options"), context + " options");
        parsed.options_json = stage.at("options").dump();
      }
      if (stage.contains("capabilities")) {
        parsed.capabilities = ParseStringArray(
            stage.at("capabilities"), context + " capabilities");
      }
      result.stages_.push_back(std::move(parsed));
    }

    if (manifest.contains("states")) {
      const Json& states = manifest.at("states");
      if (!states.is_array()) {
        Fail("manifest field 'states' must be an array");
      }
      result.states_.reserve(states.size());
      for (std::size_t index = 0; index < states.size(); ++index) {
        const Json& state = states[index];
        const std::string context =
            "state[" + std::to_string(index) + "]";
        ValidateKeys(
            state,
            {
                "name",
                "kind",
                "input",
                "output",
                "lifetime",
                "release_after",
                "sequence_axis",
                "metadata",
            },
            {
                "name",
                "kind",
                "input",
                "output",
                "lifetime",
                "release_after",
            },
            context);
        PipelineState parsed{
            .name = RequireString(state, "name", context),
            .kind = RequireString(state, "kind", context),
            .input = Endpoint::Parse(RequireString(state, "input", context)),
            .output = Endpoint::Parse(RequireString(state, "output", context)),
            .lifetime = RequireString(state, "lifetime", context),
            .release_after =
                RequireString(state, "release_after", context),
        };
        if (state.contains("sequence_axis")) {
          if (!state.at("sequence_axis").is_number_unsigned() &&
              !(state.at("sequence_axis").is_number_integer() &&
                state.at("sequence_axis").get<std::int64_t>() >= 0)) {
            Fail(context + " sequence_axis must be a non-negative integer");
          }
          parsed.sequence_axis =
              state.at("sequence_axis").get<std::size_t>();
        }
        if (state.contains("metadata")) {
          RequireObject(state.at("metadata"), context + " metadata");
          parsed.metadata_json = state.at("metadata").dump();
        }
        result.states_.push_back(std::move(parsed));
      }
    }

    if (manifest.contains("assets")) {
      const Json& assets = manifest.at("assets");
      if (!assets.is_array()) {
        Fail("manifest field 'assets' must be an array");
      }
      for (std::size_t index = 0; index < assets.size(); ++index) {
        const Json& asset = assets[index];
        const std::string context =
            "asset[" + std::to_string(index) + "]";
        ValidateKeys(
            asset,
            {"path", "required"},
            {"path"},
            context);
        result.assets_.push_back({
            .path = ParseSafePath(
                RequireString(asset, "path", context), context),
            .required = asset.value("required", true),
        });
      }
    }
    if (manifest.contains("required_capabilities")) {
      result.required_capabilities_ = ParseStringArray(
          manifest.at("required_capabilities"),
          "required_capabilities");
    }

    const Json& component_files = root.at("component_files");
    RequireObject(component_files, "component_files");
    for (const auto& [name, path] : component_files.items()) {
      if (!path.is_string()) {
        Fail("component_files values must be strings");
      }
      result.component_files_.emplace(
          name,
          ParseSafePath(path.get<std::string>(), "component_files." + name));
    }

    ValidateManifest(result);
    return result;
  } catch (const Error&) {
    throw;
  } catch (const nlohmann::json::exception& error) {
    throw Error(
        ErrorCode::pipeline_manifest,
        "Could not parse pipeline.json: " + std::string(error.what()));
  } catch (const std::exception& error) {
    throw Error(
        ErrorCode::pipeline_manifest,
        "Invalid pipeline.json: " + std::string(error.what()));
  }
}

PipelineManifest PipelineManifest::Load(
    const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw Error(
        ErrorCode::pipeline_manifest,
        "Could not open pipeline manifest: " + path.string());
  }
  std::ostringstream document;
  document << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    throw Error(
        ErrorCode::pipeline_manifest,
        "Could not read pipeline manifest: " + path.string());
  }
  return Parse(document.str());
}

std::vector<std::string> PipelineManifest::SupportedCapabilities() {
  const auto& capabilities = SupportedCapabilityNames();
  return {capabilities.begin(), capabilities.end()};
}

std::string_view PipelineManifest::schema_version() const noexcept {
  return schema_version_;
}

std::string_view PipelineManifest::profile() const noexcept {
  return profile_;
}

std::string_view PipelineManifest::profile_version() const noexcept {
  return profile_version_;
}

std::string_view PipelineManifest::model_type() const noexcept {
  return model_type_;
}

std::string_view PipelineManifest::metadata_json() const noexcept {
  return metadata_json_;
}

const std::vector<PipelineComponent>& PipelineManifest::components()
    const noexcept {
  return components_;
}

const std::vector<PipelineConnection>& PipelineManifest::connections()
    const noexcept {
  return connections_;
}

const std::vector<PipelineInput>& PipelineManifest::inputs() const noexcept {
  return inputs_;
}

const std::vector<PipelineOutput>& PipelineManifest::outputs() const noexcept {
  return outputs_;
}

const std::vector<PipelineStage>& PipelineManifest::stages() const noexcept {
  return stages_;
}

const std::vector<PipelineState>& PipelineManifest::states() const noexcept {
  return states_;
}

const std::vector<PipelineAsset>& PipelineManifest::assets() const noexcept {
  return assets_;
}

const std::vector<std::string>&
PipelineManifest::required_capabilities() const noexcept {
  return required_capabilities_;
}

const std::unordered_map<std::string, std::filesystem::path>&
PipelineManifest::component_files() const noexcept {
  return component_files_;
}

const PipelineComponent& PipelineManifest::Component(
    std::string_view name) const {
  const PipelineComponent* component = FindComponent(components_, name);
  if (component == nullptr) {
    throw Error(
        ErrorCode::pipeline_manifest,
        "Pipeline has no component '" + std::string(name) + "'");
  }
  return *component;
}

const TensorSpec& PipelineManifest::Input(const Endpoint& endpoint) const {
  return RequireEndpoint(components_, endpoint, true, "Pipeline input");
}

const TensorSpec& PipelineManifest::Output(const Endpoint& endpoint) const {
  return RequireEndpoint(components_, endpoint, false, "Pipeline output");
}

PipelinePackage::PipelinePackage(
    std::filesystem::path root,
    PipelineManifest manifest,
    std::unordered_map<std::string, Model> components)
    : root_(std::move(root)),
      manifest_(std::move(manifest)),
      components_(std::move(components)) {
  if (components_.size() != manifest_.components().size()) {
    throw Error(
        ErrorCode::pipeline_manifest,
        "Loaded component sessions do not match pipeline.json");
  }
  for (const auto& component : manifest_.components()) {
    const auto found = components_.find(component.name);
    if (found == components_.end()) {
      throw Error(
          ErrorCode::pipeline_manifest,
          "Component session '" + component.name + "' is missing");
    }
    ValidateModelMetadata(component, found->second.metadata());
  }
}

PipelinePackage PipelinePackage::Load(
    const std::filesystem::path& directory,
    const RuntimeOptions& options) {
  if (!std::filesystem::is_directory(directory)) {
    throw Error(
        ErrorCode::invalid_argument,
        "Pipeline package directory does not exist: " + directory.string());
  }
  const std::filesystem::path root = std::filesystem::canonical(directory);
  const std::filesystem::path manifest_path =
      ResolveInside(root, "pipeline.json", "pipeline.json");
  PipelineManifest manifest = PipelineManifest::Load(manifest_path);

  std::unordered_map<std::string, Model> components;
  components.reserve(manifest.components().size());
  for (const auto& component : manifest.components()) {
    const auto model_path = ResolveInside(
        root,
        manifest.component_files().at(component.name),
        "Component '" + component.name + "'");
    components.emplace(
        component.name,
        Model::Load(
            model_path,
            ComponentRuntimeOptions(options, component)));
  }
  for (const auto& asset : manifest.assets()) {
    const auto path = root / asset.path;
    if (asset.required) {
      (void)ResolveInside(
          root, asset.path, "Required asset '" + asset.path.generic_string() + "'");
    } else if (std::filesystem::exists(path)) {
      (void)ResolveInside(
          root, asset.path, "Optional asset '" + asset.path.generic_string() + "'");
    }
  }
  return PipelinePackage(root, std::move(manifest), std::move(components));
}

const std::filesystem::path& PipelinePackage::root() const noexcept {
  return root_;
}

const PipelineManifest& PipelinePackage::manifest() const noexcept {
  return manifest_;
}

const Model& PipelinePackage::Component(std::string_view name) const {
  const auto found = components_.find(std::string(name));
  if (found == components_.end()) {
    throw Error(
        ErrorCode::pipeline_manifest,
        "Pipeline has no loaded component '" + std::string(name) + "'");
  }
  return found->second;
}

std::unordered_map<std::string, std::vector<std::string>>
PipelinePackage::execution_providers() const {
  std::unordered_map<std::string, std::vector<std::string>> result;
  result.reserve(components_.size());
  for (const auto& [name, component] : components_) {
    result.emplace(name, component.metadata().execution_providers);
  }
  return result;
}

}  // namespace onnx_world_model
