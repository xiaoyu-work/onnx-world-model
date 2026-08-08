#include "onnx_world_model/pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

#include "onnx_world_model/error.hpp"

namespace onnx_world_model {
namespace {

using Json = nlohmann::json;

[[noreturn]] void Fail(std::string message) {
  throw Error(ErrorCode::pipeline_manifest, std::move(message));
}

void RequireObject(const Json& value, std::string_view context) {
  if (!value.is_object()) {
    Fail(std::string(context) + " must be a JSON object");
  }
}

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

std::string Lower(std::string_view value);

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
    bool case_insensitive = false) {
  std::unordered_set<std::string> seen;
  for (const auto& value : values) {
    const std::string key = case_insensitive ? Lower(value) : value;
    if (!seen.insert(key).second) {
      Fail(std::string(context) + " '" + value + "' is duplicated");
    }
  }
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

bool ShapesCompatible(const TensorSpec& source, const TensorSpec& target) {
  if (source.data_type != target.data_type ||
      source.shape.size() != target.shape.size()) {
    return false;
  }
  for (std::size_t axis = 0; axis < source.shape.size(); ++axis) {
    if (source.shape[axis] >= 0 && target.shape[axis] >= 0 &&
        source.shape[axis] != target.shape[axis]) {
      return false;
    }
  }
  return true;
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

const TensorSpec* FindSpec(
    const std::vector<TensorSpec>& specs,
    std::string_view name) {
  const auto found = std::ranges::find(specs, name, &TensorSpec::name);
  return found == specs.end() ? nullptr : &*found;
}

const PipelineComponent* FindComponent(
    const std::vector<PipelineComponent>& components,
    std::string_view name) {
  const auto found =
      std::ranges::find(components, name, &PipelineComponent::name);
  return found == components.end() ? nullptr : &*found;
}

const TensorSpec& RequireEndpoint(
    const std::vector<PipelineComponent>& components,
    const Endpoint& endpoint,
    bool input,
    std::string_view context) {
  const PipelineComponent* component =
      FindComponent(components, endpoint.component);
  if (component == nullptr) {
    Fail(
        std::string(context) + " references unknown component '" +
        endpoint.component + "'");
  }
  const auto& specs = input ? component->metadata.inputs
                            : component->metadata.outputs;
  const TensorSpec* spec = FindSpec(specs, endpoint.port);
  if (spec == nullptr) {
    Fail(
        std::string(context) + " references unknown " +
        (input ? "input" : "output") + " '" + endpoint.qualified() + "'");
  }
  return *spec;
}

void RequireContextEndpoint(
    const std::vector<PipelineComponent>& components,
    const Endpoint& endpoint,
    std::string_view context) {
  const PipelineComponent* component =
      FindComponent(components, endpoint.component);
  if (component == nullptr ||
      (FindSpec(component->metadata.inputs, endpoint.port) == nullptr &&
       FindSpec(component->metadata.outputs, endpoint.port) == nullptr)) {
    Fail(
        std::string(context) + " references unknown context endpoint '" +
        endpoint.qualified() + "'");
  }
}

struct ProgramDefinition {
  std::set<std::string> capabilities;
  std::set<std::string> required;
  std::set<std::string> allowed;
};

const ProgramDefinition& TransformDefinition(std::string_view kind) {
  static const std::unordered_map<std::string, ProgramDefinition> definitions{
      {"cast", {{"tensor_cast"}, {}, {"to"}}},
      {
          "reshape",
          {
              {"tensor_reshape"},
              {},
              {"shape", "input_layout", "output_layout"},
          },
      },
      {
          "normalize",
          {
              {"tensor_normalize"},
              {},
              {"mean", "std", "scale", "shift"},
          },
      },
      {
          "sample",
          {
              {"stochastic_sampling"},
              {},
              {"distribution", "seed_input"},
          },
      },
      {
          "patchify",
          {
              {"tensor_patchify"},
              {},
              {
                  "spatial_patch_size",
                  "temporal_patch_size",
                  "input_layout",
                  "output_layout",
                  "channel_order",
              },
          },
      },
      {
          "unpatchify",
          {
              {"tensor_patchify"},
              {},
              {
                  "spatial_patch_size",
                  "temporal_patch_size",
                  "input_layout",
                  "output_layout",
                  "channel_order",
              },
          },
      },
      {
          "scheduler_step",
          {
              {"iterative_scheduler"},
              {},
              {"scheduler_asset", "stage", "state", "timestep_input"},
          },
      },
      {"concat", {{"tensor_concat"}, {}, {"axis"}}},
      {
          "slice",
          {
              {"tensor_slice"},
              {},
              {"axes", "starts", "ends", "steps"},
          },
      },
      {
          "video_diffusion_finalize",
          {
              {
                  "iterative_scheduler",
                  "tensor_cast",
                  "tensor_patchify",
                  "tensor_reshape",
              },
              {
                  "scheduler_asset",
                  "state",
                  "spatial_patch_size",
                  "latent_channels",
                  "input_layout",
                  "output_layout",
                  "source_dtype",
                  "target_dtype",
              },
              {
                  "scheduler_asset",
                  "state",
                  "spatial_patch_size",
                  "latent_channels",
                  "input_layout",
                  "output_layout",
                  "source_dtype",
                  "target_dtype",
              },
          },
      },
      {
          "audio_diffusion_finalize",
          {
              {"iterative_scheduler", "tensor_cast", "tensor_reshape"},
              {
                  "scheduler_asset",
                  "state",
                  "input_layout",
                  "output_layout",
                  "source_dtype",
                  "target_dtype",
              },
              {
                  "scheduler_asset",
                  "state",
                  "input_layout",
                  "output_layout",
                  "source_dtype",
                  "target_dtype",
              },
          },
      },
  };
  const auto found = definitions.find(std::string(kind));
  if (found == definitions.end()) {
    Fail("Unknown pipeline transform '" + std::string(kind) + "'");
  }
  return found->second;
}

const ProgramDefinition& GeneratorDefinition(std::string_view kind) {
  static const std::unordered_map<std::string, ProgramDefinition> definitions{
      {
          "empty_tensor",
          {{}, {}, {"shape", "dynamic_axes", "fill"}},
      },
      {
          "zeros",
          {{}, {}, {"shape", "shape_from", "dtype"}},
      },
      {
          "causal_attention_mask",
          {
              {"attention_mask_program"},
              {"sequence_input"},
              {
                  "sequence_input",
                  "past_state",
                  "visible_value",
                  "masked_value",
              },
          },
      },
      {
          "multimodal_position_ids",
          {
              {"position_program"},
              {"source", "axes"},
              {
                  "source",
                  "axes",
                  "mrope_sections",
                  "temporal_margin",
                  "reset_spatial",
                  "past_state",
              },
          },
      },
      {
          "packed_sequence_layout",
          {
              {"packed_sequence_program"},
              {"modality"},
              {
                  "modality",
                  "source",
                  "layout",
                  "understanding_prefix",
                  "index_kind",
              },
          },
      },
      {
          "scheduler_timesteps",
          {
              {"iterative_scheduler"},
              {"stage"},
              {"stage", "modality"},
          },
      },
      {
          "action_domain_ids",
          {
              {"action_domain_program"},
              {"domain_input"},
              {
                  "domain_input",
                  "default",
                  "domain_map",
                  "padded_dimension",
              },
          },
      },
  };
  const auto found = definitions.find(std::string(kind));
  if (found == definitions.end()) {
    Fail("Unknown generated input program '" + std::string(kind) + "'");
  }
  return found->second;
}

void ValidateProgramParameters(
    const Json& parameters,
    const ProgramDefinition& definition,
    std::string_view context) {
  RequireObject(parameters, context);
  std::set<std::string> actual;
  for (const auto& [name, value] : parameters.items()) {
    (void)value;
    actual.insert(name);
    if (!definition.allowed.contains(name)) {
      Fail(
          std::string(context) + " has unknown parameter '" + name + "'");
    }
  }
  for (const auto& required : definition.required) {
    if (!actual.contains(required)) {
      Fail(
          std::string(context) + " is missing required parameter '" +
          required + "'");
    }
  }
}

const std::set<std::string>& AllowedStageOptions(std::string_view kind) {
  static const std::unordered_map<std::string, std::set<std::string>> options{
      {"single_pass", {}},
      {
          "autoregressive",
          {
              "tokenizer_asset",
              "sampling",
              "stop",
              "max_tokens",
              "state_names",
          },
      },
      {
          "iterative",
          {
              "scheduler",
              "default_steps",
              "timestep",
              "state_inputs",
              "initial_state_inputs",
              "prediction_type",
              "packed_modalities",
          },
      },
      {
          "state_transition",
          {"state_names", "max_steps", "stop"},
      },
      {"composite", {"stages"}},
      {"on_demand", {"presence"}},
  };
  const auto found = options.find(std::string(kind));
  if (found == options.end()) {
    Fail("Unknown stage kind '" + std::string(kind) + "'");
  }
  return found->second;
}

void ValidateStageOptions(const PipelineStage& stage) {
  const Json options = Json::parse(stage.options_json);
  const auto& allowed = AllowedStageOptions(stage.kind);
  for (const auto& [name, value] : options.items()) {
    (void)value;
    if (!allowed.contains(name)) {
      Fail(
          "Stage '" + stage.name + "' has unknown option '" + name + "'");
    }
  }
}

void ValidateManifest(const PipelineManifest& manifest) {
  const auto& components = manifest.components();
  if (components.empty()) {
    Fail("Pipeline manifest must contain at least one component");
  }

  std::vector<std::string> component_names;
  component_names.reserve(components.size());
  for (const auto& component : components) {
    ValidateComponentName(component.name);
    ValidateToken(component.role, "Component role");
    ValidateToken(component.run_on, "Component run_on");
    static const std::unordered_set<std::string> roles{
        "encoder",     "decoder", "embedding", "projector", "dynamics",
        "observation", "action",  "policy",    "value",     "reward",
        "sampler",     "transform", "generic",
    };
    static const std::unordered_set<std::string> phases{
        "always", "init", "warmup", "prefill", "decode",
        "step",   "refine", "finalize", "on_demand",
    };
    if (!roles.contains(component.role)) {
      Fail(
          "Component '" + component.name + "' has unknown role '" +
          component.role + "'");
    }
    if (!phases.contains(component.run_on)) {
      Fail(
          "Component '" + component.name + "' has unknown run_on phase '" +
          component.run_on + "'");
    }
    component_names.push_back(component.name);
  }
  CheckUnique(component_names, "Component name", true);

  static const std::unordered_set<std::string> stage_kinds{
      "single_pass",
      "autoregressive",
      "iterative",
      "state_transition",
      "composite",
      "on_demand",
  };
  std::vector<std::string> stage_names;
  std::unordered_set<std::string> staged_components;
  for (const auto& stage : manifest.stages()) {
    ValidateToken(stage.name, "Stage name");
    if (!stage_kinds.contains(stage.kind)) {
      Fail("Unknown stage kind '" + stage.kind + "'");
    }
    static const std::unordered_set<std::string> phases{
        "always", "init", "warmup", "prefill", "decode",
        "step",   "refine", "finalize", "on_demand",
    };
    if (!phases.contains(stage.run_on)) {
      Fail(
          "Stage '" + stage.name + "' has unknown run_on phase '" +
          stage.run_on + "'");
    }
    if (stage.components.empty()) {
      Fail("Stage '" + stage.name + "' must contain at least one component");
    }
    for (const auto& component : stage.components) {
      const PipelineComponent* definition =
          FindComponent(components, component);
      if (definition == nullptr) {
        Fail(
            "Stage '" + stage.name + "' references unknown component '" +
            component + "'");
      }
      if (stage.run_on != "always" && definition->run_on != "always" &&
          stage.run_on != definition->run_on) {
        Fail(
            "Component '" + component + "' and stage '" + stage.name +
            "' have incompatible run_on phases");
      }
      staged_components.insert(component);
    }
    if (stage.kind == "on_demand") {
      const bool has_presence =
          std::ranges::any_of(stage.components, [&components](const auto& name) {
            return FindComponent(components, name)->presence.has_value();
          });
      if (!has_presence) {
        Fail(
            "On-demand stage '" + stage.name +
            "' has no component presence condition");
      }
    }
    ValidateStageOptions(stage);
    stage_names.push_back(stage.name);
  }
  CheckUnique(stage_names, "Stage name");
  for (const auto& component : components) {
    if (!staged_components.contains(component.name)) {
      Fail(
          "Component '" + component.name +
          "' belongs to no declared stage");
    }
  }

  std::unordered_set<std::string> initial_producers;
  std::unordered_set<std::string> recurrent_producers;
  std::unordered_map<std::string, std::vector<std::string>> component_edges;
  std::set<std::string> transform_capabilities;
  for (const auto& connection : manifest.connections()) {
    const TensorSpec& source = RequireEndpoint(
        components, connection.source, false, "Connection");
    const TensorSpec& target = RequireEndpoint(
        components, connection.target, true, "Connection");
    for (const auto& endpoint : connection.context) {
      RequireContextEndpoint(components, endpoint, "Connection");
    }
    if (!connection.transform.has_value() &&
        !ShapesCompatible(source, target)) {
      Fail(
          "Connection '" + connection.source.qualified() + "' -> '" +
          connection.target.qualified() +
          "' has incompatible dtype, rank, or static dimensions");
    }
    if (!connection.transform.has_value() &&
        connection.parameters_json != "{}") {
      Fail("Connection parameters require a transform");
    }
    if (!connection.transform.has_value() && !connection.context.empty()) {
      Fail("Connection context requires a transform");
    }
    auto& producers =
        connection.recurrent ? recurrent_producers : initial_producers;
    if (!producers.insert(connection.target.qualified()).second) {
      Fail(
          "Input '" + connection.target.qualified() +
          "' has more than one producer for the same lifecycle");
    }
    if (connection.transform.has_value()) {
      const auto& definition = TransformDefinition(*connection.transform);
      transform_capabilities.insert(
          definition.capabilities.begin(), definition.capabilities.end());
      ValidateProgramParameters(
          Json::parse(connection.parameters_json),
          definition,
          "Transform '" + *connection.transform + "'");
    }

    if (connection.recurrent) {
      bool scoped = false;
      for (const auto& stage : manifest.stages()) {
        const bool looping =
            stage.kind == "autoregressive" || stage.kind == "iterative" ||
            stage.kind == "state_transition";
        if (looping &&
            std::ranges::find(stage.components, connection.source.component) !=
                stage.components.end() &&
            std::ranges::find(stage.components, connection.target.component) !=
                stage.components.end()) {
          if (std::ranges::find(
                  stage.capabilities, "loop_carried_state") ==
              stage.capabilities.end()) {
            Fail(
                "Looping stage '" + stage.name +
                "' owns recurrent state but does not provide "
                "'loop_carried_state'");
          }
          scoped = true;
          break;
        }
      }
      if (!scoped) {
        Fail(
            "Recurrent connection '" + connection.source.qualified() +
            "' -> '" + connection.target.qualified() +
            "' is not scoped to a shared looping stage");
      }
    } else {
      component_edges[connection.source.component].push_back(
          connection.target.component);
    }
  }

  std::unordered_map<std::string, int> visit_state;
  std::function<void(const std::string&)> visit =
      [&](const std::string& component) {
        int& state = visit_state[component];
        if (state == 1) {
          Fail(
              "Non-recurrent pipeline connections contain a cycle through "
              "component '" +
              component + "'");
        }
        if (state == 2) {
          return;
        }
        state = 1;
        for (const auto& target : component_edges[component]) {
          visit(target);
        }
        state = 2;
      };
  for (const auto& component : components) {
    visit(component.name);
  }

  std::unordered_set<std::string> declared_inputs;
  std::vector<std::string> external_names;
  std::set<std::string> generated_capabilities;
  for (const auto& input : manifest.inputs()) {
    (void)RequireEndpoint(components, input.port, true, "Pipeline input");
    if (!declared_inputs.insert(input.port.qualified()).second) {
      Fail("Input '" + input.port.qualified() + "' is declared more than once");
    }
    if (input.kind == PipelineInputKind::external) {
      external_names.push_back(input.name);
      if (!input.generator_json.empty() || !input.value_json.empty()) {
        Fail(
            "External input '" + input.port.qualified() +
            "' cannot define generator or value");
      }
    } else if (input.kind == PipelineInputKind::generated) {
      if (input.generator_kind.empty() || input.generator_json.empty() ||
          !input.value_json.empty()) {
        Fail(
            "Generated input '" + input.port.qualified() +
            "' must define only an executable generator recipe");
      }
      const auto& definition = GeneratorDefinition(input.generator_kind);
      generated_capabilities.insert(
          definition.capabilities.begin(), definition.capabilities.end());
      ValidateProgramParameters(
          Json::parse(input.generator_json),
          definition,
          "Generated input '" + input.port.qualified() + "'");
    } else if (input.kind == PipelineInputKind::stateful) {
      if (!input.generator_json.empty() || !input.value_json.empty()) {
        Fail(
            "Stateful input '" + input.port.qualified() +
            "' cannot define a generator or value");
      }
    } else if (input.kind == PipelineInputKind::defaulted) {
      if (input.value_json.empty() || !input.generator_json.empty()) {
        Fail(
            "Defaulted input '" + input.port.qualified() +
            "' must define only a tensor value");
      }
    }
  }
  CheckUnique(external_names, "External input alias");

  for (const auto& component : components) {
    for (const auto& input : component.metadata.inputs) {
      const std::string endpoint = component.name + "." + input.name;
      const int initial_count =
          (initial_producers.contains(endpoint) ? 1 : 0) +
          (declared_inputs.contains(endpoint) ? 1 : 0);
      if (initial_count != 1) {
        Fail(
            "Component input '" + endpoint +
            "' must have exactly one initial source");
      }
    }
  }

  std::vector<std::string> output_names;
  for (const auto& output : manifest.outputs()) {
    if (output.port.has_value()) {
      (void)RequireEndpoint(
          components, *output.port, false, "Pipeline output");
    } else if (output.state.has_value()) {
      const auto found = std::ranges::find(
          manifest.states(), *output.state, &PipelineState::name);
      if (found == manifest.states().end()) {
        Fail(
            "Pipeline output references unknown state '" + *output.state + "'");
      }
    } else {
      Fail("Pipeline output must reference a component port or state");
    }
    output_names.push_back(output.name);
  }
  CheckUnique(output_names, "Pipeline output alias");

  std::unordered_set<std::string> stage_name_set(
      stage_names.begin(), stage_names.end());
  std::set<std::pair<std::string, std::string>> recurrent_edges;
  for (const auto& connection : manifest.connections()) {
    if (connection.recurrent) {
      recurrent_edges.emplace(
          connection.source.qualified(), connection.target.qualified());
    }
  }
  std::set<std::pair<std::string, std::string>> declared_states;
  std::vector<std::string> state_names;
  std::unordered_set<std::string> state_inputs;
  static const std::unordered_set<std::string> state_kinds{
      "kv_cache",
      "diffusion_latent",
      "action_state",
      "recurrent",
  };
  static const std::unordered_set<std::string> state_lifetimes{
      "iteration",
      "sequence",
      "request",
      "session",
  };
  for (const auto& state : manifest.states()) {
    state_names.push_back(state.name);
    if (!state_kinds.contains(state.kind)) {
      Fail("State '" + state.name + "' has unknown kind '" + state.kind + "'");
    }
    if (!state_lifetimes.contains(state.lifetime)) {
      Fail(
          "State '" + state.name + "' has unknown lifetime '" +
          state.lifetime + "'");
    }
    const TensorSpec& input =
        RequireEndpoint(components, state.input, true, "State");
    (void)RequireEndpoint(components, state.output, false, "State");
    const auto edge =
        std::make_pair(state.output.qualified(), state.input.qualified());
    if (!recurrent_edges.contains(edge)) {
      Fail(
          "State '" + state.name +
          "' does not match a recurrent connection");
    }
    if (!declared_states.insert(edge).second) {
      Fail("Recurrent state edge is declared more than once");
    }
    if (!state_inputs.insert(state.input.qualified()).second) {
      Fail("State input '" + state.input.qualified() + "' is duplicated");
    }
    if (!stage_name_set.contains(state.release_after)) {
      Fail(
          "State '" + state.name + "' releases after unknown stage '" +
          state.release_after + "'");
    }
    if (state.sequence_axis.has_value() &&
        *state.sequence_axis >= input.shape.size()) {
      Fail(
          "State '" + state.name + "' sequence_axis is outside input rank");
    }
  }
  CheckUnique(state_names, "State name");
  if (!manifest.profile().empty() && declared_states != recurrent_edges) {
    Fail("Every recurrent connection requires an explicit state lifecycle");
  }

  const std::set<std::string> declared_capabilities(
      manifest.required_capabilities().begin(),
      manifest.required_capabilities().end());
  if (!recurrent_producers.empty() &&
      !declared_capabilities.contains("loop_carried_state")) {
    Fail(
        "Pipeline has recurrent connections but does not require "
        "'loop_carried_state'");
  }
  for (const auto& capability : transform_capabilities) {
    if (!declared_capabilities.contains(capability)) {
      Fail(
          "Pipeline transform requires undeclared capability '" + capability +
          "'");
    }
  }
  for (const auto& capability : generated_capabilities) {
    if (!declared_capabilities.contains(capability)) {
      Fail(
          "Generated input requires undeclared capability '" + capability +
          "'");
    }
  }
  std::set<std::string> provided_capabilities = transform_capabilities;
  provided_capabilities.insert(
      generated_capabilities.begin(), generated_capabilities.end());
  for (const auto& component : components) {
    provided_capabilities.insert(
        component.capabilities.begin(), component.capabilities.end());
  }
  for (const auto& stage : manifest.stages()) {
    provided_capabilities.insert(
        stage.capabilities.begin(), stage.capabilities.end());
  }
  for (const auto& capability : declared_capabilities) {
    if (!provided_capabilities.contains(capability)) {
      Fail(
          "Required capability '" + capability +
          "' is not provided by a component, stage, transform, or generator");
    }
  }

  if (!manifest.profile().empty()) {
    for (const auto& input : manifest.inputs()) {
      if (input.semantic.empty()) {
        Fail(
            "Executable profile requires semantic name for input '" +
            input.port.qualified() + "'");
      }
    }
    for (const auto& component : components) {
      if (component.preferred_execution_providers.empty()) {
        Fail(
            "Executable profile requires execution-provider hints for component '" +
            component.name + "'");
      }
      if (!component.parameter_data_type.has_value()) {
        Fail(
            "Executable profile requires parameter dtype for component '" +
            component.name + "'");
      }
    }

    std::set<std::string> referenced_assets;
    for (const auto& stage : manifest.stages()) {
      const Json options = Json::parse(stage.options_json);
      if (stage.kind == "autoregressive") {
        for (const std::string_view required :
             {"tokenizer_asset", "sampling", "stop"}) {
          if (!options.contains(required)) {
            Fail(
                "Autoregressive stage '" + stage.name +
                "' is missing required option '" + std::string(required) + "'");
          }
        }
        if (options.at("tokenizer_asset").is_string()) {
          referenced_assets.insert(
              options.at("tokenizer_asset").get<std::string>());
        }
      } else if (stage.kind == "iterative") {
        for (const std::string_view required :
             {"scheduler", "default_steps", "timestep", "state_inputs"}) {
          if (!options.contains(required)) {
            Fail(
                "Iterative stage '" + stage.name +
                "' is missing required option '" + std::string(required) + "'");
          }
        }
        const Json& scheduler = options.at("scheduler");
        if (scheduler.is_object() && scheduler.contains("config_asset") &&
            scheduler.at("config_asset").is_string()) {
          referenced_assets.insert(
              scheduler.at("config_asset").get<std::string>());
        }
      }
    }
    std::set<std::string> asset_paths;
    std::vector<std::string> asset_names;
    for (const auto& asset : manifest.assets()) {
      const std::string path = asset.path.generic_string();
      asset_paths.insert(path);
      asset_names.push_back(path);
    }
    CheckUnique(asset_names, "Asset path", true);
    for (const auto& asset : referenced_assets) {
      if (!asset_paths.contains(asset)) {
        Fail("Pipeline stage references undeclared asset '" + asset + "'");
      }
    }
  } else {
    std::vector<std::string> asset_names;
    for (const auto& asset : manifest.assets()) {
      asset_names.push_back(asset.path.generic_string());
    }
    CheckUnique(asset_names, "Asset path", true);
  }

  std::unordered_set<std::string> all_asset_paths;
  for (const auto& asset : manifest.assets()) {
    all_asset_paths.insert(asset.path.generic_string());
  }
  const auto validate_parameter_endpoint =
      [&components](const Json& value, std::string_view context) {
        if (!value.is_string()) {
          Fail(std::string(context) + " must be a qualified endpoint string");
        }
        const Endpoint endpoint = Endpoint::Parse(value.get<std::string>());
        RequireContextEndpoint(components, endpoint, context);
      };
  for (const auto& connection : manifest.connections()) {
    const Json parameters = Json::parse(connection.parameters_json);
    if (parameters.contains("state")) {
      if (!parameters.at("state").is_string() ||
          std::ranges::find(
              state_names,
              parameters.at("state").get<std::string>()) ==
              state_names.end()) {
        Fail(
            "Connection '" + connection.source.qualified() + "' -> '" +
            connection.target.qualified() + "' references an unknown state");
      }
    }
    if (parameters.contains("stage")) {
      if (!parameters.at("stage").is_string() ||
          !stage_name_set.contains(
              parameters.at("stage").get<std::string>())) {
        Fail(
            "Connection '" + connection.source.qualified() + "' -> '" +
            connection.target.qualified() + "' references an unknown stage");
      }
    }
    if (parameters.contains("scheduler_asset")) {
      if (!parameters.at("scheduler_asset").is_string() ||
          !all_asset_paths.contains(
              parameters.at("scheduler_asset").get<std::string>())) {
        Fail(
            "Connection '" + connection.source.qualified() + "' -> '" +
            connection.target.qualified() +
            "' references an undeclared scheduler asset");
      }
    }
    if (parameters.contains("timestep_input")) {
      validate_parameter_endpoint(
          parameters.at("timestep_input"),
          "Connection timestep_input");
    }
  }
  for (const auto& input : manifest.inputs()) {
    if (input.generator_kind.empty()) {
      continue;
    }
    const Json parameters = Json::parse(input.generator_json);
    if (parameters.contains("stage")) {
      if (!parameters.at("stage").is_string() ||
          !stage_name_set.contains(
              parameters.at("stage").get<std::string>())) {
        Fail(
            "Generated input '" + input.port.qualified() +
            "' references an unknown stage");
      }
    }
    for (const std::string_view key : {"source", "sequence_input"}) {
      if (parameters.contains(key)) {
        validate_parameter_endpoint(
            parameters.at(key),
            "Generated input '" + input.port.qualified() + "' " +
                std::string(key));
      }
    }
    if (parameters.contains("past_state")) {
      const Json& states = parameters.at("past_state");
      if (!states.is_array() ||
          std::ranges::any_of(states, [&state_names](const Json& state) {
            return !state.is_string() ||
                   std::ranges::find(
                       state_names, state.get<std::string>()) ==
                       state_names.end();
          })) {
        Fail(
            "Generated input '" + input.port.qualified() +
            "' references an unknown past state");
      }
    }
    if (parameters.contains("dynamic_axes")) {
      const Json& axes = parameters.at("dynamic_axes");
      RequireObject(
          axes, "Generated input '" + input.port.qualified() + "' dynamic_axes");
      const PipelineComponent& component =
          manifest.Component(input.port.component);
      const auto symbols =
          component.input_dimension_symbols.find(input.port.port);
      if (symbols == component.input_dimension_symbols.end()) {
        Fail(
            "Generated input '" + input.port.qualified() +
            "' has no declared dimension symbols");
      }
      std::vector<std::string> non_batch_symbols;
      for (const auto& symbol : symbols->second) {
        if (!symbol.empty() && symbol != "b" && symbol != "batch") {
          non_batch_symbols.push_back(symbol);
        }
      }
      for (const auto& [symbol, value] : axes.items()) {
        const bool exact =
            std::ranges::find(symbols->second, symbol) !=
            symbols->second.end();
        const bool unambiguous_alias =
            axes.size() == 1 && non_batch_symbols.size() == 1;
        if (!value.is_number_integer() ||
            value.get<std::int64_t>() < 0 ||
            (!exact && !unambiguous_alias)) {
          Fail(
              "Generated input '" + input.port.qualified() +
              "' references unknown dynamic axis '" + symbol + "'");
        }
      }
    }
  }

  const auto& files = manifest.component_files();
  if (files.size() != components.size()) {
    Fail("'component_files' must contain exactly one path per component");
  }
  std::vector<std::string> file_paths;
  const bool single_component = components.size() == 1;
  for (const auto& component : components) {
    if (!files.contains(component.name)) {
      Fail(
          "'component_files' is missing component '" + component.name + "'");
    }
    const std::string path = files.at(component.name).generic_string();
    const std::string expected =
        single_component ? "model.onnx"
                         : component.name + "/model.onnx";
    if (path != expected) {
      Fail(
          "Component '" + component.name + "' must use standard path '" +
          expected + "', got '" + path + "'");
    }
    file_paths.push_back(path);
  }
  CheckUnique(file_paths, "Component file path", true);
  std::unordered_set<std::string> reserved_paths{"pipeline.json"};
  for (const auto& path : file_paths) {
    reserved_paths.insert(Lower(path));
    reserved_paths.insert(Lower(path + ".data"));
  }
  for (const auto& asset : manifest.assets()) {
    if (reserved_paths.contains(Lower(asset.path.generic_string()))) {
      Fail(
          "Asset '" + asset.path.generic_string() +
          "' collides with a package-owned file");
    }
  }
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
    const bool has_provider_hint =
        !component.preferred_execution_providers.empty();
    const bool supports_cpu =
        !has_provider_hint ||
        std::ranges::any_of(
            component.preferred_execution_providers,
            [](const std::string& provider) {
              return Lower(provider) == "cpu";
            });
    if (!supports_cpu) {
      throw Error(
          ErrorCode::runtime_load,
          "Component '" + component.name +
              "' does not permit the runtime's currently available CPU "
              "execution provider");
    }
    const auto model_path = ResolveInside(
        root,
        manifest.component_files().at(component.name),
        "Component '" + component.name + "'");
    components.emplace(component.name, Model::Load(model_path, options));
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

}  // namespace onnx_world_model
