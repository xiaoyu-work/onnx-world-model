/**
 * @agent-file
 * @agent-purpose: Validates the semantics of a parsed Mobius PipelineManifest: dataflow acyclicity, endpoint and dtype agreement, transform and generator programs, stage options, capabilities, and recurrent state lifecycles.
 * @agent-public-api: FindComponent, RequireEndpoint, SupportedCapabilityNames, ValidateManifest
 * @agent-invariants: This pass runs on an already-parsed manifest and never mutates it; every violation throws ErrorCode::pipeline_manifest. Transform and generator kinds, stage option keys, and capability names are closed sets, so an unknown or unsupported value is rejected here instead of silently selecting a fallback. Non-recurrent connections must form an acyclic graph, and every declared capability must appear in SupportedCapabilityNames.
 * @agent-side-effects: none
 */

#include "pipeline_manifest_validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "onnx_world_model/error.hpp"
#include "pipeline_manifest_common.hpp"

namespace onnx_world_model::detail {
namespace {

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

const TensorSpec* FindSpec(
    const std::vector<TensorSpec>& specs,
    std::string_view name) {
  const auto found = std::ranges::find(specs, name, &TensorSpec::name);
  return found == specs.end() ? nullptr : &*found;
}

}  // namespace

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

namespace {

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

const std::unordered_map<std::string, ProgramDefinition>&
TransformDefinitions() {
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
  return definitions;
}

const ProgramDefinition& TransformDefinition(std::string_view kind) {
  const auto& definitions = TransformDefinitions();
  const auto found = definitions.find(std::string(kind));
  if (found == definitions.end()) {
    Fail("Unknown pipeline transform '" + std::string(kind) + "'");
  }
  return found->second;
}

const std::unordered_map<std::string, ProgramDefinition>&
GeneratorDefinitions() {
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
  return definitions;
}

const ProgramDefinition& GeneratorDefinition(std::string_view kind) {
  const auto& definitions = GeneratorDefinitions();
  const auto found = definitions.find(std::string(kind));
  if (found == definitions.end()) {
    Fail("Unknown generated input program '" + std::string(kind) + "'");
  }
  return found->second;
}

//: Stage capabilities the runtime implements, and the stage option each one
//: describes. A stage must advertise the capability when it declares the
//: option, and declare the option when it advertises the capability.
const std::unordered_map<std::string, std::string>& StageCapabilityOptions() {
  static const std::unordered_map<std::string, std::string> options{
      {"classifier_free_guidance", "guidance"},
      {"conditioned_diffusion", "conditioning"},
  };
  return options;
}

}  // namespace

const std::set<std::string>& SupportedCapabilityNames() {
  static const std::set<std::string> capabilities = [] {
    std::set<std::string> result{"loop_carried_state"};
    for (const auto& [name, definition] : TransformDefinitions()) {
      (void)name;
      result.insert(
          definition.capabilities.begin(), definition.capabilities.end());
    }
    for (const auto& [name, definition] : GeneratorDefinitions()) {
      (void)name;
      result.insert(
          definition.capabilities.begin(), definition.capabilities.end());
    }
    for (const auto& [capability, option] : StageCapabilityOptions()) {
      (void)option;
      result.insert(capability);
    }
    return result;
  }();
  return capabilities;
}

namespace {

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
              "guidance",
              "conditioning",
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

void RequireOptionKeys(
    const Json& value,
    const std::set<std::string>& allowed,
    const std::set<std::string>& required,
    std::string_view context) {
  RequireObject(value, context);
  for (const auto& [name, item] : value.items()) {
    (void)item;
    if (!allowed.contains(name)) {
      Fail(std::string(context) + " has unknown field '" + name + "'");
    }
  }
  for (const auto& name : required) {
    if (!value.contains(name)) {
      Fail(
          std::string(context) + " is missing required field '" + name + "'");
    }
  }
}

std::string RequireOptionString(
    const Json& value,
    std::string_view field,
    std::string_view context) {
  const auto found = value.find(std::string(field));
  if (found == value.end() || !found->is_string() ||
      found->get_ref<const std::string&>().empty()) {
    Fail(
        std::string(context) + " field '" + std::string(field) +
        "' must be a non-empty string");
  }
  return found->get<std::string>();
}

bool StageOwns(const PipelineStage& stage, std::string_view component) {
  return std::ranges::find(stage.components, component) !=
         stage.components.end();
}

void RequireStageCapability(
    const PipelineStage& stage,
    std::string_view capability,
    std::string_view context) {
  if (std::ranges::find(stage.capabilities, capability) ==
      stage.capabilities.end()) {
    Fail(
        std::string(context) + " requires stage capability '" +
        std::string(capability) + "'");
  }
}

// Classifier-free guidance: the runtime evaluates the stage twice per step,
// once with the declared conditioning input and once with the host-supplied
// unconditional value, and combines the guided predictions.
void ValidateGuidanceOption(
    const PipelineManifest& manifest,
    const PipelineStage& stage,
    const Json& guidance) {
  const std::string context = "Stage '" + stage.name + "' guidance";
  RequireOptionKeys(
      guidance,
      {
          "kind",
          "conditioning_input",
          "unconditional_input",
          "scale_option",
          "default_scale",
          "combine",
          "outputs",
      },
      {"kind", "conditioning_input"},
      context);
  RequireStageCapability(stage, "classifier_free_guidance", context);
  const std::string kind = RequireOptionString(guidance, "kind", context);
  if (kind != "classifier_free") {
    Fail(context + " has unsupported kind '" + kind + "'");
  }
  const Endpoint conditioning = Endpoint::Parse(
      RequireOptionString(guidance, "conditioning_input", context));
  if (!StageOwns(stage, conditioning.component)) {
    Fail(
        context + " conditions on '" + conditioning.qualified() +
        "', which the stage does not run");
  }
  (void)RequireEndpoint(
      manifest.components(), conditioning, true, context);
  const auto declared = std::ranges::find(
      manifest.inputs(), conditioning, &PipelineInput::port);
  if (declared == manifest.inputs().end() ||
      declared->kind != PipelineInputKind::external) {
    Fail(
        context + " conditioning input '" + conditioning.qualified() +
        "' must be a declared external input");
  }
  if (guidance.contains("unconditional_input")) {
    (void)RequireOptionString(guidance, "unconditional_input", context);
  }
  if (guidance.contains("scale_option")) {
    (void)RequireOptionString(guidance, "scale_option", context);
  }
  if (guidance.contains("default_scale") &&
      !guidance.at("default_scale").is_number()) {
    Fail(context + " field 'default_scale' must be a number");
  }
  if (guidance.contains("combine")) {
    const std::string combine =
        RequireOptionString(guidance, "combine", context);
    if (combine != "unconditional + scale * (conditional - unconditional)") {
      Fail(context + " has unsupported combination rule '" + combine + "'");
    }
  }
  if (guidance.contains("outputs")) {
    const std::vector<std::string> outputs =
        ParseStringArray(guidance.at("outputs"), context + " outputs");
    for (const auto& name : outputs) {
      const Endpoint endpoint = Endpoint::Parse(name);
      if (!StageOwns(stage, endpoint.component)) {
        Fail(
            context + " guides '" + endpoint.qualified() +
            "', which the stage does not produce");
      }
      (void)RequireEndpoint(
          manifest.components(), endpoint, false, context);
    }
  }
}

//: Resampling filters the runtime can apply to conditioning frames.
const std::set<std::string>& ConditioningResampleFilters() {
  static const std::set<std::string> filters{
      "bilinear", "bicubic", "nearest", "lanczos"};
  return filters;
}

//: The only conditioning resize strategy the runtime implements: frames are
//: scaled straight to the requested output geometry.
constexpr std::string_view kConditioningResizeStrategy = "stretch_to_target";

void ValidateConditioningPreprocessing(
    const Json& preprocessing,
    std::string_view context) {
  RequireOptionKeys(
      preprocessing,
      {"resize", "resample", "normalize", "rescale_factor", "convert_rgb"},
      {},
      context);
  for (const auto& field : {"resize", "resample"}) {
    if (!preprocessing.contains(field)) {
      continue;
    }
    const std::string value =
        RequireOptionString(preprocessing, field, context);
    // 'resample' always names a filter; 'resize' may name either the filter
    // or the resize strategy.
    if (ConditioningResampleFilters().contains(value)) {
      continue;
    }
    if (std::string(field) == "resize" &&
        value == kConditioningResizeStrategy) {
      continue;
    }
    Fail(
        std::string(context) + " field '" + std::string(field) +
        "' has unsupported value '" + value + "'");
  }
  if (preprocessing.contains("rescale_factor")) {
    const Json& factor = preprocessing.at("rescale_factor");
    if (!factor.is_number() ||
        std::abs(factor.get<double>() - 1.0 / 255.0) > 1e-9) {
      Fail(std::string(context) + " field 'rescale_factor' must be 1/255");
    }
  }
  if (preprocessing.contains("convert_rgb") &&
      (!preprocessing.at("convert_rgb").is_boolean() ||
       !preprocessing.at("convert_rgb").get<bool>())) {
    Fail(std::string(context) + " field 'convert_rgb' must be true");
  }
  if (!preprocessing.contains("normalize")) {
    return;
  }
  const Json& normalize = preprocessing.at("normalize");
  RequireOptionKeys(
      normalize, {"mean", "std"}, {}, std::string(context) + " normalize");
  for (const auto& field : {"mean", "std"}) {
    if (!normalize.contains(field)) {
      continue;
    }
    const Json& values = normalize.at(field);
    if (!values.is_array() || values.size() != 3 ||
        !std::ranges::all_of(values, [](const Json& value) {
          return value.is_number();
        })) {
      Fail(
          std::string(context) + " normalize '" + std::string(field) +
          "' must be three numbers");
    }
    if (std::string(field) == "std" &&
        std::ranges::any_of(values, [](const Json& value) {
          return value.get<double>() == 0.0;
        })) {
      Fail(std::string(context) + " normalize 'std' must be non-zero");
    }
  }
}

// Conditioned diffusion: a host encodes conditioning media through the named
// encoder stage, packs it with the declared layout, and keeps the conditioned
// rows of the diffusion state out of the noisy index sets.
void ValidateConditioningOption(
    const PipelineManifest& manifest,
    const PipelineStage& stage,
    const Json& conditioning) {
  const std::string stage_context = "Stage '" + stage.name + "' conditioning";
  RequireObject(conditioning, stage_context);
  RequireStageCapability(stage, "conditioned_diffusion", stage_context);
  if (conditioning.empty()) {
    Fail(stage_context + " must describe at least one modality");
  }
  for (const auto& [modality, spec] : conditioning.items()) {
    const std::string context =
        stage_context + " modality '" + modality + "'";
    RequireOptionKeys(
        spec,
        {
            "encoder_stage",
            "encoder_input",
            "encoder_output",
            "state",
            "conditioned_latent_frames_option",
            "default_conditioned_latent_frames",
            "timestep_token_indexes_input",
            "preprocessing",
            "packing",
        },
        {"encoder_stage", "encoder_input", "encoder_output", "state", "packing"},
        context);
    const std::string encoder_stage =
        RequireOptionString(spec, "encoder_stage", context);
    const auto encoder = std::ranges::find(
        manifest.stages(), encoder_stage, &PipelineStage::name);
    if (encoder == manifest.stages().end()) {
      Fail(context + " references unknown stage '" + encoder_stage + "'");
    }
    const Endpoint encoder_input =
        Endpoint::Parse(RequireOptionString(spec, "encoder_input", context));
    const Endpoint encoder_output =
        Endpoint::Parse(RequireOptionString(spec, "encoder_output", context));
    if (!StageOwns(*encoder, encoder_input.component) ||
        !StageOwns(*encoder, encoder_output.component)) {
      Fail(context + " encoder ports do not belong to '" + encoder_stage + "'");
    }
    (void)RequireEndpoint(
        manifest.components(), encoder_input, true, context);
    (void)RequireEndpoint(
        manifest.components(), encoder_output, false, context);
    const std::string state_name =
        RequireOptionString(spec, "state", context);
    const auto state = std::ranges::find(
        manifest.states(), state_name, &PipelineState::name);
    if (state == manifest.states().end()) {
      Fail(context + " references unknown state '" + state_name + "'");
    }
    if (!StageOwns(stage, state->input.component)) {
      Fail(
          context + " state '" + state_name +
          "' is not owned by the conditioned stage");
    }
    if (spec.contains("conditioned_latent_frames_option")) {
      (void)RequireOptionString(
          spec, "conditioned_latent_frames_option", context);
    }
    if (spec.contains("default_conditioned_latent_frames")) {
      const Json& frames = spec.at("default_conditioned_latent_frames");
      if (!frames.is_array() ||
          !std::ranges::all_of(frames, [](const Json& value) {
            return value.is_number_integer() && value.get<std::int64_t>() >= 0;
          })) {
        Fail(
            context +
            " field 'default_conditioned_latent_frames' must be "
            "non-negative integers");
      }
    }
    if (spec.contains("timestep_token_indexes_input")) {
      const Endpoint indexes = Endpoint::Parse(
          RequireOptionString(spec, "timestep_token_indexes_input", context));
      (void)RequireEndpoint(manifest.components(), indexes, true, context);
    }
    if (spec.contains("preprocessing")) {
      ValidateConditioningPreprocessing(
          spec.at("preprocessing"), context + " preprocessing");
    }
    const Json& packing = spec.at("packing");
    RequireOptionKeys(
        packing,
        {
            "spatial_patch_size",
            "temporal_patch_size",
            "input_layout",
            "output_layout",
            "channel_order",
        },
        {"spatial_patch_size", "input_layout", "output_layout", "channel_order"},
        context + " packing");
    for (const auto& field : {"spatial_patch_size", "temporal_patch_size"}) {
      if (packing.contains(field) &&
          (!packing.at(field).is_number_integer() ||
           packing.at(field).get<std::int64_t>() <= 0)) {
        Fail(
            context + " packing field '" + std::string(field) +
            "' must be a positive integer");
      }
    }
    // The runtime's video unpatchify packs one latent frame per token row.
    if (packing.value("temporal_patch_size", std::int64_t{1}) != 1) {
      Fail(context + " packing requires temporal_patch_size 1");
    }
    const std::string input_layout =
        RequireOptionString(packing, "input_layout", context + " packing");
    const std::string output_layout =
        RequireOptionString(packing, "output_layout", context + " packing");
    const std::string channel_order =
        RequireOptionString(packing, "channel_order", context + " packing");
    if (input_layout != "BCTHW" || output_layout != "NC" ||
        channel_order != "patch_height_patch_width_channel") {
      Fail(context + " packing uses an unsupported conditioning layout");
    }
  }
}

void ValidateStagePrograms(const PipelineManifest& manifest) {
  for (const auto& stage : manifest.stages()) {
    const Json options = Json::parse(stage.options_json);
    if (options.contains("guidance")) {
      ValidateGuidanceOption(manifest, stage, options.at("guidance"));
    }
    if (options.contains("conditioning")) {
      ValidateConditioningOption(manifest, stage, options.at("conditioning"));
    }
    // An advertised capability must correspond to an executable option, so a
    // package cannot claim behavior the stage never describes.
    for (const auto& capability : stage.capabilities) {
      const auto found = StageCapabilityOptions().find(capability);
      if (found != StageCapabilityOptions().end() &&
          !options.contains(found->second)) {
        Fail(
            "Stage '" + stage.name + "' provides '" + capability +
            "' but declares no '" + found->second + "' option");
      }
    }
  }
}

}  // namespace

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

  ValidateStagePrograms(manifest);

  const std::set<std::string> declared_capabilities(
      manifest.required_capabilities().begin(),
      manifest.required_capabilities().end());
  // A required capability is honored when this runtime implements it or when
  // a component or stage in the manifest declares that it provides it. Only
  // names nothing can satisfy are rejected.
  std::set<std::string> known_capabilities = SupportedCapabilityNames();
  for (const auto& component : components) {
    known_capabilities.insert(
        component.capabilities.begin(), component.capabilities.end());
  }
  for (const auto& stage : manifest.stages()) {
    known_capabilities.insert(
        stage.capabilities.begin(), stage.capabilities.end());
  }
  for (const auto& capability : declared_capabilities) {
    if (!known_capabilities.contains(capability)) {
      Fail(
          "Pipeline requires capability '" + capability +
          "', which this runtime does not implement and no component or "
          "stage provides");
    }
  }
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

}  // namespace onnx_world_model::detail
