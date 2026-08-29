/**
 * @agent-file
 * @agent-purpose: Defines the six stage kinds this runtime executes -- their manifest names, their execution strategies, and their allowed manifest option names -- and the non-throwing lookups every other file resolves a kind through.
 * @agent-public-api: detail::StageKindDefinitions, detail::FindStageKind, detail::StageKindIndex
 * @agent-invariants: This table is the definition of "supported stage kind" for the whole library: manifest validation rejects a kind that is absent here, the scheduler builds one permit bucket per entry in this order, telemetry sizes its admission array the same way, and the session resolves a stage's execution strategy from it. The option name arrays are namespace-scope constexpr objects, so the spans in the definitions view storage that outlives the program's use of them and a definition can be copied by value. Both lookups are linear over six entries, are noexcept, allocate nothing, and compare the manifest spelling exactly -- there is no normalization, no aliasing, and no fallback kind.
 * @agent-side-effects: none
 */

#include "stage_registry.hpp"

#include <algorithm>

namespace onnx_world_model::detail {
namespace {

//: One array per kind, at namespace scope so every span below views storage
//: with static duration. Each list is exactly the option set manifest
//: validation has always accepted for that kind.
constexpr std::array<std::string_view, 0> kSinglePassOptions{};

constexpr std::array<std::string_view, 5> kAutoregressiveOptions{
    "tokenizer_asset",
    "sampling",
    "stop",
    "max_tokens",
    "state_names",
};

constexpr std::array<std::string_view, 9> kIterativeOptions{
    "scheduler",
    "guidance",
    "conditioning",
    "default_steps",
    "timestep",
    "state_inputs",
    "initial_state_inputs",
    "prediction_type",
    "packed_modalities",
};

constexpr std::array<std::string_view, 3> kStateTransitionOptions{
    "state_names",
    "max_steps",
    "stop",
};

constexpr std::array<std::string_view, 1> kCompositeOptions{"stages"};

constexpr std::array<std::string_view, 1> kOnDemandOptions{"presence"};

}  // namespace

const std::array<StageKindDefinition, kStageKindCount>&
StageKindDefinitions() noexcept {
  static constexpr std::array<StageKindDefinition, kStageKindCount>
      definitions{{
          {"single_pass",
           StageExecutionStrategy::single_pass,
           kSinglePassOptions},
          {"autoregressive",
           StageExecutionStrategy::autoregressive,
           kAutoregressiveOptions},
          {"iterative",
           StageExecutionStrategy::iterative,
           kIterativeOptions},
          // Stepped exactly like single_pass. The kind still exists on its
          // own because a manifest declares it and because its option set and
          // its state-lifecycle rules differ.
          {"state_transition",
           StageExecutionStrategy::single_pass,
           kStateTransitionOptions},
          {"composite",
           StageExecutionStrategy::single_pass,
           kCompositeOptions},
          {"on_demand",
           StageExecutionStrategy::single_pass,
           kOnDemandOptions},
      }};
  return definitions;
}

const StageKindDefinition* FindStageKind(std::string_view name) noexcept {
  const auto& definitions = StageKindDefinitions();
  const auto found = std::ranges::find(
      definitions, name, &StageKindDefinition::name);
  return found == definitions.end() ? nullptr : &*found;
}

std::optional<std::size_t> StageKindIndex(std::string_view name) noexcept {
  const auto& definitions = StageKindDefinitions();
  const auto found = std::ranges::find(
      definitions, name, &StageKindDefinition::name);
  if (found == definitions.end()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(found - definitions.begin());
}

}  // namespace onnx_world_model::detail
