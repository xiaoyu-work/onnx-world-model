/**
 * @agent-file
 * @agent-purpose: Declares the one internal registry of the stage kinds this runtime executes -- each kind's manifest name, the execution strategy the stage state machine drives it with, and the exact option names its manifest options object may carry -- so manifest validation, admission scheduling, telemetry, and execution all read the same list instead of repeating it.
 * @agent-public-api: onnx_world_model::detail::kStageKindCount, onnx_world_model::detail::StageExecutionStrategy, onnx_world_model::detail::StageKindDefinition, onnx_world_model::detail::StageKindDefinitions, onnx_world_model::detail::FindStageKind, onnx_world_model::detail::StageKindIndex
 * @agent-invariants: Internal header that is not installed and depends on nothing but the standard library, which is what lets manifest validation, the scheduler, telemetry, and the session all include it without a cycle. The registry is the single source of truth: a stage kind exists here or it does not exist at all, and kStageKindCount is the size every per-kind array in this library is declared with. Definition order is the order every per-kind reading is built in, so an index into StageKindDefinitions() is the same key in the scheduler's buckets, in telemetry's admission array, and in the public per-stage-kind maps. `name` and `allowed_options` view objects with static storage duration, so a definition and the spans inside it outlive every caller and can be copied freely. A strategy is how a stage is stepped, not what it is called: state_transition, composite, and on_demand execute as one pass exactly as single_pass does, and no strategy here recurses into a composite stage's children. Lookup never throws and never allocates: FindStageKind returns null and StageKindIndex returns no value for a name this runtime does not execute, so each caller decides whether that is a manifest error, an unconstrained admission bucket, or an internal invariant failure.
 * @agent-side-effects: none
 */

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string_view>

namespace onnx_world_model::detail {

//: How many stage kinds this runtime executes. Every per-kind array in this
//: library is sized with this rather than a literal, so adding a kind to the
//: registry cannot leave a stale array behind.
inline constexpr std::size_t kStageKindCount = 6;

//: How the stage state machine drives one kind. Deliberately coarser than the
//: kind itself: it says how a stage is stepped, not what it is called.
enum class StageExecutionStrategy {
  //: Exactly one pass, then completion. state_transition, composite, and
  //: on_demand share this with single_pass because that is what they have
  //: always done; in particular nothing here recurses into a composite
  //: stage's declared child stages.
  single_pass,
  //: One token per step until the token budget is spent or every lane latched
  //: its end-of-sequence token.
  autoregressive,
  //: One scheduler iteration per step until the run's target cursor is
  //: reached.
  iterative,
};

//: One stage kind. `name` is the string a manifest declares and the public
//: PipelineStage::kind reports, so it is the contract; the rest is internal.
struct StageKindDefinition {
  //: The manifest spelling. Static storage duration.
  std::string_view name;
  StageExecutionStrategy strategy;
  //: Exactly the option names this kind's manifest options object may carry;
  //: empty means the kind takes no options. Views an array with static
  //: storage duration, so it outlives every definition copy.
  std::span<const std::string_view> allowed_options;
};

//: Every stage kind, in the order every per-kind reading is built in.
[[nodiscard]] const std::array<StageKindDefinition, kStageKindCount>&
StageKindDefinitions() noexcept;

//: The definition named `name`, or null for a kind this runtime does not
//: execute. It never throws, so a caller that reached it from a manifest
//: reports its own manifest error and a caller that reached it from an
//: already-validated stage treats null as an internal invariant failure.
[[nodiscard]] const StageKindDefinition* FindStageKind(
    std::string_view name) noexcept;

//: `name`'s index into StageKindDefinitions(), or no value for a kind this
//: runtime does not execute. That index is the shared key: the scheduler's
//: bucket, telemetry's admission entry, and the public per-stage-kind maps
//: all use it.
[[nodiscard]] std::optional<std::size_t> StageKindIndex(
    std::string_view name) noexcept;

}  // namespace onnx_world_model::detail
