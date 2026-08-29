/**
 * @agent-file
 * @agent-purpose: Implements the one-pass stage executor -- the strategy single_pass, state_transition, composite, and on_demand all execute through -- and the factory that turns a detail::StageExecutionStrategy into the executor that owns that strategy's per-run state.
 * @agent-public-api: detail::MakeOnePassStageExecutor, detail::MakeStageExecutor
 * @agent-invariants: The one pass is latched by the executor rather than by the run's emitted-step count, so it executes exactly once however the run is driven: the first StepLocked runs the stage with the run's own inputs and reports a transition event, and every later call reports completion. The latch is set only after StepStage returns, so a failed pass leaves nothing half-executed behind -- the run closes on that failure either way. Final outputs are the last executed step's outputs, which is exactly what RunStage returned for these kinds; a run that completed without executing anything reports the session's current public outputs instead, and it never consumed its inputs. The factory switches exhaustively over the registry's strategies and constructs under the session lock before the run claims the session's slot, so a construction failure -- an autoregressive stage with no resolvable token budget, for instance -- leaves that slot free; a strategy the switch cannot name fails loudly with ErrorCode::pipeline_manifest rather than degrading into one silent pass.
 * @agent-side-effects: Constructing an executor may already mutate the session through its StageExecutionHost -- the autoregressive strategy stores its external inputs and may reseed the random engine -- and every step runs component inference through that same host.
 */

#include "stage_executor.hpp"

#include <memory>
#include <optional>
#include <utility>

#include "onnx_world_model/error.hpp"

namespace onnx_world_model::detail {
namespace {

// state_transition, composite, and on_demand share single_pass's strategy:
// exactly one pass, which is what RunStage did for every one of them.
class OnePassStageExecutor final : public StageExecutor {
 public:
  OnePassStageExecutor(
      StageExecutionHost& host,
      const PipelineStage& stage) noexcept
      : host_(&host), stage_(&stage) {}

  [[nodiscard]] std::optional<StageEvent> StepLocked(
      StageStepContext& context) override {
    if (executed_) {
      return std::nullopt;
    }
    NamedTensors step_outputs = host_->StepStage(
        stage_->name,
        context.ConsumeInputs(),
        context.overrides(),
        context.options());
    last_step_outputs_ = step_outputs;
    executed_ = true;

    StageEvent event;
    event.kind = StageEventKind::transition;
    event.outputs = std::move(step_outputs);
    return event;
  }

  [[nodiscard]] NamedTensors FinalOutputsLocked(StageStepContext&) override {
    if (last_step_outputs_.has_value()) {
      // Exactly what the last executed step returned, which is what RunStage
      // returned for this stage.
      return *last_step_outputs_;
    }
    // A stage that had no work left executed nothing, so its result is the
    // session's current public outputs.
    return host_->CollectOutputs();
  }

 private:
  StageExecutionHost* host_;
  const PipelineStage* stage_;
  std::optional<NamedTensors> last_step_outputs_;
  bool executed_{false};
};

}  // namespace

std::unique_ptr<StageExecutor> MakeOnePassStageExecutor(
    StageExecutionHost& host,
    const PipelineStage& stage) {
  return std::make_unique<OnePassStageExecutor>(host, stage);
}

std::unique_ptr<StageExecutor> MakeStageExecutor(
    StageExecutionStrategy strategy,
    StageExecutionHost& host,
    const PipelineStage& stage,
    const NamedTensors& inputs,
    const NamedTensors&,
    const PipelineRunOptions& options) {
  switch (strategy) {
    case StageExecutionStrategy::autoregressive:
      return MakeAutoregressiveStageExecutor(host, stage, inputs, options);
    case StageExecutionStrategy::iterative:
      return MakeIterativeStageExecutor(host, stage, options);
    case StageExecutionStrategy::single_pass:
      return MakeOnePassStageExecutor(host, stage);
  }
  // Unreachable while the switch covers the registry, and deliberately loud
  // rather than a fallback to one pass if a strategy is ever added without a
  // body here.
  throw Error(
      ErrorCode::pipeline_manifest,
      "Stage '" + stage.name + "' has no execution strategy");
}

}  // namespace onnx_world_model::detail
