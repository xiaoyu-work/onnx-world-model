/**
 * @agent-file
 * @agent-purpose: Implements the iterative stage executor: the absolute scheduler cursor a run drives its stage to, the one scheduler step each StepLocked performs, and the outputs the run completes with.
 * @agent-public-api: detail::MakeIterativeStageExecutor
 * @agent-invariants: The target cursor is resolved once, while the run is being built, so an option change cannot move it mid-run and a stage that has already reached it completes immediately -- without executing a step and therefore without consuming the run's inputs, which is exactly what RunStage did for an exhausted stage. The stop predicate reads the session's own stage cursor rather than this run's step count, so a run that starts partway through performs only the remaining steps and a session-level release or reset is observed at the next step. Each executed step reports an iteration event carrying the session's public outputs as of that step, and the run's final outputs are the last such step's outputs, or the session's current outputs when the run executed nothing.
 * @agent-side-effects: Each step runs one scheduler iteration of the stage through the StageExecutionHost, which advances the session's stage cursor, endpoint values, and recurrent state.
 */

#include "stage_executor.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace onnx_world_model::detail {
namespace {

class IterativeStageExecutor final : public StageExecutor {
 public:
  IterativeStageExecutor(
      StageExecutionHost& host,
      const PipelineStage& stage,
      const PipelineRunOptions& options)
      : host_(&host),
        stage_(&stage),
        // Read once so a later option change cannot move the target mid-run.
        target_iterations_(host.InferenceSteps(stage.name, options)) {}

  [[nodiscard]] std::optional<StageEvent> StepLocked(
      StageStepContext& context) override {
    if (host_->StageIterations(stage_->name) >= target_iterations_) {
      return std::nullopt;
    }
    NamedTensors step_outputs = host_->StepStage(
        stage_->name,
        context.ConsumeInputs(),
        context.overrides(),
        context.options());
    last_step_outputs_ = step_outputs;

    StageEvent event;
    event.kind = StageEventKind::iteration;
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
  std::size_t target_iterations_{0};
  std::optional<NamedTensors> last_step_outputs_;
};

}  // namespace

std::unique_ptr<StageExecutor> MakeIterativeStageExecutor(
    StageExecutionHost& host,
    const PipelineStage& stage,
    const PipelineRunOptions& options) {
  return std::make_unique<IterativeStageExecutor>(host, stage, options);
}

}  // namespace onnx_world_model::detail
