/**
 * @agent-file
 * @agent-purpose: Declares the internal strategy objects the StageRun state machine steps a stage through -- the narrow StageExecutionHost a strategy may reach the session by, the StageStepContext that hands a run's own inputs to the first step that actually executes, the StageExecutor one strategy body lives behind, and the factory that resolves a detail::StageExecutionStrategy into one of them.
 * @agent-public-api: onnx_world_model::detail::StageExecutionHost, onnx_world_model::detail::StageStepContext, onnx_world_model::detail::StageExecutor, onnx_world_model::detail::MakeStageExecutor, onnx_world_model::detail::MakeAutoregressiveStageExecutor, onnx_world_model::detail::MakeIterativeStageExecutor, onnx_world_model::detail::MakeOnePassStageExecutor
 * @agent-invariants: Internal header that is not installed and that no test includes; the only executor a caller sees is the behavior of StageRun and RunStage. An executor owns exactly one strategy's per-run state -- the autoregressive sampling plan, generated history, and per-lane end-of-sequence latch, the iterative target cursor, the one-pass latch, and each strategy's last step outputs -- and deliberately nothing else: the run identity and slot, admission leases, telemetry, cancellation linkage, the emitted-step count, and the terminal event stay on StageRun::Impl, and none of that state is in SessionState, so a snapshot can never capture half of an in-flight run. The session lock is held by StageRun for every StepLocked and FinalOutputsLocked call, so an executor never takes a lock, and StageExecutionHost exposes no mutex, run slot, scheduler, telemetry, snapshot, or SessionState -- only the session operations a strategy body actually performs. A host reference and the PipelineStage handed to the factory both outlive the executor, because StageRun::Impl owns the executor and holds the session, and therefore the immutable manifest, alive through a shared_ptr; StageLogits returns a reference into the session's endpoint values that stays valid only until the next host call. StepLocked returns no value when the strategy is complete, which is the only way a run completes, and every event it does return carries its kind, its tokens, and its outputs while StageRun stamps the stage name and iteration and counts the step. Construction resolves everything a run needs -- stored external inputs, the sampling configuration, the seed, the end-of-sequence set, the prompt-derived token budget, the iterative target -- exactly once, and it happens under the session lock before the run claims the session's slot, so a construction failure leaves that slot free.
 * @agent-side-effects: none; every side effect a strategy has is performed by the StageExecutionHost implementation it is given, which runs components through ONNX Runtime, advances the session's seeded random engine, and mutates the session's external, endpoint, and state values.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "onnx_world_model/pipeline.hpp"
#include "stage_registry.hpp"

namespace onnx_world_model::detail {

//: Everything a strategy body may do to the session it runs in, and nothing
//: else. The implementation is a file-local adapter over
//: PipelineSession::Impl, so an executor cannot reach the session's mutex,
//: run slot, scheduler, telemetry, or SessionState even by accident. Every
//: method is called with the session lock already held by StageRun.
class StageExecutionHost {
 public:
  StageExecutionHost() = default;
  virtual ~StageExecutionHost() = default;

  StageExecutionHost(const StageExecutionHost&) = delete;
  StageExecutionHost& operator=(const StageExecutionHost&) = delete;
  StageExecutionHost(StageExecutionHost&&) = delete;
  StageExecutionHost& operator=(StageExecutionHost&&) = delete;

  //: Binds caller-supplied tensors to the manifest's external inputs. Only
  //: the autoregressive strategy calls it directly, because it has to read
  //: the prompt back before the first step runs.
  virtual void StoreExternalInputs(const NamedTensors& inputs) = 0;

  //: Reseeds the session's random engine, which is what makes a sampled run
  //: reproducible. Called at most once per run, while it is being built.
  virtual void SeedRandomEngine(std::uint64_t seed) = 0;

  //: The token budget for `stage`, resolved from the run options and the
  //: stage's manifest options against the prompt already stored.
  [[nodiscard]] virtual std::size_t MaximumTokens(
      const PipelineStage& stage,
      const PipelineRunOptions& options) const = 0;

  //: The absolute iteration cursor an iterative stage is driven to.
  [[nodiscard]] virtual std::size_t InferenceSteps(
      std::string_view stage_name,
      const PipelineRunOptions& options) const = 0;

  //: One component pass or scheduler step of `stage_name`, returning the
  //: session's public outputs as of that step. This is the only method that
  //: executes anything.
  [[nodiscard]] virtual NamedTensors StepStage(
      std::string_view stage_name,
      const NamedTensors& inputs,
      const NamedTensors& overrides,
      const PipelineRunOptions& options) = 0;

  //: How many iterations of `stage_name` this session has already performed,
  //: across every run. It is not const because a stage that has never run
  //: starts its cursor here, exactly as the execution paths do.
  [[nodiscard]] virtual std::size_t StageIterations(
      const std::string& stage_name) = 0;

  //: The logits the last pass of `stage` produced. The reference points into
  //: the session's endpoint values and stays valid only until the next host
  //: call, so a caller samples from it immediately rather than storing it.
  [[nodiscard]] virtual const Tensor& StageLogits(
      const PipelineStage& stage) const = 0;

  //: One sampled token per batch lane, advancing the session's random engine.
  [[nodiscard]] virtual Tensor SampleTokens(
      const Tensor& logits,
      const nlohmann::json& sampling,
      const PipelineRunOptions& options,
      const std::vector<std::vector<std::int64_t>>& history) = 0;

  //: One arg-max token per batch lane, reading nothing but the logits.
  [[nodiscard]] virtual Tensor GreedyTokens(const Tensor& logits) const = 0;

  //: Feeds `tokens` back into the stage's text.token_ids external input.
  virtual void SetStageTokenInput(
      const PipelineStage& stage,
      const Tensor& tokens) = 0;

  //: The session's public outputs as they stand right now.
  [[nodiscard]] virtual NamedTensors CollectOutputs() const = 0;
};

//: The per-step binding set a run hands its executor: the run's own inputs,
//: overrides, and internal PipelineRunOptions -- the options whose
//: cancellation token is the run's linked one -- plus the run's shared
//: consumed-inputs flag. It refers to StageRun::Impl's members rather than
//: copying them, so it is built fresh for each step and never outlives it.
class StageStepContext {
 public:
  StageStepContext(
      const NamedTensors& inputs,
      const NamedTensors& overrides,
      const PipelineRunOptions& options,
      bool& consumed_inputs) noexcept
      : inputs_(&inputs),
        overrides_(&overrides),
        options_(&options),
        consumed_inputs_(&consumed_inputs) {}

  //: The run's inputs the first time an executor actually runs a step, and
  //: nothing every time after that. A strategy that completes without
  //: executing -- an iterative stage already at its target -- never calls
  //: this, so its inputs are never marked consumed.
  [[nodiscard]] NamedTensors ConsumeInputs() {
    if (*consumed_inputs_) {
      return {};
    }
    *consumed_inputs_ = true;
    return *inputs_;
  }

  [[nodiscard]] const NamedTensors& overrides() const noexcept {
    return *overrides_;
  }

  [[nodiscard]] const PipelineRunOptions& options() const noexcept {
    return *options_;
  }

 private:
  const NamedTensors* inputs_;
  const NamedTensors* overrides_;
  const PipelineRunOptions* options_;
  bool* consumed_inputs_;
};

//: One stage-execution strategy's body. StageRun::Impl owns exactly one of
//: these for the life of a run and calls it with the session lock held.
class StageExecutor {
 public:
  StageExecutor() = default;
  virtual ~StageExecutor() = default;

  StageExecutor(const StageExecutor&) = delete;
  StageExecutor& operator=(const StageExecutor&) = delete;
  StageExecutor(StageExecutor&&) = delete;
  StageExecutor& operator=(StageExecutor&&) = delete;

  //: Performs one step and describes it, or reports completion by returning
  //: no value. The returned event carries its own kind, tokens, and outputs;
  //: the stage name, the iteration index, the step count, and the terminal
  //: event belong to StageRun.
  [[nodiscard]] virtual std::optional<StageEvent> StepLocked(
      StageStepContext& context) = 0;

  //: What this run's completed event and Finish() report. Called exactly once
  //: per run, right after StepLocked reported completion.
  [[nodiscard]] virtual NamedTensors FinalOutputsLocked(
      StageStepContext& context) = 0;
};

//: One token per step until the budget is spent or every lane latched its
//: end-of-sequence token. Construction stores `inputs` as external values and
//: resolves the sampling plan, the seed, the stop tokens, and the budget.
[[nodiscard]] std::unique_ptr<StageExecutor> MakeAutoregressiveStageExecutor(
    StageExecutionHost& host,
    const PipelineStage& stage,
    const NamedTensors& inputs,
    const PipelineRunOptions& options);

//: One scheduler iteration per step until the stage cursor reaches the target
//: construction resolved from `options`.
[[nodiscard]] std::unique_ptr<StageExecutor> MakeIterativeStageExecutor(
    StageExecutionHost& host,
    const PipelineStage& stage,
    const PipelineRunOptions& options);

//: Exactly one pass, then completion. single_pass, state_transition,
//: composite, and on_demand all execute through this.
[[nodiscard]] std::unique_ptr<StageExecutor> MakeOnePassStageExecutor(
    StageExecutionHost& host,
    const PipelineStage& stage);

//: The executor `strategy` names. The switch is exhaustive, so a strategy
//: added to stage_registry is a compile error here rather than a silent one
//: pass. `overrides` completes the run's binding set that StageStepContext
//: carries into every step; no strategy resolves construction-time
//: configuration from it today, so it is accepted and not read.
[[nodiscard]] std::unique_ptr<StageExecutor> MakeStageExecutor(
    StageExecutionStrategy strategy,
    StageExecutionHost& host,
    const PipelineStage& stage,
    const NamedTensors& inputs,
    const NamedTensors& overrides,
    const PipelineRunOptions& options);

}  // namespace onnx_world_model::detail
