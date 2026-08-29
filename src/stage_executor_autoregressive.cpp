/**
 * @agent-file
 * @agent-purpose: Implements the autoregressive stage executor: the sampling plan, seed, end-of-sequence set, and token budget one run resolves before it starts, the decode-sample-feed-back step it repeats, the per-lane end-of-sequence latch that stops it, and the batch-major generated_token_ids it completes with.
 * @agent-public-api: detail::MakeAutoregressiveStageExecutor
 * @agent-invariants: Everything the loop needs is resolved while the run is being built and never again: the run's inputs are stored as external values there, so the token budget is measured against the prompt this run started from rather than against the single column the first step replaces it with; the sampling object comes from the stage's manifest options with the run options overriding do_sample; a seed in the run options reseeds the session engine exactly once; and the stop tokens are the stage's declared eos_token_ids. A step is complete-checked first, so a run whose budget is spent or whose lanes have all latched executes nothing more. Within a step the order is fixed: one stage pass, a cancellation poll, sampling or arg-max from the logits that pass produced, a second cancellation poll, the per-lane substitution that rewrites a finished lane's token to the first declared end-of-sequence token and latches a lane that just emitted one, the append to the generated history, and the feed-back of that same tensor into the stage's token input. The token event therefore carries exactly the tensor that was fed back. Reaching an all-lanes-finished state only requests a stop; the following step reports it, so every run still ends with one terminal completed event. Final outputs are the session's public outputs plus a batch-major [batch, steps] generated_token_ids added with emplace, so a manifest output that already carries that name is never displaced, and a run that generated nothing adds nothing.
 * @agent-side-effects: Construction stores the run's inputs into the session and may reseed its random engine; each step runs component inference, advances that engine when sampling, and rewrites the stage's text.token_ids external input.
 */

#include "stage_executor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <utility>
#include <vector>

namespace onnx_world_model::detail {
namespace {

using Json = nlohmann::json;

[[nodiscard]] Tensor Int64Tensor(
    std::vector<std::int64_t> shape,
    const std::vector<std::int64_t>& values) {
  return Tensor::FromValues<std::int64_t>(
      std::move(shape), std::span(values));
}

class AutoregressiveStageExecutor final : public StageExecutor {
 public:
  AutoregressiveStageExecutor(
      StageExecutionHost& host,
      const PipelineStage& stage,
      const NamedTensors& inputs,
      const PipelineRunOptions& options)
      : host_(&host), stage_(&stage) {
    host_->StoreExternalInputs(inputs);
    const Json stage_options = Json::parse(stage.options_json);
    sampling_ = stage_options.value("sampling", Json::object());
    do_sample_ = options.integers.contains("do_sample")
                     ? options.integers.at("do_sample") != 0
                     : sampling_.value("do_sample", false);
    if (options.integers.contains("seed")) {
      host_->SeedRandomEngine(
          static_cast<std::uint64_t>(options.integers.at("seed")));
    }
    if (stage_options.contains("stop") &&
        stage_options.at("stop").is_object() &&
        stage_options.at("stop").contains("eos_token_ids")) {
      for (const auto& token : stage_options.at("stop").at("eos_token_ids")) {
        eos_tokens_.insert(token.get<std::int64_t>());
      }
    }
    // Measured before the first step replaces the prompt with a single
    // token, so the budget reflects the prompt this run started from.
    maximum_tokens_ = host_->MaximumTokens(stage, options);
  }

  [[nodiscard]] std::optional<StageEvent> StepLocked(
      StageStepContext& context) override {
    if (stop_requested_ || emitted_steps_ >= maximum_tokens_) {
      return std::nullopt;
    }
    NamedTensors step_outputs = host_->StepStage(
        stage_->name,
        context.ConsumeInputs(),
        context.overrides(),
        context.options());
    // Sampling reads the whole logits tensor on the host, so it is bracketed
    // rather than checked inside its per-token loops.
    context.options().cancellation.ThrowIfCancellationRequested();
    Tensor tokens =
        do_sample_
            ? host_->SampleTokens(
                  host_->StageLogits(*stage_),
                  sampling_,
                  context.options(),
                  generated_)
            : host_->GreedyTokens(host_->StageLogits(*stage_));
    context.options().cancellation.ThrowIfCancellationRequested();
    auto mutable_values = std::span(
        reinterpret_cast<std::int64_t*>(tokens.mutable_bytes().data()),
        tokens.element_count());
    if (finished_lanes_.empty()) {
      finished_lanes_.assign(mutable_values.size(), false);
    }
    const std::int64_t eos = eos_tokens_.empty() ? 0 : *eos_tokens_.begin();
    for (std::size_t batch_index = 0;
         batch_index < mutable_values.size();
         ++batch_index) {
      if (finished_lanes_[batch_index]) {
        mutable_values[batch_index] = eos;
      } else if (eos_tokens_.contains(mutable_values[batch_index])) {
        finished_lanes_[batch_index] = true;
      }
    }
    const auto values = tokens.values<std::int64_t>();
    generated_.emplace_back(values.begin(), values.end());
    host_->SetStageTokenInput(*stage_, tokens);
    if (!eos_tokens_.empty() &&
        std::ranges::all_of(
            finished_lanes_, [](bool value) { return value; })) {
      // The old loop broke here; the incremental run reports the stop on the
      // next Step instead, so every run ends with one completed event.
      stop_requested_ = true;
    }
    ++emitted_steps_;

    StageEvent event;
    event.kind = StageEventKind::token;
    event.token_ids = std::move(tokens);
    event.outputs = std::move(step_outputs);
    return event;
  }

  // Packs the generated history batch-major, the layout the previous
  // RunAutoregressive published, and adds it without displacing a manifest
  // output of the same name.
  [[nodiscard]] NamedTensors FinalOutputsLocked(StageStepContext&) override {
    NamedTensors result = host_->CollectOutputs();
    if (generated_.empty()) {
      return result;
    }
    const std::size_t batch = generated_.front().size();
    std::vector<std::int64_t> flattened(batch * generated_.size());
    for (std::size_t step = 0; step < generated_.size(); ++step) {
      for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
        flattened[batch_index * generated_.size() + step] =
            generated_[step][batch_index];
      }
    }
    result.emplace(
        "generated_token_ids",
        Int64Tensor(
            {
                static_cast<std::int64_t>(batch),
                static_cast<std::int64_t>(generated_.size()),
            },
            flattened));
    return result;
  }

 private:
  StageExecutionHost* host_;
  const PipelineStage* stage_;
  Json sampling_;
  bool do_sample_{false};
  std::set<std::int64_t> eos_tokens_;
  std::size_t maximum_tokens_{0};
  // The generated history, the per-lane end-of-sequence latch, and this
  // run's own step count: the loop state that used to be locals of
  // RunAutoregressive, kept here so a session snapshot never captures half of
  // an in-flight decode.
  std::vector<std::vector<std::int64_t>> generated_;
  std::vector<bool> finished_lanes_;
  std::size_t emitted_steps_{0};
  bool stop_requested_{false};
};

}  // namespace

std::unique_ptr<StageExecutor> MakeAutoregressiveStageExecutor(
    StageExecutionHost& host,
    const PipelineStage& stage,
    const NamedTensors& inputs,
    const PipelineRunOptions& options) {
  return std::make_unique<AutoregressiveStageExecutor>(
      host, stage, inputs, options);
}

}  // namespace onnx_world_model::detail
