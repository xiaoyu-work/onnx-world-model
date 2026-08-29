/**
 * @agent-file
 * @agent-purpose: Implements the opt-in telemetry collector behind PipelineTelemetryOptions: the pre-populated per-epoch counter slab, the atomic slab publication that makes ResetTelemetry a new epoch rather than a barrier, the component and stage recorders that classify an outcome by ErrorCode, the admission and device-transfer hooks, and the immutable PipelineTelemetrySnapshot that Pipeline::telemetry_snapshot() reports.
 * @agent-public-api: detail::PipelineTelemetry::PipelineTelemetry, detail::PipelineTelemetry::slab, detail::PipelineTelemetry::Reset, detail::PipelineTelemetry::Snapshot, detail::MakePipelineTelemetry, detail::SnapshotPipelineTelemetry, detail::ResetPipelineTelemetry, detail::RecordAdmissionQueued, detail::RecordAdmissionOutcome, detail::RecordDeviceToHostCopy, detail::RecordStageStep, detail::RecordStageCompletion, detail::TelemetryComponentScope constructor, destructor, RecordSuccess and RecordCurrentException, detail::TelemetryStageScope constructor, destructor, RecordSuccess and RecordCurrentException
 * @agent-invariants: Observability only: nothing here changes what the runtime executes, in what order, or with what result, and no recording path can throw into the code it measures -- every recorder is noexcept and a null collector makes each one a single branch. A slab is fully pre-populated when it is built, from the manifest's component and stage names plus SupportedStageKinds(), so a recording thread only ever looks an entry up; it never inserts, never rehashes, and never allocates. Counters are relaxed atomics and maxima are relaxed compare-exchange loops, so no reader may draw a happens-before conclusion from one. Reset serializes only its cold slab construction and publication so concurrent resets cannot publish epochs out of order; every recorder holds the slab it loaded for its whole operation without taking that mutex, so an execution that straddles a reset lands entirely in the epoch it started in and is absent from the new one. Snapshot loads one slab and copies each counter individually: fields are mutually consistent about their epoch and deliberately not one atomic instant, because the alternative is a lock on the execution path. Outcomes are classified by the ErrorCode of the exception being handled -- cancelled and deadline_exceeded are their own buckets and everything else is a failure -- never by its message, and RecordCurrentException rethrows into its own handler chain to read that code, so it must be called from inside a catch block. A recorder that is destroyed without an explicit outcome records a failure, so an unwind through a path with no catch cannot silently drop one. Byte totals come from Tensor::size_bytes(), and a tensor with no buffer contributes nothing rather than throwing; component input residency uses the same predicate Tensor::CopyToCpu() uses to decide it can alias -- canonical CPU device and host-accessible -- so "device-resident" here means exactly "would have to be copied to be read on the host". Admission recording takes an index into SupportedStageKinds(), and an out-of-range index records nothing because the snapshot has no key for it.
 * @agent-side-effects: none beyond mutating its own counters: no I/O, no device work, and no call back into scheduler, session, or cancellation code.
 */

#include "pipeline_telemetry.hpp"

#include "pipeline_scheduler.hpp"

namespace onnx_world_model::detail {
namespace {

//: The clock every duration here is measured with. Steady, because these are
//: elapsed times rather than timestamps.
using Clock = std::chrono::steady_clock;

[[nodiscard]] std::uint64_t ElapsedNanoseconds(
    Clock::time_point started) noexcept {
  const auto elapsed = Clock::now() - started;
  const auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  // A steady clock cannot go backwards, but clamping costs one predictable
  // branch and keeps a counter from wrapping if a platform ever disagrees.
  return nanoseconds <= 0 ? 0 : static_cast<std::uint64_t>(nanoseconds);
}

void RecordDuration(
    TelemetryCounter& total,
    TelemetryCounter& maximum,
    std::uint64_t duration_ns) noexcept {
  AddCounter(total, duration_ns);
  RaiseCounterMaximum(maximum, duration_ns);
}

//: Exactly the condition Tensor::CopyToCpu() uses to hand its source back
//: without copying. Anything else has to be materialized before the host can
//: read it, which is what makes it device-resident for these counters.
[[nodiscard]] bool IsCanonicalHostTensor(const Tensor& tensor) noexcept {
  // Checked first because it is noexcept and false for a tensor with no
  // buffer at all, which is what keeps device() from throwing below.
  return tensor.is_host_accessible() && tensor.device() == TensorDevice{};
}

//: Summed Tensor::size_bytes() of `tensors`. A tensor with no buffer has no
//: byte count to report and contributes nothing rather than throwing, because
//: a metrics path must never fail the work it measures.
[[nodiscard]] std::uint64_t TotalBytes(const NamedTensors& tensors) noexcept {
  std::uint64_t total = 0;
  for (const auto& [name, tensor] : tensors) {
    if (tensor.buffer() == nullptr) {
      continue;
    }
    total += static_cast<std::uint64_t>(tensor.size_bytes());
  }
  return total;
}

//: The counter an ErrorCode belongs in. Cancellation and deadlines are
//: outcomes of their own; everything else is a failure.
[[nodiscard]] TelemetryCounter& OutcomeBucket(
    ErrorCode code,
    TelemetryCounter& cancelled,
    TelemetryCounter& deadline,
    TelemetryCounter& failed) noexcept {
  switch (code) {
    case ErrorCode::cancelled:
      return cancelled;
    case ErrorCode::deadline_exceeded:
      return deadline;
    default:
      return failed;
  }
}

}  // namespace

PipelineTelemetry::PipelineTelemetry(const PipelineManifest& manifest) {
  component_names_.reserve(manifest.components().size());
  for (const PipelineComponent& component : manifest.components()) {
    component_names_.push_back(component.name);
  }
  stage_names_.reserve(manifest.stages().size());
  for (const PipelineStage& stage : manifest.stages()) {
    stage_names_.push_back(stage.name);
  }
  slab_.store(
      BuildSlab(next_epoch_++),
      std::memory_order_release);
}

std::shared_ptr<TelemetrySlab> PipelineTelemetry::BuildSlab(
    std::uint64_t epoch) const {
  auto slab = std::make_shared<TelemetrySlab>();
  slab->epoch = epoch;
  // Reserved and filled here, once, so that no execution ever inserts into
  // these maps: a recording thread finds an entry or finds nothing.
  slab->components.reserve(component_names_.size());
  for (const std::string& name : component_names_) {
    slab->components.try_emplace(name);
  }
  slab->stages.reserve(stage_names_.size());
  for (const std::string& name : stage_names_) {
    slab->stages.try_emplace(name);
  }
  return slab;
}

std::shared_ptr<TelemetrySlab> PipelineTelemetry::slab() const noexcept {
  return slab_.load(std::memory_order_acquire);
}

void PipelineTelemetry::Reset() {
  // Reset is a cold control path. Serializing construction and publication
  // keeps epochs monotonic even when Pipeline copies reset concurrently;
  // recording never takes this mutex.
  std::scoped_lock lock(reset_mutex_);
  slab_.store(
      BuildSlab(next_epoch_++),
      std::memory_order_release);
}

PipelineTelemetrySnapshot PipelineTelemetry::Snapshot() const {
  const std::shared_ptr<TelemetrySlab> slab = this->slab();
  PipelineTelemetrySnapshot result;
  result.enabled = true;
  result.epoch = slab->epoch;

  result.components.reserve(slab->components.size());
  for (const auto& [name, entry] : slab->components) {
    PipelineComponentStats stats;
    stats.successful_calls = entry.successful_calls.load(std::memory_order_relaxed);
    stats.failed_calls = entry.failed_calls.load(std::memory_order_relaxed);
    stats.cancelled_calls = entry.cancelled_calls.load(std::memory_order_relaxed);
    stats.deadline_exceeded_calls =
        entry.deadline_exceeded_calls.load(std::memory_order_relaxed);
    stats.total_duration_ns =
        entry.total_duration_ns.load(std::memory_order_relaxed);
    stats.max_duration_ns = entry.max_duration_ns.load(std::memory_order_relaxed);
    stats.input_bytes = entry.input_bytes.load(std::memory_order_relaxed);
    stats.output_bytes = entry.output_bytes.load(std::memory_order_relaxed);
    result.components.emplace(name, stats);
  }

  result.stages.reserve(slab->stages.size());
  for (const auto& [name, entry] : slab->stages) {
    PipelineStageStats stats;
    stats.successful_executions =
        entry.successful_executions.load(std::memory_order_relaxed);
    stats.failed_executions =
        entry.failed_executions.load(std::memory_order_relaxed);
    stats.cancelled_executions =
        entry.cancelled_executions.load(std::memory_order_relaxed);
    stats.deadline_exceeded_executions =
        entry.deadline_exceeded_executions.load(std::memory_order_relaxed);
    stats.steps = entry.steps.load(std::memory_order_relaxed);
    stats.completions = entry.completions.load(std::memory_order_relaxed);
    stats.total_execution_duration_ns =
        entry.total_execution_duration_ns.load(std::memory_order_relaxed);
    stats.max_execution_duration_ns =
        entry.max_execution_duration_ns.load(std::memory_order_relaxed);
    result.stages.emplace(name, stats);
  }

  // Every executable stage kind gets an entry whether or not admission ever
  // constrained it, exactly as the scheduling reading does, so a caller reads
  // a kind without testing for its key first.
  const auto& kinds = SupportedStageKinds();
  for (std::size_t index = 0; index < kinds.size(); ++index) {
    const TelemetryAdmissionEntry& entry = slab->admission[index];
    PipelineAdmissionStats stats;
    stats.queued_acquisitions =
        entry.queued_acquisitions.load(std::memory_order_relaxed);
    stats.admitted_acquisitions =
        entry.admitted_acquisitions.load(std::memory_order_relaxed);
    stats.cancelled_while_queued =
        entry.cancelled_while_queued.load(std::memory_order_relaxed);
    stats.deadline_while_queued =
        entry.deadline_while_queued.load(std::memory_order_relaxed);
    stats.total_wait_ns = entry.total_wait_ns.load(std::memory_order_relaxed);
    stats.max_wait_ns = entry.max_wait_ns.load(std::memory_order_relaxed);
    result.admission_by_stage_kind.emplace(std::string(kinds[index]), stats);
  }

  result.transfers.device_to_host_copies =
      slab->transfers.device_to_host_copies.load(std::memory_order_relaxed);
  result.transfers.device_to_host_bytes =
      slab->transfers.device_to_host_bytes.load(std::memory_order_relaxed);
  result.transfers.component_input_bytes_device_resident =
      slab->transfers.component_input_bytes_device_resident.load(
          std::memory_order_relaxed);
  result.transfers.component_input_bytes_host =
      slab->transfers.component_input_bytes_host.load(
          std::memory_order_relaxed);
  return result;
}

PipelineTelemetryPtr MakePipelineTelemetry(
    const PipelineTelemetryOptions& options,
    const PipelineManifest& manifest) {
  if (!options.enabled) {
    // Disabled is not a collector that ignores everything: it is no collector
    // at all, which is what makes every recording site one null test.
    return nullptr;
  }
  return std::make_shared<PipelineTelemetry>(manifest);
}

PipelineTelemetrySnapshot SnapshotPipelineTelemetry(
    const PipelineTelemetryPtr& telemetry) {
  if (telemetry == nullptr) {
    // A disabled or moved-from Pipeline collected nothing, so the honest
    // answer is the disabled reading rather than an error or a map of zeros
    // that would imply collection happened.
    return {};
  }
  return telemetry->Snapshot();
}

void ResetPipelineTelemetry(const PipelineTelemetryPtr& telemetry) {
  if (telemetry == nullptr) {
    return;
  }
  telemetry->Reset();
}

void RecordAdmissionQueued(
    const PipelineTelemetryPtr& telemetry,
    std::size_t kind_index) noexcept {
  if (telemetry == nullptr || kind_index >= SupportedStageKinds().size()) {
    return;
  }
  const std::shared_ptr<TelemetrySlab> slab = telemetry->slab();
  AddCounter(slab->admission[kind_index].queued_acquisitions, 1);
}

void RecordAdmissionOutcome(
    const PipelineTelemetryPtr& telemetry,
    std::size_t kind_index,
    TelemetryAdmissionOutcome outcome,
    std::uint64_t wait_ns) noexcept {
  if (telemetry == nullptr || kind_index >= SupportedStageKinds().size()) {
    return;
  }
  const std::shared_ptr<TelemetrySlab> slab = telemetry->slab();
  TelemetryAdmissionEntry& entry = slab->admission[kind_index];
  switch (outcome) {
    case TelemetryAdmissionOutcome::admitted:
      AddCounter(entry.admitted_acquisitions, 1);
      break;
    case TelemetryAdmissionOutcome::cancelled:
      AddCounter(entry.cancelled_while_queued, 1);
      break;
    case TelemetryAdmissionOutcome::deadline_exceeded:
      AddCounter(entry.deadline_while_queued, 1);
      break;
  }
  RecordDuration(entry.total_wait_ns, entry.max_wait_ns, wait_ns);
}

void RecordDeviceToHostCopy(
    const PipelineTelemetryPtr& telemetry,
    std::uint64_t bytes) noexcept {
  if (telemetry == nullptr) {
    return;
  }
  const std::shared_ptr<TelemetrySlab> slab = telemetry->slab();
  AddCounter(slab->transfers.device_to_host_copies, 1);
  AddCounter(slab->transfers.device_to_host_bytes, bytes);
}

void RecordStageStep(
    const PipelineTelemetryPtr& telemetry,
    std::string_view stage) noexcept {
  if (telemetry == nullptr) {
    return;
  }
  const std::shared_ptr<TelemetrySlab> slab = telemetry->slab();
  const auto found = slab->stages.find(stage);
  if (found == slab->stages.end()) {
    return;
  }
  AddCounter(found->second.steps, 1);
}

void RecordStageCompletion(
    const PipelineTelemetryPtr& telemetry,
    std::string_view stage) noexcept {
  if (telemetry == nullptr) {
    return;
  }
  const std::shared_ptr<TelemetrySlab> slab = telemetry->slab();
  const auto found = slab->stages.find(stage);
  if (found == slab->stages.end()) {
    return;
  }
  AddCounter(found->second.completions, 1);
}

TelemetryComponentScope::TelemetryComponentScope(
    const PipelineTelemetryPtr& telemetry,
    std::string_view component,
    const NamedTensors& inputs) noexcept {
  if (telemetry == nullptr) {
    return;
  }
  // Loaded once and held for the whole call, so a reset in the middle of a
  // component pass cannot split that pass across two epochs.
  slab_ = telemetry->slab();
  const auto found = slab_->components.find(component);
  if (found == slab_->components.end()) {
    // Only reachable for a component the manifest this slab was built from
    // does not declare, which the package loader already rejects. There is no
    // key to record under, so this scope stays inert rather than inventing
    // one.
    slab_.reset();
    return;
  }
  entry_ = &found->second;

  std::uint64_t host_bytes = 0;
  std::uint64_t device_bytes = 0;
  for (const auto& [name, tensor] : inputs) {
    if (tensor.buffer() == nullptr) {
      continue;
    }
    const auto bytes = static_cast<std::uint64_t>(tensor.size_bytes());
    if (IsCanonicalHostTensor(tensor)) {
      host_bytes += bytes;
    } else {
      device_bytes += bytes;
    }
  }
  AddCounter(entry_->input_bytes, host_bytes + device_bytes);
  AddCounter(slab_->transfers.component_input_bytes_host, host_bytes);
  AddCounter(
      slab_->transfers.component_input_bytes_device_resident, device_bytes);

  // Started last, so the presentation accounting above is not charged to the
  // component's own duration.
  started_ = Clock::now();
  pending_ = true;
}

TelemetryComponentScope::~TelemetryComponentScope() {
  if (!pending_ || entry_ == nullptr) {
    return;
  }
  // Unwound without an explicit outcome, which only happens on a path with no
  // catch of its own. It still ended somehow, so it is recorded as a failure
  // rather than dropped.
  RecordOutcome(entry_->failed_calls);
}

void TelemetryComponentScope::RecordOutcome(TelemetryCounter& bucket) noexcept {
  if (!pending_ || entry_ == nullptr) {
    return;
  }
  pending_ = false;
  AddCounter(bucket, 1);
  RecordDuration(
      entry_->total_duration_ns,
      entry_->max_duration_ns,
      ElapsedNanoseconds(started_));
}

void TelemetryComponentScope::RecordSuccess(
    const NamedTensors& outputs) noexcept {
  if (!pending_ || entry_ == nullptr) {
    return;
  }
  TelemetryComponentEntry& entry = *entry_;
  RecordOutcome(entry.successful_calls);
  AddCounter(entry.output_bytes, TotalBytes(outputs));
}

void TelemetryComponentScope::RecordCurrentException() noexcept {
  if (!pending_ || entry_ == nullptr) {
    return;
  }
  TelemetryComponentEntry& entry = *entry_;
  try {
    // Rethrowing the exception being handled is how its ErrorCode is read
    // without the caller having to classify it, which is what keeps the
    // classification in one place.
    throw;
  } catch (const Error& error) {
    RecordOutcome(OutcomeBucket(
        error.code(),
        entry.cancelled_calls,
        entry.deadline_exceeded_calls,
        entry.failed_calls));
  } catch (...) {
    RecordOutcome(entry.failed_calls);
  }
}

TelemetryStageScope::TelemetryStageScope(
    const PipelineTelemetryPtr& telemetry,
    std::string_view stage) noexcept {
  if (telemetry == nullptr) {
    return;
  }
  slab_ = telemetry->slab();
  const auto found = slab_->stages.find(stage);
  if (found == slab_->stages.end()) {
    // An unknown stage name fails the execution itself with
    // ErrorCode::invalid_argument and belongs to no stage, so there is
    // nothing to attribute it to.
    slab_.reset();
    return;
  }
  entry_ = &found->second;
  started_ = Clock::now();
  pending_ = true;
}

TelemetryStageScope::~TelemetryStageScope() {
  if (!pending_ || entry_ == nullptr) {
    return;
  }
  RecordOutcome(entry_->failed_executions);
}

void TelemetryStageScope::RecordOutcome(TelemetryCounter& bucket) noexcept {
  if (!pending_ || entry_ == nullptr) {
    return;
  }
  pending_ = false;
  AddCounter(bucket, 1);
  RecordDuration(
      entry_->total_execution_duration_ns,
      entry_->max_execution_duration_ns,
      ElapsedNanoseconds(started_));
}

void TelemetryStageScope::RecordSuccess() noexcept {
  if (entry_ == nullptr) {
    return;
  }
  RecordOutcome(entry_->successful_executions);
}

void TelemetryStageScope::RecordCurrentException() noexcept {
  if (!pending_ || entry_ == nullptr) {
    return;
  }
  TelemetryStageEntry& entry = *entry_;
  try {
    throw;
  } catch (const Error& error) {
    RecordOutcome(OutcomeBucket(
        error.code(),
        entry.cancelled_executions,
        entry.deadline_exceeded_executions,
        entry.failed_executions));
  } catch (...) {
    RecordOutcome(entry.failed_executions);
  }
}

}  // namespace onnx_world_model::detail
