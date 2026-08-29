/**
 * @agent-file
 * @agent-purpose: Implements the opt-in telemetry collector behind PipelineTelemetryOptions: the pre-populated per-epoch counter slab, the atomic slab publication that makes ResetTelemetry a new epoch rather than a barrier, the component and stage recorders that classify an outcome by ErrorCode, the admission and device-transfer hooks, the per-run ONNX Runtime trace prefix, its file discovery and record retention, and the immutable PipelineTelemetrySnapshot that Pipeline::telemetry_snapshot() reports.
 * @agent-public-api: detail::ValidatePipelineTelemetryOptions, detail::PipelineTelemetry::PipelineTelemetry, detail::PipelineTelemetry::slab, detail::PipelineTelemetry::tracing, detail::PipelineTelemetry::trace_directory, detail::PipelineTelemetry::max_trace_records, detail::PipelineTelemetry::AllocateTraceId, detail::PipelineTelemetry::Reset, detail::PipelineTelemetry::Snapshot, detail::MakePipelineTelemetry, detail::SnapshotPipelineTelemetry, detail::ResetPipelineTelemetry, detail::RecordAdmissionQueued, detail::RecordAdmissionOutcome, detail::RecordDeviceToHostCopy, detail::RecordStageStep, detail::RecordStageCompletion, detail::TelemetryComponentScope constructor, destructor, profile_prefix, RecordSuccess and RecordCurrentException, detail::TelemetryStageScope constructor, destructor, RecordSuccess and RecordCurrentException
 * @agent-invariants: Observability never changes execution and every recorder is noexcept. Counter slabs are pre-populated, use relaxed atomics, and stay alive across epoch resets. Trace prefixes combine a process-lifetime nonce, process-wide ID, epoch, component, and pid; discovery accepts exactly one non-empty matching JSON file. Record slots are reserved before discovery, so the cap cannot be exceeded and dropped calls require no directory scan. Concurrent vector order is unspecified, trace_id recovers start order, and filesystem failures affect telemetry only.
 * @agent-side-effects: none beyond mutating its own counters on the counter paths: no I/O, no device work, and no call back into scheduler, session, or cancellation code. Validation may create the configured trace directory, and a traced call reads that directory's entries and one file size to publish its record.
 */

#include "pipeline_telemetry.hpp"

#include <random>
#include <system_error>

#include "pipeline_scheduler.hpp"

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace onnx_world_model::detail {
namespace {

//: The clock every duration here is measured with. Steady, because these are
//: elapsed times rather than timestamps.
using Clock = std::chrono::steady_clock;

//: The characters a component name may contribute to a file name. Everything
//: else is replaced, so a manifest name can never escape the configured trace
//: directory, invent a path separator, or produce a name a platform refuses.
[[nodiscard]] bool IsPortableNameCharacter(char character) noexcept {
  const auto value = static_cast<unsigned char>(character);
  if (value > 127) {
    return false;
  }
  return (value >= '0' && value <= '9') || (value >= 'A' && value <= 'Z') ||
         (value >= 'a' && value <= 'z') || character == '.' ||
         character == '_' || character == '-';
}

//: `component` reduced to a file-name fragment. Unsupported characters become
//: underscores rather than being dropped, so two different components cannot
//: collapse onto one fragment, and trailing dots and spaces are removed
//: because Windows silently strips them from a file name.
[[nodiscard]] std::string SanitizeComponentName(std::string_view component) {
  std::string sanitized;
  sanitized.reserve(component.size());
  for (const char character : component) {
    sanitized.push_back(
        IsPortableNameCharacter(character) ? character : '_');
  }
  while (!sanitized.empty() &&
         (sanitized.back() == '.' || sanitized.back() == ' ')) {
    sanitized.pop_back();
  }
  if (sanitized.empty()) {
    // Only reachable for a name made entirely of characters that cannot
    // appear in a file name. The epoch, trace identifier, and process id that
    // follow still make the prefix unique.
    sanitized = "component";
  }
  return sanitized;
}

//: This process, as ONNX Runtime's file names never carry it. It is what
//: keeps two processes tracing into one shared directory apart.
[[nodiscard]] std::uint64_t CurrentProcessId() noexcept {
#if defined(_WIN32)
  return static_cast<std::uint64_t>(::_getpid());
#else
  return static_cast<std::uint64_t>(::getpid());
#endif
}

//: A process-lifetime namespace. The wall-clock value keeps old files from a
//: prior process (even one with a recycled pid) out of this process's prefix,
//: and random entropy makes simultaneously started processes distinct.
[[nodiscard]] const std::string& ProcessTraceNamespace() {
  static const std::string value = [] {
    const auto started =
        std::chrono::system_clock::now().time_since_epoch().count();
    std::uint64_t entropy = static_cast<std::uint64_t>(started);
    try {
      std::random_device random;
      entropy ^= static_cast<std::uint64_t>(random()) << 32U;
      entropy ^= static_cast<std::uint64_t>(random());
    } catch (...) {
      entropy ^= CurrentProcessId();
    }
    return std::to_string(started) + "-" + std::to_string(entropy);
  }();
  return value;
}

//: Shared by every collector in this process. A Pipeline's trace ids remain
//: monotonic, but need not be contiguous when several Pipelines trace at once.
[[nodiscard]] std::atomic<std::uint64_t>& NextProcessTraceId() {
  static std::atomic<std::uint64_t> next{1};
  return next;
}

//: The unique prefix one traced call hands to ONNX Runtime. The component
//: names it, the process namespace and pid separate process lifetimes, and
//: the epoch and process-wide trace identifier separate concurrent collectors
//: and calls; ONNX Runtime then appends its own local timestamp and `.json`.
[[nodiscard]] std::filesystem::path TraceFilePrefix(
    const std::filesystem::path& directory,
    std::string_view component,
    std::uint64_t epoch,
    std::uint64_t trace_id) {
  std::string name = SanitizeComponentName(component);
  name += ".e";
  name += std::to_string(epoch);
  name += ".n";
  name += ProcessTraceNamespace();
  name += ".r";
  name += std::to_string(trace_id);
  name += ".p";
  name += std::to_string(CurrentProcessId());
  return directory / name;
}

//: What one trace-file scan found. `failed` is the only field that matters
//: when it is true: the record is published with no path and no size.
struct TraceDiscovery {
  std::filesystem::path path;
  std::uint64_t size_bytes{0};
  bool failed{true};
};

//: The absolute form of `path` when the filesystem will produce one, and
//: `path` itself otherwise. A record that cannot be made absolute is still
//: more useful than no record, so this never fails the discovery.
[[nodiscard]] std::filesystem::path ResolvePath(
    const std::filesystem::path& path) {
  std::error_code code;
  std::filesystem::path resolved = std::filesystem::weakly_canonical(path, code);
  if (!code && !resolved.empty()) {
    return resolved;
  }
  code.clear();
  resolved = std::filesystem::absolute(path, code);
  if (!code && !resolved.empty()) {
    return resolved;
  }
  return path;
}

//: Finds the one file ONNX Runtime wrote for `prefix`. It scans only the
//: directory the prefix names -- never a subdirectory and never the working
//: directory -- and accepts exactly one match whose name begins with the
//: prefix's file name followed by the separator ONNX Runtime inserts and ends
//: with `.json`. Zero matches, several matches, a filesystem refusal, or an
//: empty file is a profiling failure, which is a fact about the trace and
//: never about the model call that produced it.
[[nodiscard]] TraceDiscovery DiscoverTraceFile(
    const std::filesystem::path& prefix) {
  using StringType = std::filesystem::path::string_type;
  // Built through std::filesystem::path so the literals are in the platform's
  // native character type without this file naming that type.
  static const StringType separator = std::filesystem::path("_").native();
  static const StringType json_suffix = std::filesystem::path(".json").native();

  const std::filesystem::path directory = prefix.parent_path();
  const StringType expected = prefix.filename().native() + separator;

  std::error_code code;
  std::filesystem::directory_iterator entry(directory, code);
  if (code) {
    return {};
  }
  const std::filesystem::directory_iterator end;
  TraceDiscovery discovery;
  std::size_t matches = 0;
  for (; entry != end; entry.increment(code)) {
    if (code) {
      return {};
    }
    const StringType name = entry->path().filename().native();
    if (!name.starts_with(expected) || !name.ends_with(json_suffix)) {
      continue;
    }
    ++matches;
    if (matches > 1) {
      // Two files under a prefix that is unique per call means the naming
      // assumption no longer holds. Reporting a failure is honest; picking
      // one would be a guess.
      return {};
    }
    const auto size = std::filesystem::file_size(entry->path(), code);
    if (code || size == 0) {
      // ONNX Runtime creates the file when the run starts, so an empty file
      // is a run that failed before it wrote anything -- a trace that exists
      // but says nothing.
      code.clear();
      return {};
    }
    discovery.path = ResolvePath(entry->path());
    discovery.size_bytes = static_cast<std::uint64_t>(size);
    discovery.failed = false;
  }
  return matches == 1 ? discovery : TraceDiscovery{};
}

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

//: The same classification as a value, for the one trace record a traced call
//: publishes. It is derived from the ErrorCode exactly as the counters are,
//: so a record can never disagree with the counter it was recorded beside.
[[nodiscard]] PipelineCallOutcome OutcomeOf(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::cancelled:
      return PipelineCallOutcome::cancelled;
    case ErrorCode::deadline_exceeded:
      return PipelineCallOutcome::deadline_exceeded;
    default:
      return PipelineCallOutcome::failure;
  }
}

}  // namespace

void ValidatePipelineTelemetryOptions(const PipelineTelemetryOptions& options) {
  if (options.trace_directory.empty()) {
    // Counters only, which is the configuration every earlier release had and
    // the one that touches no filesystem at all.
    return;
  }
  if (!options.enabled) {
    throw Error(
        ErrorCode::invalid_argument,
        "Telemetry trace_directory requires telemetry to be enabled");
  }
  if (options.max_trace_records == 0) {
    throw Error(
        ErrorCode::invalid_argument,
        "Telemetry max_trace_records must be greater than zero when a trace "
        "directory is configured");
  }
  // Created here, once, on the cold path: a component call must never have to
  // create a directory, and a directory that cannot exist has to fail loading
  // rather than every run.
  std::error_code code;
  std::filesystem::create_directories(options.trace_directory, code);
  code.clear();
  if (!std::filesystem::is_directory(options.trace_directory, code)) {
    throw Error(
        ErrorCode::invalid_argument,
        "Telemetry trace_directory is not a usable directory: " +
            options.trace_directory.string());
  }
}

PipelineTelemetry::PipelineTelemetry(
    const PipelineTelemetryOptions& options,
    const PipelineManifest& manifest)
    : trace_directory_(options.trace_directory),
      max_trace_records_(options.max_trace_records) {
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

bool PipelineTelemetry::tracing() const noexcept {
  return !trace_directory_.empty();
}

const std::filesystem::path& PipelineTelemetry::trace_directory()
    const noexcept {
  return trace_directory_;
}

std::size_t PipelineTelemetry::max_trace_records() const noexcept {
  return max_trace_records_;
}

std::uint64_t PipelineTelemetry::AllocateTraceId() noexcept {
  return NextProcessTraceId().fetch_add(1, std::memory_order_relaxed);
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

  result.dropped_traces = slab->traces.dropped.load(std::memory_order_relaxed);
  result.failed_traces = slab->traces.failed.load(std::memory_order_relaxed);
  {
    // The one lock a reading takes, and it guards the record vector alone: no
    // execution path holds it while running a model, and no scheduler or
    // session lock is involved, so copying records can never block work.
    std::scoped_lock lock(slab->traces.mutex);
    result.traces = slab->traces.records;
  }
  return result;
}

PipelineTelemetryPtr MakePipelineTelemetry(
    const PipelineTelemetryOptions& options,
    const PipelineManifest& manifest) {
  if (!options.enabled) {
    // Disabled is not a collector that ignores everything: it is no collector
    // at all, which is what makes every recording site one null test.
    // Validation still runs, so a trace directory supplied without enabling
    // telemetry is rejected rather than quietly ignored.
    ValidatePipelineTelemetryOptions(options);
    return nullptr;
  }
  // Idempotent, and repeated here because a Pipeline may be constructed
  // directly from a package without going through Pipeline::Load.
  ValidatePipelineTelemetryOptions(options);
  return std::make_shared<PipelineTelemetry>(options, manifest);
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

  if (telemetry->tracing()) {
    // The only allocating work a recorder does before the call, and it is
    // done here rather than lazily so the prefix exists before the model
    // runs. A failure to build it disables tracing for this call alone and is
    // counted as a profiling failure: the call itself still runs and is still
    // measured.
    max_trace_records_ = telemetry->max_trace_records();
    trace_id_ = telemetry->AllocateTraceId();
    try {
      component_.assign(component);
      profile_prefix_ = TraceFilePrefix(
          telemetry->trace_directory(), component, slab_->epoch, trace_id_);
    } catch (...) {
      profile_prefix_.clear();
      component_.clear();
      AddCounter(slab_->traces.failed, 1);
    }
  }

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

  // Started last, so neither the presentation accounting above nor the trace
  // prefix is charged to the component's own duration.
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
  RecordOutcome(entry_->failed_calls, PipelineCallOutcome::failure);
}

const std::filesystem::path& TelemetryComponentScope::profile_prefix()
    const noexcept {
  return profile_prefix_;
}

void TelemetryComponentScope::RecordOutcome(
    TelemetryCounter& bucket,
    PipelineCallOutcome outcome) noexcept {
  if (!pending_ || entry_ == nullptr) {
    return;
  }
  pending_ = false;
  // Read before anything else this function does, so trace discovery is never
  // charged to the component's measured duration.
  const std::uint64_t duration_ns = ElapsedNanoseconds(started_);
  AddCounter(bucket, 1);
  RecordDuration(
      entry_->total_duration_ns, entry_->max_duration_ns, duration_ns);
  if (!profile_prefix_.empty()) {
    FinalizeTrace(outcome, duration_ns);
  }
}

void TelemetryComponentScope::FinalizeTrace(
    PipelineCallOutcome outcome,
    std::uint64_t duration_ns) noexcept {
  bool counted_failure = false;
  bool reserved_record = false;
  try {
    {
      std::scoped_lock lock(slab_->traces.mutex);
      if (slab_->traces.records.size() +
              slab_->traces.pending_records >=
          max_trace_records_) {
        AddCounter(slab_->traces.dropped, 1);
        return;
      }
      ++slab_->traces.pending_records;
      reserved_record = true;
    }

    const TraceDiscovery discovery = DiscoverTraceFile(profile_prefix_);
    if (discovery.failed) {
      AddCounter(slab_->traces.failed, 1);
      counted_failure = true;
    }
    PipelineTraceRecord record;
    record.epoch = slab_->epoch;
    record.trace_id = trace_id_;
    record.component = component_;
    record.path = discovery.path;
    record.outcome = outcome;
    record.duration_ns = duration_ns;
    record.size_bytes = discovery.size_bytes;
    record.profiling_failed = discovery.failed;

    std::scoped_lock lock(slab_->traces.mutex);
    --slab_->traces.pending_records;
    reserved_record = false;
    slab_->traces.records.push_back(std::move(record));
  } catch (...) {
    // A recorder may never throw into the code it measures, so a failure to
    // allocate or publish a record is reported as a profiling failure and
    // nothing more.
    if (!counted_failure) {
      AddCounter(slab_->traces.failed, 1);
    }
    if (reserved_record) {
      try {
        std::scoped_lock lock(slab_->traces.mutex);
        --slab_->traces.pending_records;
      } catch (...) {
      }
    }
  }
}

void TelemetryComponentScope::RecordSuccess(
    const NamedTensors& outputs) noexcept {
  if (!pending_ || entry_ == nullptr) {
    return;
  }
  TelemetryComponentEntry& entry = *entry_;
  // The outcome first, because it is what reads the clock: summing the
  // returned bytes must not land inside the duration this call is charged.
  RecordOutcome(entry.successful_calls, PipelineCallOutcome::success);
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
    RecordOutcome(
        OutcomeBucket(
            error.code(),
            entry.cancelled_calls,
            entry.deadline_exceeded_calls,
            entry.failed_calls),
        OutcomeOf(error.code()));
  } catch (...) {
    RecordOutcome(entry.failed_calls, PipelineCallOutcome::failure);
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
