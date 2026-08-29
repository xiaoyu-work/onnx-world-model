/**
 * @agent-file
 * @agent-purpose: Declares the internal opt-in telemetry collector a Pipeline shares with its admission scheduler and with every session it creates -- the immutable per-epoch counter slab, the relaxed-atomic counter helpers, the RAII component and stage recorders, the admission and device-transfer recording hooks, and the epoch reset that publishes a fresh slab.
 * @agent-public-api: onnx_world_model::detail::PipelineTelemetryPtr, onnx_world_model::detail::TelemetryCounter, onnx_world_model::detail::AddCounter, onnx_world_model::detail::RaiseCounterMaximum, onnx_world_model::detail::TelemetryComponentEntry, onnx_world_model::detail::TelemetryStageEntry, onnx_world_model::detail::TelemetryAdmissionEntry, onnx_world_model::detail::TelemetryTransferEntry, onnx_world_model::detail::TelemetrySlab, onnx_world_model::detail::PipelineTelemetry, onnx_world_model::detail::MakePipelineTelemetry, onnx_world_model::detail::SnapshotPipelineTelemetry, onnx_world_model::detail::ResetPipelineTelemetry, onnx_world_model::detail::TelemetryAdmissionOutcome, onnx_world_model::detail::RecordAdmissionQueued, onnx_world_model::detail::RecordAdmissionOutcome, onnx_world_model::detail::RecordDeviceToHostCopy, onnx_world_model::detail::RecordStageStep, onnx_world_model::detail::RecordStageCompletion, onnx_world_model::detail::TelemetryComponentScope, onnx_world_model::detail::TelemetryStageScope, onnx_world_model::detail::RecordStageExecution
 * @agent-invariants: Internal header that is not installed, so only the opaque PipelineTelemetry pointer appears in the public Pipeline. Telemetry is opt-in: MakePipelineTelemetry returns null when it is disabled, every entry point here accepts a null collector, and a null collector makes each recording site one branch with no lock, allocation, atomic, or clock read. An enabled collector owns one std::atomic<std::shared_ptr<TelemetrySlab>>. A slab is pre-populated when it is created -- one entry per manifest component, one per manifest stage, one per SupportedStageKinds() name -- and is never inserted into or rehashed afterwards, so a recording thread only ever finds an entry or finds nothing. Every counter is a relaxed atomic and every maximum is a relaxed compare-exchange loop, so recording is wait-free apart from the standard library's internal atomic-shared_ptr lock, and the collector never takes the scheduler, session, or cancellation lock and never calls back into runtime code. Each recorder loads the slab once and holds that shared_ptr for its whole operation, so an operation that straddles a reset finishes into the slab it started in. Reset serializes only its cold build-and-publish path, stamps a fresh pre-populated slab with the next monotonic epoch, and publishes it atomically; recording never takes that reset mutex, and in-flight work intentionally lands in the previous epoch. Snapshot loads one slab and copies its counters individually, so each field is a valid recent value while the reading as a whole is not one atomic moment -- making it one would require a lock on the execution path, which this design refuses. A null or disabled collector snapshots as enabled=false, epoch 0, and empty maps rather than throwing. Outcomes are classified by ErrorCode, never by message, and the four outcome counters of a component call and of a stage execution are mutually exclusive; a recorder whose scope ends without an explicit outcome records a failure, so an unwind can never lose one.
 * @agent-side-effects: none beyond mutating its own counters; recording performs no I/O, no allocation on the hot path, and no device work.
 */

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_world_model/error.hpp"
#include "onnx_world_model/pipeline.hpp"

namespace onnx_world_model::detail {

//: The collector pointer every recording site takes. Null means telemetry is
//: disabled, which is the default and the only zero-overhead state.
class PipelineTelemetry;

using PipelineTelemetryPtr = std::shared_ptr<PipelineTelemetry>;

//: One counter. Relaxed ordering throughout: these are statistics, not
//: synchronization, and no reader draws a happens-before conclusion from one.
using TelemetryCounter = std::atomic<std::uint64_t>;

inline void AddCounter(TelemetryCounter& counter, std::uint64_t value) noexcept {
  counter.fetch_add(value, std::memory_order_relaxed);
}

//: Raises `counter` to `value` if `value` is larger. A relaxed CAS loop rather
//: than a lock, and it retries only while a concurrent writer raised the
//: maximum above what this thread last read.
inline void RaiseCounterMaximum(
    TelemetryCounter& counter,
    std::uint64_t value) noexcept {
  std::uint64_t observed = counter.load(std::memory_order_relaxed);
  while (value > observed &&
         !counter.compare_exchange_weak(
             observed,
             value,
             std::memory_order_relaxed,
             std::memory_order_relaxed)) {
  }
}

//: Lets a recorder find an entry by std::string_view without building a
//: std::string, so no recording site allocates.
struct TelemetryNameHash {
  using is_transparent = void;

  [[nodiscard]] std::size_t operator()(std::string_view name) const noexcept {
    return std::hash<std::string_view>{}(name);
  }
};

template <typename Entry>
using TelemetryEntryMap =
    std::unordered_map<std::string, Entry, TelemetryNameHash, std::equal_to<>>;

//: One manifest component's counters. Mirrors PipelineComponentStats field for
//: field, so the snapshot is a copy rather than a translation.
struct TelemetryComponentEntry {
  TelemetryCounter successful_calls{0};
  TelemetryCounter failed_calls{0};
  TelemetryCounter cancelled_calls{0};
  TelemetryCounter deadline_exceeded_calls{0};
  TelemetryCounter total_duration_ns{0};
  TelemetryCounter max_duration_ns{0};
  TelemetryCounter input_bytes{0};
  TelemetryCounter output_bytes{0};
};

//: One manifest stage's counters, mirroring PipelineStageStats.
struct TelemetryStageEntry {
  TelemetryCounter successful_executions{0};
  TelemetryCounter failed_executions{0};
  TelemetryCounter cancelled_executions{0};
  TelemetryCounter deadline_exceeded_executions{0};
  TelemetryCounter steps{0};
  TelemetryCounter completions{0};
  TelemetryCounter total_execution_duration_ns{0};
  TelemetryCounter max_execution_duration_ns{0};
};

//: One stage kind's admission counters, mirroring PipelineAdmissionStats.
struct TelemetryAdmissionEntry {
  TelemetryCounter queued_acquisitions{0};
  TelemetryCounter admitted_acquisitions{0};
  TelemetryCounter cancelled_while_queued{0};
  TelemetryCounter deadline_while_queued{0};
  TelemetryCounter total_wait_ns{0};
  TelemetryCounter max_wait_ns{0};
};

//: The pipeline-wide device-boundary counters, mirroring
//: PipelineTransferStats.
struct TelemetryTransferEntry {
  TelemetryCounter device_to_host_copies{0};
  TelemetryCounter device_to_host_bytes{0};
  TelemetryCounter component_input_bytes_device_resident{0};
  TelemetryCounter component_input_bytes_host{0};
};

//: One epoch's counters. Its maps are filled once, when it is built, and are
//: never inserted into again, so a recording thread performs a lookup and
//: never a rehash. Non-copyable because its counters are atomics; it is only
//: ever reached through a shared_ptr.
struct TelemetrySlab {
  std::uint64_t epoch{0};
  TelemetryEntryMap<TelemetryComponentEntry> components;
  TelemetryEntryMap<TelemetryStageEntry> stages;
  //: One entry per SupportedStageKinds() name, in that order.
  std::array<TelemetryAdmissionEntry, 6> admission;
  TelemetryTransferEntry transfers;
};

//: The shared collector. One is created per Pipeline construction when
//: telemetry is enabled, and it is shared by every copy of that Pipeline, its
//: admission scheduler, every session it creates, every fork of those
//: sessions, and every StageRun they produce.
class PipelineTelemetry {
 public:
  //: Pre-populates the first epoch from the manifest, so every component and
  //: stage that can ever be recorded already has an entry.
  explicit PipelineTelemetry(const PipelineManifest& manifest);

  PipelineTelemetry(const PipelineTelemetry&) = delete;
  PipelineTelemetry& operator=(const PipelineTelemetry&) = delete;

  //: The slab a recorder must hold for its whole operation. One atomic load;
  //: it never returns null.
  [[nodiscard]] std::shared_ptr<TelemetrySlab> slab() const noexcept;

  //: Publishes a fresh zeroed slab stamped with the next epoch. Work already
  //: in flight keeps recording into the slab it loaded, which is the previous
  //: epoch, so a reset is a new beginning rather than a barrier.
  void Reset();

  //: One reading. Copies each counter individually from one slab, so no field
  //: comes from a different epoch than any other, while the set as a whole is
  //: not a single instant.
  [[nodiscard]] PipelineTelemetrySnapshot Snapshot() const;

 private:
  [[nodiscard]] std::shared_ptr<TelemetrySlab> BuildSlab(
      std::uint64_t epoch) const;

  //: The manifest names every slab is pre-populated with, copied once so a
  //: reset never has to reach back into the package.
  std::vector<std::string> component_names_;
  std::vector<std::string> stage_names_;
  std::mutex reset_mutex_;
  std::uint64_t next_epoch_{1};
  std::atomic<std::shared_ptr<TelemetrySlab>> slab_;
};

//: The collector for `options`, or null when telemetry is disabled. Null is
//: the whole disabled implementation: every function below accepts it.
[[nodiscard]] PipelineTelemetryPtr MakePipelineTelemetry(
    const PipelineTelemetryOptions& options,
    const PipelineManifest& manifest);

//: One reading of `telemetry`, or the disabled reading -- enabled false,
//: epoch 0, empty maps -- that a disabled or moved-from Pipeline reports.
[[nodiscard]] PipelineTelemetrySnapshot SnapshotPipelineTelemetry(
    const PipelineTelemetryPtr& telemetry);

//: Starts a new epoch, or does nothing for a null collector.
void ResetPipelineTelemetry(const PipelineTelemetryPtr& telemetry);

//: How one admission acquisition ended. `admitted` wins a race against a
//: cancellation, because the permit really was granted.
enum class TelemetryAdmissionOutcome {
  admitted,
  cancelled,
  deadline_exceeded,
};

//: Records that an acquisition of the stage kind at `kind_index` had to wait.
//: An acquisition granted immediately never calls this, which is what makes
//: `queued_acquisitions` mean "had to wait" rather than "was acquired".
void RecordAdmissionQueued(
    const PipelineTelemetryPtr& telemetry,
    std::size_t kind_index) noexcept;

//: Records how one acquisition ended and how long it waited. `wait_ns` is 0
//: for an immediate grant. An out-of-range `kind_index` -- a stage kind this
//: runtime does not execute, which manifest validation already rejects --
//: records nothing, because the snapshot has no key for it.
void RecordAdmissionOutcome(
    const PipelineTelemetryPtr& telemetry,
    std::size_t kind_index,
    TelemetryAdmissionOutcome outcome,
    std::uint64_t wait_ns) noexcept;

//: Records one device-to-host materialization this runtime performed. A
//: source that was already canonical CPU memory is handed over without a copy
//: and must not reach here.
void RecordDeviceToHostCopy(
    const PipelineTelemetryPtr& telemetry,
    std::uint64_t bytes) noexcept;

//: Records one completed non-terminal stage step.
void RecordStageStep(
    const PipelineTelemetryPtr& telemetry,
    std::string_view stage) noexcept;

//: Records one terminal `completed` stage event.
void RecordStageCompletion(
    const PipelineTelemetryPtr& telemetry,
    std::string_view stage) noexcept;

//: Measures one component's model call. Construction records what the call
//: was handed -- byte totals and where those bytes lived -- and starts the
//: clock; exactly one outcome is then recorded, and the destructor records a
//: failure if the scope unwound without one, so an outcome is never lost.
class TelemetryComponentScope {
 public:
  TelemetryComponentScope(
      const PipelineTelemetryPtr& telemetry,
      std::string_view component,
      const NamedTensors& inputs) noexcept;
  ~TelemetryComponentScope();

  TelemetryComponentScope(const TelemetryComponentScope&) = delete;
  TelemetryComponentScope& operator=(const TelemetryComponentScope&) = delete;

  //: Records a successful call and the bytes it returned.
  void RecordSuccess(const NamedTensors& outputs) noexcept;
  //: Classifies the exception currently being handled by its ErrorCode. Must
  //: be called from inside a catch block.
  void RecordCurrentException() noexcept;

 private:
  void RecordOutcome(TelemetryCounter& bucket) noexcept;

  std::shared_ptr<TelemetrySlab> slab_;
  TelemetryComponentEntry* entry_{nullptr};
  std::chrono::steady_clock::time_point started_{};
  bool pending_{false};
};

//: Measures one stage execution -- one admission lease scope. Construction
//: starts the clock, so it belongs after the permit is granted and queue wait
//: stays out of the stage's duration.
class TelemetryStageScope {
 public:
  TelemetryStageScope(
      const PipelineTelemetryPtr& telemetry,
      std::string_view stage) noexcept;
  ~TelemetryStageScope();

  TelemetryStageScope(const TelemetryStageScope&) = delete;
  TelemetryStageScope& operator=(const TelemetryStageScope&) = delete;

  void RecordSuccess() noexcept;
  //: Classifies the exception currently being handled by its ErrorCode. Must
  //: be called from inside a catch block.
  void RecordCurrentException() noexcept;

 private:
  void RecordOutcome(TelemetryCounter& bucket) noexcept;

  std::shared_ptr<TelemetrySlab> slab_;
  TelemetryStageEntry* entry_{nullptr};
  std::chrono::steady_clock::time_point started_{};
  bool pending_{false};
};

//: Runs `body` as exactly one measured stage execution of `stage` and returns
//: whatever it returns. This exists so the four execution entry points cannot
//: drift: each one acquires its permit, then wraps the rest of its work here,
//: and the outcome is recorded once, by ErrorCode, on every path out.
template <typename Function>
decltype(auto) RecordStageExecution(
    const PipelineTelemetryPtr& telemetry,
    std::string_view stage,
    Function&& body) {
  TelemetryStageScope scope(telemetry, stage);
  try {
    if constexpr (std::is_void_v<decltype(body())>) {
      std::forward<Function>(body)();
      scope.RecordSuccess();
      return;
    } else {
      decltype(auto) result = std::forward<Function>(body)();
      scope.RecordSuccess();
      return result;
    }
  } catch (...) {
    scope.RecordCurrentException();
    throw;
  }
}

}  // namespace onnx_world_model::detail
