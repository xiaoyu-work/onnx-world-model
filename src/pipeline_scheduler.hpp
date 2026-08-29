/**
 * @agent-file
 * @agent-purpose: Declares the internal admission controller a Pipeline shares with every session it creates -- the per-stage-kind permit buckets, the cancellation-aware FIFO waiting queue, the RAII PipelineLease that owns one admitted execution's permit, the read-only snapshot of that admission state a Pipeline reports as PipelineSchedulingStats, and the telemetry collector it records each acquisition's wait outcome into.
 * @agent-public-api: onnx_world_model::detail::ValidatePipelineSchedulingOptions, onnx_world_model::detail::MakePipelineScheduler, onnx_world_model::detail::StageKindPermits, onnx_world_model::detail::PipelineScheduler, onnx_world_model::detail::PipelineLease, onnx_world_model::detail::AcquireExecutionLease, onnx_world_model::detail::SnapshotSchedulingStats
 * @agent-invariants: Internal header that is not installed, so only the opaque PipelineScheduler pointer appears in the public Pipeline. This is admission scheduling and nothing else: it decides how many executions are inside the runtime at once and in what order queued ones enter, and it never merges, splits, reorders, or preempts the work. The stage kinds it accepts are not declared here: they are exactly detail::StageKindDefinitions() from stage_registry, which is also what manifest validation accepts, because a limit for a kind no manifest can declare would silently never apply. The configured limits are fixed when the scheduler is created and never change, so a bucket's limit is read without the mutex while its active count is read and written only under it; the bucket array is sized once by detail::kStageKindCount and ordered by the registry, so a resolved bucket pointer stays valid for the scheduler's lifetime and its index is the same key telemetry records under. A stage kind whose global and per-kind limits are both zero is unlimited and takes the fast path: no mutex, no allocation, an inert lease, and no telemetry at all, which is why an unlimited Pipeline reports no admission counts while its executions are still measured. Every other acquire enqueues a shared waiter and then pumps, so a caller that is immediately eligible never blocks and a caller that is not cannot be overtaken by a later waiter of its own kind; the pump scans in ticket order, skips a waiter whose kind bucket is full so a different eligible kind still enters, and stops at the first point where the global limit is reached. A lease decrements the global and per-kind counts exactly once, pumps, and notifies only the waiters it granted, so no exception, cancellation, or backend failure can leak a permit. Stats() is the only const reader, which is why the mutex is mutable: it takes that mutex, copies the global count, the queue depth, and every bucket's count, and tallies the queue by bucket, so one reading is internally consistent and observing admission cannot change it. Telemetry is recorded rather than read: an acquisition granted without waiting counts as admitted with no wait and is never counted as queued, only the path that actually waits starts a wait timer and counts as queued, and a grant that races a cancellation is recorded once, as admitted, because the permit really was granted. Lock order is CancellationState's callback mutex, then the scheduler mutex, then the session mutex, then ONNX Runtime; a registration is therefore never created or destroyed while the scheduler mutex is held, no callback ever runs model or user code, and telemetry recording -- atomic counter updates only -- takes none of those locks and is therefore ordered against nothing.
 * @agent-side-effects: Acquire blocks the calling thread while it waits for a permit, and it registers a cancellation callback that another thread -- including the shared deadline watchdog -- runs to remove that waiter.
 */

#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string_view>

#include "onnx_world_model/cancellation.hpp"
#include "onnx_world_model/pipeline.hpp"
#include "pipeline_telemetry.hpp"
#include "stage_registry.hpp"

namespace onnx_world_model::detail {

//: Rejects an empty or unknown per-kind key with ErrorCode::invalid_argument.
//: The keys it accepts are exactly the StageKindDefinitions() names.
//: Pipeline::Load calls this before it opens a single component file, so a
//: misspelled stage kind fails fast rather than after a package load.
void ValidatePipelineSchedulingOptions(
    const PipelineSchedulingOptions& options);

//: One stage kind's ceiling and its live count. `limit` is written once, when
//: the scheduler is built, and read without the lock; `active` is read and
//: written only under PipelineScheduler's mutex.
struct StageKindPermits {
  std::size_t limit{0};
  std::size_t active{0};
};

class PipelineScheduler;

//: One admitted execution's permit. It is a stack-scoped RAII value: a lease
//: is never stored on a StageRun or a session, so an idle handle can never
//: hold a permit. Releasing is idempotent, and the destructor is the normal
//: release path, which is what makes a permit survive an exception, a
//: cancellation, and a backend failure alike.
class PipelineLease {
 public:
  //: The inert lease the unlimited fast path returns: it owns no scheduler
  //: and releasing it does nothing.
  PipelineLease() noexcept = default;
  PipelineLease(
      std::shared_ptr<PipelineScheduler> scheduler,
      StageKindPermits* bucket) noexcept;
  ~PipelineLease();

  PipelineLease(PipelineLease&& other) noexcept;
  PipelineLease& operator=(PipelineLease&& other) noexcept;

  PipelineLease(const PipelineLease&) = delete;
  PipelineLease& operator=(const PipelineLease&) = delete;

  //: Returns the permit early. Calling it again, or destroying the lease
  //: afterwards, does nothing.
  void Release() noexcept;

 private:
  std::shared_ptr<PipelineScheduler> scheduler_;
  //: The resolved stage-kind bucket this permit was counted against, or null
  //: when only the global limit applies. It points into the scheduler's own
  //: fixed-size array, which the shared_ptr above keeps alive.
  StageKindPermits* bucket_{nullptr};
  bool released_{true};
};

//: The shared admission controller. One is created per Pipeline construction
//: and shared by every copy of that Pipeline, every session it creates, and
//: every StageRun those sessions produce.
class PipelineScheduler
    : public std::enable_shared_from_this<PipelineScheduler> {
 public:
  //: `telemetry` is the Pipeline's collector, or null when telemetry is
  //: disabled. It is recorded into and never read back here, so admission
  //: behaves identically whether or not anything is observing it.
  explicit PipelineScheduler(
      const PipelineSchedulingOptions& options,
      PipelineTelemetryPtr telemetry = {});

  PipelineScheduler(const PipelineScheduler&) = delete;
  PipelineScheduler& operator=(const PipelineScheduler&) = delete;

  //: Admits one execution of a stage of `kind`. AcquireExecutionLease is what
  //: callers use, because it also handles a null scheduler.
  [[nodiscard]] PipelineLease Acquire(
      std::string_view kind,
      const CancellationToken& cancellation);

  //: Returns one permit and lets the queue advance. Public only because
  //: PipelineLease releases through it and is not a member.
  void ReleasePermit(StageKindPermits* bucket) noexcept;

  //: One consistent reading of the admission state: the live global and
  //: per-bucket counts plus the queue's depth, broken down by the bucket each
  //: waiter is queued against. It takes the mutex, copies counters, and
  //: builds the result; it never waits, never runs a callback, and never
  //: touches session or model state, so an observer cannot perturb admission.
  //: A waiter with no bucket -- a stage kind this runtime does not execute,
  //: held back only by the global limit -- is counted in the total but in no
  //: per-kind entry.
  [[nodiscard]] PipelineSchedulingStats Stats() const;

 private:
  //: One queued acquire. Shared rather than reached through a stack pointer,
  //: so the cancelling thread can mark and drop a waiter whose own thread is
  //: simultaneously waking up.
  struct Waiter {
    StageKindPermits* bucket{nullptr};
    bool granted{false};
    bool cancelled{false};
    std::condition_variable ready;
  };

  //: True when neither the global limit nor `bucket`'s limit constrains this
  //: kind, which is the fast path every default-configured Pipeline takes.
  [[nodiscard]] bool Unlimited(const StageKindPermits* bucket) const noexcept;
  //: The bucket for `kind`, or null for a kind this runtime does not execute.
  //: The array is fixed at construction, so the returned pointer is stable.
  [[nodiscard]] StageKindPermits* BucketFor(std::string_view kind) noexcept;
  //: `bucket`'s index into StageKindDefinitions(), which is the key telemetry
  //: records admission under. A waiter with no bucket -- a stage kind this
  //: runtime does not execute -- has no key, so it returns the array size and
  //: every recording site treats that as "nothing to record".
  [[nodiscard]] std::size_t KindIndex(
      const StageKindPermits* bucket) const noexcept;

  //: Grants every waiter that fits, oldest first. Must hold the mutex.
  void PumpLocked();
  //: Drops a waiter that was cancelled or whose deadline fired, unless it was
  //: already granted. Runs on the cancelling thread, which may be the shared
  //: deadline watchdog, so it takes the mutex and touches nothing else.
  void CancelWaiter(const std::shared_ptr<Waiter>& waiter) noexcept;
  //: Gives up a queued acquire whose own thread is unwinding, which only
  //: happens if registering or waiting failed. If the pump granted it in the
  //: meantime the permit is returned here, because no lease was ever built.
  void AbandonWaiter(const std::shared_ptr<Waiter>& waiter) noexcept;

  std::size_t global_limit_{0};
  //: The Pipeline's telemetry collector, or null. Only ever recorded into,
  //: with atomic operations alone, so it can never call back into this
  //: scheduler and can never take a lock this file orders against.
  PipelineTelemetryPtr telemetry_;
  //: Mutable so a const reader -- Stats() is the only one -- can take it.
  //: Nothing under this lock waits, allocates a waiter, or calls back out.
  mutable std::mutex mutex_;
  std::size_t active_{0};
  //: Ordered by arrival, oldest at the front: the pump's scan order is this
  //: order, which is what makes admission FIFO within one stage kind.
  std::deque<std::shared_ptr<Waiter>> queue_;
  //: One entry per StageKindDefinitions() entry, in that order. Never
  //: resized, so element addresses are stable for the scheduler's lifetime.
  std::array<StageKindPermits, kStageKindCount> kinds_;
};

//: Validates `options` and returns the scheduler that enforces them, wired to
//: `telemetry` when the Pipeline enabled it.
[[nodiscard]] std::shared_ptr<PipelineScheduler> MakePipelineScheduler(
    const PipelineSchedulingOptions& options,
    PipelineTelemetryPtr telemetry = {});

//: Admits one execution of a stage of `stage_kind`, blocking until there is
//: room under both the global and the per-kind limit.
//:
//: A null scheduler and an unconstrained stage kind both return an inert
//: lease without taking a lock. Otherwise the caller takes a FIFO ticket and
//: waits; `cancellation` is polled before the ticket is taken and observed
//: while waiting, so an explicit cancel or a deadline claimed by the shared
//: watchdog releases the queue position and throws ErrorCode::cancelled or
//: ErrorCode::deadline_exceeded without the execution ever starting. A
//: cancellation that races a grant releases the permit before throwing.
//:
//: This is deliberately called before the session lock, so it never inspects
//: session state first: a queued caller may still find a conflicting active
//: run once it is admitted and fail then. Peeking before acquiring would be a
//: worse race, not a better one.
[[nodiscard]] PipelineLease AcquireExecutionLease(
    const std::shared_ptr<PipelineScheduler>& scheduler,
    std::string_view stage_kind,
    const CancellationToken& cancellation);

//: One reading of `scheduler`'s admission state, or the all-zero reading a
//: moved-from Pipeline reports. Both per-kind maps always carry every name in
//: StageKindDefinitions(), so a caller never has to test for a key.
[[nodiscard]] PipelineSchedulingStats SnapshotSchedulingStats(
    const std::shared_ptr<PipelineScheduler>& scheduler);

}  // namespace onnx_world_model::detail
