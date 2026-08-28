/**
 * @agent-file
 * @agent-purpose: Implements the shared admission controller behind PipelineSchedulingOptions: per-stage-kind permit buckets, an oldest-first work-conserving pump over a FIFO queue of waiters, cancellation- and deadline-aware waiting, the RAII PipelineLease that returns a permit exactly once, and the consistent snapshot of the live admission state that Pipeline::scheduling_stats() reports.
 * @agent-public-api: detail::SupportedStageKinds, detail::ValidatePipelineSchedulingOptions, detail::MakePipelineScheduler, detail::PipelineScheduler::PipelineScheduler, detail::PipelineScheduler::Acquire, detail::PipelineScheduler::ReleasePermit, detail::PipelineScheduler::Stats, detail::SnapshotSchedulingStats, detail::PipelineLease constructors, destructor, move operations and Release, detail::AcquireExecutionLease
 * @agent-invariants: Admission scheduling only: nothing here merges, splits, reorders, or preempts work, and no execution is ever started or resumed by this file. Limits are fixed at construction, so `limit` is read without the mutex while `active_` and every bucket's `active` are touched only under it. An acquire whose kind has neither a global nor a per-kind limit returns an inert lease without locking or allocating. Every other acquire enqueues a shared waiter and immediately pumps under the same lock hold, so an eligible caller is granted before it ever waits and no later waiter of the same kind can pass an earlier one; the pump skips a waiter whose bucket is full, so a full kind never blocks a different eligible kind, and stops as soon as the global limit is reached, because nothing behind that point can be granted either. Cancellation ordering is the whole safety argument: the token is polled before the ticket is taken, the registration is created only after the waiter is enqueued and only with the scheduler mutex released, the callback takes the scheduler mutex and marks the waiter cancelled only if it was not already granted, and the registration is destroyed with the scheduler mutex released, so an inline callback from an already-cancelled token cannot self-deadlock and ~CancellationRegistration cannot wait for a callback that wants a lock this thread holds. A cancellation that raced a grant is detected by re-polling after the registration is gone, and the permit is released before the throw. The one path with no lease to unwind -- registering or waiting itself failing -- unregisters first and then abandons the ticket, returning the permit if the pump granted it while that thread was already unwinding, so even that path cannot leak one. PipelineLease::Release decrements the global count and the bucket exactly once, pumps, and notifies only the waiters that pump granted. Stats reads rather than mutates: it takes the same mutex, copies the global count, the queue depth, and every bucket count into stack arrays, tallies the queue by bucket pointer, and then builds its maps with the lock released, so it allocates nothing while holding it, never blocks admission for longer than those reads, and reports one internally consistent moment. Its per-kind maps always carry all six SupportedStageKinds() names, and a queued waiter with no bucket is counted in the total only. A moved-from Pipeline has no scheduler, and SnapshotSchedulingStats answers that with the same all-zero shape instead of throwing.
 * @agent-side-effects: Blocks the calling thread while a permit is unavailable, and runs its own waiter-removal callback on whichever thread cancels -- an application thread or the shared deadline watchdog.
 */

#include "pipeline_scheduler.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

#include "cancellation.hpp"
#include "onnx_world_model/error.hpp"

namespace onnx_world_model::detail {

const std::array<std::string_view, 6>& SupportedStageKinds() noexcept {
  static const std::array<std::string_view, 6> kinds{
      "single_pass",
      "autoregressive",
      "iterative",
      "state_transition",
      "composite",
      "on_demand",
  };
  return kinds;
}

void ValidatePipelineSchedulingOptions(
    const PipelineSchedulingOptions& options) {
  for (const auto& [kind, limit] : options.max_concurrent_by_stage_kind) {
    (void)limit;
    if (kind.empty()) {
      throw Error(
          ErrorCode::invalid_argument,
          "Pipeline scheduling stage kind must not be empty");
    }
    if (std::ranges::find(SupportedStageKinds(), kind) ==
        SupportedStageKinds().end()) {
      std::string supported;
      for (const std::string_view name : SupportedStageKinds()) {
        if (!supported.empty()) {
          supported += ", ";
        }
        supported += name;
      }
      throw Error(
          ErrorCode::invalid_argument,
          "Unknown pipeline scheduling stage kind '" + kind +
              "'; supported stage kinds are " + supported);
    }
  }
}

PipelineScheduler::PipelineScheduler(const PipelineSchedulingOptions& options)
    : global_limit_(options.max_concurrent_executions) {
  ValidatePipelineSchedulingOptions(options);
  const auto& kinds = SupportedStageKinds();
  for (std::size_t index = 0; index < kinds.size(); ++index) {
    const auto found =
        options.max_concurrent_by_stage_kind.find(std::string(kinds[index]));
    kinds_[index].limit = found == options.max_concurrent_by_stage_kind.end()
                              ? 0
                              : found->second;
  }
}

bool PipelineScheduler::Unlimited(
    const StageKindPermits* bucket) const noexcept {
  return global_limit_ == 0 && (bucket == nullptr || bucket->limit == 0);
}

StageKindPermits* PipelineScheduler::BucketFor(
    std::string_view kind) noexcept {
  const auto& kinds = SupportedStageKinds();
  const auto found = std::ranges::find(kinds, kind);
  if (found == kinds.end()) {
    // Manifest validation already rejected every other stage kind, so this is
    // unreachable for a loaded package. A kind with no bucket is simply
    // unconstrained per kind; the global limit still applies to it.
    return nullptr;
  }
  return &kinds_[static_cast<std::size_t>(found - kinds.begin())];
}

void PipelineScheduler::PumpLocked() {
  for (auto entry = queue_.begin(); entry != queue_.end();) {
    if (global_limit_ != 0 && active_ >= global_limit_) {
      // Nothing further back can be granted either, so this is the whole
      // scan rather than an early skip.
      break;
    }
    StageKindPermits* bucket = (*entry)->bucket;
    if (bucket != nullptr && bucket->limit != 0 &&
        bucket->active >= bucket->limit) {
      // This kind is full. Skipping rather than stopping is what keeps one
      // saturated kind from blocking the head of the queue for a different,
      // still-eligible kind. A later waiter of *this* kind hits the same
      // check, so it can never pass this one.
      ++entry;
      continue;
    }
    // Copied out before the erase, because the deque is the only other owner
    // of this node and the waiting thread reads the flags below.
    const std::shared_ptr<Waiter> granted = *entry;
    entry = queue_.erase(entry);
    ++active_;
    if (bucket != nullptr) {
      ++bucket->active;
    }
    granted->granted = true;
    granted->ready.notify_one();
  }
}

void PipelineScheduler::CancelWaiter(
    const std::shared_ptr<Waiter>& waiter) noexcept {
  std::scoped_lock lock(mutex_);
  if (waiter->granted || waiter->cancelled) {
    // A grant that already happened is not undone here: the acquiring thread
    // owns that permit and releases it once it re-polls its token.
    return;
  }
  waiter->cancelled = true;
  const auto found = std::ranges::find(queue_, waiter);
  if (found != queue_.end()) {
    queue_.erase(found);
  }
  // Dropping a waiter frees no permit, so this can only be a no-op; it runs
  // anyway so that "the queue is always drained as far as capacity allows"
  // holds after every mutation without a special case.
  PumpLocked();
  waiter->ready.notify_one();
}

void PipelineScheduler::AbandonWaiter(
    const std::shared_ptr<Waiter>& waiter) noexcept {
  bool granted = false;
  {
    std::scoped_lock lock(mutex_);
    granted = waiter->granted;
    if (!granted && !waiter->cancelled) {
      waiter->cancelled = true;
      const auto found = std::ranges::find(queue_, waiter);
      if (found != queue_.end()) {
        queue_.erase(found);
      }
    }
  }
  if (granted) {
    // The pump admitted this waiter while its own thread was unwinding, and
    // no lease was ever built for it, so the permit is returned here instead
    // of being lost for the scheduler's lifetime.
    ReleasePermit(waiter->bucket);
  }
}

void PipelineScheduler::ReleasePermit(StageKindPermits* bucket) noexcept {
  std::scoped_lock lock(mutex_);
  if (active_ > 0) {
    --active_;
  }
  if (bucket != nullptr && bucket->active > 0) {
    --bucket->active;
  }
  PumpLocked();
}

PipelineSchedulingStats PipelineScheduler::Stats() const {
  std::array<std::size_t, 6> active_by_bucket{};
  std::array<std::size_t, 6> queued_by_bucket{};
  std::size_t active = 0;
  std::size_t queued = 0;
  {
    // Plain counter reads and one queue walk: nothing here waits, allocates,
    // or calls out, so observing admission cannot perturb or deadlock it.
    std::scoped_lock lock(mutex_);
    active = active_;
    queued = queue_.size();
    for (std::size_t index = 0; index < kinds_.size(); ++index) {
      active_by_bucket[index] = kinds_[index].active;
    }
    for (const std::shared_ptr<Waiter>& waiter : queue_) {
      if (waiter->bucket == nullptr) {
        // A kind this runtime does not execute, held back only by the global
        // limit. It is in the total but belongs to no per-kind entry.
        continue;
      }
      ++queued_by_bucket[static_cast<std::size_t>(
          waiter->bucket - kinds_.data())];
    }
  }

  PipelineSchedulingStats stats;
  stats.active_executions = active;
  stats.queued_executions = queued;
  // Every executable stage kind gets an entry whether or not anything is
  // using it, so a caller reads a kind without testing for its key first.
  const auto& kinds = SupportedStageKinds();
  for (std::size_t index = 0; index < kinds.size(); ++index) {
    std::string name(kinds[index]);
    stats.active_by_stage_kind.emplace(name, active_by_bucket[index]);
    stats.queued_by_stage_kind.emplace(std::move(name), queued_by_bucket[index]);
  }
  return stats;
}

PipelineLease PipelineScheduler::Acquire(
    std::string_view kind,
    const CancellationToken& cancellation) {
  StageKindPermits* bucket = BucketFor(kind);
  if (Unlimited(bucket)) {
    return {};
  }
  // Polled before a ticket is taken, so a token that is already cancelled --
  // including a deadline that has already passed -- never occupies a queue
  // position at all.
  cancellation.ThrowIfCancellationRequested();

  auto waiter = std::make_shared<Waiter>();
  waiter->bucket = bucket;
  {
    std::scoped_lock lock(mutex_);
    queue_.push_back(waiter);
    // Enqueue and pump under one hold: a caller that fits is granted here and
    // never blocks, and a caller that does not fit is already behind every
    // earlier waiter of its own kind.
    PumpLocked();
    if (waiter->granted) {
      return PipelineLease(shared_from_this(), bucket);
    }
  }

  // Created only after the waiter is queued and only with the scheduler mutex
  // released. A token that is already cancelled runs this callback inline, on
  // this thread, which is exactly why the mutex must not be held.
  std::optional<CancellationRegistration> registration;
  bool granted = false;
  try {
    if (cancellation.cancellable()) {
      registration.emplace(
          cancellation,
          [self = shared_from_this(), waiter](CancellationReason) {
            self->CancelWaiter(waiter);
          });
    }
    std::unique_lock lock(mutex_);
    waiter->ready.wait(
        lock, [&waiter] { return waiter->granted || waiter->cancelled; });
    granted = waiter->granted;
  } catch (...) {
    // Registering or waiting failed outright, which leaves a ticket nobody
    // will ever consume. Unregister first -- the mutex is already released
    // here -- and then give the ticket up, returning the permit if the pump
    // granted it while this thread was unwinding.
    registration.reset();
    AbandonWaiter(waiter);
    throw;
  }
  // Destroyed with the mutex released: ~CancellationRegistration blocks while
  // its callback runs, and that callback wants this same mutex.
  registration.reset();

  if (!granted) {
    cancellation.ThrowIfCancellationRequested();
    // Only reachable if a claimed reason vanished, which the one-way latch
    // makes impossible; reporting it beats waiting forever.
    throw Error(
        ErrorCode::cancelled,
        "Pipeline execution was removed from the admission queue");
  }

  PipelineLease lease(shared_from_this(), bucket);
  // The grant may have won a race against a cancellation that was already
  // claimed. Re-polling after the registration is gone is what makes the
  // outcome the reason the token holds rather than the order two threads
  // happened to run in; the lease releases the permit before unwinding.
  cancellation.ThrowIfCancellationRequested();
  return lease;
}

std::shared_ptr<PipelineScheduler> MakePipelineScheduler(
    const PipelineSchedulingOptions& options) {
  return std::make_shared<PipelineScheduler>(options);
}

PipelineLease::PipelineLease(
    std::shared_ptr<PipelineScheduler> scheduler,
    StageKindPermits* bucket) noexcept
    : scheduler_(std::move(scheduler)), bucket_(bucket), released_(false) {}

PipelineLease::~PipelineLease() {
  Release();
}

PipelineLease::PipelineLease(PipelineLease&& other) noexcept
    : scheduler_(std::move(other.scheduler_)),
      bucket_(other.bucket_),
      released_(other.released_) {
  other.bucket_ = nullptr;
  other.released_ = true;
}

PipelineLease& PipelineLease::operator=(PipelineLease&& other) noexcept {
  if (this != &other) {
    Release();
    scheduler_ = std::move(other.scheduler_);
    bucket_ = other.bucket_;
    released_ = other.released_;
    other.bucket_ = nullptr;
    other.released_ = true;
  }
  return *this;
}

void PipelineLease::Release() noexcept {
  if (released_) {
    return;
  }
  released_ = true;
  if (scheduler_ != nullptr) {
    scheduler_->ReleasePermit(bucket_);
  }
  scheduler_.reset();
  bucket_ = nullptr;
}

PipelineLease AcquireExecutionLease(
    const std::shared_ptr<PipelineScheduler>& scheduler,
    std::string_view stage_kind,
    const CancellationToken& cancellation) {
  if (scheduler == nullptr) {
    return {};
  }
  return scheduler->Acquire(stage_kind, cancellation);
}

PipelineSchedulingStats SnapshotSchedulingStats(
    const std::shared_ptr<PipelineScheduler>& scheduler) {
  if (scheduler == nullptr) {
    // Only a moved-from Pipeline gets here. It admits nothing, so an all-zero
    // reading -- with every stage kind still present -- is the honest answer
    // rather than an error.
    PipelineSchedulingStats stats;
    for (const std::string_view name : SupportedStageKinds()) {
      std::string kind(name);
      stats.active_by_stage_kind.emplace(kind, 0);
      stats.queued_by_stage_kind.emplace(std::move(kind), 0);
    }
    return stats;
  }
  return scheduler->Stats();
}

}  // namespace onnx_world_model::detail
