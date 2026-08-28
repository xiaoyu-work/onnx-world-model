/**
 * @agent-file
 * @agent-purpose: Implements the cancellation state machine behind CancellationToken and CancellationSource -- first-reason-wins claiming, blocking waits, the race-free callback registry that lets an in-flight ONNX Runtime call be terminated -- and the one process-wide DeadlineService that claims a deadline on time without a thread per request.
 * @agent-public-api: CancellationToken::cancelled, CancellationToken::reason, CancellationToken::deadline, CancellationToken::ThrowIfCancellationRequested, CancellationToken::WaitForCancellation, CancellationSource::CancellationSource, CancellationSource::WithDeadline, CancellationSource::WithTimeout, CancellationSource move operations and destructor, CancellationSource::token, CancellationSource::Cancel, CancellationSource::cancelled, CancellationSource::reason, CancellationSource::deadline, CancellationSource::WaitForCancellation, detail::DeadlineService, detail::CancellationState, detail::CancellationAccess, detail::CancellationRegistration
 * @agent-invariants: A reason is claimed exactly once, by a compare-exchange from none, so the first claim wins and callbacks run exactly once with that reason. A successful claim disarms the watchdog entry, then releases waiters, then runs callbacks; callbacks run while holding the same mutex Register and Unregister take, so a registration made after cancellation still runs inline and ~CancellationRegistration cannot return while its callback is running; every callback is invoked inside a catch-all because Cancel is noexcept. Waiters use their own mutex and condition variable, and Wait polls once before blocking while Claim takes that mutex before notifying, so no wakeup is lost in either direction. DeadlineService is a lazy immortal singleton: it is allocated with new, never destroyed, and never joined, so a process teardown that stops its detached worker cannot touch freed state. Its worker holds weak_ptr only, removes every due entry from both the schedule and the id index under the service mutex, and only then -- with that mutex released and before re-acquiring it -- locks each state, claims it, and drops the strong reference, so no consumer callback, state destructor, or singleton accessor ever runs under the service lock. Every wait is capped at one hour and looped, because a saturated timeout stores time_point::max and passing that to wait_until overflows. A deadline already due when the state is armed is never registered, so it stays a poll claim and an immediate Cancel() can still win. WithTimeout accepts the whole millisecond range: it compares the requested timeout against the headroom left on the steady clock, expressed in milliseconds, and saturates to time_point::max or time_point::min rather than letting the conversion or the addition overflow. A null state -- the default token and a moved-from source -- is never cancellable, never throws from a poll, and rejects a wait with ErrorCode::invalid_argument rather than blocking forever.
 * @agent-side-effects: Reads the steady clock, starts one detached watchdog thread on the first armed deadline, and runs consumer callbacks on the thread that cancels, polls, or fires the deadline.
 */

#include "cancellation.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_world_model/error.hpp"

namespace onnx_world_model {
namespace detail {

// One watchdog for the whole process. A deadline is a scheduled claim, not a
// thread: registering one costs a map node, and the single worker sleeps
// until the earliest of them.
class DeadlineService {
 public:
  using Clock = CancellationState::Clock;

  // Deliberately immortal. The worker is detached and never joined, so the
  // service must outlive every possible teardown order -- including a Windows
  // DLL_PROCESS_DETACH that kills the worker mid-wait -- rather than race a
  // destructor against it. Allocating with `new` and never deleting is the
  // whole point, not an oversight.
  [[nodiscard]] static DeadlineService& Instance() {
    static DeadlineService* const instance = new DeadlineService();
    return *instance;
  }

  DeadlineService(const DeadlineService&) = delete;
  DeadlineService& operator=(const DeadlineService&) = delete;

  // Returns the handle that disarms this entry. The state is held weakly, so
  // a registration never keeps a source alive past its last owner.
  [[nodiscard]] std::uint64_t Register(
      Clock::time_point deadline,
      std::weak_ptr<CancellationState> state) {
    std::uint64_t id = 0;
    {
      std::scoped_lock lock(mutex_);
      id = next_id_++;
      const Schedule::iterator entry =
          schedule_.emplace(deadline, Entry{id, std::move(state)});
      try {
        index_.emplace(id, entry);
      } catch (...) {
        // The two containers are one registration; leaving a scheduled entry
        // nothing can disarm would fire a deadline on a state that asked to
        // be removed. This rethrows rather than swallowing.
        schedule_.erase(entry);
        throw;
      }
      ++revision_;
    }
    condition_.notify_all();
    return id;
  }

  void Unregister(std::uint64_t id) noexcept {
    if (id == 0) {
      return;
    }
    {
      std::scoped_lock lock(mutex_);
      const auto found = index_.find(id);
      if (found == index_.end()) {
        // The worker removes an entry before it fires it, so a disarm that
        // finds nothing is the ordinary outcome of a deadline that just
        // expired, not an error.
        return;
      }
      schedule_.erase(found->second);
      index_.erase(found);
      ++revision_;
    }
    condition_.notify_all();
  }

 private:
  struct Entry {
    std::uint64_t id{0};
    std::weak_ptr<CancellationState> state;
  };
  using Schedule = std::multimap<Clock::time_point, Entry>;

  // A saturated timeout stores time_point::max, and handing that to
  // wait_until overflows the conversion every standard library does
  // internally. Capping each wait and looping keeps that instant expressible
  // at the cost of one wakeup per hour while such a source is alive.
  static constexpr std::chrono::hours kMaximumWait{1};

  DeadlineService() { std::thread(&DeadlineService::Run, this).detach(); }
  // Never runs: Instance() leaks the one instance on purpose.
  ~DeadlineService() = default;

  [[noreturn]] void Run() {
    std::vector<std::weak_ptr<CancellationState>> due;
    for (;;) {
      {
        std::unique_lock lock(mutex_);
        const Clock::time_point now = Clock::now();
        while (!schedule_.empty() && schedule_.begin()->first <= now) {
          const Schedule::iterator entry = schedule_.begin();
          // Collected before either container is touched, so a failed
          // allocation leaves the entry scheduled rather than dropped.
          due.push_back(std::move(entry->second.state));
          index_.erase(entry->second.id);
          schedule_.erase(entry);
        }
        if (due.empty()) {
          const Clock::time_point capped = now + kMaximumWait;
          const Clock::time_point until =
              schedule_.empty()
                  ? capped
                  : (std::min)(schedule_.begin()->first, capped);
          const std::uint64_t revision = revision_;
          condition_.wait_until(lock, until, [this, revision] {
            return revision_ != revision;
          });
          continue;
        }
      }
      // Outside the service lock on purpose. Claim runs consumer callbacks,
      // and dropping the last reference to a state re-enters Unregister, so
      // holding the lock here would be a self-deadlock and a re-entrancy hole
      // at once. Each strong reference dies at the end of its iteration, well
      // before the lock is taken again.
      for (std::weak_ptr<CancellationState>& weak : due) {
        if (const std::shared_ptr<CancellationState> state = weak.lock()) {
          state->Claim(CancellationReason::deadline_exceeded);
        }
      }
      due.clear();
    }
  }

  std::mutex mutex_;
  std::condition_variable condition_;
  Schedule schedule_;
  std::unordered_map<std::uint64_t, Schedule::iterator> index_;
  std::uint64_t next_id_{1};
  // Bumped by every registration change so a worker that is already asleep
  // re-reads the earliest deadline instead of waiting out the old one.
  std::uint64_t revision_{0};
};

CancellationState::~CancellationState() { DisarmDeadline(); }

void CancellationState::ArmDeadline(
    const std::shared_ptr<CancellationState>& self) {
  if (!deadline_.has_value() || *deadline_ <= Clock::now()) {
    // An already-due deadline stays a poll claim. Registering it would let
    // the watchdog claim deadline_exceeded before a caller that cancels
    // immediately after construction gets its turn, which would quietly
    // change which reason wins.
    return;
  }
  // Published before the watchdog can observe this state, so the worker never
  // reads it unset and never has to reach the singleton accessor that its own
  // constructor started.
  service_ = &DeadlineService::Instance();
  const std::uint64_t timer =
      service_->Register(*deadline_, std::weak_ptr<CancellationState>(self));
  timer_.store(timer, std::memory_order_release);
  if (reason_.load(std::memory_order_acquire) != CancellationReason::none) {
    // The deadline fired in the window between registering and storing the
    // handle, so the Claim that ran could not see it. Drop it here instead.
    DisarmDeadline();
  }
}

void CancellationState::DisarmDeadline() noexcept {
  const std::uint64_t timer = timer_.exchange(0, std::memory_order_acq_rel);
  if (timer != 0) {
    service_->Unregister(timer);
  }
}

CancellationReason CancellationState::Poll() noexcept {
  const CancellationReason claimed = reason_.load(std::memory_order_acquire);
  if (claimed != CancellationReason::none) {
    return claimed;
  }
  if (deadline_.has_value() && Clock::now() >= *deadline_) {
    Claim(CancellationReason::deadline_exceeded);
    return reason_.load(std::memory_order_acquire);
  }
  return CancellationReason::none;
}

CancellationReason CancellationState::Wait() {
  // Polling first both claims a deadline that was already due when this state
  // was armed -- the one case the watchdog deliberately does not cover -- and
  // makes a state that already stopped return without ever blocking.
  const CancellationReason polled = Poll();
  if (polled != CancellationReason::none) {
    return polled;
  }
  std::unique_lock lock(wait_mutex_);
  wait_condition_.wait(lock, [this] {
    return reason_.load(std::memory_order_acquire) != CancellationReason::none;
  });
  return reason_.load(std::memory_order_acquire);
}

void CancellationState::Claim(CancellationReason reason) noexcept {
  if (reason == CancellationReason::none) {
    return;
  }
  CancellationReason expected = CancellationReason::none;
  if (!reason_.compare_exchange_strong(
          expected,
          reason,
          std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return;
  }
  // Nothing is left for the watchdog to do, and a source that stopped early
  // should not keep the worker waking up for it.
  DisarmDeadline();
  {
    // Taking the wait mutex before notifying is what closes the lost-wakeup
    // window against a waiter that has read the reason but has not blocked
    // yet. Waiters are released before the callbacks, which are consumer
    // code, get their turn.
    const std::scoped_lock wait_lock(wait_mutex_);
  }
  wait_condition_.notify_all();
  // Holding the mutex across the callbacks is the whole point: Unregister
  // takes the same mutex, so a caller that destroys its registration cannot
  // race with a callback that is still touching what it captured.
  std::scoped_lock lock(mutex_);
  for (const auto& [registration, callback] : callbacks_) {
    (void)registration;
    if (callback == nullptr) {
      continue;
    }
    try {
      callback(reason);
    } catch (...) {
      // Cancel() is noexcept, so a misbehaving callback must not escape. The
      // remaining callbacks still run.
    }
  }
  callbacks_.clear();
}

std::uint64_t CancellationState::Register(Callback callback) {
  if (callback == nullptr) {
    throw Error(
        ErrorCode::invalid_argument,
        "Cancellation callback cannot be empty");
  }
  std::scoped_lock lock(mutex_);
  // Claim() flips the reason before it takes this mutex, so observing a
  // non-none reason here means the callbacks either already ran or are about
  // to; either way this late registration runs inline and is not stored.
  const CancellationReason claimed = reason_.load(std::memory_order_acquire);
  if (claimed != CancellationReason::none) {
    try {
      callback(claimed);
    } catch (...) {
    }
    return 0;
  }
  const std::uint64_t registration = next_registration_++;
  callbacks_.emplace_back(registration, std::move(callback));
  return registration;
}

void CancellationState::Unregister(std::uint64_t registration) noexcept {
  if (registration == 0) {
    return;
  }
  std::scoped_lock lock(mutex_);
  for (auto entry = callbacks_.begin(); entry != callbacks_.end(); ++entry) {
    if (entry->first == registration) {
      callbacks_.erase(entry);
      return;
    }
  }
}

const std::shared_ptr<CancellationState>& CancellationAccess::State(
    const CancellationToken& token) noexcept {
  return token.state_;
}

const std::shared_ptr<CancellationState>& CancellationAccess::State(
    const CancellationSource& source) noexcept {
  return source.state_;
}

void CancellationAccess::Cancel(
    const CancellationSource& source,
    CancellationReason reason) noexcept {
  if (source.state_ != nullptr) {
    source.state_->Claim(reason);
  }
}

CancellationRegistration::CancellationRegistration(
    const CancellationToken& token,
    CancellationState::Callback callback)
    : state_(CancellationAccess::State(token)) {
  if (state_ == nullptr) {
    return;
  }
  registration_ = state_->Register(std::move(callback));
}

CancellationRegistration::~CancellationRegistration() {
  if (state_ != nullptr) {
    state_->Unregister(registration_);
  }
}

}  // namespace detail

namespace {

[[nodiscard]] std::string ReasonMessage(CancellationReason reason) {
  return reason == CancellationReason::deadline_exceeded
             ? "Operation exceeded its deadline"
             : "Operation was cancelled";
}

}  // namespace

CancellationToken::CancellationToken(
    std::shared_ptr<detail::CancellationState> state) noexcept
    : state_(std::move(state)) {}

bool CancellationToken::cancelled() const noexcept {
  return reason() != CancellationReason::none;
}

CancellationReason CancellationToken::reason() const noexcept {
  return state_ == nullptr ? CancellationReason::none : state_->Poll();
}

std::optional<std::chrono::steady_clock::time_point>
CancellationToken::deadline() const noexcept {
  return state_ == nullptr ? std::nullopt : state_->deadline();
}

void CancellationToken::ThrowIfCancellationRequested() const {
  const CancellationReason claimed = reason();
  if (claimed == CancellationReason::none) {
    return;
  }
  throw Error(
      claimed == CancellationReason::deadline_exceeded
          ? ErrorCode::deadline_exceeded
          : ErrorCode::cancelled,
      ReasonMessage(claimed));
}

CancellationReason CancellationToken::WaitForCancellation() const {
  if (state_ == nullptr) {
    throw Error(
        ErrorCode::invalid_argument,
        "Cannot wait on a token that is not cancellable: nothing could ever "
        "release it");
  }
  return state_->Wait();
}

CancellationSource::CancellationSource()
    : state_(std::make_shared<detail::CancellationState>(std::nullopt)) {}

CancellationSource::CancellationSource(
    std::shared_ptr<detail::CancellationState> state) noexcept
    : state_(std::move(state)) {}

CancellationSource CancellationSource::WithDeadline(
    std::chrono::steady_clock::time_point deadline) {
  auto state = std::make_shared<detail::CancellationState>(deadline);
  // Armed here rather than in the constructor because the watchdog needs a
  // weak_ptr, which only exists once make_shared has returned. Nothing else
  // holds the state yet, so this is the one point where arming is safe.
  state->ArmDeadline(state);
  return CancellationSource(std::move(state));
}

CancellationSource CancellationSource::WithTimeout(
    std::chrono::milliseconds timeout) {
  using Clock = std::chrono::steady_clock;
  const Clock::time_point now = Clock::now();
  // `now + timeout` is what a caller means, but the steady clock counts in a
  // unit finer than a millisecond, so converting the timeout into the clock's
  // own duration overflows long before milliseconds::max(). Comparing against
  // the headroom that actually remains -- itself expressed in milliseconds,
  // where it cannot overflow -- keeps the addition in range and saturates
  // instead of wrapping into a deadline in the past.
  const auto forward = std::chrono::duration_cast<std::chrono::milliseconds>(
      Clock::time_point::max() - now);
  if (timeout >= forward) {
    return WithDeadline(Clock::time_point::max());
  }
  // On the negative side, conversion into the clock duration is the limiting
  // operation. Adding the non-negative current clock value cannot underflow
  // once that conversion is representable.
  const auto backward =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          Clock::duration::min());
  if (timeout <= backward) {
    // Saturating the other way keeps "already out of time" meaningful for a
    // timeout so negative that the sum would underflow.
    return WithDeadline(Clock::time_point::min());
  }
  return WithDeadline(now + timeout);
}

CancellationSource::CancellationSource(CancellationSource&&) noexcept = default;

CancellationSource& CancellationSource::operator=(
    CancellationSource&&) noexcept = default;

CancellationSource::~CancellationSource() = default;

CancellationToken CancellationSource::token() const noexcept {
  return CancellationToken(state_);
}

void CancellationSource::Cancel() noexcept {
  if (state_ != nullptr) {
    state_->Claim(CancellationReason::cancelled);
  }
}

bool CancellationSource::cancelled() const noexcept {
  return reason() != CancellationReason::none;
}

CancellationReason CancellationSource::reason() const noexcept {
  return state_ == nullptr ? CancellationReason::none : state_->Poll();
}

std::optional<std::chrono::steady_clock::time_point>
CancellationSource::deadline() const noexcept {
  return state_ == nullptr ? std::nullopt : state_->deadline();
}

CancellationReason CancellationSource::WaitForCancellation() const {
  if (state_ == nullptr) {
    throw Error(
        ErrorCode::invalid_argument,
        "Cannot wait on a moved-from cancellation source: nothing could ever "
        "release it");
  }
  return state_->Wait();
}

}  // namespace onnx_world_model
