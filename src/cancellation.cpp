/**
 * @agent-file
 * @agent-purpose: Implements the cancellation state machine behind CancellationToken and CancellationSource: first-reason-wins claiming, boundary deadline discovery, and the race-free callback registry that lets an in-flight ONNX Runtime call be terminated.
 * @agent-public-api: CancellationToken::cancelled, CancellationToken::reason, CancellationToken::deadline, CancellationToken::ThrowIfCancellationRequested, CancellationSource::CancellationSource, CancellationSource::WithDeadline, CancellationSource::WithTimeout, CancellationSource move operations and destructor, CancellationSource::token, CancellationSource::Cancel, CancellationSource::cancelled, CancellationSource::reason, CancellationSource::deadline, detail::CancellationState, detail::CancellationAccess, detail::CancellationRegistration
 * @agent-invariants: A reason is claimed exactly once, by a compare-exchange from none, so the first claim wins and callbacks run exactly once with that reason. Callbacks run while holding the same mutex Register and Unregister take, so a registration made after cancellation still runs inline and ~CancellationRegistration cannot return while its callback is running; every callback is invoked inside a catch-all because Cancel is noexcept. The deadline is immutable and is only observed by Poll, so this milestone reports deadline_exceeded at boundaries rather than from a timer thread. WithTimeout accepts the whole millisecond range: it compares the requested timeout against the headroom left on the steady clock, expressed in milliseconds, and saturates to time_point::max or time_point::min rather than letting the conversion or the addition overflow. A null state -- the default token and a moved-from source -- is never cancellable and never throws.
 * @agent-side-effects: Reads the steady clock and runs consumer callbacks on the thread that cancels or polls.
 */

#include "cancellation.hpp"

#include <string>
#include <utility>

#include "onnx_world_model/error.hpp"

namespace onnx_world_model {
namespace detail {

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

CancellationSource::CancellationSource()
    : state_(std::make_shared<detail::CancellationState>(std::nullopt)) {}

CancellationSource::CancellationSource(
    std::shared_ptr<detail::CancellationState> state) noexcept
    : state_(std::move(state)) {}

CancellationSource CancellationSource::WithDeadline(
    std::chrono::steady_clock::time_point deadline) {
  return CancellationSource(
      std::make_shared<detail::CancellationState>(deadline));
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

}  // namespace onnx_world_model
