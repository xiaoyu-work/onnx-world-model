/**
 * @agent-file
 * @agent-purpose: Declares the internal cancellation state, the process-wide DeadlineService that claims deadlines without a per-request thread, the CancellationAccess seam that lets src/ reach a token's or source's shared state, and the RAII CancellationRegistration that runs a callback when a token is cancelled.
 * @agent-public-api: onnx_world_model::detail::DeadlineService, onnx_world_model::detail::CancellationState, onnx_world_model::detail::CancellationAccess, onnx_world_model::detail::CancellationRegistration
 * @agent-invariants: Internal header that is not installed, so the registration and watchdog machinery stays out of the public ABI. CancellationState claims the first non-none reason with an atomic compare-exchange and only then disarms its watchdog entry, releases its waiters, and runs callbacks, and it runs the callbacks while holding the same mutex that Register and Unregister take. That single mutex closes both races a stack-local callback target has: registering after the state was already cancelled invokes the callback inline instead of dropping it, and ~CancellationRegistration cannot return while its callback is still running. Waiters block on a second, separate mutex and condition variable, so a callback can never keep a waiter parked and a waiter can never delay a registration. A callback receives the claimed reason as an argument rather than capturing a token, so the state never owns a shared_ptr back to itself, and it never escapes an exception, because Cancel is noexcept. ArmDeadline is called exactly once, right after make_shared, and only for a deadline still in the future: an already-due deadline stays a poll claim so an explicit Cancel() made immediately after construction can still win. The watchdog holds weak_ptr only, so an armed state is still destroyed on its last owning reference and its destructor disarms it; a disarm that finds nothing is the normal outcome for an entry the worker already removed.
 * @agent-side-effects: Running a callback executes consumer code, such as ONNX Runtime's SetTerminate, on the cancelling thread. Arming a deadline starts the one process-wide watchdog thread on first use. Wait blocks the calling thread.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "onnx_world_model/cancellation.hpp"

namespace onnx_world_model::detail {

//: The one process-wide deadline watchdog. It is defined in cancellation.cpp
//: because nothing outside that translation unit needs its shape; a state
//: reaches it through the raw pointer ArmDeadline stored.
class DeadlineService;

class CancellationState {
 public:
  using Clock = std::chrono::steady_clock;
  //: Receives the reason that was claimed, so a linked source can preserve it
  //: rather than flattening every stop into a plain cancellation. Taking the
  //: reason as an argument also keeps a callback from having to capture a
  //: token, which would make the state own a shared_ptr back to itself.
  using Callback = std::function<void(CancellationReason)>;

  explicit CancellationState(
      std::optional<Clock::time_point> deadline) noexcept
      : deadline_(deadline) {}

  //: Disarms this state's watchdog entry, so a source that is dropped before
  //: its deadline leaves nothing behind for the worker to wake up for.
  ~CancellationState();

  CancellationState(const CancellationState&) = delete;
  CancellationState& operator=(const CancellationState&) = delete;

  [[nodiscard]] const std::optional<Clock::time_point>& deadline()
      const noexcept {
    return deadline_;
  }

  //: The current reason, claiming deadline_exceeded first when the fixed
  //: deadline has passed and nothing else claimed the state yet. Every
  //: boundary check goes through here, and it is also what claims a deadline
  //: that was already due when the state was armed.
  [[nodiscard]] CancellationReason Poll() noexcept;

  //: Blocks until a reason is claimed and returns it. It polls once before
  //: sleeping, and Claim takes the wait mutex before notifying, so no wakeup
  //: can be lost between the two.
  [[nodiscard]] CancellationReason Wait();

  //: Claims `reason` if the state is still uncancelled and, on success only,
  //: disarms the watchdog, releases every waiter, and runs every registered
  //: callback exactly once.
  void Claim(CancellationReason reason) noexcept;

  //: Hands a still-future deadline to the shared watchdog so it is claimed at
  //: the deadline even when nothing polls. `self` must own this state, and
  //: this must be called exactly once, immediately after make_shared, before
  //: the state is published anywhere else.
  void ArmDeadline(const std::shared_ptr<CancellationState>& self);

  //: Registers `callback` and returns its removal handle. A state that is
  //: already cancelled runs the callback inline and returns 0, so a caller
  //: that registers after the fact still observes the cancellation.
  [[nodiscard]] std::uint64_t Register(Callback callback);
  //: Removes a registration. It blocks while that callback is running, so the
  //: object a callback captures can be destroyed right after this returns.
  void Unregister(std::uint64_t registration) noexcept;

 private:
  //: Drops this state's watchdog entry, at most once. Finding no entry is the
  //: expected outcome when the worker is the one firing this deadline.
  void DisarmDeadline() noexcept;

  std::atomic<CancellationReason> reason_{CancellationReason::none};
  const std::optional<Clock::time_point> deadline_;
  //: Written by ArmDeadline before the watchdog can observe this state, and
  //: only ever read after a non-zero timer_ was exchanged out, so the worker
  //: thread reaches the service through this pointer instead of re-entering
  //: the singleton accessor that started it.
  DeadlineService* service_{nullptr};
  std::atomic<std::uint64_t> timer_{0};
  std::mutex mutex_;
  std::vector<std::pair<std::uint64_t, Callback>> callbacks_;
  std::uint64_t next_registration_{1};
  //: Deliberately not mutex_: a waiter must not be held up by a callback, and
  //: a callback must not be held up by a waiter.
  std::mutex wait_mutex_;
  std::condition_variable wait_condition_;
};

//: The only seam between the public cancellation types and src/. It keeps the
//: shared state and the reason-preserving cancel out of the installed header
//: while letting the runtime link one source into another.
struct CancellationAccess {
  [[nodiscard]] static const std::shared_ptr<CancellationState>& State(
      const CancellationToken& token) noexcept;
  [[nodiscard]] static const std::shared_ptr<CancellationState>& State(
      const CancellationSource& source) noexcept;
  //: Cancels `source` while preserving `reason`, so a deadline that fired on
  //: an external token stays a deadline on the source it is linked into
  //: rather than being flattened into a plain cancellation.
  static void Cancel(
      const CancellationSource& source,
      CancellationReason reason) noexcept;
};

//: Runs `callback` when `token` is cancelled, and guarantees the callback is
//: neither missed nor still running once this object is destroyed. That is
//: what makes a stack-local Ort::RunOptions a legal callback target.
class CancellationRegistration {
 public:
  CancellationRegistration(
      const CancellationToken& token,
      CancellationState::Callback callback);
  ~CancellationRegistration();

  CancellationRegistration(const CancellationRegistration&) = delete;
  CancellationRegistration& operator=(const CancellationRegistration&) = delete;
  CancellationRegistration(CancellationRegistration&&) = delete;
  CancellationRegistration& operator=(CancellationRegistration&&) = delete;

 private:
  std::shared_ptr<CancellationState> state_;
  std::uint64_t registration_{0};
};

}  // namespace onnx_world_model::detail
