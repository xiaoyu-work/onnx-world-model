/**
 * @agent-file
 * @agent-purpose: Standalone test executable for CancellationToken and CancellationSource: default inertness, first-reason-wins claiming, boundary deadline discovery, the shared deadline watchdog that releases a blocking WaitForCancellation, saturating timeouts across the whole millisecond range, error codes, move semantics, and the thread-safety and callback-lifetime guarantees the public API establishes.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as cancellation_test; it counts failures through local Check and CheckThrowsCode helpers and returns a non-zero exit code when any check fails. It includes public headers only and touches no filesystem, no ONNX Runtime library, and no model, so it always runs. No assertion sleeps and none asserts an upper bound on latency: every wait runs on a separate thread through WaitBounded, which fails and then cancels its source if the watchdog has not fired within two seconds, so a broken watchdog reports a failure instead of hanging CTest. The only timing assertions are lower bounds -- a waiter never reports before its own deadline -- and "a later deadline already fired" is used in place of sleeping to prove an earlier reason was not overwritten.
 * @agent-side-effects: Writes failure descriptions to stderr.
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "onnx_world_model/cancellation.hpp"
#include "onnx_world_model/error.hpp"

namespace {

using onnx_world_model::CancellationReason;
using onnx_world_model::CancellationSource;
using onnx_world_model::CancellationToken;
using onnx_world_model::ErrorCode;

using Clock = std::chrono::steady_clock;

// Long enough that an ordinary scheduling hiccup cannot exhaust it, short
// enough that a broken watchdog fails the run quickly.
constexpr std::chrono::seconds kWaitBudget{2};

int failures = 0;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    ++failures;
  }
}

template <typename Function>
void CheckThrowsCode(
    Function&& function,
    ErrorCode expected,
    const char* message) {
  try {
    function();
    Check(false, message);
  } catch (const onnx_world_model::Error& error) {
    if (error.code() != expected) {
      std::cerr << "FAILED: " << message << " (got: " << error.what() << ")\n";
      ++failures;
    }
  }
}

// What a bounded wait observed: whether it finished inside the budget, the
// reason it reported, and when it came back.
struct WaitOutcome {
  bool completed{false};
  CancellationReason reason{CancellationReason::none};
  Clock::time_point woke{};
};

// Waits on `token` from another thread with a hard time bound. A watchdog
// that never fires must not hang CTest, so an exhausted budget records the
// failure and then cancels `source` to release the waiter before joining it.
[[nodiscard]] WaitOutcome WaitBounded(
    const CancellationToken& token,
    CancellationSource& source,
    const char* message) {
  std::future<WaitOutcome> waiter = std::async(std::launch::async, [token] {
    WaitOutcome outcome;
    outcome.reason = token.WaitForCancellation();
    outcome.woke = Clock::now();
    outcome.completed = true;
    return outcome;
  });
  if (waiter.wait_for(kWaitBudget) != std::future_status::ready) {
    Check(false, message);
    source.Cancel();
    (void)waiter.get();
    return WaitOutcome{};
  }
  return waiter.get();
}

void TestDefaultTokenIsInert() {
  const CancellationToken token;

  Check(!token.cancellable(), "A default token is not cancellable");
  Check(!token.cancelled(), "A default token is never cancelled");
  Check(
      token.reason() == CancellationReason::none,
      "A default token reports no reason");
  Check(
      !token.deadline().has_value(),
      "A default token carries no deadline");
  try {
    token.ThrowIfCancellationRequested();
  } catch (...) {
    Check(false, "A default token never throws");
  }
}

void TestSourceProducesACancellableToken() {
  CancellationSource source;
  const CancellationToken token = source.token();

  Check(token.cancellable(), "A source token is cancellable");
  Check(!token.cancelled(), "A fresh source token is not cancelled");
  Check(
      !token.deadline().has_value(),
      "A source without a deadline exposes none");
  Check(!source.cancelled(), "A fresh source is not cancelled");

  source.Cancel();

  Check(token.cancelled(), "Cancel is visible through every token copy");
  Check(
      token.reason() == CancellationReason::cancelled,
      "An explicit cancel reports the cancelled reason");
  Check(source.cancelled(), "The source observes its own cancellation");
  CheckThrowsCode(
      [&token] { token.ThrowIfCancellationRequested(); },
      ErrorCode::cancelled,
      "A cancelled token throws ErrorCode::cancelled");
}

void TestCancelIsIdempotentAndFirstReasonWins() {
  CancellationSource source;
  source.Cancel();
  source.Cancel();

  Check(
      source.reason() == CancellationReason::cancelled,
      "Cancelling twice keeps the first reason");

  // A deadline that passes after an explicit cancel must not overwrite it.
  CancellationSource cancelled_first =
      CancellationSource::WithTimeout(std::chrono::milliseconds{0});
  // Claim the reason with an explicit cancel before anything polls, which is
  // what makes this a first-reason-wins assertion rather than a race.
  cancelled_first.Cancel();

  Check(
      cancelled_first.reason() == CancellationReason::cancelled,
      "An explicit cancel that lands first survives a passed deadline");
  CheckThrowsCode(
      [&cancelled_first] {
        cancelled_first.token().ThrowIfCancellationRequested();
      },
      ErrorCode::cancelled,
      "A source cancelled before its deadline was polled reports cancelled");
}

void TestExpiredDeadlineIsClaimedAtTheBoundary() {
  CancellationSource source =
      CancellationSource::WithTimeout(std::chrono::milliseconds{0});
  const CancellationToken token = source.token();

  Check(
      token.deadline().has_value(),
      "A timeout source exposes its absolute deadline");
  Check(token.cancelled(), "A zero timeout is exceeded immediately");
  Check(
      token.reason() == CancellationReason::deadline_exceeded,
      "An expired deadline reports deadline_exceeded, not cancelled");
  CheckThrowsCode(
      [&token] { token.ThrowIfCancellationRequested(); },
      ErrorCode::deadline_exceeded,
      "An expired deadline throws ErrorCode::deadline_exceeded");

  // The deadline was already claimed, so an explicit cancel cannot demote it.
  source.Cancel();
  Check(
      token.reason() == CancellationReason::deadline_exceeded,
      "A cancel after a claimed deadline keeps deadline_exceeded");
}

void TestFutureDeadlineIsNotCancelled() {
  const CancellationSource source = CancellationSource::WithDeadline(
      std::chrono::steady_clock::now() + std::chrono::hours{1});
  const CancellationToken token = source.token();

  Check(
      !token.cancelled(),
      "A deadline an hour out does not cancel the token");
  Check(
      token.reason() == CancellationReason::none,
      "A future deadline reports no reason");
  Check(
      token.deadline().has_value() &&
          *token.deadline() > std::chrono::steady_clock::now(),
      "A future deadline is exposed as a future instant");
}

void TestNegativeTimeoutIsAnImmediateDeadline() {
  const CancellationSource source =
      CancellationSource::WithTimeout(std::chrono::milliseconds{-5});

  Check(
      source.reason() == CancellationReason::deadline_exceeded,
      "A negative timeout means the deadline already passed");
}

// The steady clock counts in a unit finer than a millisecond, so a large
// millisecond count overflows both the conversion into the clock's duration
// and the addition to now(). Saturating is what keeps "wait essentially
// forever" from wrapping into "already out of time".
void TestSaturatingTimeoutIsNotImmediatelyExpired() {
  for (const std::chrono::milliseconds timeout : {
           std::chrono::milliseconds::max(),
           std::chrono::milliseconds{std::chrono::milliseconds::max().count() /
                                     2},
           std::chrono::milliseconds{1'000'000'000'000},
       }) {
    const CancellationSource source = CancellationSource::WithTimeout(timeout);
    Check(
        source.reason() == CancellationReason::none,
        "A timeout beyond the clock's range is not already expired");
    Check(
        source.deadline().has_value() &&
            *source.deadline() > std::chrono::steady_clock::now(),
        "A saturated timeout still exposes a future instant");
    Check(
        !source.token().cancelled(),
        "A token from a saturated timeout is live");
  }

  // The other end saturates too, so the most negative timeout stays an
  // already-expired deadline instead of wrapping forward.
  const CancellationSource expired =
      CancellationSource::WithTimeout(std::chrono::milliseconds::min());
  Check(
      expired.reason() == CancellationReason::deadline_exceeded,
      "The most negative timeout is still an expired deadline");
  const auto clock_minimum =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::duration::min());
  const CancellationSource just_below_clock_minimum =
      CancellationSource::WithTimeout(
          clock_minimum - std::chrono::milliseconds{1});
  Check(
      just_below_clock_minimum.reason() ==
          CancellationReason::deadline_exceeded,
      "A timeout just below the clock conversion limit saturates backward");
}

void TestMovedFromSourceIsInert() {
  CancellationSource source;
  const CancellationToken token = source.token();
  CancellationSource moved = std::move(source);

  Check(
      !source.token().cancellable(),  // NOLINT(bugprone-use-after-move)
      "A moved-from source hands out the never-cancellable token");
  Check(
      source.reason() == CancellationReason::none,
      "A moved-from source reports no reason");
  source.Cancel();
  Check(
      !token.cancelled(),
      "Cancelling a moved-from source does nothing");

  moved.Cancel();
  Check(
      token.cancelled(),
      "The moved-to source still owns the original state");
}

void TestConcurrentCancelClaimsExactlyOneReason() {
  constexpr std::size_t kThreads = 8;
  CancellationSource source;
  const CancellationToken token = source.token();

  std::mutex mutex;
  std::condition_variable condition;
  bool go = false;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (std::size_t index = 0; index < kThreads; ++index) {
    threads.emplace_back([&source, &mutex, &condition, &go] {
      std::unique_lock lock(mutex);
      condition.wait(lock, [&go] { return go; });
      lock.unlock();
      source.Cancel();
    });
  }
  {
    std::scoped_lock lock(mutex);
    go = true;
  }
  condition.notify_all();
  for (auto& thread : threads) {
    thread.join();
  }

  Check(
      token.reason() == CancellationReason::cancelled,
      "Concurrent cancels agree on one reason");
}

void TestCancelUnblocksAnotherThreadAtItsNextBoundary() {
  CancellationSource source;
  const CancellationToken token = source.token();

  std::mutex mutex;
  std::condition_variable condition;
  bool worker_ready = false;
  std::atomic<bool> observed_cancellation{false};

  std::thread worker([&] {
    {
      std::scoped_lock lock(mutex);
      worker_ready = true;
    }
    condition.notify_all();
    // Spins on the public boundary check, which is exactly how the runtime
    // observes a cancellation another thread requested.
    while (!token.cancelled()) {
    }
    try {
      token.ThrowIfCancellationRequested();
    } catch (const onnx_world_model::Error& error) {
      observed_cancellation = error.code() == ErrorCode::cancelled;
    }
  });

  {
    std::unique_lock lock(mutex);
    condition.wait(lock, [&worker_ready] { return worker_ready; });
  }
  source.Cancel();
  worker.join();

  Check(
      observed_cancellation.load(),
      "A cancel from another thread is observed at the next boundary");
}

void TestWaitingOnAnUncancellableStateIsRejected() {
  const CancellationToken token;
  CheckThrowsCode(
      [&token] { (void)token.WaitForCancellation(); },
      ErrorCode::invalid_argument,
      "Waiting on the default token is rejected instead of blocking forever");

  CancellationSource source;
  const CancellationSource moved = std::move(source);
  Check(
      moved.token().cancellable(),
      "The moved-to source still owns a cancellable state");
  CheckThrowsCode(
      [&source] {
        (void)source.WaitForCancellation();  // NOLINT(bugprone-use-after-move)
      },
      ErrorCode::invalid_argument,
      "Waiting on a moved-from source is rejected");
}

// The whole point of the watchdog: nothing in this test ever polls the token,
// so only a background claim can release the waiter.
void TestTheWatchdogReleasesAWaiterAtItsDeadline() {
  CancellationSource source =
      CancellationSource::WithTimeout(std::chrono::milliseconds{20});
  const CancellationToken token = source.token();
  const Clock::time_point deadline = source.deadline().value();

  const WaitOutcome outcome = WaitBounded(
      token,
      source,
      "A future deadline releases a waiter that never polls");

  Check(outcome.completed, "The waiter came back inside its budget");
  Check(
      outcome.reason == CancellationReason::deadline_exceeded,
      "A waiter released by the watchdog reports deadline_exceeded");
  Check(
      !outcome.completed || outcome.woke >= deadline,
      "The watchdog never releases a waiter before its deadline");
  Check(
      token.reason() == CancellationReason::deadline_exceeded,
      "The state stays claimed after the waiter returned");
}

void TestAnExplicitCancelReleasesAWaiter() {
  CancellationSource source;
  const CancellationToken token = source.token();

  std::mutex mutex;
  std::condition_variable condition;
  bool waiter_ready = false;
  // Wait() polls once before it blocks, so this handshake only narrows the
  // window; the reported reason is correct either way.
  std::future<CancellationReason> waiter =
      std::async(std::launch::async, [&, token] {
        {
          std::scoped_lock lock(mutex);
          waiter_ready = true;
        }
        condition.notify_all();
        return token.WaitForCancellation();
      });
  {
    std::unique_lock lock(mutex);
    condition.wait(lock, [&waiter_ready] { return waiter_ready; });
  }
  source.Cancel();

  if (waiter.wait_for(kWaitBudget) != std::future_status::ready) {
    Check(false, "An explicit cancel releases a waiter");
    return;
  }
  Check(
      waiter.get() == CancellationReason::cancelled,
      "A waiter released by an explicit cancel reports cancelled");
  Check(
      !source.deadline().has_value(),
      "A source with no deadline still releases its waiters");
}

void TestACancelBeforeADeadlineSurvivesALaterOne() {
  CancellationSource cancelled =
      CancellationSource::WithTimeout(std::chrono::milliseconds{40});
  const CancellationToken token = cancelled.token();
  cancelled.Cancel();

  const WaitOutcome first = WaitBounded(
      token,
      cancelled,
      "A cancelled source releases its waiter immediately");
  Check(
      first.reason == CancellationReason::cancelled,
      "A cancel that lands before the deadline is the reason a waiter sees");

  // Rather than sleeping past the first deadline and asserting that nothing
  // happened, wait for a strictly later one to fire: once the watchdog has
  // demonstrably passed that instant it has also passed the earlier one.
  CancellationSource sentinel =
      CancellationSource::WithTimeout(std::chrono::milliseconds{80});
  const WaitOutcome later = WaitBounded(
      sentinel.token(),
      sentinel,
      "A sentinel deadline later than the cancelled one fires");
  Check(
      later.reason == CancellationReason::deadline_exceeded,
      "The sentinel source stopped because its own deadline passed");
  Check(
      !later.completed || later.woke > *cancelled.deadline(),
      "The sentinel woke after the cancelled source's deadline had passed");
  Check(
      token.reason() == CancellationReason::cancelled,
      "A deadline that passes after an explicit cancel never demotes it");
}

void TestSeveralDeadlinesFireIndependently() {
  struct Scheduled {
    CancellationSource source;
    Clock::time_point deadline;
  };

  // Registered out of order so the schedule is exercised as an ordered
  // container rather than a queue that happens to be sorted already.
  std::vector<Scheduled> scheduled;
  for (const int milliseconds : {60, 20, 80, 40}) {
    CancellationSource source =
        CancellationSource::WithTimeout(std::chrono::milliseconds{milliseconds});
    const Clock::time_point deadline = source.deadline().value();
    scheduled.push_back(Scheduled{std::move(source), deadline});
  }

  for (Scheduled& entry : scheduled) {
    const WaitOutcome outcome = WaitBounded(
        entry.source.token(),
        entry.source,
        "Every scheduled deadline releases its own waiter");
    Check(
        outcome.reason == CancellationReason::deadline_exceeded,
        "Each source reports its own deadline rather than a neighbour's");
    Check(
        !outcome.completed || outcome.woke >= entry.deadline,
        "No waiter is released before its own deadline");
  }

  // Dropping every source unregisters its watchdog entry; a later deadline
  // still fires, which is what proves the schedule survived the removals.
  scheduled.clear();
  CancellationSource after =
      CancellationSource::WithTimeout(std::chrono::milliseconds{20});
  const WaitOutcome outcome = WaitBounded(
      after.token(),
      after,
      "The watchdog still fires after every earlier source was dropped");
  Check(
      outcome.reason == CancellationReason::deadline_exceeded,
      "A deadline armed after a batch of removals still fires");
}

}  // namespace

int main() {
  TestDefaultTokenIsInert();
  TestSourceProducesACancellableToken();
  TestCancelIsIdempotentAndFirstReasonWins();
  TestExpiredDeadlineIsClaimedAtTheBoundary();
  TestFutureDeadlineIsNotCancelled();
  TestNegativeTimeoutIsAnImmediateDeadline();
  TestSaturatingTimeoutIsNotImmediatelyExpired();
  TestMovedFromSourceIsInert();
  TestConcurrentCancelClaimsExactlyOneReason();
  TestCancelUnblocksAnotherThreadAtItsNextBoundary();
  TestWaitingOnAnUncancellableStateIsRejected();
  TestTheWatchdogReleasesAWaiterAtItsDeadline();
  TestAnExplicitCancelReleasesAWaiter();
  TestACancelBeforeADeadlineSurvivesALaterOne();
  TestSeveralDeadlinesFireIndependently();

  if (failures != 0) {
    std::cerr << failures << " cancellation checks failed\n";
    return 1;
  }
  std::cout << "cancellation tests passed\n";
  return 0;
}
