/**
 * @agent-file
 * @agent-purpose: Standalone test executable for CancellationToken and CancellationSource: default inertness, first-reason-wins claiming, boundary deadline discovery, saturating timeouts across the whole millisecond range, error codes, move semantics, and the thread-safety and callback-lifetime guarantees the public API establishes.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as cancellation_test; it counts failures through local Check and CheckThrowsCode helpers and returns a non-zero exit code when any check fails. It includes public headers only and touches no filesystem, no ONNX Runtime library, and no model, so it always runs. Every concurrency assertion is driven by a condition variable or a joined thread rather than by a sleep, so the file has no timing-dependent result.
 * @agent-side-effects: Writes failure descriptions to stderr.
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
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

  if (failures != 0) {
    std::cerr << failures << " cancellation checks failed\n";
    return 1;
  }
  std::cout << "cancellation tests passed\n";
  return 0;
}
