#pragma once

/**
 * @agent-file
 * @agent-purpose: Declares explicit cancellation and deadline support: the CancellationReason taxonomy, the copyable observer CancellationToken, and the move-only CancellationSource that owns the cancellable state.
 * @agent-public-api: CancellationReason, CancellationToken, CancellationSource
 * @agent-invariants: A default-constructed CancellationToken owns no state, is never cancellable, and never throws; only a CancellationSource produces a cancellable token. A source's optional deadline is fixed at construction and never changes. The first non-none reason wins: an explicit Cancel() after a deadline was already observed leaves the reason deadline_exceeded, and a deadline that passes after Cancel() leaves it cancelled. Cancel() is noexcept and safe to call from any thread, including while another thread blocks inside the work the token guards. Deadlines are enforced where a token is polled -- ThrowIfCancellationRequested, cancelled(), and reason() all claim deadline_exceeded when the deadline has passed -- so this milestone fires a deadline at execution boundaries rather than from a background timer. A moved-from source owns no state, so its token is the never-cancellable token and its Cancel() does nothing.
 * @agent-side-effects: Cancel(), and a poll that observes a passed deadline, run the callbacks internal consumers registered on the state; those callbacks never escape an exception.
 */

#include <chrono>
#include <memory>
#include <optional>

#include "onnx_world_model/error.hpp"

namespace onnx_world_model {

namespace detail {

class CancellationState;
struct CancellationAccess;

}  // namespace detail

//: Why a token reports that its work should stop. `none` is the only value a
//: live, uncancelled token reports.
enum class CancellationReason {
  none,
  cancelled,
  deadline_exceeded,
};

class CancellationSource;

//: A copyable, thread-safe observer of one CancellationSource's state.
//:
//: The default token is deliberately inert: it owns no state, reports
//: `cancellable() == false`, and never throws, so every API that accepts a
//: token keeps its uncancellable behavior when a caller passes nothing.
//:
//: A token is an observer only. It cannot cancel, and it cannot extend or
//: shorten the deadline its source fixed.
class CancellationToken {
 public:
  CancellationToken() noexcept = default;

  //: False for the default token, which no source can ever cancel.
  [[nodiscard]] bool cancellable() const noexcept {
    return state_ != nullptr;
  }
  //: True once this token's source was cancelled or its deadline passed.
  //: Polling is what discovers a passed deadline in this milestone, so this
  //: can become true without any other thread doing anything.
  [[nodiscard]] bool cancelled() const noexcept;
  //: The first reason that was claimed, or `none` while the work may proceed.
  [[nodiscard]] CancellationReason reason() const noexcept;
  //: The absolute steady-clock instant this token's work must stop by, if the
  //: source was given one. Fixed for the token's lifetime.
  [[nodiscard]] std::optional<std::chrono::steady_clock::time_point> deadline()
      const noexcept;
  //: Throws Error with ErrorCode::cancelled or ErrorCode::deadline_exceeded
  //: when this token has a reason, and returns otherwise. This is the check
  //: every cancellable operation makes at its own boundaries.
  void ThrowIfCancellationRequested() const;

 private:
  explicit CancellationToken(
      std::shared_ptr<detail::CancellationState> state) noexcept;

  std::shared_ptr<detail::CancellationState> state_;

  friend class CancellationSource;
  friend struct detail::CancellationAccess;
};

//: The owning half of a cancellation state. It is move-only because exactly
//: one owner decides when the work it guards stops; observers hold tokens.
//:
//: A default-constructed source is cancellable and has no deadline. The
//: deadline factories fix an absolute steady-clock instant or a timeout
//: relative to construction, and that instant never changes afterwards.
class CancellationSource {
 public:
  CancellationSource();
  //: A source whose work must stop by an absolute steady-clock instant. An
  //: instant already in the past means the source is exceeded from the start.
  [[nodiscard]] static CancellationSource WithDeadline(
      std::chrono::steady_clock::time_point deadline);
  //: A source whose work must stop `timeout` after this call. A zero or
  //: negative timeout means the source is exceeded from the start; it is not
  //: rejected, because "already out of time" is a meaningful request. The
  //: whole millisecond range is accepted: a timeout the steady clock cannot
  //: represent saturates to the clock's furthest instant instead of
  //: overflowing into a deadline in the past.
  [[nodiscard]] static CancellationSource WithTimeout(
      std::chrono::milliseconds timeout);

  CancellationSource(CancellationSource&&) noexcept;
  CancellationSource& operator=(CancellationSource&&) noexcept;
  ~CancellationSource();

  CancellationSource(const CancellationSource&) = delete;
  CancellationSource& operator=(const CancellationSource&) = delete;

  //: An observer of this source's state. Copies are cheap and every copy
  //: observes the same state.
  [[nodiscard]] CancellationToken token() const noexcept;
  //: Requests cancellation. Safe from any thread at any time, including while
  //: another thread is blocked inside the guarded work, and idempotent: the
  //: first claimed reason wins.
  void Cancel() noexcept;
  [[nodiscard]] bool cancelled() const noexcept;
  [[nodiscard]] CancellationReason reason() const noexcept;
  [[nodiscard]] std::optional<std::chrono::steady_clock::time_point> deadline()
      const noexcept;

 private:
  explicit CancellationSource(
      std::shared_ptr<detail::CancellationState> state) noexcept;

  std::shared_ptr<detail::CancellationState> state_;

  friend struct detail::CancellationAccess;
};

}  // namespace onnx_world_model
