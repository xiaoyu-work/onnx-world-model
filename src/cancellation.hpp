/**
 * @agent-file
 * @agent-purpose: Declares the internal cancellation state, the CancellationAccess seam that lets src/ reach a token's or source's shared state, and the RAII CancellationRegistration that runs a callback when a token is cancelled.
 * @agent-public-api: onnx_world_model::detail::CancellationState, onnx_world_model::detail::CancellationAccess, onnx_world_model::detail::CancellationRegistration
 * @agent-invariants: Internal header that is not installed, so the registration machinery stays out of the public ABI. CancellationState claims the first non-none reason with an atomic compare-exchange and only then runs callbacks, and it runs them while holding the same mutex that Register and Unregister take. That single mutex closes both races a stack-local callback target has: registering after the state was already cancelled invokes the callback inline instead of dropping it, and ~CancellationRegistration cannot return while its callback is still running. A callback receives the claimed reason as an argument rather than capturing a token, so the state never owns a shared_ptr back to itself, and it never escapes an exception, because Cancel is noexcept.
 * @agent-side-effects: Running a callback executes consumer code, such as ONNX Runtime's SetTerminate, on the cancelling thread.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "onnx_world_model/cancellation.hpp"

namespace onnx_world_model::detail {

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

  CancellationState(const CancellationState&) = delete;
  CancellationState& operator=(const CancellationState&) = delete;

  [[nodiscard]] const std::optional<Clock::time_point>& deadline()
      const noexcept {
    return deadline_;
  }

  //: The current reason, claiming deadline_exceeded first when the fixed
  //: deadline has passed and nothing else claimed the state yet. Polling is
  //: how a deadline is discovered in this milestone, so every boundary check
  //: goes through here.
  [[nodiscard]] CancellationReason Poll() noexcept;

  //: Claims `reason` if the state is still uncancelled and, on success only,
  //: runs every registered callback exactly once.
  void Claim(CancellationReason reason) noexcept;

  //: Registers `callback` and returns its removal handle. A state that is
  //: already cancelled runs the callback inline and returns 0, so a caller
  //: that registers after the fact still observes the cancellation.
  [[nodiscard]] std::uint64_t Register(Callback callback);
  //: Removes a registration. It blocks while that callback is running, so the
  //: object a callback captures can be destroyed right after this returns.
  void Unregister(std::uint64_t registration) noexcept;

 private:
  std::atomic<CancellationReason> reason_{CancellationReason::none};
  const std::optional<Clock::time_point> deadline_;
  std::mutex mutex_;
  std::vector<std::pair<std::uint64_t, Callback>> callbacks_;
  std::uint64_t next_registration_{1};
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
