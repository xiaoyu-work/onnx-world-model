#pragma once

/**
 * @agent-file
 * @agent-purpose: Declares the library-wide error taxonomy: the ErrorCode category enum and the Error exception every public entry point throws on failure.
 * @agent-public-api: ErrorCode, Error
 * @agent-invariants: Error derives from std::runtime_error and always carries a stable ErrorCode; the enumerator set is append-only because pybind11 maps the code onto the Python exception hierarchy -- cancelled to CancelledError, deadline_exceeded to DeadlineExceededError, and every other code to their WorldModelError base.
 * @agent-side-effects: none
 */

#include <stdexcept>
#include <string>
#include <utility>

namespace onnx_world_model {

enum class ErrorCode {
  invalid_argument,
  model_contract,
  runtime_load,
  runtime_execution,
  state,
  pipeline_manifest,
  //: An operation stopped because a CancellationToken was cancelled.
  cancelled,
  //: An operation stopped because a CancellationToken's deadline passed.
  deadline_exceeded,
};

class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, std::string message)
      : std::runtime_error(std::move(message)), code_(code) {}

  [[nodiscard]] ErrorCode code() const noexcept { return code_; }

 private:
  ErrorCode code_;
};

}  // namespace onnx_world_model
