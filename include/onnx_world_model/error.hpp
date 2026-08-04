#pragma once

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
