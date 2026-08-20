/**
 * @agent-file
 * @agent-purpose: Declares the RAII dynamic-library handle and the process-wide ONNX Runtime C API initializer used to load ORT without link-time coupling.
 * @agent-public-api: onnx_world_model::detail::DynamicLibrary, onnx_world_model::detail::InitializeOrtApi
 * @agent-invariants: Internal header that is not installed; DynamicLibrary is non-copyable, and InitializeOrtApi may be called only for one library path per process.
 * @agent-side-effects: none in this header; the declared functions load a shared library into the process.
 */

#pragma once

#include <filesystem>
#include <string_view>

namespace onnx_world_model::detail {

class DynamicLibrary {
 public:
  explicit DynamicLibrary(const std::filesystem::path& path);
  ~DynamicLibrary();

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  [[nodiscard]] void* Symbol(std::string_view name) const;

 private:
  void* handle_{nullptr};
};

void InitializeOrtApi(const std::filesystem::path& library_path);

}  // namespace onnx_world_model::detail
