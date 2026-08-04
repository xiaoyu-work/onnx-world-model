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
