#include "dynamic_library.hpp"

#include <cstring>
#include <mutex>
#include <string>

#include "onnx_world_model/error.hpp"
#include "onnxruntime_cxx_api.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace onnx_world_model::detail {
namespace {

[[nodiscard]] std::string LibraryKey(const std::filesystem::path& path) {
  if (path.empty()) {
    return "<default>";
  }
  return std::filesystem::absolute(path).lexically_normal().generic_string();
}

[[nodiscard]] std::string LastLibraryError() {
#ifdef _WIN32
  return "Windows error " + std::to_string(GetLastError());
#else
  const char* error = dlerror();
  return error == nullptr ? "unknown dynamic loader error" : error;
#endif
}

}  // namespace

DynamicLibrary::DynamicLibrary(const std::filesystem::path& path) {
#ifdef _WIN32
  const auto library_path =
      path.empty() ? std::filesystem::path(L"onnxruntime.dll") : path;
  handle_ = reinterpret_cast<void*>(LoadLibraryW(library_path.c_str()));
#else
#if defined(__APPLE__)
  const auto library_path =
      path.empty() ? std::filesystem::path("libonnxruntime.dylib") : path;
#else
  const auto library_path =
      path.empty() ? std::filesystem::path("libonnxruntime.so") : path;
#endif
  handle_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  if (handle_ == nullptr) {
    throw Error(
        ErrorCode::runtime_load,
        "Failed to load ONNX Runtime library '" +
            (path.empty() ? std::string("<default>") : path.string()) +
            "': " + LastLibraryError());
  }
}

DynamicLibrary::~DynamicLibrary() {
  if (handle_ == nullptr) {
    return;
  }
#ifdef _WIN32
  FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
  dlclose(handle_);
#endif
}

void* DynamicLibrary::Symbol(std::string_view name) const {
#ifdef _WIN32
  void* symbol = reinterpret_cast<void*>(
      GetProcAddress(reinterpret_cast<HMODULE>(handle_), std::string(name).c_str()));
#else
  dlerror();
  void* symbol = dlsym(handle_, std::string(name).c_str());
#endif
  if (symbol == nullptr) {
    throw Error(
        ErrorCode::runtime_load,
        "Failed to resolve ONNX Runtime symbol '" + std::string(name) +
            "': " + LastLibraryError());
  }
  return symbol;
}

void InitializeOrtApi(const std::filesystem::path& library_path) {
  static std::mutex mutex;
  static DynamicLibrary* loaded_library = nullptr;
  static std::string loaded_key;

  const std::string requested_key = LibraryKey(library_path);
  std::scoped_lock lock(mutex);
  if (loaded_library != nullptr) {
    if (loaded_key != requested_key) {
      throw Error(
          ErrorCode::runtime_load,
          "ONNX Runtime is already initialized from '" + loaded_key +
              "' and cannot be reinitialized from '" + requested_key + "'");
    }
    return;
  }

  auto* library = new DynamicLibrary(library_path);
  try {
    using GetApiBaseFunction = const OrtApiBase*(ORT_API_CALL*)();
    const void* symbol = library->Symbol("OrtGetApiBase");
    static_assert(sizeof(GetApiBaseFunction) == sizeof(symbol));
    GetApiBaseFunction get_api_base;
    std::memcpy(&get_api_base, &symbol, sizeof(get_api_base));
    const OrtApiBase* api_base = get_api_base();
    if (api_base == nullptr) {
      throw Error(
          ErrorCode::runtime_load,
          "OrtGetApiBase returned null for ONNX Runtime library");
    }
    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    if (api == nullptr) {
      throw Error(
          ErrorCode::runtime_load,
          "ONNX Runtime library '" + requested_key +
              "' does not support C API version " +
              std::to_string(ORT_API_VERSION) + " (runtime version: " +
              api_base->GetVersionString() + ")");
    }
    Ort::InitApi(api);
  } catch (...) {
    delete library;
    throw;
  }

  // Keep the library loaded for process lifetime because Ort::InitApi stores
  // pointers into its function table.
  loaded_library = library;
  loaded_key = requested_key;
}

}  // namespace onnx_world_model::detail
