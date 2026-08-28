/**
 * @agent-file
 * @agent-purpose: Implements the ONNX Runtime ModelBackend: it shares one process-wide Ort::Env, creates component sessions, applies RuntimeOptions and execution providers, and uses I/O binding to retain ORT-owned output tensors on their assigned devices.
 * @agent-public-api: CreateOrtBackend, GetAvailableOrtProviders
 * @agent-invariants: This is the only translation unit besides dynamic_library.cpp that includes ONNX Runtime headers; ORT is initialized through InitializeOrtApi before the process-wide Ort::Env or any session is created. Every component session shares that environment while retaining its own session options. The per-port memory plan is read once in the constructor with GetMemoryInfoForInputs and GetMemoryInfoForOutputs, checked against the input and output counts, cached as non-owning Ort::ConstMemoryInfo views that stay valid for the session's lifetime, and published on each metadata TensorSpec::device, so Run reuses that plan instead of querying it again and a caller can see where ONNX Runtime actually placed every port. Opt-in I/O binding places each output in the device memory assigned by ORT graph partitioning; the resulting TensorBuffer owns the Ort::Value and a flattened set of session, binding, and aliased-input lifetime roots. ORT-backed inputs bind without copying only when their original dtype and shape match the enclosing Tensor view and their device matches the destination input plan; incompatible, foreign, or reshaped device buffers are synchronously materialized and retained on CPU for the complete Run call. Requested provider lists fail when no available provider can satisfy them instead of silently selecting CPU. The cancellable Run override owns one fresh Ort::RunOptions per call and never reuses or un-terminates it, registers its SetTerminate callback after that object so the registration is destroyed first, and decides that an Ort::Exception was a cancellation from the token rather than from ONNX Runtime's message text. That callback is what the shared deadline watchdog reaches, so a deadline interrupts an in-flight Run at the next graph node without any boundary being polled; a single long-running kernel still finishes first, because ORT checks its terminate flag between nodes and not inside one.
 * @agent-side-effects: Loads the ONNX Runtime shared library, reads model files from disk, allocates ORT sessions and output buffers, may transfer foreign device inputs to CPU, and runs inference.
 */

#include "ort_backend.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cancellation.hpp"
#include "dynamic_library.hpp"
#include "onnx_world_model/error.hpp"
#include "onnxruntime_cxx_api.h"

namespace onnx_world_model::detail {
namespace {

[[nodiscard]] DataType FromOrtDataType(ONNXTensorElementDataType data_type) {
  switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DataType::float32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DataType::float16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return DataType::bfloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return DataType::float64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DataType::int64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DataType::int32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return DataType::int16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return DataType::int8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return DataType::uint64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return DataType::uint32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return DataType::uint16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DataType::uint8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return DataType::boolean;
    default:
      throw Error(
          ErrorCode::model_contract,
          "Model tensor uses unsupported ONNX data type " +
              std::to_string(static_cast<int>(data_type)));
  }
}

[[nodiscard]] ONNXTensorElementDataType ToOrtDataType(DataType data_type) {
  switch (data_type) {
    case DataType::float32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DataType::float16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DataType::bfloat16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    case DataType::float64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case DataType::int64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DataType::int32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DataType::int16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case DataType::int8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case DataType::uint64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case DataType::uint32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case DataType::uint16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case DataType::uint8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DataType::boolean:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  }
  throw Error(ErrorCode::invalid_argument, "Unsupported tensor data type");
}

[[nodiscard]] ::GraphOptimizationLevel ToOrtGraphOptimizationLevel(
    GraphOptimizationLevel level) {
  switch (level) {
    case GraphOptimizationLevel::disabled:
      return ORT_DISABLE_ALL;
    case GraphOptimizationLevel::basic:
      return ORT_ENABLE_BASIC;
    case GraphOptimizationLevel::extended:
      return ORT_ENABLE_EXTENDED;
    case GraphOptimizationLevel::all:
      return ORT_ENABLE_ALL;
  }
  return ORT_ENABLE_ALL;
}

[[nodiscard]] TensorSpec ReadTensorSpec(
    const Ort::Session& session,
    std::size_t index,
    bool input,
    Ort::AllocatorWithDefaultOptions& allocator) {
  auto name = input ? session.GetInputNameAllocated(index, allocator)
                    : session.GetOutputNameAllocated(index, allocator);
  Ort::TypeInfo type_info =
      input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    throw Error(
        ErrorCode::model_contract,
        "Model value '" + std::string(name.get()) + "' must be a tensor");
  }
  const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  return TensorSpec{
      .name = name.get(),
      .data_type = FromOrtDataType(tensor_info.GetElementType()),
      .shape = tensor_info.GetShape(),
  };
}

[[nodiscard]] std::string AnnotateExecutionFailure(std::string message) {
  // Oversized intermediates surface as several different CUDA statuses: a
  // kernel that rejects its launch configuration, an allocator that runs out of
  // device memory, or a cuBLAS/cuDNN handle that cannot reserve its workspace.
  // All of them are size problems rather than bad graphs, and the raw status
  // gives a caller nothing to act on.
  static constexpr std::string_view kOversizeSignals[] = {
      "cudaErrorInvalidValue",
      "cudaErrorMemoryAllocation",
      "out of memory",
      "resource allocation failed",
      "CUBLAS_STATUS_ALLOC_FAILED",
      "Failed to allocate memory for requested buffer",
  };
  const bool oversized = std::any_of(
      std::begin(kOversizeSignals),
      std::end(kOversizeSignals),
      [&message](std::string_view signal) {
        return message.find(signal) != std::string::npos;
      });
  if (oversized) {
    message +=
        ". This execution provider could not run the stage at this size, "
        "either because an intermediate tensor exceeds what its kernels can "
        "index or because it could not reserve device memory. Run the stage "
        "over smaller inputs, for example fewer video frames per decode or a "
        "lower resolution, place this component on the CPU provider, or free "
        "device memory held by other processes.";
  }
  return message;
}

[[nodiscard]] Ort::Value MakeOrtTensor(
    const Tensor& tensor,
    const Ort::MemoryInfo& memory_info) {
  const auto& shape = tensor.shape();
  return Ort::Value::CreateTensor(
      memory_info,
      const_cast<std::byte*>(tensor.bytes().data()),
      tensor.size_bytes(),
      shape.data(),
      shape.size(),
      ToOrtDataType(tensor.data_type()));
}

using ProviderOptions =
    std::unordered_map<std::string, std::string>;

struct ResolvedProvider {
  std::string name;
  std::string available_name;
  ProviderOptions options;
};

[[nodiscard]] bool SupportsProviderRegistration(std::string_view name) {
  static const std::unordered_set<std::string> supported{
      "azure",
      "coreml",
      "cpu",
      "cuda",
      "dml",
      "js",
      "nvtensorrtrtx",
      "openvino",
      "qnn",
      "tensorrt",
      "vitisai",
      "webgpu",
      "webnn",
      "xnnpack",
  };
  return supported.contains(std::string(name));
}

[[nodiscard]] std::string JoinProviders(
    const std::vector<std::string>& providers) {
  std::ostringstream result;
  for (std::size_t index = 0; index < providers.size(); ++index) {
    if (index != 0) {
      result << ", ";
    }
    result << providers[index];
  }
  return result.str();
}

[[nodiscard]] std::unordered_map<std::string, ProviderOptions>
NormalizeProviderOptions(const RuntimeOptions& options) {
  std::unordered_map<std::string, ProviderOptions> normalized;
  normalized.reserve(options.provider_options.size());
  for (const auto& [name, values] : options.provider_options) {
    const std::string key = NormalizeExecutionProviderName(name);
    if (!normalized.emplace(key, values).second) {
      throw Error(
          ErrorCode::invalid_argument,
          "Provider options were supplied more than once for '" + key + "'");
    }
  }
  return normalized;
}

[[nodiscard]] std::vector<ResolvedProvider> ResolveProviders(
    const RuntimeOptions& options) {
  const std::vector<std::string> available =
      Ort::GetAvailableProviders();
  std::unordered_map<std::string, std::string> available_by_name;
  available_by_name.reserve(available.size());
  for (const auto& name : available) {
    available_by_name.emplace(
        NormalizeExecutionProviderName(name), name);
  }

  std::vector<std::string> requested = options.providers;
  if (requested.empty()) {
    requested.emplace_back("cpu");
  }
  const auto provider_options = NormalizeProviderOptions(options);
  std::unordered_set<std::string> seen;
  std::vector<ResolvedProvider> resolved;
  std::vector<std::string> unsupported;
  for (const auto& requested_name : requested) {
    const std::string name =
        NormalizeExecutionProviderName(requested_name);
    if (!seen.insert(name).second) {
      throw Error(
          ErrorCode::invalid_argument,
          "Execution provider '" + name + "' was requested more than once");
    }
    const auto available_provider = available_by_name.find(name);
    if (available_provider == available_by_name.end()) {
      continue;
    }
    if (!SupportsProviderRegistration(name)) {
      unsupported.push_back(available_provider->second);
      continue;
    }
    const auto values = provider_options.find(name);
    resolved.push_back({
        .name = name,
        .available_name = available_provider->second,
        .options =
            values == provider_options.end() ? ProviderOptions{} : values->second,
    });
  }
  for (const auto& [name, values] : provider_options) {
    (void)values;
    if (!seen.contains(name)) {
        throw Error(
            ErrorCode::invalid_argument,
            "Provider options were supplied for unrequested provider '" +
                name + "'");
    }
  }
  if (resolved.empty()) {
    const std::string unsupported_message =
        unsupported.empty()
            ? std::string{}
            : ". Available but unsupported by this runtime: " +
                  JoinProviders(unsupported);
    throw Error(
        ErrorCode::runtime_load,
        "None of the requested execution providers are available and "
        "supported (" +
            JoinProviders(requested) + "). Available providers: " +
            JoinProviders(available) + unsupported_message);
  }
  const auto cpu = std::ranges::find(
      resolved, std::string("cpu"), &ResolvedProvider::name);
  if (cpu != resolved.end() && std::next(cpu) != resolved.end()) {
    throw Error(
        ErrorCode::invalid_argument,
        "The CPU execution provider must be last because ONNX Runtime uses "
        "it as the fallback provider");
  }
  return resolved;
}

[[nodiscard]] bool ParseBooleanOption(
    std::string_view value,
    std::string_view option,
    std::string_view provider) {
  std::string normalized;
  normalized.reserve(value.size());
  for (const unsigned char character : value) {
    normalized.push_back(
        static_cast<char>(std::tolower(character)));
  }
  if (normalized == "1" || normalized == "true" ||
      normalized == "on") {
    return true;
  }
  if (normalized == "0" || normalized == "false" ||
      normalized == "off") {
    return false;
  }
  throw Error(
      ErrorCode::invalid_argument,
      "Provider option '" + std::string(option) + "' for '" +
          std::string(provider) + "' must be true or false");
}

[[nodiscard]] std::string GenericProviderName(
    const ResolvedProvider& provider) {
  static const std::unordered_map<std::string, std::string> names{
      {"dml", "DML"},
      {"nvtensorrtrtx", "NvTensorRtRtx"},
  };
  const auto known = names.find(provider.name);
  if (known != names.end()) {
    return known->second;
  }
  return provider.available_name;
}

void AppendProvider(
    Ort::SessionOptions& session_options,
    const ResolvedProvider& provider) {
  try {
    if (provider.name == "cpu") {
      bool use_arena = true;
      for (const auto& [name, value] : provider.options) {
        if (name != "use_arena") {
          throw Error(
              ErrorCode::invalid_argument,
              "Unknown CPU provider option '" + name + "'");
        }
        use_arena = ParseBooleanOption(value, name, provider.name);
      }
      if (use_arena) {
        session_options.EnableCpuMemArena();
      } else {
        session_options.DisableCpuMemArena();
      }
    } else if (provider.name == "cuda") {
      Ort::CUDAProviderOptions cuda_options;
      cuda_options.Update(provider.options);
      session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
    } else if (provider.name == "tensorrt") {
      Ort::TensorRTProviderOptions tensorrt_options;
      tensorrt_options.Update(provider.options);
      session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
    } else if (provider.name == "openvino") {
      session_options.AppendExecutionProvider_OpenVINO_V2(provider.options);
    } else if (provider.name == "vitisai") {
      session_options.AppendExecutionProvider_VitisAI(provider.options);
    } else {
      session_options.AppendExecutionProvider(
          GenericProviderName(provider), provider.options);
    }
  } catch (const Error&) {
    throw;
  } catch (const Ort::Exception& exception) {
    throw Error(
        ErrorCode::runtime_load,
        "Failed to configure execution provider '" +
            provider.available_name + "': " + exception.what());
  }
}

Ort::Env& SharedOrtEnvironment() {
  static Ort::Env environment(
      ORT_LOGGING_LEVEL_WARNING,
      "onnx-world-model");
  return environment;
}

[[nodiscard]] std::string DeviceTypeForMemory(
    const Ort::ConstMemoryInfo& memory_info) {
  if (memory_info.GetDeviceType() == OrtMemoryInfoDeviceType_CPU) {
    return "cpu";
  }
  const std::string allocator =
      NormalizeExecutionProviderName(memory_info.GetAllocatorName());
  if (!allocator.empty() && allocator != "cpu") {
    return allocator;
  }
  switch (memory_info.GetDeviceType()) {
    case OrtMemoryInfoDeviceType_GPU:
      return "gpu";
    case OrtMemoryInfoDeviceType_FPGA:
      return "fpga";
    case OrtMemoryInfoDeviceType_NPU:
      return "npu";
    case OrtMemoryInfoDeviceType_CPU:
      return "cpu";
  }
  throw Error(
      ErrorCode::runtime_execution,
      "ONNX Runtime returned an unknown tensor device type");
}

[[nodiscard]] TensorDevice TensorDeviceForMemory(
    const Ort::ConstMemoryInfo& memory_info) {
  // Pinned CPU-input/output allocators owned by an accelerator may report
  // that accelerator's ordinal even though the memory itself is host memory.
  // TensorDevice reserves nonzero ids for actual devices, so every CPU-typed
  // allocation is canonically cpu:0.
  if (memory_info.GetDeviceType() == OrtMemoryInfoDeviceType_CPU) {
    return TensorDevice{};
  }
  return TensorDevice(
      DeviceTypeForMemory(memory_info),
      memory_info.GetDeviceId());
}

class OrtTensorBuffer final : public TensorBuffer {
 public:
  OrtTensorBuffer(
      Ort::Value value,
      std::shared_ptr<Ort::Session> session_keep_alive,
      std::shared_ptr<Ort::IoBinding> run_keep_alive,
      const std::vector<std::shared_ptr<TensorBuffer>>& input_buffers)
      : value_(std::move(value)) {
    if (!value_.IsTensor()) {
      throw Error(
          ErrorCode::runtime_execution,
          "ONNX Runtime returned a non-tensor value");
    }
    if (session_keep_alive == nullptr) {
      throw Error(
          ErrorCode::runtime_execution,
          "ONNX Runtime tensor storage has no owning session");
    }
    if (run_keep_alive == nullptr) {
      throw Error(
          ErrorCode::runtime_execution,
          "ONNX Runtime tensor storage has no owning I/O binding");
    }
    AddLifetimeOwner(std::move(session_keep_alive));
    const auto type_info = value_.GetTensorTypeAndShapeInfo();
    data_type_ = FromOrtDataType(type_info.GetElementType());
    shape_ = type_info.GetShape();
    size_bytes_ = value_.GetTensorSizeInBytes();
    data_ = value_.GetTensorRawData();
    const auto memory_info = value_.GetTensorMemoryInfo();
    device_ = TensorDeviceForMemory(memory_info);
    host_accessible_ =
        memory_info.GetDeviceType() == OrtMemoryInfoDeviceType_CPU ||
        memory_info.GetDeviceMemoryType() ==
            OrtDeviceMemoryType_HOST_ACCESSIBLE;
    if (!RetainAliasedInputs(input_buffers)) {
      AddLifetimeOwner(std::move(run_keep_alive));
    }
  }

  [[nodiscard]] DataType data_type() const noexcept {
    return data_type_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& shape() const noexcept {
    return shape_;
  }

  [[nodiscard]] const Ort::Value& value() const noexcept {
    return value_;
  }

  [[nodiscard]] const TensorDevice& device() const noexcept override {
    return device_;
  }

  [[nodiscard]] std::size_t size_bytes() const noexcept override {
    return size_bytes_;
  }

  [[nodiscard]] bool is_host_accessible() const noexcept override {
    return host_accessible_;
  }

  [[nodiscard]] const void* data() const noexcept override {
    return data_;
  }

  [[nodiscard]] std::span<const std::byte> bytes() const override {
    if (!host_accessible_) {
      throw Error(
          ErrorCode::invalid_argument,
          "ONNX Runtime device tensor is not host-accessible");
    }
    return {
        static_cast<const std::byte*>(data_),
        size_bytes_,
    };
  }

  void CopyToCpu(std::span<std::byte> destination) const override {
    if (destination.size() != size_bytes_) {
      throw Error(
          ErrorCode::invalid_argument,
          "ONNX Runtime tensor copy destination has the wrong byte size");
    }
    try {
      if (host_accessible_) {
        std::memcpy(destination.data(), data_, size_bytes_);
        return;
      }
      Ort::MemoryInfo cpu_memory = Ort::MemoryInfo::CreateCpu(
          OrtDeviceAllocator,
          OrtMemTypeDefault);
      Ort::Value cpu_value = Ort::Value::CreateTensor(
          cpu_memory,
          destination.data(),
          destination.size(),
          shape_.data(),
          shape_.size(),
          ToOrtDataType(data_type_));
      const Ort::Status status = SharedOrtEnvironment().CopyTensor(
          value_,
          cpu_value,
          nullptr);
      if (!status.IsOK()) {
        throw Error(
            ErrorCode::runtime_execution,
            "ONNX Runtime device-to-CPU copy failed: " +
                status.GetErrorMessage());
      }
    } catch (const Error&) {
      throw;
    } catch (const Ort::Exception& exception) {
      throw Error(
          ErrorCode::runtime_execution,
          "ONNX Runtime device-to-CPU copy failed: " +
              std::string(exception.what()));
    }
  }

 private:
  template <typename Owner>
  void AddLifetimeOwner(std::shared_ptr<Owner> owner) {
    if (owner == nullptr) {
      return;
    }
    std::shared_ptr<const void> erased = std::move(owner);
    const auto found = std::ranges::find(
        lifetime_owners_,
        erased.get(),
        [](const std::shared_ptr<const void>& value) {
          return value.get();
        });
    if (found == lifetime_owners_.end()) {
      lifetime_owners_.push_back(std::move(erased));
    }
  }

  [[nodiscard]] bool RetainAliasedInputs(
      const std::vector<std::shared_ptr<TensorBuffer>>& input_buffers) {
    bool retained = false;
    const auto output_begin = reinterpret_cast<std::uintptr_t>(data_);
    const auto output_end =
        size_bytes_ > std::numeric_limits<std::uintptr_t>::max() - output_begin
            ? std::numeric_limits<std::uintptr_t>::max()
            : output_begin + size_bytes_;
    for (const auto& input : input_buffers) {
      if (input == nullptr || input->data() == nullptr) {
        continue;
      }
      bool aliases = input->data() == data_;
      if (!aliases && host_accessible_ && input->is_host_accessible()) {
        const auto input_begin =
            reinterpret_cast<std::uintptr_t>(input->data());
        const auto input_end =
            input->size_bytes() >
                    std::numeric_limits<std::uintptr_t>::max() - input_begin
                ? std::numeric_limits<std::uintptr_t>::max()
                : input_begin + input->size_bytes();
        aliases =
            output_begin < input_end && input_begin < output_end;
      }
      if (aliases) {
        retained = true;
        const auto ort_input =
            std::dynamic_pointer_cast<OrtTensorBuffer>(input);
        if (ort_input != nullptr) {
          for (const auto& owner : ort_input->lifetime_owners_) {
            AddLifetimeOwner(owner);
          }
        } else {
          AddLifetimeOwner(input);
        }
      }
    }
    return retained;
  }

  std::vector<std::shared_ptr<const void>> lifetime_owners_;
  Ort::Value value_;
  DataType data_type_{DataType::float32};
  std::vector<std::int64_t> shape_;
  std::size_t size_bytes_{0};
  const void* data_{nullptr};
  TensorDevice device_;
  bool host_accessible_{false};
};

[[nodiscard]] Tensor WrapOrtTensor(
    Ort::Value value,
    const std::shared_ptr<Ort::Session>& session_keep_alive,
    const std::shared_ptr<Ort::IoBinding>& run_keep_alive,
    const std::vector<std::shared_ptr<TensorBuffer>>& input_buffers) {
  auto buffer = std::make_shared<OrtTensorBuffer>(
      std::move(value),
      session_keep_alive,
      run_keep_alive,
      input_buffers);
  const DataType data_type = buffer->data_type();
  std::vector<std::int64_t> shape = buffer->shape();
  return Tensor::FromBuffer(
      data_type,
      std::move(shape),
      std::move(buffer));
}

class OrtBackend final : public ModelBackend {
 public:
  OrtBackend(const std::filesystem::path& model_path, const RuntimeOptions& options)
      : device_outputs_(options.device_outputs),
        memory_info_(Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault)) {
    if (options.intra_op_threads < 0 || options.inter_op_threads < 0) {
      throw Error(
          ErrorCode::invalid_argument,
          "Thread counts must be zero (automatic) or positive");
    }
    if (options.intra_op_threads > 0) {
      session_options_.SetIntraOpNumThreads(options.intra_op_threads);
    }
    if (options.inter_op_threads > 0) {
      session_options_.SetInterOpNumThreads(options.inter_op_threads);
    }
    session_options_.SetLogSeverityLevel(options.log_severity);
    session_options_.SetGraphOptimizationLevel(
        ToOrtGraphOptimizationLevel(options.graph_optimization));
    const auto providers = ResolveProviders(options);
    bool has_cpu_fallback = false;
    for (const auto& provider : providers) {
      AppendProvider(session_options_, provider);
      metadata_.execution_providers.push_back(provider.available_name);
      has_cpu_fallback = has_cpu_fallback || provider.name == "cpu";
    }
    if (!has_cpu_fallback) {
      session_options_.AddConfigEntry(
          "session.disable_cpu_ep_fallback", "1");
    }
    session_ = std::make_shared<Ort::Session>(
        SharedOrtEnvironment(),
        model_path.c_str(),
        session_options_);

    Ort::AllocatorWithDefaultOptions allocator;
    metadata_.inputs.reserve(session_->GetInputCount());
    for (std::size_t index = 0; index < session_->GetInputCount(); ++index) {
      metadata_.inputs.push_back(
          ReadTensorSpec(*session_, index, true, allocator));
    }
    metadata_.outputs.reserve(session_->GetOutputCount());
    for (std::size_t index = 0; index < session_->GetOutputCount(); ++index) {
      metadata_.outputs.push_back(
          ReadTensorSpec(*session_, index, false, allocator));
    }

    // Graph partitioning has already decided where each port lives, and that
    // decision cannot change for the life of the session, so it is read once
    // here rather than on every Run. Ort::ConstMemoryInfo is a non-owning view
    // of memory ONNX Runtime owns inside the session; session_ is held by this
    // backend and released after these vectors, so every view stays valid for
    // as long as the backend does.
    input_memory_ = session_->GetMemoryInfoForInputs();
    output_memory_ = session_->GetMemoryInfoForOutputs();
    if (input_memory_.size() != metadata_.inputs.size() ||
        output_memory_.size() != metadata_.outputs.size()) {
      throw Error(
          ErrorCode::runtime_load,
          "ONNX Runtime reported a memory plan that does not cover every "
          "model input and output");
    }
    for (std::size_t index = 0; index < metadata_.inputs.size(); ++index) {
      if (static_cast<const OrtMemoryInfo*>(input_memory_[index]) != nullptr) {
        metadata_.inputs[index].device =
            TensorDeviceForMemory(input_memory_[index]);
      }
    }
    for (std::size_t index = 0; index < metadata_.outputs.size(); ++index) {
      if (static_cast<const OrtMemoryInfo*>(output_memory_[index]) != nullptr) {
        metadata_.outputs[index].device =
            TensorDeviceForMemory(output_memory_[index]);
      }
    }
  }

  [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
    return metadata_;
  }

  [[nodiscard]] NamedTensors Run(const NamedTensors& inputs) const override {
    return Run(inputs, CancellationToken{});
  }

  [[nodiscard]] NamedTensors Run(
      const NamedTensors& inputs,
      const CancellationToken& cancellation) const override {
    cancellation.ThrowIfCancellationRequested();
    try {
      auto binding = std::make_shared<Ort::IoBinding>(*session_);
      std::vector<Tensor> cpu_inputs;
      cpu_inputs.reserve(metadata_.inputs.size());
      std::vector<Ort::Value> input_values;
      input_values.reserve(metadata_.inputs.size());
      std::vector<std::shared_ptr<TensorBuffer>> input_buffers;
      input_buffers.reserve(metadata_.inputs.size());
      for (std::size_t index = 0; index < metadata_.inputs.size(); ++index) {
        const TensorSpec& spec = metadata_.inputs[index];
        const Tensor& input = inputs.at(spec.name);
        const auto ort_buffer =
            std::dynamic_pointer_cast<OrtTensorBuffer>(input.buffer());
        const bool device_compatible =
            ort_buffer != nullptr && spec.device.has_value() &&
            ort_buffer->device() == *spec.device;
        if (ort_buffer != nullptr &&
            ort_buffer->data_type() == input.data_type() &&
            ort_buffer->shape() == input.shape() &&
            device_compatible) {
          binding->BindInput(spec.name.c_str(), ort_buffer->value());
          input_buffers.push_back(input.buffer());
        } else {
          cpu_inputs.push_back(input.CopyToCpu());
          input_values.push_back(
              MakeOrtTensor(cpu_inputs.back(), memory_info_));
          binding->BindInput(spec.name.c_str(), input_values.back());
          input_buffers.push_back(cpu_inputs.back().buffer());
        }
      }

      binding->SynchronizeInputs();
      for (std::size_t index = 0; index < metadata_.outputs.size(); ++index) {
        const OrtMemoryInfo* location = output_memory_[index];
        binding->BindOutput(
            metadata_.outputs[index].name.c_str(),
            !device_outputs_ || location == nullptr
                ? static_cast<const OrtMemoryInfo*>(memory_info_)
                : location);
      }

      // A fresh RunOptions per call, never reused and never un-terminated: a
      // terminated one stays terminated, so sharing it would silently poison
      // the next inference. The registration is created after it and
      // therefore destroyed before it, and unregistering blocks until any
      // callback already running has returned, so SetTerminate can never
      // touch this object after it dies. ONNX Runtime only checks the
      // terminate flag between graph nodes, so a single long kernel still
      // runs to completion before the call unwinds.
      Ort::RunOptions run_options;
      const detail::CancellationRegistration termination(
          cancellation,
          [&run_options](CancellationReason) { run_options.SetTerminate(); });

      session_->Run(run_options, *binding);
      binding->SynchronizeOutputs();
      auto outputs = binding->GetOutputValues();
      if (outputs.size() != metadata_.outputs.size()) {
        throw Error(
            ErrorCode::runtime_execution,
            "ONNX Runtime returned an unexpected number of outputs");
      }
      NamedTensors result;
      result.reserve(outputs.size());
      for (std::size_t index = 0; index < outputs.size(); ++index) {
        result.emplace(
            metadata_.outputs[index].name,
            WrapOrtTensor(
                std::move(outputs[index]),
                session_,
                binding,
                input_buffers));
      }
      cancellation.ThrowIfCancellationRequested();
      return result;
    } catch (const Ort::Exception& exception) {
      // A terminated run surfaces as an ordinary ORT failure whose message is
      // not part of any contract, so the token -- not the text -- decides
      // whether this was a cancellation.
      cancellation.ThrowIfCancellationRequested();
      throw Error(
          ErrorCode::runtime_execution,
          "ONNX Runtime inference failed: " +
              AnnotateExecutionFailure(exception.what()));
    }
  }

 private:
  Ort::SessionOptions session_options_;
  std::shared_ptr<Ort::Session> session_;
  bool device_outputs_{false};
  Ort::MemoryInfo memory_info_;
  //: Non-owning views of the per-port memory plan ONNX Runtime owns inside
  //: session_. They are read once in the constructor and are valid for as
  //: long as session_ is, which is the lifetime of this backend.
  std::vector<Ort::ConstMemoryInfo> input_memory_;
  std::vector<Ort::ConstMemoryInfo> output_memory_;
  ModelMetadata metadata_;
};

}  // namespace

ModelBackendPtr CreateOrtBackend(
    const std::filesystem::path& model_path,
    const RuntimeOptions& options) {
  if (model_path.empty()) {
    throw Error(ErrorCode::invalid_argument, "Model path cannot be empty");
  }
  if (!std::filesystem::is_regular_file(model_path)) {
    throw Error(
        ErrorCode::invalid_argument,
        "Model file does not exist: " + model_path.string());
  }
  if (options.log_severity < ORT_LOGGING_LEVEL_VERBOSE ||
      options.log_severity > ORT_LOGGING_LEVEL_FATAL) {
    throw Error(
        ErrorCode::invalid_argument,
        "log_severity must be between 0 (verbose) and 4 (fatal)");
  }

  InitializeOrtApi(options.ort_library_path);
  try {
    return std::make_shared<OrtBackend>(model_path, options);
  } catch (const Error&) {
    throw;
  } catch (const Ort::Exception& exception) {
    throw Error(
        ErrorCode::runtime_load,
        "Failed to create ONNX Runtime session: " +
            std::string(exception.what()));
  }
}

std::vector<std::string> GetAvailableOrtProviders(
    const std::filesystem::path& library_path) {
  InitializeOrtApi(library_path);
  try {
    return Ort::GetAvailableProviders();
  } catch (const Ort::Exception& exception) {
    throw Error(
        ErrorCode::runtime_load,
        "Failed to query ONNX Runtime execution providers: " +
            std::string(exception.what()));
  }
}

void RegisterOrtExecutionProviderLibrary(
    std::string_view registration_name,
    const std::filesystem::path& provider_library_path,
    const std::filesystem::path& ort_library_path) {
  if (registration_name.empty()) {
    throw Error(
        ErrorCode::invalid_argument,
        "Execution provider registration name cannot be empty");
  }
  if (!std::filesystem::is_regular_file(provider_library_path)) {
    throw Error(
        ErrorCode::invalid_argument,
        "Execution provider library does not exist: " +
            provider_library_path.string());
  }
  InitializeOrtApi(ort_library_path);
  try {
    SharedOrtEnvironment().RegisterExecutionProviderLibrary(
        std::string(registration_name).c_str(),
        std::filesystem::canonical(provider_library_path).native());
  } catch (const Ort::Exception& exception) {
    throw Error(
        ErrorCode::runtime_load,
        "Failed to register ONNX Runtime execution provider library '" +
            std::string(registration_name) + "': " + exception.what());
  }
}

}  // namespace onnx_world_model::detail
