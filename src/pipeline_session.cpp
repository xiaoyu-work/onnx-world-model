#include "onnx_world_model/pipeline.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "onnx_world_model/error.hpp"

namespace onnx_world_model {
namespace {

using Json = nlohmann::json;

[[noreturn]] void ExecutionError(std::string message) {
  throw Error(ErrorCode::runtime_execution, std::move(message));
}

const PipelineStage& FindStage(
    const PipelineManifest& manifest,
    std::string_view name) {
  const auto found =
      std::ranges::find(manifest.stages(), name, &PipelineStage::name);
  if (found == manifest.stages().end()) {
    throw Error(
        ErrorCode::invalid_argument,
        "Pipeline has no stage '" + std::string(name) + "'");
  }
  return *found;
}

const PipelineInput* FindDeclaredInput(
    const PipelineManifest& manifest,
    const Endpoint& endpoint) {
  const auto found = std::ranges::find(
      manifest.inputs(), endpoint, &PipelineInput::port);
  return found == manifest.inputs().end() ? nullptr : &*found;
}

const PipelineState* FindInputState(
    const PipelineManifest& manifest,
    const Endpoint& endpoint) {
  const auto found = std::ranges::find(
      manifest.states(), endpoint, &PipelineState::input);
  return found == manifest.states().end() ? nullptr : &*found;
}

const PipelineConnection* FindConnection(
    const PipelineManifest& manifest,
    const Endpoint& target,
    bool recurrent) {
  const auto found = std::ranges::find_if(
      manifest.connections(),
      [&target, recurrent](const PipelineConnection& connection) {
        return connection.target == target &&
               connection.recurrent == recurrent;
      });
  return found == manifest.connections().end() ? nullptr : &*found;
}

const PipelineConnection& FindStateConnection(
    const PipelineManifest& manifest,
    const PipelineState& state) {
  const auto found = std::ranges::find_if(
      manifest.connections(),
      [&state](const PipelineConnection& connection) {
        return connection.recurrent &&
               connection.source == state.output &&
               connection.target == state.input;
      });
  if (found == manifest.connections().end()) {
    ExecutionError(
        "State '" + state.name + "' has no recurrent connection");
  }
  return *found;
}

bool Contains(
    const std::vector<std::string>& values,
    std::string_view value) {
  return std::ranges::find(values, value) != values.end();
}

std::size_t CheckedElementCount(
    const std::vector<std::int64_t>& shape) {
  std::size_t result = 1;
  for (const std::int64_t dimension : shape) {
    if (dimension < 0) {
      ExecutionError("Generated tensor shape has an unresolved dimension");
    }
    const auto converted = static_cast<std::size_t>(dimension);
    if (converted != 0 &&
        result > std::numeric_limits<std::size_t>::max() / converted) {
      ExecutionError("Generated tensor element count overflows size_t");
    }
    result *= converted;
  }
  return result;
}

template <typename T>
void FillTensor(Tensor& tensor, T value) {
  auto bytes = tensor.mutable_bytes();
  auto values = std::span(
      reinterpret_cast<T*>(bytes.data()),
      tensor.element_count());
  std::ranges::fill(values, value);
}

Tensor FilledTensor(
    DataType data_type,
    std::vector<std::int64_t> shape,
    double value) {
  Tensor result(data_type, std::move(shape));
  switch (data_type) {
    case DataType::float32:
      FillTensor(result, static_cast<float>(value));
      break;
    case DataType::float64:
      FillTensor(result, value);
      break;
    case DataType::int64:
      FillTensor(result, static_cast<std::int64_t>(value));
      break;
    case DataType::int32:
      FillTensor(result, static_cast<std::int32_t>(value));
      break;
    case DataType::int16:
      FillTensor(result, static_cast<std::int16_t>(value));
      break;
    case DataType::int8:
      FillTensor(result, static_cast<std::int8_t>(value));
      break;
    case DataType::uint64:
      FillTensor(result, static_cast<std::uint64_t>(value));
      break;
    case DataType::uint32:
      FillTensor(result, static_cast<std::uint32_t>(value));
      break;
    case DataType::uint16:
      FillTensor(result, static_cast<std::uint16_t>(value));
      break;
    case DataType::uint8:
      FillTensor(result, static_cast<std::uint8_t>(value));
      break;
    case DataType::boolean:
      FillTensor(result, value != 0.0);
      break;
    case DataType::float16:
    case DataType::bfloat16:
      if (value != 0.0) {
        ExecutionError(
            "Non-zero float16/bfloat16 generated constants are not supported");
      }
      break;
  }
  return result;
}

float HalfToFloat(std::uint16_t half) {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(half & 0x8000U) << 16U;
  std::uint32_t exponent = (half >> 10U) & 0x1FU;
  std::uint32_t mantissa = half & 0x03FFU;
  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 127U - 15U + 1U;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1U;
        --exponent;
      }
      mantissa &= 0x03FFU;
      bits = sign | (exponent << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 127U - 15U) << 23U) |
           (mantissa << 13U);
  }
  return std::bit_cast<float>(bits);
}

std::uint16_t FloatToHalf(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;
  if (exponent == 0xFFU) {
    return static_cast<std::uint16_t>(
        sign | 0x7C00U | (mantissa == 0 ? 0U : 0x0200U));
  }
  int adjusted = static_cast<int>(exponent) - 127 + 15;
  if (adjusted >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (adjusted <= 0) {
    if (adjusted < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t normalized = mantissa | 0x800000U;
    const int shift = 14 - adjusted;
    const std::uint32_t rounded =
        (normalized + (1U << (shift - 1))) >> shift;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  std::uint32_t rounded = mantissa + 0x1000U;
  if ((rounded & 0x800000U) != 0) {
    rounded = 0;
    ++adjusted;
    if (adjusted >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(adjusted) << 10U) |
      ((rounded >> 13U) & 0x03FFU));
}

float ReadFloat(const Tensor& tensor, std::size_t index) {
  switch (tensor.data_type()) {
    case DataType::float32:
      return tensor.values<float>()[index];
    case DataType::float64:
      return static_cast<float>(tensor.values<double>()[index]);
    case DataType::float16: {
      const auto values = std::span(
          reinterpret_cast<const std::uint16_t*>(tensor.bytes().data()),
          tensor.element_count());
      return HalfToFloat(values[index]);
    }
    case DataType::bfloat16: {
      const auto values = std::span(
          reinterpret_cast<const std::uint16_t*>(tensor.bytes().data()),
          tensor.element_count());
      return std::bit_cast<float>(
          static_cast<std::uint32_t>(values[index]) << 16U);
    }
    case DataType::int64:
      return static_cast<float>(tensor.values<std::int64_t>()[index]);
    case DataType::int32:
      return static_cast<float>(tensor.values<std::int32_t>()[index]);
    case DataType::int16:
      return static_cast<float>(tensor.values<std::int16_t>()[index]);
    case DataType::int8:
      return static_cast<float>(tensor.values<std::int8_t>()[index]);
    case DataType::uint64:
      return static_cast<float>(tensor.values<std::uint64_t>()[index]);
    case DataType::uint32:
      return static_cast<float>(tensor.values<std::uint32_t>()[index]);
    case DataType::uint16:
      return static_cast<float>(tensor.values<std::uint16_t>()[index]);
    case DataType::uint8:
      return static_cast<float>(tensor.values<std::uint8_t>()[index]);
    case DataType::boolean:
      return tensor.bytes()[index] == std::byte{0} ? 0.0F : 1.0F;
  }
  ExecutionError("Unsupported tensor data type");
}

void WriteFloat(Tensor& tensor, std::size_t index, float value) {
  switch (tensor.data_type()) {
    case DataType::float32: {
      auto values = std::span(
          reinterpret_cast<float*>(tensor.mutable_bytes().data()),
          tensor.element_count());
      values[index] = value;
      return;
    }
    case DataType::float64: {
      auto values = std::span(
          reinterpret_cast<double*>(tensor.mutable_bytes().data()),
          tensor.element_count());
      values[index] = value;
      return;
    }
    case DataType::float16: {
      auto values = std::span(
          reinterpret_cast<std::uint16_t*>(tensor.mutable_bytes().data()),
          tensor.element_count());
      values[index] = FloatToHalf(value);
      return;
    }
    case DataType::bfloat16: {
      auto values = std::span(
          reinterpret_cast<std::uint16_t*>(tensor.mutable_bytes().data()),
          tensor.element_count());
      const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
      const std::uint32_t rounding =
          0x7FFFU + ((bits >> 16U) & 1U);
      values[index] =
          static_cast<std::uint16_t>((bits + rounding) >> 16U);
      return;
    }
    case DataType::int64: {
      auto values = std::span(
          reinterpret_cast<std::int64_t*>(tensor.mutable_bytes().data()),
          tensor.element_count());
      values[index] = static_cast<std::int64_t>(value);
      return;
    }
    case DataType::int32: {
      auto values = std::span(
          reinterpret_cast<std::int32_t*>(tensor.mutable_bytes().data()),
          tensor.element_count());
      values[index] = static_cast<std::int32_t>(value);
      return;
    }
    case DataType::int16: {
      auto values = std::span(
          reinterpret_cast<std::int16_t*>(tensor.mutable_bytes().data()),
          tensor.element_count());
      values[index] = static_cast<std::int16_t>(value);
      return;
    }
    case DataType::int8: {
      auto values = std::span(
          reinterpret_cast<std::int8_t*>(tensor.mutable_bytes().data()),
          tensor.element_count());
      values[index] = static_cast<std::int8_t>(value);
      return;
    }
    case DataType::uint64: {
      auto values = std::span(
          reinterpret_cast<std::uint64_t*>(tensor.mutable_bytes().data()),
          tensor.element_count());
      values[index] = static_cast<std::uint64_t>(value);
      return;
    }
    case DataType::uint32: {
      auto values = std::span(
          reinterpret_cast<std::uint32_t*>(tensor.mutable_bytes().data()),
          tensor.element_count());
      values[index] = static_cast<std::uint32_t>(value);
      return;
    }
    case DataType::uint16: {
      auto values = std::span(
          reinterpret_cast<std::uint16_t*>(tensor.mutable_bytes().data()),
          tensor.element_count());
      values[index] = static_cast<std::uint16_t>(value);
      return;
    }
    case DataType::uint8: {
      auto values = std::span(
          reinterpret_cast<std::uint8_t*>(tensor.mutable_bytes().data()),
          tensor.element_count());
      values[index] = static_cast<std::uint8_t>(value);
      return;
    }
    case DataType::boolean:
      tensor.mutable_bytes()[index] =
          value == 0.0F ? std::byte{0} : std::byte{1};
      return;
  }
  ExecutionError("Unsupported tensor data type");
}

bool IsIntegral(DataType type) {
  return type != DataType::float16 && type != DataType::bfloat16 &&
         type != DataType::float32 && type != DataType::float64;
}

bool IsUnsigned(DataType type) {
  return type == DataType::uint8 || type == DataType::uint16 ||
         type == DataType::uint32 || type == DataType::uint64 ||
         type == DataType::boolean;
}

std::uint64_t ReadUnsignedInteger(
    const Tensor& tensor,
    std::size_t index) {
  switch (tensor.data_type()) {
    case DataType::uint64:
      return tensor.values<std::uint64_t>()[index];
    case DataType::uint32:
      return tensor.values<std::uint32_t>()[index];
    case DataType::uint16:
      return tensor.values<std::uint16_t>()[index];
    case DataType::uint8:
      return tensor.values<std::uint8_t>()[index];
    case DataType::boolean:
      return tensor.bytes()[index] == std::byte{0} ? 0U : 1U;
    case DataType::int64:
      return static_cast<std::uint64_t>(
          tensor.values<std::int64_t>()[index]);
    case DataType::int32:
      return static_cast<std::uint64_t>(
          tensor.values<std::int32_t>()[index]);
    case DataType::int16:
      return static_cast<std::uint64_t>(
          tensor.values<std::int16_t>()[index]);
    case DataType::int8:
      return static_cast<std::uint64_t>(
          tensor.values<std::int8_t>()[index]);
    default:
      ExecutionError("Source tensor is not integral");
  }
}

std::int64_t ReadSignedInteger(
    const Tensor& tensor,
    std::size_t index) {
  switch (tensor.data_type()) {
    case DataType::int64:
      return tensor.values<std::int64_t>()[index];
    case DataType::int32:
      return tensor.values<std::int32_t>()[index];
    case DataType::int16:
      return tensor.values<std::int16_t>()[index];
    case DataType::int8:
      return tensor.values<std::int8_t>()[index];
    default:
      return static_cast<std::int64_t>(
          ReadUnsignedInteger(tensor, index));
  }
}

template <typename T, typename Value>
void WriteIntegerValue(Tensor& tensor, std::size_t index, Value value) {
  auto values = std::span(
      reinterpret_cast<T*>(tensor.mutable_bytes().data()),
      tensor.element_count());
  values[index] = static_cast<T>(value);
}

template <typename Value>
void WriteInteger(
    Tensor& tensor,
    std::size_t index,
    Value value) {
  switch (tensor.data_type()) {
    case DataType::int64:
      WriteIntegerValue<std::int64_t>(tensor, index, value);
      return;
    case DataType::int32:
      WriteIntegerValue<std::int32_t>(tensor, index, value);
      return;
    case DataType::int16:
      WriteIntegerValue<std::int16_t>(tensor, index, value);
      return;
    case DataType::int8:
      WriteIntegerValue<std::int8_t>(tensor, index, value);
      return;
    case DataType::uint64:
      WriteIntegerValue<std::uint64_t>(tensor, index, value);
      return;
    case DataType::uint32:
      WriteIntegerValue<std::uint32_t>(tensor, index, value);
      return;
    case DataType::uint16:
      WriteIntegerValue<std::uint16_t>(tensor, index, value);
      return;
    case DataType::uint8:
      WriteIntegerValue<std::uint8_t>(tensor, index, value);
      return;
    case DataType::boolean:
      tensor.mutable_bytes()[index] =
          value == 0 ? std::byte{0} : std::byte{1};
      return;
    default:
      ExecutionError("Target tensor is not integral");
  }
}

Tensor CastTensor(const Tensor& source, DataType target_type) {
  if (source.data_type() == target_type) {
    return source;
  }
  Tensor result(target_type, source.shape());
  if (IsIntegral(source.data_type()) && IsIntegral(target_type)) {
    for (std::size_t index = 0; index < source.element_count(); ++index) {
      if (IsUnsigned(source.data_type())) {
        WriteInteger(
            result, index, ReadUnsignedInteger(source, index));
      } else {
        WriteInteger(result, index, ReadSignedInteger(source, index));
      }
    }
    return result;
  }
  if (IsIntegral(source.data_type()) &&
      target_type == DataType::float64) {
    auto values = std::span(
        reinterpret_cast<double*>(result.mutable_bytes().data()),
        result.element_count());
    for (std::size_t index = 0; index < source.element_count(); ++index) {
      values[index] =
          IsUnsigned(source.data_type())
              ? static_cast<double>(ReadUnsignedInteger(source, index))
              : static_cast<double>(ReadSignedInteger(source, index));
    }
    return result;
  }
  if (!IsIntegral(source.data_type()) && IsIntegral(target_type)) {
    for (std::size_t index = 0; index < source.element_count(); ++index) {
      const double value =
          source.data_type() == DataType::float64
              ? source.values<double>()[index]
              : static_cast<double>(ReadFloat(source, index));
      WriteInteger(result, index, value);
    }
    return result;
  }
  for (std::size_t index = 0; index < source.element_count(); ++index) {
    WriteFloat(result, index, ReadFloat(source, index));
  }
  return result;
}

Tensor EulerStep(
    const Tensor& state,
    const Tensor& velocity,
    float delta) {
  if (state.shape() != velocity.shape()) {
    ExecutionError(
        "Scheduler state and velocity must have identical shapes");
  }
  Tensor result(state.data_type(), state.shape());
  for (std::size_t index = 0; index < state.element_count(); ++index) {
    WriteFloat(
        result,
        index,
        ReadFloat(state, index) + delta * ReadFloat(velocity, index));
  }
  return result;
}

Tensor EulerStepIndexed(
    const Tensor& state,
    const Tensor& velocity,
    const Tensor* indexes,
    float delta) {
  if (state.shape() == velocity.shape() && indexes == nullptr) {
    return EulerStep(state, velocity, delta);
  }
  if (state.shape().size() != 2 || velocity.shape().size() != 2 ||
      state.shape()[1] != velocity.shape()[1] ||
      indexes == nullptr ||
      indexes->data_type() != DataType::int64 ||
      indexes->element_count() !=
          static_cast<std::size_t>(velocity.shape()[0])) {
    ExecutionError(
        "Indexed scheduler update has incompatible state, velocity, or indexes");
  }
  Tensor result = state;
  const auto rows = indexes->values<std::int64_t>();
  const std::size_t width =
      static_cast<std::size_t>(state.shape()[1]);
  for (std::size_t velocity_row = 0;
       velocity_row < rows.size();
       ++velocity_row) {
    if (rows[velocity_row] < 0 ||
        rows[velocity_row] >= state.shape()[0]) {
      ExecutionError("Scheduler token index is outside the state tensor");
    }
    const std::size_t state_row =
        static_cast<std::size_t>(rows[velocity_row]);
    for (std::size_t column = 0; column < width; ++column) {
      const std::size_t state_index = state_row * width + column;
      const std::size_t velocity_index =
          velocity_row * width + column;
      WriteFloat(
          result,
          state_index,
          ReadFloat(state, state_index) +
              delta * ReadFloat(velocity, velocity_index));
    }
  }
  return result;
}

Tensor Int64Tensor(
    std::vector<std::int64_t> shape,
    const std::vector<std::int64_t>& values) {
  return Tensor::FromValues<std::int64_t>(
      std::move(shape), std::span(values));
}

Tensor FloatTensor(
    std::vector<std::int64_t> shape,
    const std::vector<float>& values) {
  return Tensor::FromValues<float>(std::move(shape), std::span(values));
}

std::vector<std::string> TopologicalComponents(
    const PipelineManifest& manifest,
    const PipelineStage& stage) {
  std::unordered_map<std::string, std::size_t> indegree;
  std::unordered_map<std::string, std::vector<std::string>> successors;
  for (const auto& component : stage.components) {
    indegree.emplace(component, 0);
  }
  for (const auto& connection : manifest.connections()) {
    if (connection.recurrent ||
        connection.source.component == connection.target.component ||
        !indegree.contains(connection.source.component) ||
        !indegree.contains(connection.target.component)) {
      continue;
    }
    successors[connection.source.component].push_back(
        connection.target.component);
    ++indegree.at(connection.target.component);
  }

  std::deque<std::string> ready;
  for (const auto& component : stage.components) {
    if (indegree.at(component) == 0) {
      ready.push_back(component);
    }
  }
  std::vector<std::string> result;
  result.reserve(stage.components.size());
  while (!ready.empty()) {
    std::string component = std::move(ready.front());
    ready.pop_front();
    result.push_back(component);
    for (const auto& successor : successors[component]) {
      if (--indegree.at(successor) == 0) {
        ready.push_back(successor);
      }
    }
  }
  if (result.size() != stage.components.size()) {
    ExecutionError("Stage '" + stage.name + "' contains a dataflow cycle");
  }
  return result;
}

}  // namespace

struct PipelineSession::Impl {
  struct SchedulerHistory {
    std::vector<Tensor> model_outputs;
    std::optional<Tensor> last_sample;
    std::size_t lower_order{0};
    std::size_t previous_order{1};
  };

  explicit Impl(std::shared_ptr<const PipelinePackage> pipeline_package)
      : package(std::move(pipeline_package)) {}

  std::shared_ptr<const PipelinePackage> package;
  mutable std::mutex mutex;
  NamedTensors external_values;
  NamedTensors endpoint_values;
  NamedTensors state_values;
  std::unordered_map<std::string, std::size_t> stage_iterations;
  mutable std::unordered_map<std::string, SchedulerHistory>
      scheduler_histories;
  mutable std::unordered_map<std::string, std::vector<std::int64_t>>
      position_cursors;
  std::mt19937_64 random_engine{0};

  [[nodiscard]] const PipelineManifest& manifest() const noexcept {
    return package->manifest();
  }

  [[nodiscard]] bool ComponentPresent(
      const PipelineComponent& component) const {
    if (!component.presence.has_value()) {
      return true;
    }
    return std::ranges::any_of(
        manifest().inputs(),
        [this, &component](const PipelineInput& input) {
          return input.presence == *component.presence &&
                 external_values.contains(input.port.qualified());
        });
  }

  [[nodiscard]] Tensor AdaptExternal(
      const Tensor& tensor,
      const TensorSpec& spec) const {
    if (tensor.shape().size() == spec.shape.size()) {
      return tensor;
    }
    if (tensor.shape().size() + 1 == spec.shape.size() &&
        (spec.shape[0] < 0 || spec.shape[0] == 1)) {
      std::vector<std::int64_t> shape{1};
      shape.insert(shape.end(), tensor.shape().begin(), tensor.shape().end());
      return Tensor::FromBytes(
          tensor.data_type(), std::move(shape), tensor.bytes());
    }
    if (tensor.shape().size() == spec.shape.size() + 1 &&
        tensor.shape()[0] == 1) {
      std::vector<std::int64_t> shape(
          tensor.shape().begin() + 1, tensor.shape().end());
      return Tensor::FromBytes(
          tensor.data_type(), std::move(shape), tensor.bytes());
    }
    return tensor;
  }

  void StoreExternalInputs(const NamedTensors& inputs) {
    for (const auto& [name, tensor] : inputs) {
      bool matched = false;
      for (const auto& input : manifest().inputs()) {
        if (input.kind != PipelineInputKind::external) {
          continue;
        }
        if (name == input.name || name == input.semantic ||
            name == input.port.qualified()) {
          external_values.insert_or_assign(
              input.port.qualified(),
              AdaptExternal(tensor, manifest().Input(input.port)));
          matched = true;
        }
      }
      if (!matched) {
        throw Error(
            ErrorCode::invalid_argument,
            "Unknown pipeline input '" + name + "'");
      }
    }
  }

  [[nodiscard]] const Tensor* Override(
      const NamedTensors& overrides,
      const PipelineInput& input) const {
    for (const auto& name :
         {input.port.qualified(), input.semantic, input.name}) {
      if (name.empty()) {
        continue;
      }
      const auto found = overrides.find(name);
      if (found != overrides.end()) {
        return &found->second;
      }
    }
    return nullptr;
  }

  [[nodiscard]] const Tensor* Override(
      const NamedTensors& overrides,
      const Endpoint& endpoint) const {
    const auto found = overrides.find(endpoint.qualified());
    return found == overrides.end() ? nullptr : &found->second;
  }

  [[nodiscard]] std::optional<std::int64_t> SymbolValue(
      std::string_view symbol) const {
    std::optional<std::int64_t> result;
    const auto bind = [&result, symbol](
                          const Tensor& tensor,
                          const std::vector<std::string>& symbols) {
      for (std::size_t axis = 0; axis < symbols.size(); ++axis) {
        if (symbols[axis] != symbol) {
          continue;
        }
        const std::int64_t value = tensor.shape()[axis];
        if (result.has_value() && *result != value) {
          ExecutionError(
              "Symbolic dimension '" + std::string(symbol) +
              "' has conflicting concrete values");
        }
        result = value;
      }
    };

    for (const auto& input : manifest().inputs()) {
      const auto value = external_values.find(input.port.qualified());
      if (value == external_values.end()) {
        continue;
      }
      const auto& component = manifest().Component(input.port.component);
      const auto symbols =
          component.input_dimension_symbols.find(input.port.port);
      if (symbols != component.input_dimension_symbols.end()) {
        bind(value->second, symbols->second);
      }
    }
    for (const auto& component : manifest().components()) {
      for (const auto& output : component.metadata.outputs) {
        const auto value =
            endpoint_values.find(component.name + "." + output.name);
        if (value == endpoint_values.end()) {
          continue;
        }
        const auto symbols =
            component.output_dimension_symbols.find(output.name);
        if (symbols != component.output_dimension_symbols.end()) {
          bind(value->second, symbols->second);
        }
      }
    }
    return result;
  }

  [[nodiscard]] std::optional<std::int64_t> BatchSize() const {
    std::optional<std::int64_t> result;
    for (const auto& input : manifest().inputs()) {
      const auto value = external_values.find(input.port.qualified());
      if (value == external_values.end() || value->second.shape().empty()) {
        continue;
      }
      const auto& component = manifest().Component(input.port.component);
      const auto symbols =
          component.input_dimension_symbols.find(input.port.port);
      if (symbols == component.input_dimension_symbols.end() ||
          symbols->second.empty() ||
          (symbols->second[0] != "b" && symbols->second[0] != "batch")) {
        continue;
      }
      const std::int64_t batch = value->second.shape()[0];
      if (result.has_value() && *result != batch) {
        ExecutionError("Pipeline external inputs have conflicting batch sizes");
      }
      result = batch;
    }
    return result;
  }

  [[nodiscard]] const Tensor* KnownValue(const Endpoint& endpoint) const {
    if (const PipelineState* state =
            FindInputState(manifest(), endpoint)) {
      const auto found = state_values.find(state->name);
      if (found != state_values.end()) {
        return &found->second;
      }
    }
    const auto external = external_values.find(endpoint.qualified());
    if (external != external_values.end()) {
      return &external->second;
    }
    const auto direct = endpoint_values.find(endpoint.qualified());
    if (direct != endpoint_values.end()) {
      return &direct->second;
    }
    if (const PipelineConnection* connection =
            FindConnection(manifest(), endpoint, false)) {
      const auto source =
          endpoint_values.find(connection->source.qualified());
      if (source != endpoint_values.end() &&
          !connection->transform.has_value()) {
        return &source->second;
      }
    }
    return nullptr;
  }

  [[nodiscard]] std::vector<std::int64_t> GeneratedShape(
      const PipelineInput& input,
      const Json& parameters) const {
    const TensorSpec& spec = manifest().Input(input.port);
    if (parameters.contains("shape_from")) {
      const Endpoint source =
          Endpoint::Parse(parameters.at("shape_from").get<std::string>());
      const Tensor* value = KnownValue(source);
      if (value == nullptr) {
        ExecutionError(
            "Generated input '" + input.port.qualified() +
            "' shape_from value is unavailable");
      }
      return value->shape();
    }
    if (parameters.contains("shape")) {
      const Json& shape = parameters.at("shape");
      if (!shape.is_array()) {
        ExecutionError("Generated tensor shape must be an array");
      }
      std::vector<std::int64_t> result;
      result.reserve(shape.size());
      for (const auto& dimension : shape) {
        if (dimension.is_number_integer()) {
          result.push_back(dimension.get<std::int64_t>());
        } else if (dimension.is_string()) {
          const auto value =
              SymbolValue(dimension.get<std::string>());
          if (!value.has_value()) {
            ExecutionError(
                "Generated tensor shape references an unbound symbol");
          }
          result.push_back(*value);
        } else {
          ExecutionError("Generated tensor shape has an invalid dimension");
        }
      }
      return result;
    }

    const auto& component = manifest().Component(input.port.component);
    const auto symbols =
        component.input_dimension_symbols.find(input.port.port);
    std::vector<std::string> dimension_symbols(spec.shape.size());
    if (symbols != component.input_dimension_symbols.end()) {
      dimension_symbols = symbols->second;
    }
    Json dynamic_axes = Json::object();
    if (parameters.contains("dynamic_axes")) {
      dynamic_axes = parameters.at("dynamic_axes");
    }

    std::vector<std::int64_t> result = spec.shape;
    for (std::size_t axis = 0; axis < result.size(); ++axis) {
      if (result[axis] >= 0) {
        continue;
      }
      const std::string& symbol = dimension_symbols[axis];
      if (!symbol.empty() && dynamic_axes.contains(symbol)) {
        result[axis] = dynamic_axes.at(symbol).get<std::int64_t>();
        continue;
      }
      if ((symbol == "b" || symbol == "batch") && BatchSize().has_value()) {
        result[axis] = *BatchSize();
        continue;
      }
      if (!symbol.empty()) {
        const auto value = SymbolValue(symbol);
        if (value.has_value()) {
          result[axis] = *value;
          continue;
        }
      }
      if (dynamic_axes.size() == 1) {
        result[axis] =
            dynamic_axes.begin().value().get<std::int64_t>();
        continue;
      }
      ExecutionError(
          "Generated input '" + input.port.qualified() +
          "' has an unresolved dimension");
    }
    return result;
  }

  [[nodiscard]] std::size_t StateSequenceLength(
      const Json& state_names) const {
    if (!state_names.is_array()) {
      ExecutionError("past_state must be an array of state names");
    }
    std::size_t length = 0;
    for (const auto& name : state_names) {
      if (!name.is_string()) {
        ExecutionError("past_state entries must be strings");
      }
      const auto state =
          std::ranges::find(
              manifest().states(),
              name.get<std::string>(),
              &PipelineState::name);
      if (state == manifest().states().end() ||
          !state->sequence_axis.has_value()) {
        continue;
      }
      const auto value = state_values.find(state->name);
      if (value != state_values.end()) {
        length = std::max(
            length,
            static_cast<std::size_t>(
                value->second.shape()[*state->sequence_axis]));
      }
    }
    return length;
  }

  [[nodiscard]] std::size_t TokenCount(std::string_view modality) const {
    std::string port;
    if (modality == "text" || modality == "und") {
      port = "input_ids";
    } else {
      port = std::string(modality) + "_tokens";
    }
    const Tensor* value = KnownValue({"generator", port});
    if (value == nullptr || value->shape().empty()) {
      return 0;
    }
    if (port == "input_ids" && value->shape().size() == 2) {
      return static_cast<std::size_t>(value->shape()[1]);
    }
    return static_cast<std::size_t>(value->shape()[0]);
  }

  [[nodiscard]] std::size_t NoisyTokenCount(
      std::string_view modality,
      const NamedTensors& overrides) const {
    const Endpoint indexes{
        "generator",
        std::string(modality) + "_timestep_token_indexes",
    };
    if (const PipelineInput* input =
            FindDeclaredInput(manifest(), indexes)) {
      if (const Tensor* value = Override(overrides, *input)) {
        return value->element_count();
      }
    }
    const auto current = endpoint_values.find(indexes.qualified());
    if (current != endpoint_values.end()) {
      return current->second.element_count();
    }
    return TokenCount(modality);
  }

  [[nodiscard]] std::size_t PackedOffset(
      std::string_view modality) const {
    std::size_t offset = 0;
    for (const std::string_view current :
         {"text", "vision", "sound", "action"}) {
      if (current == modality) {
        return offset;
      }
      offset += TokenCount(current);
    }
    return offset;
  }

  [[nodiscard]] std::size_t PackedLength() const {
    return TokenCount("text") + TokenCount("vision") +
           TokenCount("sound") + TokenCount("action");
  }

  [[nodiscard]] Json SchedulerModeOverrides(
      std::string_view stage_name,
      const PipelineRunOptions& options) const {
    const Json stage_options =
        Json::parse(FindStage(manifest(), stage_name).options_json);
    if (!stage_options.contains("scheduler") ||
        !stage_options.at("scheduler").is_object()) {
      return Json::object();
    }
    const Json& scheduler = stage_options.at("scheduler");
    if (!scheduler.contains("mode_overrides") ||
        !scheduler.at("mode_overrides").is_object()) {
      return Json::object();
    }
    const Json& modes = scheduler.at("mode_overrides");
    std::string mode;
    const auto requested = options.strings.find("mode");
    if (requested != options.strings.end()) {
      mode = requested->second;
    } else if (options.strings.contains("action_domain") &&
               modes.contains("action")) {
      mode = "action";
    } else if (modes.contains("image_to_video")) {
      mode = "image_to_video";
    }
    if (mode.empty()) {
      return Json::object();
    }
    if (!modes.contains(mode) || !modes.at(mode).is_object()) {
      throw Error(
          ErrorCode::invalid_argument,
          "Unknown scheduler mode '" + mode + "'");
    }
    return modes.at(mode);
  }

  [[nodiscard]] std::size_t InferenceSteps(
      std::string_view stage_name,
      const PipelineRunOptions& options) const {
    const auto override = options.integers.find("num_inference_steps");
    if (override != options.integers.end()) {
      if (override->second <= 0) {
        throw Error(
            ErrorCode::invalid_argument,
            "num_inference_steps must be positive");
      }
      return static_cast<std::size_t>(override->second);
    }
    const Json mode = SchedulerModeOverrides(stage_name, options);
    if (mode.contains("num_inference_steps") &&
        mode.at("num_inference_steps").is_number_integer() &&
        mode.at("num_inference_steps").get<std::int64_t>() > 0) {
      return static_cast<std::size_t>(
          mode.at("num_inference_steps").get<std::int64_t>());
    }
    const PipelineStage& stage = FindStage(manifest(), stage_name);
    const Json stage_options = Json::parse(stage.options_json);
    if (stage_options.contains("default_steps") &&
        stage_options.at("default_steps").is_number_integer()) {
      const auto steps =
          stage_options.at("default_steps").get<std::int64_t>();
      if (steps > 0) {
        return static_cast<std::size_t>(steps);
      }
    }
    return 1;
  }

  [[nodiscard]] Json SchedulerConfig(std::string_view stage_name) const {
    const PipelineStage& stage = FindStage(manifest(), stage_name);
    const Json stage_options = Json::parse(stage.options_json);
    if (!stage_options.contains("scheduler") ||
        !stage_options.at("scheduler").is_object()) {
      ExecutionError(
          "Iterative stage '" + std::string(stage_name) +
          "' has no scheduler configuration");
    }
    const Json& scheduler = stage_options.at("scheduler");
    const std::string kind = scheduler.value("kind", std::string{});
    if (kind != "FlowMatchEulerDiscreteScheduler" &&
        kind != "UniPCMultistepScheduler") {
      ExecutionError(
          "Unsupported scheduler kind '" + kind + "'");
    }
    const std::string asset =
        scheduler.at("config_asset").get<std::string>();
    const std::filesystem::path path = package->root() / asset;
    std::ifstream stream(path);
    if (!stream) {
      ExecutionError(
          "Could not open scheduler asset '" + path.string() + "'");
    }
    std::ostringstream text;
    text << stream.rdbuf();
    try {
      return Json::parse(text.str());
    } catch (const Json::exception& error) {
      ExecutionError(
          "Could not parse scheduler asset '" + path.string() +
          "': " + error.what());
    }
  }

  [[nodiscard]] std::vector<float> SchedulerSigmas(
      std::string_view stage_name,
      const PipelineRunOptions& options) const {
    const Json config = SchedulerConfig(stage_name);
    const Json mode = SchedulerModeOverrides(stage_name, options);
    const std::size_t steps = InferenceSteps(stage_name, options);
    const double train_steps =
        config.value("num_train_timesteps", 1000.0);
    const std::string scheduler_kind =
        config.value("_class_name", std::string{});
    if (scheduler_kind == "UniPCMultistepScheduler") {
      if (!config.value("use_flow_sigmas", false)) {
        ExecutionError(
            "UniPC execution currently requires use_flow_sigmas=true");
      }
      const bool use_karras =
          options.integers.contains("use_karras_sigmas")
              ? options.integers.at("use_karras_sigmas") != 0
              : mode.value(
                    "use_karras_sigmas",
                    config.value("use_karras_sigmas", false));
      std::vector<double> sigmas(steps);
      if (use_karras) {
        if (!config.contains("sigma_min") ||
            config.at("sigma_min").is_null() ||
            !config.contains("sigma_max") ||
            config.at("sigma_max").is_null()) {
          ExecutionError(
              "UniPC Karras flow schedule requires sigma_min and sigma_max");
        }
        constexpr double rho = 7.0;
        const double minimum = config.at("sigma_min").get<double>();
        const double maximum = config.at("sigma_max").get<double>();
        const double min_root = std::pow(minimum, 1.0 / rho);
        const double max_root = std::pow(maximum, 1.0 / rho);
        for (std::size_t index = 0; index < steps; ++index) {
          const double ratio =
              steps == 1 ? 0.0
                         : static_cast<double>(index) /
                               static_cast<double>(steps - 1);
          const double raw = std::pow(
              max_root + ratio * (min_root - max_root), rho);
          sigmas[index] = raw / (raw + 1.0);
        }
      } else {
        const auto shift_override = options.numbers.find("flow_shift");
        const double shift =
            shift_override != options.numbers.end()
                ? shift_override->second
                : mode.value(
                      "flow_shift", config.value("flow_shift", 1.0));
        for (std::size_t index = 0; index < steps; ++index) {
          const double ratio =
              static_cast<double>(index) / static_cast<double>(steps);
          double sigma =
              1.0 + ratio * (1.0 / train_steps - 1.0);
          sigma =
              shift * sigma / (1.0 + (shift - 1.0) * sigma);
          sigmas[index] = sigma;
        }
      }
      if (!sigmas.empty() && std::abs(sigmas.front() - 1.0) < 1e-6) {
        sigmas.front() -= 1e-6;
      }
      if (config.contains("shift_terminal") &&
          !config.at("shift_terminal").is_null()) {
        const double terminal =
            config.at("shift_terminal").get<double>();
        const double scale =
            (1.0 - sigmas.back()) / (1.0 - terminal);
        for (double& sigma : sigmas) {
          sigma = 1.0 - (1.0 - sigma) / scale;
        }
      }
      const std::string final_type =
          config.value("final_sigmas_type", std::string("zero"));
      const float terminal =
          final_type == "zero"
              ? 0.0F
              : static_cast<float>(sigmas.back());
      if (final_type != "zero" && final_type != "sigma_min") {
        ExecutionError(
            "Unsupported UniPC final_sigmas_type '" + final_type + "'");
      }
      std::vector<float> result;
      result.reserve(steps + 1);
      for (const double sigma : sigmas) {
        result.push_back(static_cast<float>(sigma));
      }
      result.push_back(terminal);
      return result;
    }
    if (scheduler_kind != "FlowMatchEulerDiscreteScheduler") {
      ExecutionError(
          "Scheduler asset class does not match the declared stage scheduler");
    }
    if (config.contains("fixed_step_sampler_config") ||
        config.value("stochastic_sampling", false)) {
      ExecutionError(
          "Fixed-step stochastic FlowMatch schedulers are not supported");
    }
    const double sigma_min = 1.0 / train_steps;
    const double sigma_max = 1.0;
    std::vector<double> sigmas(steps);
    for (std::size_t index = 0; index < steps; ++index) {
      const double ratio =
          steps == 1 ? 0.0
                     : static_cast<double>(index) /
                           static_cast<double>(steps - 1);
      sigmas[index] =
          sigma_max + ratio * (sigma_min - sigma_max);
    }

    const bool dynamic = config.value("use_dynamic_shifting", false);
    if (dynamic) {
      const auto mu = options.numbers.find("mu");
      if (mu == options.numbers.end()) {
        throw Error(
            ErrorCode::invalid_argument,
            "Dynamic scheduler shifting requires numeric option 'mu'");
      }
      const std::string type =
          config.value("time_shift_type", std::string("exponential"));
      for (double& sigma : sigmas) {
        const double ratio = 1.0 / sigma - 1.0;
        if (type == "exponential") {
          const double numerator = std::exp(mu->second);
          sigma = numerator / (numerator + ratio);
        } else if (type == "linear") {
          sigma = mu->second / (mu->second + ratio);
        } else {
          ExecutionError(
              "Unsupported scheduler time_shift_type '" + type + "'");
        }
      }
    } else {
      const auto override = options.numbers.find("flow_shift");
      const double shift =
          override == options.numbers.end()
              ? mode.value(
                    "flow_shift", config.value("shift", 1.0))
              : override->second;
      for (double& sigma : sigmas) {
        sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma);
      }
    }

    if (config.contains("shift_terminal") &&
        !config.at("shift_terminal").is_null()) {
      const double terminal =
          config.at("shift_terminal").get<double>();
      const double one_minus_last = 1.0 - sigmas.back();
      const double scale = one_minus_last / (1.0 - terminal);
      for (double& sigma : sigmas) {
        sigma = 1.0 - (1.0 - sigma) / scale;
      }
    }

    const bool use_karras =
        options.integers.contains("use_karras_sigmas")
            ? options.integers.at("use_karras_sigmas") != 0
            : mode.value(
                  "use_karras_sigmas",
                  config.value("use_karras_sigmas", false));
    if (use_karras) {
      constexpr double rho = 7.0;
      const double min_root = std::pow(sigmas.back(), 1.0 / rho);
      const double max_root = std::pow(sigmas.front(), 1.0 / rho);
      for (std::size_t index = 0; index < steps; ++index) {
        const double ratio =
            steps == 1 ? 0.0
                       : static_cast<double>(index) /
                             static_cast<double>(steps - 1);
        sigmas[index] = std::pow(
            max_root + ratio * (min_root - max_root), rho);
      }
    } else if (config.value("use_exponential_sigmas", false)) {
      const double log_max = std::log(sigmas.front());
      const double log_min = std::log(sigmas.back());
      for (std::size_t index = 0; index < steps; ++index) {
        const double ratio =
            steps == 1 ? 0.0
                       : static_cast<double>(index) /
                             static_cast<double>(steps - 1);
        sigmas[index] =
            std::exp(log_max + ratio * (log_min - log_max));
      }
    } else if (config.value("use_beta_sigmas", false)) {
      ExecutionError("Beta sigma schedules are not supported");
    }

    const bool invert = config.value("invert_sigmas", false);
    std::vector<float> result;
    result.reserve(steps + 1);
    for (const double sigma : sigmas) {
      result.push_back(
          static_cast<float>(invert ? 1.0 - sigma : sigma));
    }
    result.push_back(invert ? 1.0F : 0.0F);
    return result;
  }

  [[nodiscard]] Tensor SelectSchedulerRows(
      const Tensor& state,
      const Tensor& velocity,
      const Tensor* indexes) const {
    if (state.shape() == velocity.shape() && indexes == nullptr) {
      return state;
    }
    if (state.shape().size() != 2 || velocity.shape().size() != 2 ||
        state.shape()[1] != velocity.shape()[1] ||
        indexes == nullptr ||
        indexes->data_type() != DataType::int64 ||
        indexes->element_count() !=
            static_cast<std::size_t>(velocity.shape()[0])) {
      ExecutionError("UniPC scheduler row selection is incompatible");
    }
    Tensor result(state.data_type(), velocity.shape());
    const auto rows = indexes->values<std::int64_t>();
    const std::size_t width =
        static_cast<std::size_t>(state.shape()[1]);
    for (std::size_t row = 0; row < rows.size(); ++row) {
      if (rows[row] < 0 || rows[row] >= state.shape()[0]) {
        ExecutionError("Scheduler token index is outside the state tensor");
      }
      for (std::size_t column = 0; column < width; ++column) {
        WriteFloat(
            result,
            row * width + column,
            ReadFloat(
                state,
                static_cast<std::size_t>(rows[row]) * width + column));
      }
    }
    return result;
  }

  [[nodiscard]] Tensor ScatterSchedulerRows(
      const Tensor& state,
      const Tensor& updated,
      const Tensor* indexes) const {
    if (state.shape() == updated.shape() && indexes == nullptr) {
      return updated;
    }
    Tensor result = state;
    const auto rows = indexes->values<std::int64_t>();
    const std::size_t width =
        static_cast<std::size_t>(state.shape()[1]);
    for (std::size_t row = 0; row < rows.size(); ++row) {
      for (std::size_t column = 0; column < width; ++column) {
        WriteFloat(
            result,
            static_cast<std::size_t>(rows[row]) * width + column,
            ReadFloat(updated, row * width + column));
      }
    }
    return result;
  }

  [[nodiscard]] Tensor UniPCStep(
      std::string_view state_name,
      std::string_view stage_name,
      const Tensor& state,
      const Tensor& velocity,
      const Tensor* indexes,
      const PipelineRunOptions& options) const {
    const Json config = SchedulerConfig(stage_name);
    if (!config.value("use_flow_sigmas", false) ||
        config.value("prediction_type", std::string{}) != "flow_prediction" ||
        !config.value("predict_x0", true) ||
        config.value("solver_type", std::string("bh2")) != "bh2" ||
        config.value("thresholding", false)) {
      ExecutionError(
          "This UniPC implementation requires flow_prediction, predict_x0, "
          "solver_type=bh2, and thresholding=false");
    }
    const std::int64_t configured_order =
        config.value("solver_order", 2);
    if (configured_order < 1 || configured_order > 2) {
      ExecutionError("UniPC solver_order must be 1 or 2");
    }
    const std::size_t steps = InferenceSteps(stage_name, options);
    const auto iteration_entry =
        stage_iterations.find(std::string(stage_name));
    const std::size_t index = std::min(
        iteration_entry == stage_iterations.end()
            ? std::size_t{0}
            : iteration_entry->second,
        steps - 1);
    const auto sigmas = SchedulerSigmas(stage_name, options);
    Tensor sample = SelectSchedulerRows(state, velocity, indexes);
    Tensor converted(sample.data_type(), sample.shape());
    for (std::size_t element = 0; element < sample.element_count(); ++element) {
      WriteFloat(
          converted,
          element,
          ReadFloat(sample, element) -
              sigmas[index] * ReadFloat(velocity, element));
    }

    SchedulerHistory& history =
        scheduler_histories[std::string(state_name)];
    if (index == 0) {
      history = SchedulerHistory{};
    }

    const Json disabled =
        config.value("disable_corrector", Json::array());
    const bool corrector_disabled =
        index == 0 ||
        std::ranges::any_of(disabled, [index](const Json& value) {
          return value.is_number_integer() &&
                 value.get<std::int64_t>() ==
                     static_cast<std::int64_t>(index - 1);
        });
    if (!corrector_disabled && history.last_sample.has_value() &&
        !history.model_outputs.empty()) {
      const float sigma_t = sigmas[index];
      const float sigma_s0 = sigmas[index - 1];
      const float alpha_t = 1.0F - sigma_t;
      const float alpha_s0 = 1.0F - sigma_s0;
      const float lambda_t =
          std::log(alpha_t) - std::log(sigma_t);
      const float lambda_s0 =
          std::log(alpha_s0) - std::log(sigma_s0);
      const float h = lambda_t - lambda_s0;
      const float phi = std::expm1(-h);
      const Tensor& previous_output = history.model_outputs.back();
      float previous_coefficient = 0.0F;
      float current_coefficient = 0.5F;
      float previous_ratio = 1.0F;
      if (history.previous_order == 2 &&
          history.model_outputs.size() >= 2) {
        const float sigma_previous = sigmas[index - 2];
        const float lambda_previous =
            std::log(1.0F - sigma_previous) -
            std::log(sigma_previous);
        previous_ratio =
            (lambda_previous - lambda_s0) / h;
        const float hh = -h;
        const float first_phi = phi / hh - 1.0F;
        const float second_phi =
            first_phi / hh - 0.5F;
        const float first_b = first_phi / phi;
        const float second_b = second_phi * 2.0F / phi;
        previous_coefficient =
            (second_b - first_b) / (previous_ratio - 1.0F);
        current_coefficient = first_b - previous_coefficient;
      }
      Tensor corrected(sample.data_type(), sample.shape());
      for (std::size_t element = 0;
           element < sample.element_count();
           ++element) {
        const float base =
            sigma_t / sigma_s0 *
                ReadFloat(*history.last_sample, element) -
            alpha_t * phi * ReadFloat(previous_output, element);
        const float difference =
            ReadFloat(converted, element) -
            ReadFloat(previous_output, element);
        float correction = current_coefficient * difference;
        if (previous_coefficient != 0.0F) {
          const float previous_difference =
              (ReadFloat(
                   history.model_outputs[
                       history.model_outputs.size() - 2],
                   element) -
               ReadFloat(previous_output, element)) /
              previous_ratio;
          correction += previous_coefficient * previous_difference;
        }
        WriteFloat(
            corrected,
            element,
            base - alpha_t * phi * correction);
      }
      sample = std::move(corrected);
    }

    history.model_outputs.push_back(converted);
    if (history.model_outputs.size() >
        static_cast<std::size_t>(configured_order)) {
      history.model_outputs.erase(history.model_outputs.begin());
    }
    const std::size_t remaining = steps - index;
    const std::size_t base_order =
        config.value("lower_order_final", true)
            ? std::min<std::size_t>(configured_order, remaining)
            : static_cast<std::size_t>(configured_order);
    const std::size_t order =
        std::min(base_order, history.lower_order + 1);

    const float sigma_t = sigmas[index + 1];
    const float sigma_s0 = sigmas[index];
    const float alpha_t = 1.0F - sigma_t;
    const float alpha_s0 = 1.0F - sigma_s0;
    const float lambda_t =
        sigma_t == 0.0F
            ? std::numeric_limits<float>::infinity()
            : std::log(alpha_t) - std::log(sigma_t);
    const float lambda_s0 =
        std::log(alpha_s0) - std::log(sigma_s0);
    const float h = lambda_t - lambda_s0;
    const float phi = std::expm1(-h);
    const Tensor& current_output = history.model_outputs.back();
    Tensor predicted(sample.data_type(), sample.shape());
    for (std::size_t element = 0;
         element < sample.element_count();
         ++element) {
      float value =
          sigma_t / sigma_s0 * ReadFloat(sample, element) -
          alpha_t * phi * ReadFloat(current_output, element);
      if (order == 2 && history.model_outputs.size() >= 2) {
        const float sigma_previous = sigmas[index - 1];
        const float lambda_previous =
            std::log(1.0F - sigma_previous) -
            std::log(sigma_previous);
        const float ratio =
            (lambda_previous - lambda_s0) / h;
        const float derivative =
            (ReadFloat(
                 history.model_outputs[
                     history.model_outputs.size() - 2],
                 element) -
             ReadFloat(current_output, element)) /
            ratio;
        value -= alpha_t * phi * 0.5F * derivative;
      }
      WriteFloat(predicted, element, value);
    }
    history.last_sample = sample;
    history.previous_order = order;
    history.lower_order = std::min(
        history.lower_order + 1,
        static_cast<std::size_t>(configured_order));
    return ScatterSchedulerRows(state, predicted, indexes);
  }

  [[nodiscard]] Tensor Generate(
      const PipelineInput& input,
      const NamedTensors& overrides,
      const PipelineRunOptions& options) const {
    if (const Tensor* value = Override(overrides, input)) {
      if (input.generator_kind == "multimodal_position_ids") {
        if (value->data_type() != DataType::int64 ||
            (value->shape().size() != 2 &&
             value->shape().size() != 3)) {
          ExecutionError(
              "Overridden multimodal position IDs must be an int64 rank-2 "
              "or rank-3 tensor");
        }
        const std::size_t axes =
            static_cast<std::size_t>(value->shape()[0]);
        const std::size_t batch =
            value->shape().size() == 3
                ? static_cast<std::size_t>(value->shape()[1])
                : 1;
        const std::size_t length =
            static_cast<std::size_t>(value->shape().back());
        const std::size_t axis_stride = batch * length;
        const auto positions = value->values<std::int64_t>();
        std::vector<std::int64_t> cursors(batch, 0);
        for (std::size_t batch_index = 0;
             batch_index < batch;
             ++batch_index) {
          std::int64_t maximum = -1;
          for (std::size_t axis = 0; axis < axes; ++axis) {
            for (std::size_t index = 0; index < length; ++index) {
              maximum = std::max(
                  maximum,
                  positions[axis * axis_stride +
                            batch_index * length + index]);
            }
          }
          cursors[batch_index] = maximum + 1;
        }
        position_cursors.insert_or_assign(
            input.port.qualified(), std::move(cursors));
      }
      return *value;
    }
    const Json parameters = Json::parse(input.generator_json);
    if (input.generator_kind == "zeros" ||
        input.generator_kind == "empty_tensor") {
      const TensorSpec& spec = manifest().Input(input.port);
      return Tensor::Zeros(
          spec.data_type, GeneratedShape(input, parameters));
    }
    if (input.generator_kind == "causal_attention_mask") {
      const Endpoint source = Endpoint::Parse(
          parameters.at("sequence_input").get<std::string>());
      const Tensor* sequence = KnownValue(source);
      if (sequence == nullptr || sequence->shape().size() < 2) {
        ExecutionError(
            "Causal attention mask sequence input is unavailable");
      }
      const std::int64_t batch = sequence->shape()[0];
      const std::int64_t current_length = sequence->shape()[1];
      const std::size_t past_length =
          parameters.contains("past_state")
              ? StateSequenceLength(parameters.at("past_state"))
              : 0;
      const double visible =
          parameters.value("visible_value", 1.0);
      return FilledTensor(
          manifest().Input(input.port).data_type,
          {
              batch,
              static_cast<std::int64_t>(past_length) + current_length,
          },
          visible);
    }
    if (input.generator_kind == "multimodal_position_ids") {
      const Endpoint source =
          Endpoint::Parse(parameters.at("source").get<std::string>());
      const Tensor* sequence = KnownValue(source);
      if (sequence == nullptr) {
        ExecutionError("Position-id source is unavailable");
      }
      const std::int64_t axes =
          parameters.at("axes").get<std::int64_t>();
      const std::size_t past_length =
          parameters.contains("past_state")
              ? StateSequenceLength(parameters.at("past_state"))
              : 0;
      std::vector<std::int64_t> shape;
      std::size_t length;
      if (sequence->shape().size() == 2) {
        length = static_cast<std::size_t>(sequence->shape()[1]);
        shape = {
            axes,
            sequence->shape()[0],
            static_cast<std::int64_t>(length),
        };
      } else {
        length = std::max(PackedLength(), sequence->element_count());
        shape = {axes, static_cast<std::int64_t>(length)};
      }
      std::vector<std::int64_t> values(
          CheckedElementCount(shape));
      const std::size_t batch =
          shape.size() == 3 ? static_cast<std::size_t>(shape[1]) : 1;
      const std::size_t axis_stride = batch * length;
      for (std::int64_t axis = 0; axis < axes; ++axis) {
        for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
          for (std::size_t index = 0; index < length; ++index) {
            values[static_cast<std::size_t>(axis) * axis_stride +
                   batch_index * length + index] =
                static_cast<std::int64_t>(past_length + index);
          }
        }
      }
      if (sequence->shape().size() == 2 && axes >= 3 &&
          sequence->data_type() == DataType::int64) {
        auto& cursors = position_cursors[input.port.qualified()];
        if (cursors.size() != batch) {
          cursors.assign(
              batch, static_cast<std::int64_t>(past_length));
        }
        const auto ids = sequence->values<std::int64_t>();
        const auto image_features =
            endpoint_values.find("reasoner_vision_encoder.image_features");
        for (std::size_t batch_index = 0;
             batch_index < batch;
             ++batch_index) {
          const std::size_t batch_offset = batch_index * length;
          if (past_length != 0) {
            const std::int64_t start = cursors[batch_index];
            for (std::int64_t axis = 0; axis < axes; ++axis) {
              for (std::size_t index = 0; index < length; ++index) {
                values[static_cast<std::size_t>(axis) * axis_stride +
                       batch_offset + index] =
                    start + static_cast<std::int64_t>(index);
              }
            }
            cursors[batch_index] += static_cast<std::int64_t>(length);
            continue;
          }

          const Json metadata = Json::parse(manifest().metadata_json());
          const auto grid_value =
              endpoint_values.find("reasoner_vision_encoder.grid_thw");
          if (metadata.contains("vision_understanding") &&
              metadata.at("vision_understanding").contains("tokens") &&
              metadata.at("vision_understanding").contains("preprocessing") &&
              metadata.at("vision_understanding")
                  .at("preprocessing")
                  .contains("patchify") &&
              grid_value != endpoint_values.end() &&
              grid_value->second.data_type() == DataType::int64 &&
              grid_value->second.element_count() == 3) {
            const Json& understanding =
                metadata.at("vision_understanding");
            const Json& tokens = understanding.at("tokens");
            const std::int64_t image_token =
                tokens.value("image", std::int64_t{-1});
            const std::int64_t video_token =
                tokens.value("video", std::int64_t{-1});
            const std::int64_t merge =
                understanding.at("preprocessing")
                    .at("patchify")
                    .value("merge_size", 1);
            const auto raw_grid =
                grid_value->second.values<std::int64_t>();
            if (merge <= 0 || raw_grid[0] <= 0 || raw_grid[1] <= 0 ||
                raw_grid[2] <= 0 || raw_grid[1] % merge != 0 ||
                raw_grid[2] % merge != 0) {
              ExecutionError("Vision grid_thw is incompatible with merge size");
            }
            const std::size_t grid_time =
                static_cast<std::size_t>(raw_grid[0]);
            const std::size_t grid_height =
                static_cast<std::size_t>(raw_grid[1] / merge);
            const std::size_t grid_width =
                static_cast<std::size_t>(raw_grid[2] / merge);
            const std::size_t spatial_tokens = grid_height * grid_width;
            std::size_t position = 0;
            std::int64_t current_position = 0;
            std::size_t video_spans = 0;
            while (position < length) {
              const std::int64_t token_id =
                  ids[batch_offset + position];
              const bool is_image = token_id == image_token;
              const bool is_video = token_id == video_token;
              if (!is_image && !is_video) {
                const std::size_t start = position;
                while (position < length) {
                  const std::int64_t next_id =
                      ids[batch_offset + position];
                  if (next_id == image_token || next_id == video_token) {
                    break;
                  }
                  ++position;
                }
                for (std::size_t index = start; index < position; ++index) {
                  for (std::int64_t axis = 0; axis < axes; ++axis) {
                    values[static_cast<std::size_t>(axis) * axis_stride +
                           batch_offset + index] =
                        current_position +
                        static_cast<std::int64_t>(index - start);
                  }
                }
                current_position +=
                    static_cast<std::int64_t>(position - start);
                continue;
              }

              const std::size_t start = position;
              while (position < length &&
                     ids[batch_offset + position] == token_id) {
                ++position;
              }
              const std::size_t count = position - start;
              if (spatial_tokens == 0 || count % spatial_tokens != 0) {
                ExecutionError(
                    "Visual token span does not match vision grid_thw");
              }
              const std::size_t span_time = count / spatial_tokens;
              if (is_video && span_time != 1) {
                ExecutionError(
                    "Video position indexing requires one token span per frame");
              }
              if (is_video) {
                ++video_spans;
              }
              for (std::size_t index = 0; index < count; ++index) {
                const std::size_t time = index / spatial_tokens;
                const std::size_t spatial = index % spatial_tokens;
                values[batch_offset + start + index] =
                    current_position + static_cast<std::int64_t>(time);
                values[axis_stride + batch_offset + start + index] =
                    current_position +
                    static_cast<std::int64_t>(spatial / grid_width);
                values[2 * axis_stride + batch_offset + start + index] =
                    current_position +
                    static_cast<std::int64_t>(spatial % grid_width);
              }
              current_position += static_cast<std::int64_t>(
                  std::max({span_time, grid_height, grid_width}));
            }
            if (video_spans != 0 && video_spans != grid_time) {
              ExecutionError(
                  "Video token span count does not match vision grid time");
            }
            cursors[batch_index] = current_position;
            continue;
          }

          std::size_t visual_count = 0;
          if (image_features != endpoint_values.end() &&
              !image_features->second.shape().empty()) {
            visual_count = static_cast<std::size_t>(
                image_features->second.shape()[0]);
          }
          std::size_t visual_start = length;
          if (visual_count > 0 && visual_count <= length) {
            for (std::size_t candidate = 0;
                 candidate + visual_count <= length;
                 ++candidate) {
              const std::int64_t token = ids[batch_offset + candidate];
              const bool repeated = std::ranges::all_of(
                  ids.begin() +
                      static_cast<std::ptrdiff_t>(batch_offset + candidate),
                  ids.begin() + static_cast<std::ptrdiff_t>(
                                    batch_offset + candidate + visual_count),
                  [token](std::int64_t value) { return value == token; });
              if (repeated) {
                visual_start = candidate;
                break;
              }
            }
          }
          std::size_t grid_time = 1;
          std::size_t grid_height = static_cast<std::size_t>(
              std::sqrt(static_cast<double>(visual_count)));
          std::size_t grid_width = grid_height;
          const auto fallback_grid_value =
              endpoint_values.find("reasoner_vision_encoder.grid_thw");
          if (fallback_grid_value != endpoint_values.end() &&
              fallback_grid_value->second.data_type() == DataType::int64 &&
              fallback_grid_value->second.element_count() == 3) {
            const auto raw_grid =
                fallback_grid_value->second.values<std::int64_t>();
            const std::int64_t merge =
                metadata.contains("vision_understanding")
                    ? metadata.at("vision_understanding")
                          .at("preprocessing")
                          .at("patchify")
                          .value("merge_size", 1)
                    : 1;
            if (merge <= 0 || raw_grid[0] <= 0 || raw_grid[1] <= 0 ||
                raw_grid[2] <= 0 || raw_grid[1] % merge != 0 ||
                raw_grid[2] % merge != 0) {
              ExecutionError("Vision grid_thw is incompatible with merge size");
            }
            grid_time = static_cast<std::size_t>(raw_grid[0]);
            grid_height =
                static_cast<std::size_t>(raw_grid[1] / merge);
            grid_width =
                static_cast<std::size_t>(raw_grid[2] / merge);
          }
          if (visual_start == length ||
              grid_time * grid_height * grid_width != visual_count) {
            cursors[batch_index] = static_cast<std::int64_t>(length);
            continue;
          }

          const std::int64_t visual_base =
              static_cast<std::int64_t>(visual_start);
          for (std::size_t token = 0; token < visual_count; ++token) {
            const std::size_t position = visual_start + token;
            const std::size_t time =
                token / (grid_height * grid_width);
            const std::size_t spatial =
                token % (grid_height * grid_width);
            values[batch_offset + position] =
                visual_base + static_cast<std::int64_t>(time);
            values[axis_stride + batch_offset + position] =
                visual_base +
                static_cast<std::int64_t>(spatial / grid_width);
            values[2 * axis_stride + batch_offset + position] =
                visual_base +
                static_cast<std::int64_t>(spatial % grid_width);
          }
          const std::size_t visual_end = visual_start + visual_count;
          std::int64_t next =
              visual_base +
              static_cast<std::int64_t>(
                  std::max(grid_height, grid_width));
          for (std::size_t position = visual_end;
               position < length;
               ++position, ++next) {
            for (std::int64_t axis = 0; axis < axes; ++axis) {
              values[static_cast<std::size_t>(axis) * axis_stride +
                     batch_offset + position] = next;
            }
          }
          cursors[batch_index] = next;
        }
      }
      if (sequence->shape().size() != 2 && axes >= 3 &&
          TokenCount("vision") > 0 &&
          options.integers.contains("video_latent_frames") &&
          options.integers.contains("video_latent_height") &&
          options.integers.contains("video_latent_width")) {
        const Json metadata = Json::parse(manifest().metadata_json());
        const std::int64_t patch =
            metadata.contains("packing")
                ? metadata.at("packing").value("latent_patch_size", 1)
                : 1;
        if (patch <= 0) {
          ExecutionError("latent_patch_size must be positive");
        }
        const std::int64_t batch_count =
            options.integers.contains("video_batch")
                ? options.integers.at("video_batch")
                : 1;
        const std::int64_t frames =
            options.integers.at("video_latent_frames");
        const std::int64_t height =
            options.integers.at("video_latent_height") / patch;
        const std::int64_t width =
            options.integers.at("video_latent_width") / patch;
        const std::size_t vision_count = TokenCount("vision");
        if (batch_count > 0 && frames > 0 && height > 0 && width > 0 &&
            vision_count ==
                static_cast<std::size_t>(
                    batch_count * frames * height * width)) {
          const std::size_t offset = PackedOffset("vision");
          const std::int64_t temporal_margin =
              parameters.value("temporal_margin", 0);
          const bool reset_spatial =
              parameters.value("reset_spatial", false);
          for (std::size_t token = 0; token < vision_count; ++token) {
            const std::int64_t spatial_index =
                static_cast<std::int64_t>(token) % (height * width);
            const std::int64_t temporal_index =
                (static_cast<std::int64_t>(token) / (height * width)) %
                frames;
            values[offset + token] =
                static_cast<std::int64_t>(TokenCount("text")) +
                temporal_margin + temporal_index;
            values[axis_stride + offset + token] =
                (reset_spatial ? 0
                               : static_cast<std::int64_t>(
                                     TokenCount("text"))) +
                spatial_index / width;
            values[2 * axis_stride + offset + token] =
                (reset_spatial ? 0
                               : static_cast<std::int64_t>(
                                     TokenCount("text"))) +
                spatial_index % width;
          }
        }
      }
      return Int64Tensor(std::move(shape), values);
    }
    if (input.generator_kind == "packed_sequence_layout") {
      const std::string modality =
          parameters.at("modality").get<std::string>();
      const std::string index_kind =
          parameters.value("index_kind", input.port.port);
      if (index_kind == "und_len") {
        const std::vector<std::int64_t> values{
            static_cast<std::int64_t>(TokenCount("text"))};
        return Int64Tensor({1}, values);
      }
      const bool noisy_indexes =
          index_kind.find("timestep_token_indexes") != std::string::npos ||
          index_kind.find("mse_loss_indexes") != std::string::npos;
      const std::size_t count =
          noisy_indexes ? NoisyTokenCount(modality, overrides)
                        : TokenCount(modality);
      if (index_kind.find("mse_loss_indexes") != std::string::npos) {
        const Endpoint timestep_indexes{
            "generator",
            modality + "_timestep_token_indexes",
        };
        const Tensor* index_tensor = nullptr;
        if (const PipelineInput* declaration =
                FindDeclaredInput(manifest(), timestep_indexes)) {
          index_tensor = Override(overrides, *declaration);
        }
        if (index_tensor == nullptr) {
          const auto current =
              endpoint_values.find(timestep_indexes.qualified());
          if (current != endpoint_values.end()) {
            index_tensor = &current->second;
          }
        }
        if (index_tensor == nullptr ||
            index_tensor->data_type() != DataType::int64) {
          ExecutionError(
              "MSE loss indexes require int64 timestep-token indexes");
        }
        const auto token_indexes =
            index_tensor->values<std::int64_t>();
        std::vector<std::int64_t> values(token_indexes.size());
        const std::size_t offset = PackedOffset(modality);
        const std::size_t token_count = TokenCount(modality);
        for (std::size_t index = 0; index < token_indexes.size(); ++index) {
          if (token_indexes[index] < 0 ||
              static_cast<std::size_t>(token_indexes[index]) >= token_count) {
            ExecutionError(
                "Timestep-token index is outside the modality token range");
          }
          values[index] =
              static_cast<std::int64_t>(offset) + token_indexes[index];
        }
        return Int64Tensor(
            {static_cast<std::int64_t>(values.size())}, values);
      }
      const bool joint_indexes =
          index_kind.find("sequence_indexes") != std::string::npos ||
          index_kind.find("mse_loss_indexes") != std::string::npos ||
          index_kind == "text_indexes";
      const std::size_t offset =
          joint_indexes ? PackedOffset(modality) : 0;
      std::vector<std::int64_t> values(count);
      for (std::size_t index = 0; index < count; ++index) {
        values[index] =
            static_cast<std::int64_t>(offset + index);
      }
      return Int64Tensor(
          {static_cast<std::int64_t>(count)}, values);
    }
    if (input.generator_kind == "scheduler_timesteps") {
      const std::string stage =
          parameters.at("stage").get<std::string>();
      const std::string modality =
          parameters.value("modality", "vision");
      const std::size_t count =
          NoisyTokenCount(modality, overrides);
      const std::size_t steps = InferenceSteps(stage, options);
      const auto found_iteration = stage_iterations.find(stage);
      const std::size_t iteration = std::min(
          found_iteration == stage_iterations.end()
              ? std::size_t{0}
              : found_iteration->second,
          steps - 1);
      const Json scheduler = SchedulerConfig(stage);
      const float train_steps = scheduler.value(
          "num_train_timesteps", 1000.0F);
      const auto sigmas = SchedulerSigmas(stage, options);
      const float timestep =
          sigmas[iteration] * train_steps;
      const std::vector<float> values(count, timestep);
      return FloatTensor(
          {static_cast<std::int64_t>(count)}, values);
    }
    if (input.generator_kind == "action_domain_ids") {
      const std::string option_name =
          parameters.at("domain_input").get<std::string>();
      const auto provided = options.strings.find(option_name);
      const std::string domain =
          provided != options.strings.end()
              ? provided->second
              : parameters.value("default", std::string{});
      const Json& domain_map = parameters.at("domain_map");
      if (!domain_map.contains(domain) ||
          !domain_map.at(domain).is_number_integer()) {
        ExecutionError("Unknown action domain '" + domain + "'");
      }
      const std::size_t count =
          input.port.port.find("pred") != std::string::npos
              ? NoisyTokenCount("action", overrides)
              : TokenCount("action");
      const std::vector<std::int64_t> values(
          count, domain_map.at(domain).get<std::int64_t>());
      return Int64Tensor(
          {static_cast<std::int64_t>(count)}, values);
    }
    ExecutionError(
        "Generated input program '" + input.generator_kind +
        "' requires an override in this execution mode");
  }

  [[nodiscard]] Tensor DefaultValue(const PipelineInput& input) const {
    const TensorSpec& spec = manifest().Input(input.port);
    const Json value = Json::parse(input.value_json);
    std::vector<std::int64_t> shape = spec.shape;
    if (std::ranges::any_of(shape, [](std::int64_t dimension) {
          return dimension < 0;
        })) {
      ExecutionError(
          "Defaulted input '" + input.port.qualified() +
          "' has a dynamic shape");
    }
    if (!value.is_number() && !value.is_boolean()) {
      ExecutionError(
          "Defaulted tensor values currently require a scalar JSON value");
    }
    return FilledTensor(
        spec.data_type,
        std::move(shape),
        value.is_boolean() ? (value.get<bool>() ? 1.0 : 0.0)
                           : value.get<double>());
  }

  [[nodiscard]] std::int64_t RequiredIntegerOption(
      const PipelineRunOptions& options,
      std::string_view name) const {
    const auto found = options.integers.find(std::string(name));
    if (found == options.integers.end() || found->second <= 0) {
      throw Error(
          ErrorCode::invalid_argument,
          "Pipeline stage requires positive integer option '" +
              std::string(name) + "'");
    }
    return found->second;
  }

  [[nodiscard]] Tensor UnpatchifyVideo(
      const Tensor& packed,
      const PipelineConnection& connection,
      const PipelineRunOptions& options) const {
    if (packed.shape().size() != 2) {
      ExecutionError("Packed video latent must have rank 2");
    }
    const Json parameters = Json::parse(connection.parameters_json);
    const std::int64_t patch =
        parameters.at("spatial_patch_size").get<std::int64_t>();
    const std::int64_t channels =
        parameters.at("latent_channels").get<std::int64_t>();
    const std::int64_t batch =
        options.integers.contains("video_batch")
            ? options.integers.at("video_batch")
            : 1;
    const std::int64_t frames =
        RequiredIntegerOption(options, "video_latent_frames");
    const std::int64_t height =
        RequiredIntegerOption(options, "video_latent_height");
    const std::int64_t width =
        RequiredIntegerOption(options, "video_latent_width");
    if (patch <= 0 || channels <= 0 || batch <= 0) {
      throw Error(
          ErrorCode::invalid_argument,
          "Video patch size, channels, and batch must be positive");
    }
    if (height % patch != 0 || width % patch != 0) {
      ExecutionError(
          "Video latent height and width must be divisible by patch size");
    }
    const std::int64_t grid_height = height / patch;
    const std::int64_t grid_width = width / patch;
    const std::int64_t expected_tokens =
        batch * frames * grid_height * grid_width;
    if (packed.shape()[0] != expected_tokens ||
        packed.shape()[1] != patch * patch * channels) {
      ExecutionError(
          "Packed video latent shape does not match requested BCTHW shape");
    }

    const DataType target_type =
        manifest().Input(connection.target).data_type;
    Tensor result(
        target_type, {batch, channels, frames, height, width});
    for (std::int64_t batch_index = 0; batch_index < batch; ++batch_index) {
      for (std::int64_t frame = 0; frame < frames; ++frame) {
        for (std::int64_t grid_y = 0; grid_y < grid_height; ++grid_y) {
          for (std::int64_t grid_x = 0; grid_x < grid_width; ++grid_x) {
            const std::int64_t token =
                ((batch_index * frames + frame) * grid_height + grid_y) *
                    grid_width +
                grid_x;
            for (std::int64_t patch_x = 0; patch_x < patch; ++patch_x) {
              for (std::int64_t patch_y = 0; patch_y < patch; ++patch_y) {
                for (std::int64_t channel = 0; channel < channels; ++channel) {
                  const std::int64_t packed_channel =
                      (patch_y * patch + patch_x) * channels + channel;
                  const std::size_t source_index =
                      static_cast<std::size_t>(
                          token * packed.shape()[1] + packed_channel);
                  const std::size_t target_index =
                      static_cast<std::size_t>(
                          ((((batch_index * channels + channel) * frames +
                              frame) *
                                 height +
                             grid_y * patch + patch_y) *
                                width +
                            grid_x * patch + patch_x));
                  WriteFloat(
                      result, target_index, ReadFloat(packed, source_index));
                }
              }
            }
          }
        }
      }
    }
    return result;
  }

  [[nodiscard]] Tensor ReshapeAudio(
      const Tensor& packed,
      const PipelineConnection& connection,
      const PipelineRunOptions& options) const {
    if (packed.shape().size() != 2) {
      ExecutionError("Packed audio latent must have rank 2");
    }
    const std::int64_t batch =
        options.integers.contains("audio_batch")
            ? options.integers.at("audio_batch")
            : 1;
    if (batch <= 0) {
      throw Error(
          ErrorCode::invalid_argument,
          "audio_batch must be positive");
    }
    if (packed.shape()[0] % batch != 0) {
      ExecutionError("Packed audio length is not divisible by audio_batch");
    }
    const std::int64_t frames = packed.shape()[0] / batch;
    const std::int64_t channels = packed.shape()[1];
    const DataType target_type =
        manifest().Input(connection.target).data_type;
    Tensor result(target_type, {batch, channels, frames});
    for (std::int64_t batch_index = 0; batch_index < batch; ++batch_index) {
      for (std::int64_t frame = 0; frame < frames; ++frame) {
        for (std::int64_t channel = 0; channel < channels; ++channel) {
          const std::size_t source_index =
              static_cast<std::size_t>(
                  (batch_index * frames + frame) * channels + channel);
          const std::size_t target_index =
              static_cast<std::size_t>(
                  (batch_index * channels + channel) * frames + frame);
          WriteFloat(result, target_index, ReadFloat(packed, source_index));
        }
      }
    }
    return result;
  }

  [[nodiscard]] Tensor ApplyConnection(
      const PipelineConnection& connection,
      const Tensor& source,
      const NamedTensors& overrides,
      const PipelineRunOptions& options) const {
    if (const Tensor* value = Override(overrides, connection.target)) {
      return *value;
    }
    if (!connection.transform.has_value()) {
      return source;
    }
    if (*connection.transform == "cast") {
      return CastTensor(
          source, manifest().Input(connection.target).data_type);
    }
    if (*connection.transform == "reshape") {
      const Json parameters = Json::parse(connection.parameters_json);
      std::vector<std::int64_t> shape;
      if (parameters.contains("shape")) {
        for (const auto& dimension : parameters.at("shape")) {
          shape.push_back(dimension.get<std::int64_t>());
        }
      } else {
        const TensorSpec& target = manifest().Input(connection.target);
        shape = target.shape;
      }
      const auto dynamic = std::ranges::find(shape, std::int64_t{-1});
      if (dynamic != shape.end()) {
        if (std::ranges::find(
                std::next(dynamic), shape.end(), std::int64_t{-1}) !=
            shape.end()) {
          ExecutionError("Reshape has more than one inferred dimension");
        }
        *dynamic = 1;
        const std::size_t known = CheckedElementCount(shape);
        if (known == 0 || source.element_count() % known != 0) {
          ExecutionError("Reshape cannot infer its dynamic dimension");
        }
        *dynamic = static_cast<std::int64_t>(
            source.element_count() / known);
      }
      if (CheckedElementCount(shape) != source.element_count()) {
        ExecutionError("Reshape transform changes tensor element count");
      }
      return Tensor::FromBytes(
          source.data_type(), std::move(shape), source.bytes());
    }
    if (*connection.transform == "scheduler_step") {
      const Json parameters = Json::parse(connection.parameters_json);
      const std::string state_name =
          parameters.at("state").get<std::string>();
      Tensor current;
      const auto state = state_values.find(state_name);
      if (state != state_values.end()) {
        current = state->second;
      } else {
        current = ResolveInput(connection.target, overrides, options);
      }
      const std::string stage =
          parameters.at("stage").get<std::string>();
      const std::size_t steps = InferenceSteps(stage, options);
      const Tensor* indexes = nullptr;
      if (parameters.contains("timestep_input")) {
        std::string index_endpoint =
            parameters.at("timestep_input").get<std::string>();
        const std::string suffix = "_timesteps";
        if (index_endpoint.ends_with(suffix)) {
          index_endpoint.replace(
              index_endpoint.size() - suffix.size(),
              suffix.size(),
              "_timestep_token_indexes");
          const auto found = endpoint_values.find(index_endpoint);
          if (found != endpoint_values.end()) {
            indexes = &found->second;
          }
        }
      }
      const Json scheduler = SchedulerConfig(stage);
      if (scheduler.value("_class_name", std::string{}) ==
          "UniPCMultistepScheduler") {
        return UniPCStep(
            state_name,
            stage,
            current,
            source,
            indexes,
            options);
      }
      const auto sigmas = SchedulerSigmas(stage, options);
      const auto iteration = stage_iterations.find(stage);
      const std::size_t index = std::min(
          iteration == stage_iterations.end()
              ? std::size_t{0}
              : iteration->second,
          steps - 1);
      return EulerStepIndexed(
          current,
          source,
          indexes,
          sigmas[index + 1] - sigmas[index]);
    }
    if (*connection.transform == "video_diffusion_finalize") {
      const Json parameters = Json::parse(connection.parameters_json);
      const std::string state_name =
          parameters.at("state").get<std::string>();
      const auto state = state_values.find(state_name);
      if (state == state_values.end()) {
        ExecutionError(
            "Video finalization requires state '" + state_name + "'");
      }
      return UnpatchifyVideo(
          state->second, connection, options);
    }
    if (*connection.transform == "audio_diffusion_finalize") {
      const Json parameters = Json::parse(connection.parameters_json);
      const std::string state_name =
          parameters.at("state").get<std::string>();
      const auto state = state_values.find(state_name);
      if (state == state_values.end()) {
        ExecutionError(
            "Audio finalization requires state '" + state_name + "'");
      }
      return ReshapeAudio(state->second, connection, options);
    }
    ExecutionError(
        "Transform '" + *connection.transform +
        "' requires an override for target '" +
        connection.target.qualified() + "'");
  }

  [[nodiscard]] Tensor ResolveInput(
      const Endpoint& endpoint,
      const NamedTensors& overrides,
      const PipelineRunOptions& options) const {
    if (const Tensor* value = Override(overrides, endpoint)) {
      return *value;
    }
    if (const PipelineState* state =
            FindInputState(manifest(), endpoint)) {
      const auto current = state_values.find(state->name);
      if (current != state_values.end()) {
        return current->second;
      }
    }
    if (const PipelineConnection* connection =
            FindConnection(manifest(), endpoint, false)) {
      const auto source =
          endpoint_values.find(connection->source.qualified());
      if (source == endpoint_values.end()) {
        ExecutionError(
            "Input '" + endpoint.qualified() + "' depends on unavailable output '" +
            connection->source.qualified() + "'");
      }
      return ApplyConnection(
          *connection, source->second, overrides, options);
    }

    const PipelineInput* declaration =
        FindDeclaredInput(manifest(), endpoint);
    if (declaration == nullptr) {
      ExecutionError(
          "Input '" + endpoint.qualified() + "' has no runtime source");
    }
    switch (declaration->kind) {
      case PipelineInputKind::external: {
        const auto found = external_values.find(endpoint.qualified());
        if (found == external_values.end()) {
          const auto modality = options.strings.find("vision_modality");
          if (modality != options.strings.end()) {
            const Json metadata = Json::parse(manifest().metadata_json());
            if (metadata.contains("vision_understanding")) {
              const Json& understanding =
                  metadata.at("vision_understanding");
              const Json& routing = understanding.at("routing");
              const auto target = routing.find(modality->second);
              if (target != routing.end() && target->is_string() &&
                  target->get<std::string>() == endpoint.qualified()) {
                const std::string encoder =
                    understanding.at("encoder").get<std::string>();
                const auto features =
                    endpoint_values.find(encoder + ".image_features");
                if (features == endpoint_values.end()) {
                  ExecutionError(
                      "Vision feature route is unavailable for modality '" +
                      modality->second + "'");
                }
                return features->second;
              }
            }
          }
          if (!declaration->required) {
            const TensorSpec& spec = manifest().Input(endpoint);
            std::vector<std::int64_t> shape = spec.shape;
            for (auto& dimension : shape) {
              if (dimension < 0) {
                dimension = 0;
              }
            }
            return Tensor::Zeros(spec.data_type, std::move(shape));
          }
          ExecutionError(
              "Required pipeline input '" + declaration->name +
              "' was not provided");
        }
        return found->second;
      }
      case PipelineInputKind::generated:
        return Generate(*declaration, overrides, options);
      case PipelineInputKind::stateful: {
        if (const PipelineState* state =
                FindInputState(manifest(), endpoint)) {
          const auto found = state_values.find(state->name);
          if (found != state_values.end()) {
            return found->second;
          }
        }
        ExecutionError(
            "Stateful input '" + endpoint.qualified() +
            "' has not been initialized");
      }
      case PipelineInputKind::defaulted:
        return DefaultValue(*declaration);
    }
    ExecutionError("Unknown pipeline input source");
  }

  [[nodiscard]] NamedTensors CollectOutputs() const {
    NamedTensors result;
    for (const auto& output : manifest().outputs()) {
      if (output.port.has_value()) {
        const auto found =
            endpoint_values.find(output.port->qualified());
        if (found != endpoint_values.end()) {
          result.emplace(output.name, found->second);
        }
      } else if (output.state.has_value()) {
        const auto found = state_values.find(*output.state);
        if (found != state_values.end()) {
          result.emplace(output.name, found->second);
        }
      }
    }
    return result;
  }

  [[nodiscard]] NamedTensors StepStage(
      std::string_view stage_name,
      const NamedTensors& inputs,
      const NamedTensors& overrides,
      const PipelineRunOptions& options) {
    (void)options;
    const PipelineStage& stage = FindStage(manifest(), stage_name);
    StoreExternalInputs(inputs);

    for (const auto& component_name :
         TopologicalComponents(manifest(), stage)) {
      const PipelineComponent& component =
          manifest().Component(component_name);
      if (!ComponentPresent(component)) {
        continue;
      }
      const bool reuse_prefill_embedding =
          stage.kind == "autoregressive" &&
          stage_iterations[stage.name] == 0 &&
          inputs.empty() &&
          component.role == "embedding" &&
          std::ranges::all_of(
              component.metadata.outputs,
              [this, &component](const TensorSpec& output) {
                return endpoint_values.contains(
                    component.name + "." + output.name);
              });
      if (reuse_prefill_embedding) {
        continue;
      }
      NamedTensors model_inputs;
      model_inputs.reserve(component.metadata.inputs.size());
      for (const auto& spec : component.metadata.inputs) {
        const Endpoint endpoint{component.name, spec.name};
        Tensor value = ResolveInput(endpoint, overrides, options);
        endpoint_values.insert_or_assign(endpoint.qualified(), value);
        model_inputs.emplace(spec.name, std::move(value));
      }
      NamedTensors model_outputs =
          package->Component(component.name).Run(model_inputs);
      for (auto& [name, tensor] : model_outputs) {
        endpoint_values.insert_or_assign(
            component.name + "." + name, std::move(tensor));
      }
    }

    for (const auto& state : manifest().states()) {
      if (!Contains(stage.components, state.output.component)) {
        continue;
      }
      const auto produced =
          endpoint_values.find(state.output.qualified());
      if (produced == endpoint_values.end()) {
        continue;
      }
      const PipelineConnection& connection =
          FindStateConnection(manifest(), state);
      state_values.insert_or_assign(
          state.name,
          ApplyConnection(
              connection, produced->second, overrides, options));
    }
    ++stage_iterations[stage.name];
    return CollectOutputs();
  }

  [[nodiscard]] const Tensor& StageLogits(
      const PipelineStage& stage) const {
    for (auto component = stage.components.rbegin();
         component != stage.components.rend();
         ++component) {
      const auto found =
          endpoint_values.find(*component + ".logits");
      if (found != endpoint_values.end()) {
        return found->second;
      }
    }
    ExecutionError(
        "Autoregressive stage '" + stage.name +
        "' produced no logits output");
  }

  [[nodiscard]] Tensor GreedyTokens(const Tensor& logits) const {
    if (logits.shape().size() < 2 || logits.shape().back() <= 0 ||
        logits.shape()[0] <= 0) {
      ExecutionError("Logits must have batch and vocabulary dimensions");
    }
    const std::size_t batch =
        static_cast<std::size_t>(logits.shape()[0]);
    const std::size_t vocabulary =
        static_cast<std::size_t>(logits.shape().back());
    const std::size_t per_batch = logits.element_count() / batch;
    if (per_batch < vocabulary) {
      ExecutionError("Logits tensor has an invalid shape");
    }
    std::vector<std::int64_t> tokens(batch);
    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
      const std::size_t begin =
          batch_index * per_batch + per_batch - vocabulary;
      std::size_t best = 0;
      float best_value = ReadFloat(logits, begin);
      for (std::size_t token = 1; token < vocabulary; ++token) {
        const float value = ReadFloat(logits, begin + token);
        if (value > best_value) {
          best = token;
          best_value = value;
        }
      }
      tokens[batch_index] = static_cast<std::int64_t>(best);
    }
    return Int64Tensor(
        {static_cast<std::int64_t>(batch), 1}, tokens);
  }

  [[nodiscard]] Tensor SampleTokens(
      const Tensor& logits,
      const Json& sampling,
      const PipelineRunOptions& options,
      const std::vector<std::vector<std::int64_t>>& history) {
    if (logits.shape().size() < 2 || logits.shape().back() <= 0 ||
        logits.shape()[0] <= 0) {
      ExecutionError("Logits must have batch and vocabulary dimensions");
    }
    const std::size_t batch =
        static_cast<std::size_t>(logits.shape()[0]);
    const std::size_t vocabulary =
        static_cast<std::size_t>(logits.shape().back());
    const std::size_t per_batch = logits.element_count() / batch;
    const double temperature =
        options.numbers.contains("temperature")
            ? options.numbers.at("temperature")
            : sampling.value("temperature", 1.0);
    const double top_p =
        options.numbers.contains("top_p")
            ? options.numbers.at("top_p")
            : sampling.value("top_p", 1.0);
    const std::int64_t top_k =
        options.integers.contains("top_k")
            ? options.integers.at("top_k")
            : sampling.value("top_k", 0);
    const double repetition_penalty =
        options.numbers.contains("repetition_penalty")
            ? options.numbers.at("repetition_penalty")
            : sampling.value("repetition_penalty", 1.0);
    if (temperature <= 0.0 || top_p <= 0.0 || top_p > 1.0 ||
        top_k < 0 || repetition_penalty <= 0.0) {
      throw Error(
          ErrorCode::invalid_argument,
          "Invalid temperature, top_p, top_k, or repetition_penalty");
    }

    std::vector<std::int64_t> tokens(batch);
    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
      const std::size_t begin =
          batch_index * per_batch + per_batch - vocabulary;
      std::vector<double> scores(vocabulary);
      for (std::size_t token = 0; token < vocabulary; ++token) {
        scores[token] =
            static_cast<double>(ReadFloat(logits, begin + token));
      }
      if (repetition_penalty != 1.0) {
        std::set<std::int64_t> previous;
        for (const auto& step : history) {
          if (batch_index < step.size()) {
            previous.insert(step[batch_index]);
          }
        }
        for (const std::int64_t token : previous) {
          if (token < 0 ||
              static_cast<std::size_t>(token) >= vocabulary) {
            continue;
          }
          double& score = scores[static_cast<std::size_t>(token)];
          score = score < 0.0 ? score * repetition_penalty
                              : score / repetition_penalty;
        }
      }

      std::vector<std::size_t> candidates(vocabulary);
      for (std::size_t token = 0; token < vocabulary; ++token) {
        candidates[token] = token;
      }
      std::ranges::sort(
          candidates,
          [&scores](std::size_t left, std::size_t right) {
            return scores[left] > scores[right];
          });
      if (top_k > 0 &&
          static_cast<std::size_t>(top_k) < candidates.size()) {
        candidates.resize(static_cast<std::size_t>(top_k));
      }

      const double maximum = scores[candidates.front()] / temperature;
      std::vector<double> weights;
      weights.reserve(candidates.size());
      double total = 0.0;
      for (const std::size_t token : candidates) {
        const double weight =
            std::exp(scores[token] / temperature - maximum);
        weights.push_back(weight);
        total += weight;
      }
      if (top_p < 1.0) {
        double cumulative = 0.0;
        std::size_t keep = 0;
        for (; keep < weights.size(); ++keep) {
          cumulative += weights[keep] / total;
          if (cumulative >= top_p) {
            ++keep;
            break;
          }
        }
        keep = std::max<std::size_t>(keep, 1);
        candidates.resize(keep);
        weights.resize(keep);
      }
      std::discrete_distribution<std::size_t> distribution(
          weights.begin(), weights.end());
      tokens[batch_index] =
          static_cast<std::int64_t>(
              candidates[distribution(random_engine)]);
    }
    return Int64Tensor(
        {static_cast<std::int64_t>(batch), 1}, tokens);
  }

  void SetStageTokenInput(
      const PipelineStage& stage,
      const Tensor& tokens) {
    bool updated = false;
    for (const auto& input : manifest().inputs()) {
      if (input.kind == PipelineInputKind::external &&
          input.semantic == "text.token_ids" &&
          Contains(stage.components, input.port.component)) {
        external_values.insert_or_assign(
            input.port.qualified(),
            AdaptExternal(tokens, manifest().Input(input.port)));
        updated = true;
      }
    }
    if (!updated) {
      ExecutionError(
          "Autoregressive stage '" + stage.name +
          "' has no text.token_ids external input");
    }
  }

  [[nodiscard]] std::size_t MaximumTokens(
      const PipelineStage& stage,
      const PipelineRunOptions& options) const {
    std::optional<std::size_t> requested_count;
    const auto requested = options.integers.find("max_tokens");
    if (requested != options.integers.end()) {
      if (requested->second <= 0) {
        throw Error(
            ErrorCode::invalid_argument,
            "max_tokens must be positive");
      }
      requested_count = static_cast<std::size_t>(requested->second);
    }
    const Json stage_options = Json::parse(stage.options_json);
    if (!requested_count.has_value() &&
        stage_options.contains("max_tokens")) {
      const Json& maximum = stage_options.at("max_tokens");
      if (maximum.is_number_integer() &&
          maximum.get<std::int64_t>() > 0) {
        requested_count = static_cast<std::size_t>(
            maximum.get<std::int64_t>());
      } else if (maximum.is_object() && maximum.contains("default") &&
          maximum.at("default").is_number_integer() &&
          maximum.at("default").get<std::int64_t>() > 0) {
        requested_count = static_cast<std::size_t>(
            maximum.at("default").get<std::int64_t>());
      }
    }
    if (!requested_count.has_value()) {
      throw Error(
          ErrorCode::invalid_argument,
          "Autoregressive stage '" + stage.name +
              "' requires max_tokens");
    }

    if (stage_options.contains("max_tokens") &&
        stage_options.at("max_tokens").is_object()) {
      const Json& maximum = stage_options.at("max_tokens");
      if (maximum.contains("limit") &&
          maximum.at("limit").is_number_integer() &&
          maximum.at("limit").get<std::int64_t>() > 0 &&
          *requested_count > static_cast<std::size_t>(
                                 maximum.at("limit").get<std::int64_t>())) {
        throw Error(
            ErrorCode::invalid_argument,
            "max_tokens exceeds the stage limit");
      }
    }

    std::size_t context_length = 0;
    for (const auto& input : manifest().inputs()) {
      if (input.kind != PipelineInputKind::external ||
          input.semantic != "text.token_ids" ||
          !Contains(stage.components, input.port.component)) {
        continue;
      }
      const auto value = external_values.find(input.port.qualified());
      if (value != external_values.end()) {
        context_length = std::max(
            context_length,
            static_cast<std::size_t>(
                value->second.shape().size() == 2
                    ? value->second.shape()[1]
                    : value->second.shape()[0]));
      }
    }
    if (stage_options.contains("stop") &&
        stage_options.at("stop").is_object() &&
        stage_options.at("stop").contains("max_sequence_length") &&
        stage_options.at("stop").at("max_sequence_length").is_number_integer()) {
      const auto maximum_sequence =
          stage_options.at("stop")
              .at("max_sequence_length")
              .get<std::int64_t>();
      if (maximum_sequence > 0) {
        if (context_length >= static_cast<std::size_t>(maximum_sequence)) {
          throw Error(
              ErrorCode::invalid_argument,
              "Prompt already reaches the maximum sequence length");
        }
        requested_count = std::min(
            *requested_count,
            static_cast<std::size_t>(maximum_sequence) - context_length);
      }
    }
    return *requested_count;
  }

  [[nodiscard]] NamedTensors RunAutoregressive(
      const PipelineStage& stage,
      const NamedTensors& inputs,
      const NamedTensors& overrides,
      const PipelineRunOptions& options) {
    StoreExternalInputs(inputs);
    const Json stage_options = Json::parse(stage.options_json);
    const Json sampling =
        stage_options.value("sampling", Json::object());
    const bool do_sample =
        options.integers.contains("do_sample")
            ? options.integers.at("do_sample") != 0
            : sampling.value("do_sample", false);
    if (options.integers.contains("seed")) {
      random_engine.seed(
          static_cast<std::uint64_t>(options.integers.at("seed")));
    }
    std::set<std::int64_t> eos_tokens;
    if (stage_options.contains("stop") &&
        stage_options.at("stop").is_object() &&
        stage_options.at("stop").contains("eos_token_ids")) {
      for (const auto& token :
           stage_options.at("stop").at("eos_token_ids")) {
        eos_tokens.insert(token.get<std::int64_t>());
      }
    }

    const std::size_t maximum = MaximumTokens(stage, options);
    std::vector<std::vector<std::int64_t>> generated;
    std::vector<bool> finished;
    for (std::size_t index = 0; index < maximum; ++index) {
      (void)StepStage(
          stage.name,
          index == 0 ? inputs : NamedTensors{},
          overrides,
          options);
      Tensor tokens =
          do_sample
              ? SampleTokens(
                    StageLogits(stage), sampling, options, generated)
              : GreedyTokens(StageLogits(stage));
      auto mutable_values = std::span(
          reinterpret_cast<std::int64_t*>(
              tokens.mutable_bytes().data()),
          tokens.element_count());
      if (finished.empty()) {
        finished.assign(mutable_values.size(), false);
      }
      const std::int64_t eos =
          eos_tokens.empty() ? 0 : *eos_tokens.begin();
      for (std::size_t batch_index = 0;
           batch_index < mutable_values.size();
           ++batch_index) {
        if (finished[batch_index]) {
          mutable_values[batch_index] = eos;
        } else if (eos_tokens.contains(mutable_values[batch_index])) {
          finished[batch_index] = true;
        }
      }
      const auto values = tokens.values<std::int64_t>();
      generated.emplace_back(values.begin(), values.end());
      SetStageTokenInput(stage, tokens);
      if (!eos_tokens.empty() &&
          std::ranges::all_of(finished, [](bool value) { return value; })) {
        break;
      }
    }

    NamedTensors result = CollectOutputs();
    if (!generated.empty()) {
      const std::size_t batch = generated.front().size();
      std::vector<std::int64_t> flattened(batch * generated.size());
      for (std::size_t step = 0; step < generated.size(); ++step) {
        for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
          flattened[batch_index * generated.size() + step] =
              generated[step][batch_index];
        }
      }
      result.emplace(
          "generated_token_ids",
          Int64Tensor(
              {
                  static_cast<std::int64_t>(batch),
                  static_cast<std::int64_t>(generated.size()),
              },
              flattened));
    }
    return result;
  }

  [[nodiscard]] NamedTensors RunStage(
      std::string_view stage_name,
      const NamedTensors& inputs,
      const NamedTensors& overrides,
      const PipelineRunOptions& options) {
    const PipelineStage& stage = FindStage(manifest(), stage_name);
    if (stage.kind == "autoregressive") {
      return RunAutoregressive(stage, inputs, overrides, options);
    }
    if (stage.kind == "iterative") {
      const std::size_t steps = InferenceSteps(stage.name, options);
      const std::size_t completed = stage_iterations[stage.name];
      if (completed >= steps) {
        return CollectOutputs();
      }
      NamedTensors result;
      for (std::size_t index = completed; index < steps; ++index) {
        result = StepStage(
            stage.name,
            index == completed ? inputs : NamedTensors{},
            overrides,
            options);
      }
      return result;
    }
    return StepStage(stage.name, inputs, overrides, options);
  }
};

Pipeline::Pipeline(PipelinePackage package)
    : package_(
          std::make_shared<const PipelinePackage>(std::move(package))) {}

Pipeline Pipeline::Load(
    const std::filesystem::path& directory,
    const RuntimeOptions& options) {
  return Pipeline(PipelinePackage::Load(directory, options));
}

const PipelineManifest& Pipeline::manifest() const noexcept {
  return package_->manifest();
}

std::unordered_map<std::string, std::vector<std::string>>
Pipeline::execution_providers() const {
  return package_->execution_providers();
}

PipelineSession Pipeline::CreateSession() const {
  return PipelineSession(package_);
}

PipelineSession::PipelineSession(
    std::shared_ptr<const PipelinePackage> package)
    : impl_(std::make_unique<Impl>(std::move(package))) {}

PipelineSession::PipelineSession(PipelineSession&&) noexcept = default;

PipelineSession& PipelineSession::operator=(PipelineSession&&) noexcept =
    default;

PipelineSession::~PipelineSession() = default;

NamedTensors PipelineSession::RunStage(
    std::string_view stage,
    const NamedTensors& inputs,
    const NamedTensors& overrides,
    const PipelineRunOptions& options) {
  std::scoped_lock lock(impl_->mutex);
  return impl_->RunStage(stage, inputs, overrides, options);
}

NamedTensors PipelineSession::StepStage(
    std::string_view stage,
    const NamedTensors& inputs,
    const NamedTensors& overrides,
    const PipelineRunOptions& options) {
  std::scoped_lock lock(impl_->mutex);
  return impl_->StepStage(stage, inputs, overrides, options);
}

NamedTensors PipelineSession::outputs() const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->CollectOutputs();
}

std::optional<Tensor> PipelineSession::state(std::string_view name) const {
  std::scoped_lock lock(impl_->mutex);
  const auto found = impl_->state_values.find(std::string(name));
  if (found == impl_->state_values.end()) {
    return std::nullopt;
  }
  return found->second;
}

void PipelineSession::ReleaseStage(std::string_view stage) {
  std::scoped_lock lock(impl_->mutex);
  (void)FindStage(impl_->manifest(), stage);
  for (const auto& state : impl_->manifest().states()) {
    if (state.release_after != stage) {
      continue;
    }
    impl_->state_values.erase(state.name);
    impl_->scheduler_histories.erase(state.name);
    if (state.kind == "kv_cache") {
      for (const auto& input : impl_->manifest().inputs()) {
        if (input.generator_kind != "multimodal_position_ids") {
          continue;
        }
        const Json parameters = Json::parse(input.generator_json);
        if (!parameters.contains("past_state") ||
            !parameters.at("past_state").is_array()) {
          continue;
        }
        const bool depends_on_state = std::ranges::any_of(
            parameters.at("past_state"),
            [&state](const Json& value) {
              return value.is_string() &&
                     value.get<std::string>() == state.name;
            });
        if (depends_on_state) {
          impl_->position_cursors.erase(input.port.qualified());
        }
      }
    }
    impl_->external_values.erase(state.input.qualified());
    impl_->endpoint_values.erase(state.input.qualified());
    impl_->endpoint_values.erase(state.output.qualified());
    for (const auto& owner : impl_->manifest().stages()) {
      if (Contains(owner.components, state.input.component) &&
          Contains(owner.components, state.output.component) &&
          (owner.kind == "autoregressive" || owner.kind == "iterative" ||
           owner.kind == "state_transition")) {
        impl_->stage_iterations.erase(owner.name);
      }
    }
  }
}

void PipelineSession::Reset() {
  std::scoped_lock lock(impl_->mutex);
  impl_->external_values.clear();
  impl_->endpoint_values.clear();
  impl_->state_values.clear();
  impl_->stage_iterations.clear();
  impl_->scheduler_histories.clear();
  impl_->position_cursors.clear();
}

}  // namespace onnx_world_model
