/**
 * @agent-file
 * @agent-purpose: Implements PipelineSession, the per-trajectory execution engine that resolves stage inputs, runs component sessions, and owns recurrent state, diffusion schedulers, guidance, and token sampling, plus its in-memory snapshot, restore, fork, and named-checkpoint operations, the StageRun state machine that drives every stage one step at a time, and the Pipeline that hands every session one shared admission scheduler and reports that scheduler's live counts.
 * @agent-public-api: Pipeline::Pipeline, Pipeline::Load, Pipeline::manifest, Pipeline::execution_providers, Pipeline::transfer_plan, Pipeline::CreateSession, Pipeline::scheduling_stats, PipelineSession move operations and destructor, PipelineSession::RunStage, PipelineSession::BeginStage, PipelineSession::StepStage, PipelineSession::outputs, PipelineSession::state, PipelineSession::ReleaseStage, PipelineSession::Reset, PipelineSession::Snapshot, PipelineSession::Restore, PipelineSession::Fork, PipelineSession::Checkpoint, PipelineSession::RestoreCheckpoint, PipelineSession::DropCheckpoint, PipelineSession::HasCheckpoint, PipelineSessionSnapshot::valid, StageRun move operations and destructor, StageRun::stage, StageRun::done, StageRun::iteration, StageRun::Step, StageRun::Finish, StageRun::RequestCancellation, StageRun::Cancel
 * @agent-invariants: All mutable state lives in the file-local SessionState bundle that PipelineSession::Impl derives from, behind impl_->mutex, so one session serves one request or trajectory and is never shared across threads without that lock. PipelineSession owns that Impl through a shared_ptr and a StageRun holds the same pointer, so a run outlives a moved or destroyed session wrapper. Device storage is preserved end to end: caller inputs, overrides, component outputs, recurrent state, public outputs, and StageEvent outputs keep the producing TensorBuffer, a transform-free connection forwards it unchanged, and external rank adaptation and the reshape transform reuse it through Tensor::FromBuffer because they only relabel axes. Every host-evaluated path -- casts, scheduler steps, guidance combination, packed video and audio finalization, token sampling, and value-reading generated-input programs -- materializes each device source exactly once at its own outer boundary and then reads only that host tensor; the per-element ReadFloat, WriteFloat, ReadInteger, and WriteInteger helpers never transfer. A stage runs its components in dependency order derived from the manifest connections. Unknown stage kinds, generator kinds, scheduler types, and option keys throw rather than falling back. ReleaseStage frees only state whose declared release_after names that stage, and Reset clears every cache plus every named checkpoint so the session can be reused while keeping the current random engine. Snapshot copies the whole SessionState bundle under the lock through Impl::CaptureLocked and records the package shared_ptr; Restore rejects a snapshot from any other PipelinePackage instance with ErrorCode::state, copies every container before taking the lock, and commits by swapping so it cannot leave partial state; Fork restores a fresh session on the same package from that snapshot and keeps that session's empty checkpoint map. Named checkpoints live on PipelineSession::Impl rather than in SessionState, so a snapshot never carries them and Restore leaves the target session's checkpoints alone; Checkpoint captures and publishes under one lock hold, RestoreCheckpoint finds, copies, and swaps a checkpoint under one lock acquisition so it is linearizable with reset and replacement, an empty name throws ErrorCode::invalid_argument, and an unknown name throws ErrorCode::state from both RestoreCheckpoint and DropCheckpoint. StageRun::Impl is the only stage state machine: RunStage drains it under one lock acquisition while BeginStage exposes it one step at a time, so complete runs preserve historical whole-stage serialization without duplicating execution logic. Begin resolves the stage kind, inputs, overrides, options, sampling configuration, seed, end-of-sequence tokens, prompt-derived token budget, and iterative target exactly once under the session lock. Autoregressive, iterative, and single-pass runs emit exactly one terminal completed event whose outputs equal the RunStage result, and every stop condition is reported by the following Step rather than folded into a step event. The run identity -- next_run_id and active_run_id -- is control metadata beside SessionState, one run is active per session at a time, a failing or cancelled run releases the slot without rolling back applied state, and a moved-from handle owns nothing so its destructor cancels nothing. Cancellation is cooperative and travels on PipelineRunOptions::cancellation: StageRun::Begin links the caller's token into the run's own CancellationSource -- copying its deadline and registering a reason-preserving callback whose registration is declared after the source so it is destroyed first -- and then replaces plan.options.cancellation with that internal token, so every downstream call observes both the caller's token and StageRun::RequestCancellation. RequestCancellation only signals that source and never takes the session mutex, which is what lets a second thread stop a step that already holds it. The token is polled at boundaries rather than per element: Step and Finish check before each step, StepStage checks on entry, between the two guidance passes, around guidance combination, and around the state transforms, RunStageComponents checks before and after each component, ApplyConnection checks before any host transform, and the autoregressive step brackets token sampling. Polling is not what enforces a deadline, though: the run's own source arms the copied deadline on the shared process-wide watchdog, so a backend already blocked inside a component pass is claimed at the deadline rather than at the next boundary. A cancelled step throws through the existing failure path, so the run slot is released, the handle closes, and everything already applied stays applied. Admission is layered above all of that and is scheduling only, never batching: Impl holds the Pipeline's shared detail::PipelineScheduler, and RunStage, StepStage, StageRun::Step, and a StageRun::Finish that still has steps to drain each take exactly one stack-scoped detail::PipelineLease before the session mutex, in the documented order CancellationState callback mutex -> scheduler mutex -> session mutex -> ONNX Runtime. BeginStage, a Finish whose run already completed, Cancel, RequestCancellation, and every query and state method take none, so an idle StageRun never holds a permit. Pipeline::scheduling_stats() only reads that same scheduler and returns a detached PipelineSchedulingStats value, so it takes no session lock, blocks no execution, and cannot admit or queue anything. Pipeline::transfer_plan() is a pass-through to the package's plan, which was computed while the package was built; Pipeline::Load forwards PipelinePlacementOptions to PipelinePackage::Load while the Pipeline(PipelinePackage, scheduling) constructor takes none, because its sessions already exist. A lease is never stored on a run or a session and its destructor is the only release path, so an exception, a cancellation, or a backend failure returns the permit. The stage kind is resolved from the immutable manifest, so admission needs no lock of its own, and no execution peeks at active_run_id before queuing: a queued caller may find a conflicting run once admitted and fail then, which is deliberate. The one window that is not "leave it applied" is the classifier-free guidance scratch state: the unconditional pass temporarily replaces the endpoint map, the position cursors, and the conditioning tensor's existing slot in external_values, and the file-local GuidanceScratch guard restores all three by swap -- static-asserted nothrow -- on every exit, so a cancellation or a backend failure inside that pass can never leave the unconditional conditioning value behind for the next step.
  * @agent-side-effects: May transfer device tensors to CPU at host-transform boundaries, runs ONNX Runtime inference through the shared PipelinePackage sessions, reads scheduler and tokenizer assets from disk, and advances the session's seeded random engine when sampling. Snapshot, Restore, Fork, and the checkpoint operations touch memory only; they perform no device transfer, no disk access, and no inference.
 */

#include "onnx_world_model/pipeline.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include <nlohmann/json.hpp>

#include "cancellation.hpp"
#include "onnx_world_model/error.hpp"
#include "pipeline_scheduler.hpp"

namespace onnx_world_model {
namespace {

using Json = nlohmann::json;

[[noreturn]] void ExecutionError(std::string message) {
  throw Error(ErrorCode::runtime_execution, std::move(message));
}

// Every checkpoint entry point names a checkpoint, and an empty name is a
// caller mistake rather than a session-state problem, so all four share this
// check and the std::string key it produces.
[[nodiscard]] std::string CheckpointKey(std::string_view name) {
  if (name.empty()) {
    throw Error(
        ErrorCode::invalid_argument,
        "Pipeline session checkpoint name must not be empty");
  }
  return std::string(name);
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

// Materializes the source once and then reads only host memory, so a device
// tensor crosses the transfer boundary exactly one time per cast.
Tensor CastTensor(const Tensor& source, DataType target_type) {
  if (source.data_type() == target_type) {
    return source;
  }
  const Tensor host = source.CopyToCpu();
  Tensor result(target_type, host.shape());
  if (IsIntegral(host.data_type()) && IsIntegral(target_type)) {
    for (std::size_t index = 0; index < host.element_count(); ++index) {
      if (IsUnsigned(host.data_type())) {
        WriteInteger(
            result, index, ReadUnsignedInteger(host, index));
      } else {
        WriteInteger(result, index, ReadSignedInteger(host, index));
      }
    }
    return result;
  }
  if (IsIntegral(host.data_type()) &&
      target_type == DataType::float64) {
    auto values = std::span(
        reinterpret_cast<double*>(result.mutable_bytes().data()),
        result.element_count());
    for (std::size_t index = 0; index < host.element_count(); ++index) {
      values[index] =
          IsUnsigned(host.data_type())
              ? static_cast<double>(ReadUnsignedInteger(host, index))
              : static_cast<double>(ReadSignedInteger(host, index));
    }
    return result;
  }
  if (!IsIntegral(host.data_type()) && IsIntegral(target_type)) {
    for (std::size_t index = 0; index < host.element_count(); ++index) {
      const double value =
          host.data_type() == DataType::float64
              ? host.values<double>()[index]
              : static_cast<double>(ReadFloat(host, index));
      WriteInteger(result, index, value);
    }
    return result;
  }
  for (std::size_t index = 0; index < host.element_count(); ++index) {
    WriteFloat(result, index, ReadFloat(host, index));
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
  const Tensor host_state = state.CopyToCpu();
  const Tensor host_velocity = velocity.CopyToCpu();
  Tensor result(host_state.data_type(), host_state.shape());
  for (std::size_t index = 0; index < host_state.element_count(); ++index) {
    WriteFloat(
        result,
        index,
        ReadFloat(host_state, index) +
            delta * ReadFloat(host_velocity, index));
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
  const Tensor host_state = state.CopyToCpu();
  const Tensor host_velocity = velocity.CopyToCpu();
  const Tensor host_indexes = indexes->CopyToCpu();
  Tensor result = host_state;
  const auto rows = host_indexes.values<std::int64_t>();
  const std::size_t width =
      static_cast<std::size_t>(host_state.shape()[1]);
  for (std::size_t velocity_row = 0;
       velocity_row < rows.size();
       ++velocity_row) {
    if (rows[velocity_row] < 0 ||
        rows[velocity_row] >= host_state.shape()[0]) {
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
          ReadFloat(host_state, state_index) +
              delta * ReadFloat(host_velocity, velocity_index));
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

struct SchedulerHistory {
  std::vector<Tensor> model_outputs;
  std::optional<Tensor> last_sample;
  std::size_t lower_order{0};
  std::size_t previous_order{1};
};

// Every mutable execution field a PipelineSession owns, in one copyable
// bundle. PipelineSession::Impl derives from it and PipelineSessionSnapshot
// stores one, so capturing, restoring, and forking move the whole bundle and
// a field added here is carried by all three without further edits.
struct SessionState {
  NamedTensors external_values;
  NamedTensors endpoint_values;
  NamedTensors state_values;
  NamedTensors guidance_values;
  std::unordered_map<std::string, std::size_t> stage_iterations;
  mutable std::unordered_map<std::string, SchedulerHistory>
      scheduler_histories;
  mutable std::unordered_map<std::string, std::vector<std::int64_t>>
      position_cursors;
  std::mt19937_64 random_engine{0};

  void Swap(SessionState& other) noexcept {
    external_values.swap(other.external_values);
    endpoint_values.swap(other.endpoint_values);
    state_values.swap(other.state_values);
    guidance_values.swap(other.guidance_values);
    stage_iterations.swap(other.stage_iterations);
    scheduler_histories.swap(other.scheduler_histories);
    position_cursors.swap(other.position_cursors);
    std::swap(random_engine, other.random_engine);
  }
};

// The scratch window a guided stage opens around its unconditional pass.
// That pass overwrites the session's endpoint map, its position cursors, and
// the one external conditioning tensor, and every one of those has to be back
// in place before any Error leaves the window -- a cancellation claimed at one
// of the new boundaries, a backend failure, or a guided output the second pass
// did not produce. Restoration therefore performs no fallible work: the copies
// are taken in the constructor, before anything is mutated, and Restore only
// swaps. The static assertions are the justification for its noexcept.
class GuidanceScratch {
 public:
  using PositionCursors =
      std::unordered_map<std::string, std::vector<std::int64_t>>;

  static_assert(std::is_nothrow_swappable_v<NamedTensors>);
  static_assert(std::is_nothrow_swappable_v<PositionCursors>);
  static_assert(std::is_nothrow_move_constructible_v<Tensor>);
  static_assert(std::is_nothrow_move_assignable_v<Tensor>);
  static_assert(std::is_nothrow_swappable_v<Tensor>);

  // `conditioning` must name the conditioning tensor's existing slot in
  // external_values. Nothing inside the window inserts into that map, so the
  // reference stays valid and the restore never has to hash the key -- or
  // allocate a node -- again.
  GuidanceScratch(
      NamedTensors& endpoints,
      PositionCursors& cursors,
      Tensor& conditioning)
      : endpoints_(endpoints),
        cursors_(cursors),
        conditioning_(conditioning),
        saved_endpoints_(endpoints),
        saved_cursors_(cursors),
        saved_conditioning_(conditioning) {}

  GuidanceScratch(const GuidanceScratch&) = delete;
  GuidanceScratch& operator=(const GuidanceScratch&) = delete;
  GuidanceScratch(GuidanceScratch&&) = delete;
  GuidanceScratch& operator=(GuidanceScratch&&) = delete;

  ~GuidanceScratch() { Restore(); }

  // Idempotent, so the success path calls it explicitly before publishing the
  // combined predictions and the destructor then does nothing.
  void Restore() noexcept {
    if (!armed_) {
      return;
    }
    armed_ = false;
    endpoints_.swap(saved_endpoints_);
    cursors_.swap(saved_cursors_);
    std::swap(conditioning_, saved_conditioning_);
  }

 private:
  NamedTensors& endpoints_;
  PositionCursors& cursors_;
  Tensor& conditioning_;
  NamedTensors saved_endpoints_;
  PositionCursors saved_cursors_;
  Tensor saved_conditioning_;
  bool armed_{true};
};

}  // namespace

struct PipelineSessionSnapshot::Impl {
  std::shared_ptr<const PipelinePackage> package;
  SessionState state;
};

struct PipelineSession::Impl : SessionState {
  Impl(
      std::shared_ptr<const PipelinePackage> pipeline_package,
      std::shared_ptr<detail::PipelineScheduler> admission)
      : package(std::move(pipeline_package)),
        admission_scheduler(std::move(admission)) {}

  std::shared_ptr<const PipelinePackage> package;
  // The Pipeline's admission controller, shared by every session that
  // Pipeline (or any copy of it) created. A StageRun reaches it through this
  // same Impl, so incremental and all-at-once execution are admitted by one
  // controller rather than two.
  std::shared_ptr<detail::PipelineScheduler> admission_scheduler;
  mutable std::mutex mutex;
  // Named checkpoints are control metadata, deliberately declared here rather
  // than in SessionState so that capturing, restoring, or forking execution
  // state never carries a checkpoint namespace along with it.
  std::unordered_map<std::string, PipelineSessionSnapshot> checkpoints;
  // The identity of the session's one incremental stage run, also control
  // metadata rather than execution state: a snapshot never carries a run, so
  // restoring or forking can never resurrect one or adopt another session's.
  // `next_run_id` only ever grows, so a stale handle can be told apart from
  // the run that currently holds the slot.
  std::uint64_t next_run_id{1};
  std::optional<std::uint64_t> active_run_id;

  // An active run owns the execution state until it completes or is
  // cancelled, and its autoregressive loop state lives on the run rather than
  // in SessionState. Rather than capture or mutate a session mid-run and
  // silently drop that loop state, every conflicting operation fails here.
  void EnsureNoActiveRunLocked(std::string_view operation) const {
    if (active_run_id.has_value()) {
      throw Error(
          ErrorCode::state,
          "Pipeline session cannot " + std::string(operation) +
              " while a stage run is active");
    }
  }

  // Captures the execution bundle without touching the lock, so a caller that
  // already holds `mutex` can capture without re-entering a public method.
  // Copying the bundle copies Tensor values, which share their storage
  // copy-on-write, so a device-backed tensor is never materialized here.
  [[nodiscard]] PipelineSessionSnapshot CaptureLocked() const {
    auto captured = std::make_shared<PipelineSessionSnapshot::Impl>();
    captured->package = package;
    captured->state = static_cast<const SessionState&>(*this);
    return PipelineSessionSnapshot(std::move(captured));
  }

  // Takes the one admission permit an execution of `stage` needs, before the
  // session lock. The stage kind comes from the immutable manifest, so this
  // needs no lock of its own, and an unknown stage name fails here with the
  // same ErrorCode::invalid_argument it produced when the lookup happened
  // under the lock. Deliberately no peek at active_run_id first: a queued
  // caller may find a conflicting run once it is admitted and fail then,
  // which is a fair outcome, while checking before waiting would only be a
  // staler race.
  [[nodiscard]] detail::PipelineLease AcquireStageLease(
      std::string_view stage,
      const CancellationToken& cancellation) const {
    return detail::AcquireExecutionLease(
        admission_scheduler, FindStage(manifest(), stage).kind, cancellation);
  }

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

  // Rank adaptation only relabels axes, so the byte layout is unchanged and
  // the original buffer -- device-backed or not -- is retained as is.
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
      return Tensor::FromBuffer(
          tensor.data_type(), std::move(shape), tensor.buffer());
    }
    if (tensor.shape().size() == spec.shape.size() + 1 &&
        tensor.shape()[0] == 1) {
      std::vector<std::int64_t> shape(
          tensor.shape().begin() + 1, tensor.shape().end());
      return Tensor::FromBuffer(
          tensor.data_type(), std::move(shape), tensor.buffer());
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

  [[nodiscard]] Json StageGuidance(std::string_view stage_name) const {
    const Json stage_options =
        Json::parse(FindStage(manifest(), stage_name).options_json);
    if (!stage_options.contains("guidance") ||
        !stage_options.at("guidance").is_object()) {
      return Json::object();
    }
    return stage_options.at("guidance");
  }

  // Names a host may use to supply the unconditional value of the guided
  // input. The manifest may declare one explicitly; otherwise the reserved
  // "unconditional:" prefix is applied to the guided port, its semantic, and
  // its alias so no extra manifest field is required.
  [[nodiscard]] std::vector<std::string> GuidanceInputNames(
      const Json& guidance) const {
    std::vector<std::string> names;
    if (guidance.contains("unconditional_input") &&
        guidance.at("unconditional_input").is_string()) {
      names.push_back(
          guidance.at("unconditional_input").get<std::string>());
    }
    const std::string conditioning =
        guidance.value("conditioning_input", std::string{});
    if (conditioning.empty()) {
      return names;
    }
    names.push_back("unconditional:" + conditioning);
    const Endpoint endpoint = Endpoint::Parse(conditioning);
    if (const PipelineInput* declared =
            FindDeclaredInput(manifest(), endpoint)) {
      if (!declared->semantic.empty()) {
        names.push_back("unconditional:" + declared->semantic);
      }
      if (!declared->name.empty()) {
        names.push_back("unconditional:" + declared->name);
      }
    }
    return names;
  }

  // Endpoints whose predictions are combined across the conditional and
  // unconditional passes. The manifest may list them; otherwise every
  // scheduler-driven recurrent output produced by the stage is guided.
  [[nodiscard]] std::vector<std::string> GuidedOutputs(
      const PipelineStage& stage,
      const Json& guidance) const {
    std::vector<std::string> result;
    if (guidance.contains("outputs") && guidance.at("outputs").is_array()) {
      for (const auto& value : guidance.at("outputs")) {
        result.push_back(value.get<std::string>());
      }
      return result;
    }
    for (const auto& connection : manifest().connections()) {
      if (!connection.recurrent ||
          !connection.transform.has_value() ||
          *connection.transform != "scheduler_step" ||
          !Contains(stage.components, connection.source.component)) {
        continue;
      }
      result.push_back(connection.source.qualified());
    }
    return result;
  }

  [[nodiscard]] double GuidanceScale(
      std::string_view stage_name,
      const Json& guidance,
      const PipelineRunOptions& options) const {
    const std::string scale_option =
        guidance.value("scale_option", std::string("guidance_scale"));
    const auto number = options.numbers.find(scale_option);
    if (number != options.numbers.end()) {
      return number->second;
    }
    const auto integer = options.integers.find(scale_option);
    if (integer != options.integers.end()) {
      return static_cast<double>(integer->second);
    }
    const Json mode = SchedulerModeOverrides(stage_name, options);
    if (mode.contains(scale_option) && mode.at(scale_option).is_number()) {
      return mode.at(scale_option).get<double>();
    }
    return guidance.value("default_scale", 1.0);
  }

  struct GuidancePlan {
    bool active{false};
    Endpoint conditioning;
    Tensor unconditional;
    double scale{1.0};
    std::vector<std::string> outputs;
  };

  [[nodiscard]] GuidancePlan ResolveGuidance(
      const PipelineStage& stage,
      const NamedTensors& overrides,
      const PipelineRunOptions& options) const {
    GuidancePlan plan;
    const Json guidance = StageGuidance(stage.name);
    if (guidance.empty()) {
      return plan;
    }
    const std::string kind = guidance.value("kind", std::string{});
    if (kind != "classifier_free") {
      ExecutionError("Unsupported stage guidance kind '" + kind + "'");
    }
    plan.scale = GuidanceScale(stage.name, guidance, options);
    plan.conditioning =
        Endpoint::Parse(guidance.at("conditioning_input").get<std::string>());
    const Tensor* unconditional = nullptr;
    for (const auto& name : GuidanceInputNames(guidance)) {
      const auto stored = guidance_values.find(name);
      if (stored != guidance_values.end()) {
        unconditional = &stored->second;
        break;
      }
      const auto supplied = overrides.find(name);
      if (supplied != overrides.end()) {
        unconditional = &supplied->second;
        break;
      }
    }
    if (plan.scale == 1.0) {
      // Classifier-free guidance at scale 1 is the conditional pass itself.
      return plan;
    }
    if (unconditional == nullptr) {
      throw Error(
          ErrorCode::invalid_argument,
          "Stage '" + stage.name + "' uses classifier-free guidance at scale " +
              std::to_string(plan.scale) + " but no unconditional value for '" +
              plan.conditioning.qualified() +
              "' was provided; supply 'unconditional:" +
              plan.conditioning.qualified() + "'");
    }
    plan.unconditional = AdaptExternal(
        *unconditional, manifest().Input(plan.conditioning));
    plan.outputs = GuidedOutputs(stage, guidance);
    if (plan.outputs.empty()) {
      ExecutionError(
          "Stage '" + stage.name +
          "' declares guidance but produces no guided prediction");
    }
    plan.active = true;
    return plan;
  }

  [[nodiscard]] static Tensor CombineGuidance(
      const Tensor& conditional,
      const Tensor& unconditional,
      double scale) {
    if (conditional.shape() != unconditional.shape() ||
        conditional.data_type() != unconditional.data_type()) {
      ExecutionError(
          "Conditional and unconditional predictions have different shapes");
    }
    const Tensor host_conditional = conditional.CopyToCpu();
    const Tensor host_unconditional = unconditional.CopyToCpu();
    Tensor result(conditional.data_type(), conditional.shape());
    for (std::size_t index = 0;
         index < host_conditional.element_count();
         ++index) {
      const double guided = ReadFloat(host_unconditional, index);
      WriteFloat(
          result,
          index,
          static_cast<float>(
              guided +
              scale * (ReadFloat(host_conditional, index) - guided)));
    }
    return result;
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
    if (indexes == nullptr) {
      ExecutionError(
          "Scheduler row scatter requires int64 token indexes");
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
    // Each operand crosses to the host once here; row selection and scattering
    // reuse these tensors throughout the complete scheduler step.
    const Tensor host_state = state.CopyToCpu();
    const Tensor host_velocity = velocity.CopyToCpu();
    std::optional<Tensor> host_indexes;
    if (indexes != nullptr) {
      host_indexes = indexes->CopyToCpu();
    }
    const Tensor* host_index_values =
        host_indexes.has_value() ? &*host_indexes : nullptr;
    Tensor sample = SelectSchedulerRows(
        host_state,
        host_velocity,
        host_index_values);
    Tensor converted(sample.data_type(), sample.shape());
    for (std::size_t element = 0; element < sample.element_count(); ++element) {
      WriteFloat(
          converted,
          element,
          ReadFloat(sample, element) -
              sigmas[index] * ReadFloat(host_velocity, element));
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
    return ScatterSchedulerRows(
        host_state,
        predicted,
        host_index_values);
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
        // Cursor tracking needs the values on the host; the override itself
        // is returned unchanged so a device tensor stays on its device.
        const Tensor host_value = value->CopyToCpu();
        const auto positions = host_value.values<std::int64_t>();
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
        // The token sequence and the vision grid each cross to the host once
        // for the whole batch loop below.
        const Tensor host_sequence = sequence->CopyToCpu();
        const auto ids = host_sequence.values<std::int64_t>();
        const auto image_features =
            endpoint_values.find("reasoner_vision_encoder.image_features");
        const auto grid_value =
            endpoint_values.find("reasoner_vision_encoder.grid_thw");
        // Only the prefill branch below reads the grid, so nothing is
        // materialized while decoding with a populated past state.
        const bool grid_present = past_length == 0 &&
                                  grid_value != endpoint_values.end() &&
                                  grid_value->second.data_type() ==
                                      DataType::int64 &&
                                  grid_value->second.element_count() == 3;
        const Tensor host_grid =
            grid_present ? grid_value->second.CopyToCpu() : Tensor{};
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
          if (metadata.contains("vision_understanding") &&
              metadata.at("vision_understanding").contains("tokens") &&
              metadata.at("vision_understanding").contains("preprocessing") &&
              metadata.at("vision_understanding")
                  .at("preprocessing")
                  .contains("patchify") &&
              grid_present) {
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
            const auto raw_grid = host_grid.values<std::int64_t>();
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
          if (grid_present) {
            const auto raw_grid = host_grid.values<std::int64_t>();
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
        const Tensor host_indexes = index_tensor->CopyToCpu();
        const auto token_indexes =
            host_indexes.values<std::int64_t>();
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
    const Tensor host_packed = packed.CopyToCpu();
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
                          token * host_packed.shape()[1] + packed_channel);
                  const std::size_t target_index =
                      static_cast<std::size_t>(
                          ((((batch_index * channels + channel) * frames +
                              frame) *
                                 height +
                             grid_y * patch + patch_y) *
                                width +
                            grid_x * patch + patch_x));
                  WriteFloat(
                      result,
                      target_index,
                      ReadFloat(host_packed, source_index));
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
    const Tensor host_packed = packed.CopyToCpu();
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
          WriteFloat(
              result, target_index, ReadFloat(host_packed, source_index));
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
    // Every branch below is a host-evaluated transform over a whole tensor,
    // so the token is checked once here rather than inside the per-element
    // ReadFloat and WriteFloat loops those branches run.
    options.cancellation.ThrowIfCancellationRequested();
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
      // The byte layout is unchanged, so the source buffer is reused and a
      // device tensor is never materialized for a shape-only view.
      return Tensor::FromBuffer(
          source.data_type(), std::move(shape), source.buffer());
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

  void RunStageComponents(
      const PipelineStage& stage,
      bool first_call_inputs_empty,
      const NamedTensors& overrides,
      const PipelineRunOptions& options) {
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
          first_call_inputs_empty &&
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
      // Per component rather than per tensor: resolving one component's
      // inputs can run generated-input programs and host transforms, and a
      // cancelled run should not start another component's worth of them.
      options.cancellation.ThrowIfCancellationRequested();
      NamedTensors model_inputs;
      model_inputs.reserve(component.metadata.inputs.size());
      for (const auto& spec : component.metadata.inputs) {
        const Endpoint endpoint{component.name, spec.name};
        Tensor value = ResolveInput(endpoint, overrides, options);
        endpoint_values.insert_or_assign(endpoint.qualified(), value);
        model_inputs.emplace(spec.name, std::move(value));
      }
      options.cancellation.ThrowIfCancellationRequested();
      NamedTensors model_outputs = package->Component(component.name)
                                       .Run(model_inputs, options.cancellation);
      for (auto& [name, tensor] : model_outputs) {
        // Component outputs keep whatever storage the backend produced, so a
        // device-backed output flows to the next component untouched.
        endpoint_values.insert_or_assign(
            component.name + "." + name, std::move(tensor));
      }
      options.cancellation.ThrowIfCancellationRequested();
    }
  }

  // Stores host-supplied unconditional values for a guided stage and returns
  // the remaining inputs, which are bound as ordinary external values.
  [[nodiscard]] NamedTensors ExtractGuidanceInputs(
      const PipelineStage& stage,
      const NamedTensors& inputs) {
    const Json guidance = StageGuidance(stage.name);
    if (guidance.empty() || inputs.empty()) {
      return inputs;
    }
    const std::vector<std::string> names = GuidanceInputNames(guidance);
    NamedTensors remaining;
    remaining.reserve(inputs.size());
    for (const auto& [name, tensor] : inputs) {
      if (std::ranges::find(names, name) != names.end()) {
        guidance_values.insert_or_assign(name, tensor);
      } else {
        remaining.emplace(name, tensor);
      }
    }
    return remaining;
  }

  [[nodiscard]] NamedTensors StepStage(
      std::string_view stage_name,
      const NamedTensors& inputs,
      const NamedTensors& overrides,
      const PipelineRunOptions& options) {
    options.cancellation.ThrowIfCancellationRequested();
    const PipelineStage& stage = FindStage(manifest(), stage_name);
    const NamedTensors external = ExtractGuidanceInputs(stage, inputs);
    StoreExternalInputs(external);

    const GuidancePlan guidance =
        ResolveGuidance(stage, overrides, options);
    RunStageComponents(stage, external.empty(), overrides, options);
    if (guidance.active) {
      // The conditional pass owns the packed layout that the scheduler
      // update and the public outputs observe; the unconditional pass only
      // contributes its prediction.
      NamedTensors conditional_outputs;
      for (const auto& name : guidance.outputs) {
        const auto produced = endpoint_values.find(name);
        if (produced == endpoint_values.end()) {
          ExecutionError(
              "Guided output '" + name + "' was not produced by stage '" +
              stage.name + "'");
        }
        conditional_outputs.emplace(name, produced->second);
      }
      const std::string conditioning = guidance.conditioning.qualified();
      const auto saved = external_values.find(conditioning);
      if (saved == external_values.end()) {
        ExecutionError(
            "Guided input '" + conditioning + "' has no conditional value");
      }
      // Arming the guard is the last thing that happens before the session's
      // conditional state is overwritten, so every exit below -- including a
      // cancellation claimed between the passes or inside the unconditional
      // one -- unwinds through it and leaves the conditional endpoints,
      // cursors, and conditioning tensor exactly as the conditional pass left
      // them. This is scratch state only; nothing the run legitimately
      // applied outside the window is touched.
      GuidanceScratch scratch(
          endpoint_values, position_cursors, saved->second);
      saved->second = guidance.unconditional;
      // Between the two passes: the conditional prediction is already held,
      // so stopping here costs one pass rather than two.
      options.cancellation.ThrowIfCancellationRequested();
      RunStageComponents(stage, external.empty(), overrides, options);
      // Guidance combination is a host transform over every guided output,
      // so it is bracketed rather than checked per element.
      options.cancellation.ThrowIfCancellationRequested();
      NamedTensors combined;
      for (const auto& name : guidance.outputs) {
        const auto produced = endpoint_values.find(name);
        if (produced == endpoint_values.end()) {
          ExecutionError(
              "Guided output '" + name +
              "' was not produced by the unconditional pass");
        }
        combined.emplace(
            name,
            CombineGuidance(
                conditional_outputs.at(name),
                produced->second,
                guidance.scale));
      }
      options.cancellation.ThrowIfCancellationRequested();
      // Closes the scratch window before the combined predictions are
      // published, so what follows writes into the conditional endpoint map.
      scratch.Restore();
      for (auto& [name, tensor] : combined) {
        endpoint_values.insert_or_assign(name, std::move(tensor));
      }
    }

    // The state transforms below are host-evaluated scheduler steps and
    // finalizations, so the token is checked once before them and once after.
    options.cancellation.ThrowIfCancellationRequested();
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
    options.cancellation.ThrowIfCancellationRequested();
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
    const Tensor host_logits = logits.CopyToCpu();
    std::vector<std::int64_t> tokens(batch);
    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
      const std::size_t begin =
          batch_index * per_batch + per_batch - vocabulary;
      std::size_t best = 0;
      float best_value = ReadFloat(host_logits, begin);
      for (std::size_t token = 1; token < vocabulary; ++token) {
        const float value = ReadFloat(host_logits, begin + token);
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
    const Tensor host_logits = logits.CopyToCpu();
    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
      const std::size_t begin =
          batch_index * per_batch + per_batch - vocabulary;
      std::vector<double> scores(vocabulary);
      for (std::size_t token = 0; token < vocabulary; ++token) {
        scores[token] =
            static_cast<double>(ReadFloat(host_logits, begin + token));
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

};

// The one state machine behind both execution paths. A StageRun holds one of
// these; PipelineSession::RunStage begins a run, drains it, and drops it, so
// full and incremental execution cannot diverge. Everything that used to be a
// local of the autoregressive loop -- the generated history, the per-lane
// end-of-sequence latch, the resolved sampling configuration, and the token
// budget -- lives here, deliberately outside SessionState so that a snapshot
// never captures half of an in-flight run.
struct StageRun::Impl {
  std::shared_ptr<PipelineSession::Impl> session;
  std::uint64_t run_id{0};
  std::string stage_name;
  std::string stage_kind;
  NamedTensors inputs;
  NamedTensors overrides;
  PipelineRunOptions options;

  // Every run owns its cancellation source, so RequestCancellation always has
  // something to signal even when the caller supplied no token. `external`
  // links a caller-supplied token into it. Declaration order is the lifetime
  // contract: `external` is destroyed before `source`, so the callback that
  // targets `source` cannot run against a destroyed object.
  CancellationSource source;
  std::optional<detail::CancellationRegistration> external;

  // The stage's own inputs are bound by the first step that actually runs, and
  // never again, exactly as the old loops passed them only on their first
  // iteration.
  bool consumed_inputs{false};
  std::size_t emitted_steps{0};
  bool completed{false};
  bool closed{false};
  NamedTensors final_outputs;
  std::optional<NamedTensors> last_step_outputs;

  // Autoregressive plan, resolved once by Begin.
  Json sampling;
  bool do_sample{false};
  std::set<std::int64_t> eos_tokens;
  std::size_t maximum_tokens{0};
  std::vector<std::vector<std::int64_t>> generated;
  std::vector<bool> finished_lanes;
  bool stop_requested{false};

  // Iterative plan: the absolute cursor this run drives the stage to, read
  // once so a later option change cannot move the target mid-run.
  std::size_t target_iterations{0};

  // Resolves everything the run will need while the caller still holds the
  // session lock, then claims the session's run slot. A failure here leaves
  // the slot free; whatever it already applied to the session stays applied,
  // which is what the previous RunStage did when it failed after storing its
  // inputs.
  [[nodiscard]] static std::unique_ptr<Impl> Begin(
      const std::shared_ptr<PipelineSession::Impl>& owner,
      std::string_view stage_name,
      const NamedTensors& inputs,
      const NamedTensors& overrides,
      const PipelineRunOptions& options) {
    // Checked before anything is resolved or stored, so a token that is
    // already cancelled -- including a reused one -- fails without claiming
    // the session's run slot and without touching the session.
    options.cancellation.ThrowIfCancellationRequested();
    const PipelineStage& stage = FindStage(owner->manifest(), stage_name);
    auto plan = std::make_unique<Impl>();
    plan->session = owner;
    plan->stage_name = stage.name;
    plan->stage_kind = stage.kind;
    plan->inputs = inputs;
    plan->overrides = overrides;
    plan->options = options;
    plan->LinkCancellation(options.cancellation);

    if (stage.kind == "autoregressive") {
      owner->StoreExternalInputs(inputs);
      const Json stage_options = Json::parse(stage.options_json);
      plan->sampling = stage_options.value("sampling", Json::object());
      plan->do_sample =
          options.integers.contains("do_sample")
              ? options.integers.at("do_sample") != 0
              : plan->sampling.value("do_sample", false);
      if (options.integers.contains("seed")) {
        owner->random_engine.seed(
            static_cast<std::uint64_t>(options.integers.at("seed")));
      }
      if (stage_options.contains("stop") &&
          stage_options.at("stop").is_object() &&
          stage_options.at("stop").contains("eos_token_ids")) {
        for (const auto& token :
             stage_options.at("stop").at("eos_token_ids")) {
          plan->eos_tokens.insert(token.get<std::int64_t>());
        }
      }
      // Measured before the first step replaces the prompt with a single
      // token, so the budget reflects the prompt this run started from.
      plan->maximum_tokens = owner->MaximumTokens(stage, options);
    } else if (stage.kind == "iterative") {
      plan->target_iterations = owner->InferenceSteps(stage.name, options);
    }

    plan->run_id = owner->next_run_id++;
    owner->active_run_id = plan->run_id;
    return plan;
  }

  // Gives this run one effective token that fires for both its own
  // RequestCancellation and the caller's token, and publishes that token as
  // the run's options so every downstream call observes both. The external
  // deadline is copied rather than shared, so the run's own source arms the
  // same instant on the shared watchdog and the reason stays a deadline
  // instead of turning into a plain cancellation.
  void LinkCancellation(const CancellationToken& caller) {
    if (const auto deadline = caller.deadline(); deadline.has_value()) {
      source = CancellationSource::WithDeadline(*deadline);
    }
    options.cancellation = source.token();
    if (!caller.cancellable()) {
      return;
    }
    external.emplace(
        caller,
        [target = &source](CancellationReason reason) {
          detail::CancellationAccess::Cancel(*target, reason);
        });
    // A caller token that was already cancelled ran that callback inline, so
    // this run is cancelled before it claims the session's run slot.
    options.cancellation.ThrowIfCancellationRequested();
  }

  // Never takes the session lock: this is the one operation a second thread
  // can perform while Step() or Finish() is executing.
  void RequestCancellation() noexcept { source.Cancel(); }

  void ReleaseSlotLocked() noexcept {
    if (session->active_run_id.has_value() &&
        *session->active_run_id == run_id) {
      session->active_run_id.reset();
    }
  }

  void EnsureActiveLocked() const {
    if (completed) {
      throw Error(
          ErrorCode::state,
          "Pipeline stage run for '" + stage_name +
              "' already reported its completed event");
    }
    if (closed) {
      throw Error(
          ErrorCode::state,
          "Pipeline stage run for '" + stage_name + "' is closed");
    }
    if (!session->active_run_id.has_value() ||
        *session->active_run_id != run_id) {
      throw Error(
          ErrorCode::state,
          "Pipeline stage run for '" + stage_name +
              "' is no longer this session's active run");
    }
  }

  [[nodiscard]] StageEvent Step() {
    {
      // A terminal handle has a deterministic state error and must not queue
      // behind unrelated work merely to discover it. The check after
      // admission remains authoritative for a concurrent state change.
      std::scoped_lock lock(session->mutex);
      if (completed || closed) {
        EnsureActiveLocked();
      }
    }
    // One Step() is one execution, so it takes one permit -- before the
    // session lock, in the documented lock order -- and returns it when this
    // scope ends, whether the step produced an event or threw. A handle
    // sitting idle between Step() calls therefore holds nothing.
    try {
      const detail::PipelineLease lease = AcquireLease();
      std::scoped_lock lock(session->mutex);
      EnsureActiveLocked();
      options.cancellation.ThrowIfCancellationRequested();
      return StepLocked();
    } catch (...) {
      // The session keeps everything the run already applied, exactly as a
      // failing RunStage left it; only the run slot and this handle close.
      // Admission-time cancellation travels this same path even though it
      // happens before the session lock is first taken.
      std::scoped_lock lock(session->mutex);
      ReleaseSlotLocked();
      closed = true;
      throw;
    }
  }

  [[nodiscard]] NamedTensors Finish() {
    {
      // A run that already reported its completed event has nothing left to
      // execute, so returning its cached outputs must not queue behind a
      // saturated scheduler.
      std::scoped_lock lock(session->mutex);
      if (completed) {
        return final_outputs;
      }
      if (closed) {
        EnsureActiveLocked();
      }
    }
    try {
      // One permit covers the whole remaining drain, matching RunStage: the
      // steps of one stage are not admitted individually.
      const detail::PipelineLease lease = AcquireLease();
      std::scoped_lock lock(session->mutex);
      return FinishLocked();
    } catch (...) {
      // AcquireLease can itself observe cancellation while queued. Closing
      // here keeps that path identical to cancellation inside FinishLocked.
      std::scoped_lock lock(session->mutex);
      ReleaseSlotLocked();
      closed = true;
      throw;
    }
  }

  // The run's own token is the one that guards a queued wait, so
  // RequestCancellation and the caller's deadline both release a Step or
  // Finish that has not been admitted yet.
  [[nodiscard]] detail::PipelineLease AcquireLease() const {
    return detail::AcquireExecutionLease(
        session->admission_scheduler, stage_kind, options.cancellation);
  }

  // Drains a run while the caller holds the session lock. Full RunStage uses
  // this path so its historical whole-stage atomicity is preserved, while
  // explicit Step() calls still release the lock between events.
  [[nodiscard]] NamedTensors FinishLocked() {
    if (completed) {
      return final_outputs;
    }
    EnsureActiveLocked();
    try {
      while (!completed) {
        options.cancellation.ThrowIfCancellationRequested();
        (void)StepLocked();
      }
      return final_outputs;
    } catch (...) {
      ReleaseSlotLocked();
      closed = true;
      throw;
    }
  }

  void Cancel() noexcept {
    try {
      std::scoped_lock lock(session->mutex);
      ReleaseSlotLocked();
      if (!completed) {
        closed = true;
      }
    } catch (...) {
      // A run that cannot take the lock keeps the slot; there is nothing
      // safe to do about it from a noexcept path.
    }
  }

  [[nodiscard]] bool Done() const noexcept {
    try {
      std::scoped_lock lock(session->mutex);
      return completed || closed;
    } catch (...) {
      return true;
    }
  }

  [[nodiscard]] std::size_t Iteration() const noexcept {
    try {
      std::scoped_lock lock(session->mutex);
      return emitted_steps;
    } catch (...) {
      return emitted_steps;
    }
  }

  [[nodiscard]] StageEvent StepLocked() {
    const PipelineStage& stage = FindStage(session->manifest(), stage_name);
    if (stage_kind == "autoregressive") {
      return AutoregressiveStepLocked(stage);
    }
    if (stage_kind == "iterative") {
      return IterativeStepLocked(stage);
    }
    return SinglePassStepLocked(stage);
  }

  [[nodiscard]] NamedTensors RunInputsLocked() {
    if (consumed_inputs) {
      return {};
    }
    consumed_inputs = true;
    return inputs;
  }

  [[nodiscard]] StageEvent AutoregressiveStepLocked(
      const PipelineStage& stage) {
    if (stop_requested || emitted_steps >= maximum_tokens) {
      return CompleteLocked();
    }
    NamedTensors step_outputs = session->StepStage(
        stage.name, RunInputsLocked(), overrides, options);
    // Sampling reads the whole logits tensor on the host, so it is bracketed
    // rather than checked inside its per-token loops.
    options.cancellation.ThrowIfCancellationRequested();
    Tensor tokens =
        do_sample
            ? session->SampleTokens(
                  session->StageLogits(stage), sampling, options, generated)
            : session->GreedyTokens(session->StageLogits(stage));
    options.cancellation.ThrowIfCancellationRequested();
    auto mutable_values = std::span(
        reinterpret_cast<std::int64_t*>(tokens.mutable_bytes().data()),
        tokens.element_count());
    if (finished_lanes.empty()) {
      finished_lanes.assign(mutable_values.size(), false);
    }
    const std::int64_t eos = eos_tokens.empty() ? 0 : *eos_tokens.begin();
    for (std::size_t batch_index = 0;
         batch_index < mutable_values.size();
         ++batch_index) {
      if (finished_lanes[batch_index]) {
        mutable_values[batch_index] = eos;
      } else if (eos_tokens.contains(mutable_values[batch_index])) {
        finished_lanes[batch_index] = true;
      }
    }
    const auto values = tokens.values<std::int64_t>();
    generated.emplace_back(values.begin(), values.end());
    session->SetStageTokenInput(stage, tokens);
    if (!eos_tokens.empty() &&
        std::ranges::all_of(
            finished_lanes, [](bool value) { return value; })) {
      // The old loop broke here; the incremental run reports the stop on the
      // next Step instead, so every run ends with one completed event.
      stop_requested = true;
    }

    StageEvent event;
    event.kind = StageEventKind::token;
    event.stage = stage.name;
    event.iteration = emitted_steps;
    event.token_ids = std::move(tokens);
    event.outputs = std::move(step_outputs);
    ++emitted_steps;
    return event;
  }

  [[nodiscard]] StageEvent IterativeStepLocked(const PipelineStage& stage) {
    if (session->stage_iterations[stage.name] >= target_iterations) {
      return CompleteLocked();
    }
    NamedTensors step_outputs = session->StepStage(
        stage.name, RunInputsLocked(), overrides, options);
    last_step_outputs = step_outputs;

    StageEvent event;
    event.kind = StageEventKind::iteration;
    event.stage = stage.name;
    event.iteration = emitted_steps;
    event.outputs = std::move(step_outputs);
    ++emitted_steps;
    return event;
  }

  // single_pass, state_transition, composite, and on_demand all execute as
  // exactly one pass, which is what RunStage did for them.
  [[nodiscard]] StageEvent SinglePassStepLocked(const PipelineStage& stage) {
    if (emitted_steps > 0) {
      return CompleteLocked();
    }
    NamedTensors step_outputs = session->StepStage(
        stage.name, RunInputsLocked(), overrides, options);
    last_step_outputs = step_outputs;

    StageEvent event;
    event.kind = StageEventKind::transition;
    event.stage = stage.name;
    event.iteration = emitted_steps;
    event.outputs = std::move(step_outputs);
    ++emitted_steps;
    return event;
  }

  [[nodiscard]] StageEvent CompleteLocked() {
    if (stage_kind == "autoregressive") {
      final_outputs = AutoregressiveResultLocked();
    } else if (last_step_outputs.has_value()) {
      // Exactly what the last executed step returned, which is what RunStage
      // returned for this stage.
      final_outputs = *last_step_outputs;
    } else {
      // A stage that had no work left executed nothing, so its result is the
      // session's current public outputs.
      final_outputs = session->CollectOutputs();
    }
    completed = true;
    ReleaseSlotLocked();

    StageEvent event;
    event.kind = StageEventKind::completed;
    event.stage = stage_name;
    event.iteration = emitted_steps;
    event.outputs = final_outputs;
    event.finished = true;
    return event;
  }

  // Packs the generated history batch-major, the layout the previous
  // RunAutoregressive published, and adds it without displacing a manifest
  // output of the same name.
  [[nodiscard]] NamedTensors AutoregressiveResultLocked() const {
    NamedTensors result = session->CollectOutputs();
    if (generated.empty()) {
      return result;
    }
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
    return result;
  }
};

StageRun::StageRun(std::unique_ptr<Impl> state) noexcept
    : impl_(std::move(state)) {}

StageRun::StageRun(StageRun&&) noexcept = default;

StageRun& StageRun::operator=(StageRun&& other) noexcept {
  if (this != &other) {
    Cancel();
    impl_ = std::move(other.impl_);
  }
  return *this;
}

StageRun::~StageRun() {
  Cancel();
}

namespace {

[[noreturn]] void ClosedRunError() {
  throw Error(
      ErrorCode::state,
      "This pipeline stage run owns no run; it was moved from");
}

}  // namespace

std::string_view StageRun::stage() const {
  if (impl_ == nullptr) {
    ClosedRunError();
  }
  return impl_->stage_name;
}

bool StageRun::done() const noexcept {
  return impl_ == nullptr || impl_->Done();
}

std::size_t StageRun::iteration() const {
  if (impl_ == nullptr) {
    ClosedRunError();
  }
  return impl_->Iteration();
}

StageEvent StageRun::Step() {
  if (impl_ == nullptr) {
    ClosedRunError();
  }
  return impl_->Step();
}

NamedTensors StageRun::Finish() {
  if (impl_ == nullptr) {
    ClosedRunError();
  }
  return impl_->Finish();
}

void StageRun::RequestCancellation() noexcept {
  if (impl_ != nullptr) {
    impl_->RequestCancellation();
  }
}

void StageRun::Cancel() noexcept {
  if (impl_ != nullptr) {
    impl_->Cancel();
  }
}

Pipeline::Pipeline(
    PipelinePackage package,
    PipelineSchedulingOptions scheduling)
    : package_(std::make_shared<const PipelinePackage>(std::move(package))),
      scheduler_(detail::MakePipelineScheduler(scheduling)) {}

Pipeline Pipeline::Load(
    const std::filesystem::path& directory,
    const RuntimeOptions& options,
    const PipelineSchedulingOptions& scheduling,
    const PipelinePlacementOptions& placement) {
  // Checked before a single component file is opened, so a misspelled stage
  // kind fails immediately rather than after a full package load. Placement is
  // checked the same way, inside PipelinePackage::Load, once the manifest
  // exists to check the component names against.
  detail::ValidatePipelineSchedulingOptions(scheduling);
  return Pipeline(
      PipelinePackage::Load(directory, options, placement), scheduling);
}

const PipelineManifest& Pipeline::manifest() const noexcept {
  return package_->manifest();
}

std::unordered_map<std::string, std::vector<std::string>>
Pipeline::execution_providers() const {
  return package_->execution_providers();
}

const PipelineTransferPlan& Pipeline::transfer_plan() const noexcept {
  return package_->transfer_plan();
}

PipelineSession Pipeline::CreateSession() const {
  return PipelineSession(package_, scheduler_);
}

PipelineSchedulingStats Pipeline::scheduling_stats() const {
  return detail::SnapshotSchedulingStats(scheduler_);
}

PipelineSession::PipelineSession(
    std::shared_ptr<const PipelinePackage> package,
    std::shared_ptr<detail::PipelineScheduler> scheduler)
    : impl_(std::make_shared<Impl>(
          std::move(package),
          std::move(scheduler))) {}

PipelineSession::PipelineSession(PipelineSession&&) noexcept = default;

PipelineSession& PipelineSession::operator=(PipelineSession&&) noexcept =
    default;

PipelineSession::~PipelineSession() = default;

// Full and incremental execution share the same StageRun state machine. A full
// run keeps the session lock for the complete drain, preserving the historical
// behavior that concurrent calls on one PipelineSession serialize and readers
// observe only the pre-stage or final state. It is also one execution, so it
// takes exactly one admission permit for the whole stage, acquired before the
// session lock in the documented order.
NamedTensors PipelineSession::RunStage(
    std::string_view stage,
    const NamedTensors& inputs,
    const NamedTensors& overrides,
    const PipelineRunOptions& options) {
  const detail::PipelineLease lease =
      impl_->AcquireStageLease(stage, options.cancellation);
  std::scoped_lock lock(impl_->mutex);
  impl_->EnsureNoActiveRunLocked("run a stage");
  auto run = StageRun::Impl::Begin(
      impl_, stage, inputs, overrides, options);
  return run->FinishLocked();
}

StageRun PipelineSession::BeginStage(
    std::string_view stage,
    const NamedTensors& inputs,
    const NamedTensors& overrides,
    const PipelineRunOptions& options) {
  std::scoped_lock lock(impl_->mutex);
  impl_->EnsureNoActiveRunLocked("begin a stage run");
  return StageRun(
      StageRun::Impl::Begin(impl_, stage, inputs, overrides, options));
}

NamedTensors PipelineSession::StepStage(
    std::string_view stage,
    const NamedTensors& inputs,
    const NamedTensors& overrides,
    const PipelineRunOptions& options) {
  // One direct step is one execution, admitted like any other.
  const detail::PipelineLease lease =
      impl_->AcquireStageLease(stage, options.cancellation);
  std::scoped_lock lock(impl_->mutex);
  impl_->EnsureNoActiveRunLocked("step a stage directly");
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
  impl_->EnsureNoActiveRunLocked("release a stage");
  const PipelineStage& released = FindStage(impl_->manifest(), stage);
  const Json guidance = impl_->StageGuidance(released.name);
  if (!guidance.empty()) {
    for (const auto& name : impl_->GuidanceInputNames(guidance)) {
      impl_->guidance_values.erase(name);
    }
  }
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
  impl_->EnsureNoActiveRunLocked("reset");
  impl_->external_values.clear();
  impl_->endpoint_values.clear();
  impl_->state_values.clear();
  impl_->guidance_values.clear();
  impl_->stage_iterations.clear();
  impl_->scheduler_histories.clear();
  impl_->position_cursors.clear();
  // Named checkpoints are session-scoped control metadata, so a reset session
  // starts with an empty checkpoint namespace as well as empty state.
  impl_->checkpoints.clear();
}

PipelineSessionSnapshot::PipelineSessionSnapshot(
    std::shared_ptr<const Impl> state)
    : impl_(std::move(state)) {}

bool PipelineSessionSnapshot::valid() const noexcept {
  return impl_ != nullptr;
}

PipelineSessionSnapshot PipelineSession::Snapshot() const {
  std::scoped_lock lock(impl_->mutex);
  // An active run keeps its decode or scheduler loop state outside
  // SessionState, so capturing here would hand back a snapshot that silently
  // drops it. Refuse instead.
  impl_->EnsureNoActiveRunLocked("capture a snapshot");
  return impl_->CaptureLocked();
}

void PipelineSession::Restore(const PipelineSessionSnapshot& snapshot) {
  if (!snapshot.valid()) {
    throw Error(
        ErrorCode::state,
        "Pipeline session snapshot was moved from and holds no state");
  }
  const PipelineSessionSnapshot::Impl& captured = *snapshot.impl_;
  // impl_->package is fixed when the session is constructed and is never
  // reassigned, so identity can be checked before the session lock is taken.
  if (captured.package != impl_->package) {
    throw Error(
        ErrorCode::state,
        "Pipeline session snapshot belongs to a different pipeline package");
  }
  // Every allocation happens before the session lock is taken and the commit
  // below only swaps, so a failure here leaves the session exactly as it was.
  SessionState restored = captured.state;
  std::scoped_lock lock(impl_->mutex);
  impl_->EnsureNoActiveRunLocked("restore a snapshot");
  impl_->Swap(restored);
}

PipelineSession PipelineSession::Fork() const {
  {
    std::scoped_lock lock(impl_->mutex);
    impl_->EnsureNoActiveRunLocked("fork");
  }
  const PipelineSessionSnapshot snapshot = Snapshot();
  // A fork belongs to the same Pipeline, so it competes for the same permits
  // rather than escaping the configured ceiling.
  PipelineSession forked(snapshot.impl_->package, impl_->admission_scheduler);
  forked.Restore(snapshot);
  // The fork deliberately keeps its freshly constructed, empty checkpoint
  // map: a snapshot carries execution state, and checkpoint names are the
  // property of the session that declared them.
  return forked;
}

void PipelineSession::Checkpoint(std::string_view name) {
  std::string key = CheckpointKey(name);
  // Capture and publish happen under one lock hold, so the stored checkpoint
  // is exactly the state observable at the instant the lock was acquired, and
  // no concurrent RunStage can slip between the two. Snapshot() is never
  // called here, so the session lock is taken exactly once.
  std::scoped_lock lock(impl_->mutex);
  impl_->EnsureNoActiveRunLocked("store a checkpoint");
  impl_->checkpoints.insert_or_assign(std::move(key), impl_->CaptureLocked());
}

void PipelineSession::RestoreCheckpoint(std::string_view name) {
  const std::string key = CheckpointKey(name);
  std::scoped_lock lock(impl_->mutex);
  impl_->EnsureNoActiveRunLocked("restore a checkpoint");
  const auto found = impl_->checkpoints.find(key);
  if (found == impl_->checkpoints.end()) {
    throw Error(
        ErrorCode::state,
        "Pipeline session has no checkpoint named '" + key + "'");
  }
  // Copy before swapping so allocation failure leaves the session untouched.
  // Lookup, copy, and commit share one lock acquisition, making restore
  // linearizable with checkpoint replacement, dropping, reset, and stage runs.
  SessionState restored = found->second.impl_->state;
  impl_->Swap(restored);
}

void PipelineSession::DropCheckpoint(std::string_view name) {
  const std::string key = CheckpointKey(name);
  std::scoped_lock lock(impl_->mutex);
  impl_->EnsureNoActiveRunLocked("drop a checkpoint");
  // Dropping an unknown checkpoint is a caller error, not a silent no-op.
  if (impl_->checkpoints.erase(key) == 0) {
    throw Error(
        ErrorCode::state,
        "Pipeline session has no checkpoint named '" + key + "'");
  }
}

bool PipelineSession::HasCheckpoint(std::string_view name) const {
  const std::string key = CheckpointKey(name);
  std::scoped_lock lock(impl_->mutex);
  return impl_->checkpoints.contains(key);
}

}  // namespace onnx_world_model
