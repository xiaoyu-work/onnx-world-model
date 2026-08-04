#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "onnx_world_model/onnx_world_model.hpp"

namespace {

using onnx_world_model::DataType;
using onnx_world_model::ModelMetadata;
using onnx_world_model::RuntimeOptions;
using onnx_world_model::StepOutput;
using onnx_world_model::Tensor;
using onnx_world_model::TensorSpec;
using onnx_world_model::WorldModel;

struct Arguments {
  std::string command;
  std::filesystem::path model_path;
  std::filesystem::path ort_library_path;
  std::int64_t batch_size{1};
  int intra_op_threads{0};
  int inter_op_threads{0};
  std::optional<std::string> observation;
  std::optional<std::string> action;
  std::optional<std::string> state;
};

[[noreturn]] void UsageError(std::string_view message) {
  throw onnx_world_model::Error(
      onnx_world_model::ErrorCode::invalid_argument,
      std::string(message));
}

void PrintUsage(std::ostream& stream) {
  stream
      << "Usage:\n"
      << "  onnx-world-model inspect <model.onnx> [options]\n"
      << "  onnx-world-model step <model.onnx> --observation CSV --action CSV "
         "[options]\n\n"
      << "Options:\n"
      << "  --ort-library PATH       Path to onnxruntime shared library\n"
      << "  --batch N                Batch size for step (default: 1)\n"
      << "  --state CSV              Explicit state; defaults to zeros\n"
      << "  --intra-op-threads N     ORT intra-op thread count\n"
      << "  --inter-op-threads N     ORT inter-op thread count\n"
      << "  -h, --help               Show this help\n";
}

[[nodiscard]] std::string_view RequireValue(
    int& index,
    int argument_count,
    char** arguments,
    std::string_view option) {
  if (++index >= argument_count) {
    UsageError("Missing value for " + std::string(option));
  }
  return arguments[index];
}

template <typename Integer>
[[nodiscard]] Integer ParseInteger(
    std::string_view text,
    std::string_view option) {
  Integer value{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    UsageError("Invalid integer for " + std::string(option) + ": " +
               std::string(text));
  }
  return value;
}

[[nodiscard]] Arguments ParseArguments(
    int argument_count,
    char** arguments) {
  if (argument_count < 2) {
    PrintUsage(std::cerr);
    UsageError("Missing command");
  }
  if (std::string_view(arguments[1]) == "-h" ||
      std::string_view(arguments[1]) == "--help") {
    PrintUsage(std::cout);
    std::exit(0);
  }
  if (argument_count < 3) {
    UsageError("Missing model path");
  }
  if (std::string_view(arguments[2]) == "-h" ||
      std::string_view(arguments[2]) == "--help") {
    PrintUsage(std::cout);
    std::exit(0);
  }

  Arguments parsed{
      .command = arguments[1],
      .model_path = arguments[2],
  };
  if (parsed.command != "inspect" && parsed.command != "step") {
    UsageError("Unknown command: " + parsed.command);
  }

  for (int index = 3; index < argument_count; ++index) {
    const std::string_view option = arguments[index];
    if (option == "-h" || option == "--help") {
      PrintUsage(std::cout);
      std::exit(0);
    }
    if (option == "--ort-library") {
      parsed.ort_library_path =
          RequireValue(index, argument_count, arguments, option);
    } else if (option == "--batch") {
      parsed.batch_size = ParseInteger<std::int64_t>(
          RequireValue(index, argument_count, arguments, option),
          option);
    } else if (option == "--intra-op-threads") {
      parsed.intra_op_threads = ParseInteger<int>(
          RequireValue(index, argument_count, arguments, option),
          option);
    } else if (option == "--inter-op-threads") {
      parsed.inter_op_threads = ParseInteger<int>(
          RequireValue(index, argument_count, arguments, option),
          option);
    } else if (option == "--observation") {
      parsed.observation =
          RequireValue(index, argument_count, arguments, option);
    } else if (option == "--action") {
      parsed.action = RequireValue(index, argument_count, arguments, option);
    } else if (option == "--state") {
      parsed.state = RequireValue(index, argument_count, arguments, option);
    } else {
      UsageError("Unknown option: " + std::string(option));
    }
  }

  if (parsed.batch_size <= 0) {
    UsageError("--batch must be positive");
  }
  if (parsed.command == "step" &&
      (!parsed.observation.has_value() || !parsed.action.has_value())) {
    UsageError("step requires --observation and --action");
  }
  return parsed;
}

[[nodiscard]] std::vector<float> ParseFloatValues(std::string_view text) {
  std::vector<float> values;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::size_t end =
        comma == std::string_view::npos ? text.size() : comma;
    std::string_view token = text.substr(begin, end - begin);
    const std::size_t first = token.find_first_not_of(" \t");
    const std::size_t last = token.find_last_not_of(" \t");
    if (first == std::string_view::npos) {
      UsageError("Tensor values cannot contain empty entries");
    }
    token = token.substr(first, last - first + 1);

    float value{};
    const auto [parsed_end, error] =
        std::from_chars(token.data(), token.data() + token.size(), value);
    if (error != std::errc{} || parsed_end != token.data() + token.size()) {
      UsageError("Invalid floating-point tensor value: " + std::string(token));
    }
    values.push_back(value);
    if (comma == std::string_view::npos) {
      break;
    }
    begin = comma + 1;
  }
  return values;
}

[[nodiscard]] std::vector<std::int64_t> ConcreteShape(
    const TensorSpec& spec,
    std::int64_t batch_size) {
  if (spec.shape.empty()) {
    UsageError("Tensor '" + spec.name + "' does not have a batch dimension");
  }
  std::vector<std::int64_t> shape = spec.shape;
  if (shape[0] >= 0 && shape[0] != batch_size) {
    UsageError(
        "Tensor '" + spec.name + "' has fixed batch " +
        std::to_string(shape[0]) + ", not " + std::to_string(batch_size));
  }
  shape[0] = batch_size;
  for (std::size_t axis = 1; axis < shape.size(); ++axis) {
    if (shape[axis] < 0) {
      UsageError(
          "CLI cannot infer dynamic non-batch dimension " +
          std::to_string(axis) + " for tensor '" + spec.name + "'");
    }
  }
  return shape;
}

[[nodiscard]] Tensor FloatTensorFromCsv(
    const TensorSpec& spec,
    std::int64_t batch_size,
    std::string_view text) {
  if (spec.data_type != DataType::float32) {
    UsageError(
        "CLI CSV input currently supports float32 models; tensor '" +
        spec.name + "' uses " + std::string(onnx_world_model::ToString(spec.data_type)));
  }
  auto shape = ConcreteShape(spec, batch_size);
  const std::vector<float> values = ParseFloatValues(text);
  std::size_t expected_count = 1;
  for (const std::int64_t dimension : shape) {
    expected_count *= static_cast<std::size_t>(dimension);
  }
  if (values.size() != expected_count) {
    UsageError(
        "Tensor '" + spec.name + "' needs " +
        std::to_string(expected_count) + " values, got " +
        std::to_string(values.size()));
  }
  return Tensor::FromValues<float>(std::move(shape), std::span(values));
}

void PrintShape(std::ostream& stream, const std::vector<std::int64_t>& shape) {
  stream << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      stream << ", ";
    }
    stream << shape[index];
  }
  stream << ']';
}

void PrintSpec(std::string_view kind, const TensorSpec& spec) {
  std::cout << kind << ' ' << spec.name << " dtype="
            << onnx_world_model::ToString(spec.data_type) << " shape=";
  PrintShape(std::cout, spec.shape);
  std::cout << '\n';
}

void PrintMetadata(const ModelMetadata& metadata) {
  for (const auto& input : metadata.inputs) {
    PrintSpec("input ", input);
  }
  for (const auto& output : metadata.outputs) {
    PrintSpec("output", output);
  }
}

void PrintTensor(std::string_view name, const Tensor& tensor) {
  std::cout << name << " dtype=" << onnx_world_model::ToString(tensor.data_type())
            << " shape=";
  PrintShape(std::cout, tensor.shape());
  std::cout << " values=[";
  if (tensor.data_type() == DataType::float32) {
    const auto values = tensor.values<float>();
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                << values[index];
    }
  } else {
    std::cout << "<binary>";
  }
  std::cout << "]\n";
}

void PrintStepOutput(const StepOutput& output) {
  PrintTensor("next_state", output.next_state);
  PrintTensor("observation_prediction", output.observation_prediction);
  PrintTensor("reward", output.reward);
  PrintTensor("continuation", output.continuation);
}

}  // namespace

int main(int argument_count, char** arguments) {
  try {
    const Arguments parsed = ParseArguments(argument_count, arguments);
    RuntimeOptions options{
        .ort_library_path = parsed.ort_library_path,
        .intra_op_threads = parsed.intra_op_threads,
        .inter_op_threads = parsed.inter_op_threads,
    };
    WorldModel model = WorldModel::Load(parsed.model_path, options);
    if (parsed.command == "inspect") {
      PrintMetadata(model.metadata());
      return 0;
    }

    Tensor observation = FloatTensorFromCsv(
        model.metadata().Input("observation"),
        parsed.batch_size,
        *parsed.observation);
    Tensor action = FloatTensorFromCsv(
        model.metadata().Input("action"),
        parsed.batch_size,
        *parsed.action);
    onnx_world_model::Rollout rollout(model);
    if (parsed.state.has_value()) {
      rollout.Reset(FloatTensorFromCsv(
          model.metadata().Input("state"),
          parsed.batch_size,
          *parsed.state));
    }
    PrintStepOutput(rollout.Step(observation, action));
    return 0;
  } catch (const onnx_world_model::Error& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 1;
  }
}
