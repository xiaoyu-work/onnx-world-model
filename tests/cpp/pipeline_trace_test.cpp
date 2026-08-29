/**
 * @agent-file
 * @agent-purpose: Standalone test executable for per-run ONNX Runtime trace profiling driven by pipeline telemetry: the profile-file prefix a component call is handed, its uniqueness across concurrent calls, the one trace record each call publishes with its outcome and file, the profiling failures that leave a record with no path, the retention cap and its dropped counter, the epoch reset that clears records while leaving files on disk, and the configuration validation that rejects an unusable trace directory while a Pipeline is being built.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as pipeline_trace_test; it counts failures through local Check and CheckThrowsCode helpers and returns a non-zero exit code when any check fails. It includes public headers only -- it never reaches into src/pipeline_telemetry.hpp -- so every claim is made through Pipeline, PipelineSession, ModelRunOptions, and Pipeline::telemetry_snapshot(). Every component is a stub ModelBackend that overrides the ModelRunOptions overload, records the prefix it was handed, and writes the file ONNX Runtime would have written -- one small JSON document named <prefix>_stub.json -- so the run needs no ONNX Runtime library and no ONNX model while still exercising the real discovery rules. All files are written inside one scratch directory created beside the test executable's working directory and removed at the end, so nothing is left behind and nothing outside it is ever touched; a stale directory from an earlier failed run is removed before the checks start. No check asserts a wall-clock magnitude: a duration is only ever compared against zero, and no assertion depends on the timestamp ONNX Runtime would put in a file name, because the file is found by prefix rather than by predicting its name. The concurrency check parks its backends on a shared Gate -- a condition variable, never a sleep -- so every prefix is provably outstanding at the same moment, bounded by kBudget so a run that never enters fails a check instead of hanging CTest.
 * @agent-side-effects: Creates, writes, and removes files under one scratch directory in the current working directory, writes failure descriptions to stderr, and starts short-lived worker threads that block inside a stub backend.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx_world_model/cancellation.hpp"
#include "onnx_world_model/error.hpp"
#include "onnx_world_model/pipeline.hpp"
#include "onnx_world_model/tensor.hpp"

namespace {

using onnx_world_model::CancellationSource;
using onnx_world_model::CancellationToken;
using onnx_world_model::DataType;
using onnx_world_model::ErrorCode;
using onnx_world_model::Model;
using onnx_world_model::ModelMetadata;
using onnx_world_model::ModelRunOptions;
using onnx_world_model::NamedTensors;
using onnx_world_model::Pipeline;
using onnx_world_model::PipelineCallOutcome;
using onnx_world_model::PipelineManifest;
using onnx_world_model::PipelinePackage;
using onnx_world_model::PipelineRunOptions;
using onnx_world_model::PipelineSchedulingOptions;
using onnx_world_model::PipelineSession;
using onnx_world_model::PipelineTelemetryOptions;
using onnx_world_model::PipelineTelemetrySnapshot;
using onnx_world_model::PipelineTraceRecord;
using onnx_world_model::Tensor;

int failures = 0;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    ++failures;
  }
}

template <typename Function>
void CheckThrowsCode(
    Function&& function,
    ErrorCode expected,
    const char* message) {
  try {
    function();
    Check(false, message);
  } catch (const onnx_world_model::Error& error) {
    if (error.code() != expected) {
      std::cerr << "FAILED: " << message << " (got: " << error.what() << ")\n";
      ++failures;
    }
  }
}

// Long enough that ordinary scheduling noise cannot exhaust it, short enough
// that a run whose backends never enter fails quickly instead of hanging
// CTest.
constexpr std::chrono::milliseconds kBudget{5000};

// -- Scratch files ----------------------------------------------------------

// Every file this test writes lives under one directory beside the working
// directory CTest runs it in, so cleanup is exact: the whole tree is removed
// and nothing outside it is ever touched.
[[nodiscard]] std::filesystem::path ScratchRoot() {
  return std::filesystem::current_path() / "pipeline_trace_test_scratch";
}

[[nodiscard]] std::size_t CountJsonFiles(const std::filesystem::path& path) {
  std::error_code code;
  std::filesystem::directory_iterator entry(path, code);
  if (code) {
    return 0;
  }
  const std::filesystem::directory_iterator end;
  std::size_t total = 0;
  for (; entry != end; entry.increment(code)) {
    if (code) {
      return total;
    }
    if (entry->path().extension() == ".json") {
      ++total;
    }
  }
  return total;
}

// -- Stub backends ----------------------------------------------------------

// What a stub does with the prefix it is handed. `normal` is what ONNX
// Runtime does: exactly one file whose name is the prefix, a separator, and
// `.json`. The others are the three ways discovery can find no usable file.
enum class TraceBehavior {
  normal,
  no_file,
  two_files,
  empty_file,
};

// Every prefix the pipeline handed a component, in call order. It is the only
// way this test can see what ModelRunOptions carried, because a prefix is
// deliberately not part of any result.
struct PrefixLog {
  std::mutex mutex;
  std::vector<std::filesystem::path> prefixes;

  void Record(const std::filesystem::path& prefix) {
    std::scoped_lock lock(mutex);
    prefixes.push_back(prefix);
  }

  [[nodiscard]] std::vector<std::filesystem::path> Snapshot() {
    std::scoped_lock lock(mutex);
    return prefixes;
  }
};

void WriteTraceFile(const std::filesystem::path& prefix, bool empty) {
  // ONNX Runtime appends its own local timestamp to the prefix; this stub
  // appends a fixed suffix instead, because the file is discovered by prefix
  // and no assertion may depend on a timestamp.
  std::filesystem::path file = prefix;
  file += "_stub.json";
  std::ofstream stream(file, std::ios::binary | std::ios::trunc);
  if (!empty) {
    stream << R"([{"cat":"Node","pid":1,"tid":1,"ts":0,"dur":1,"ph":"X",)"
           << R"("name":"stub_node","args":{"op_name":"Add"}}])";
  }
}

// Returns its input unchanged, records the prefix it was given, and writes
// whatever file `behavior` describes. `failure` makes the very next call
// throw a chosen ErrorCode after the file is written, which is how a trace of
// a call that did not finish is asserted.
class TracingBackend final : public onnx_world_model::ModelBackend {
 public:
  TracingBackend(
      std::string input,
      std::string output,
      std::shared_ptr<PrefixLog> log,
      TraceBehavior behavior = TraceBehavior::normal,
      std::shared_ptr<std::optional<ErrorCode>> failure = nullptr,
      std::shared_ptr<struct Gate> gate = nullptr)
      : input_(std::move(input)),
        output_(std::move(output)),
        log_(std::move(log)),
        behavior_(behavior),
        failure_(std::move(failure)),
        gate_(std::move(gate)) {
    metadata_.inputs.push_back({
        .name = input_,
        .data_type = DataType::float32,
        .shape = {1, 4},
    });
    metadata_.outputs.push_back({
        .name = output_,
        .data_type = DataType::float32,
        .shape = {1, 4},
    });
  }

  [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
    return metadata_;
  }

  [[nodiscard]] NamedTensors Run(const NamedTensors& inputs) const override {
    return Run(inputs, ModelRunOptions{});
  }

  [[nodiscard]] NamedTensors Run(
      const NamedTensors& inputs,
      const ModelRunOptions& options) const override;

 private:
  ModelMetadata metadata_;
  std::string input_;
  std::string output_;
  std::shared_ptr<PrefixLog> log_;
  TraceBehavior behavior_;
  std::shared_ptr<std::optional<ErrorCode>> failure_;
  std::shared_ptr<struct Gate> gate_;
};

// The handshake the concurrency check goes through: a backend announces that
// it has been handed its prefix and then waits, so every prefix is provably
// outstanding at one moment rather than merely observed in sequence.
struct Gate {
  std::mutex mutex;
  std::condition_variable condition;
  std::size_t entered{0};
  bool open{false};
};

NamedTensors TracingBackend::Run(
    const NamedTensors& inputs,
    const ModelRunOptions& options) const {
  options.cancellation.ThrowIfCancellationRequested();
  log_->Record(options.profile_file_prefix);
  if (!options.profile_file_prefix.empty()) {
    switch (behavior_) {
      case TraceBehavior::normal:
        WriteTraceFile(options.profile_file_prefix, false);
        break;
      case TraceBehavior::no_file:
        break;
      case TraceBehavior::two_files: {
        WriteTraceFile(options.profile_file_prefix, false);
        std::filesystem::path second = options.profile_file_prefix;
        second += "_stub_extra.json";
        std::ofstream stream(second, std::ios::binary | std::ios::trunc);
        stream << "[]";
        break;
      }
      case TraceBehavior::empty_file:
        WriteTraceFile(options.profile_file_prefix, true);
        break;
    }
  }
  if (gate_ != nullptr) {
    std::unique_lock lock(gate_->mutex);
    ++gate_->entered;
    gate_->condition.notify_all();
    gate_->condition.wait(lock, [this] { return gate_->open; });
  }
  if (failure_ != nullptr && failure_->has_value()) {
    throw onnx_world_model::Error(
        **failure_, "Scripted component failure for the trace test");
  }
  return {{output_, inputs.at(input_)}};
}

[[nodiscard]] bool WaitForEntries(Gate& gate, std::size_t count) {
  std::unique_lock lock(gate.mutex);
  return gate.condition.wait_for(lock, kBudget, [&gate, count] {
    return gate.entered >= count;
  });
}

void OpenGate(Gate& gate) {
  {
    std::scoped_lock lock(gate.mutex);
    gate.open = true;
  }
  gate.condition.notify_all();
}

// -- Manifests --------------------------------------------------------------

// Two components in one stage, the second deliberately named with a space and
// a plus sign: the manifest accepts both, and neither may reach a file name
// unchanged.
constexpr std::string_view kTwoComponentManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "producer",
        "role": "encoder",
        "run_on": "always",
        "inputs": [{"name": "x", "dtype": "FLOAT", "shape": [1, 4]}],
        "outputs": [{"name": "y", "dtype": "FLOAT", "shape": [1, 4]}]
      },
      {
        "name": "vision encoder+1",
        "role": "decoder",
        "run_on": "always",
        "inputs": [{"name": "z", "dtype": "FLOAT", "shape": [1, 4]}],
        "outputs": [{"name": "w", "dtype": "FLOAT", "shape": [1, 4]}]
      }
    ],
    "connections": [
      {"source": "producer.y", "target": "vision encoder+1.z"}
    ],
    "stages": [
      {
        "name": "pass",
        "kind": "single_pass",
        "components": ["producer", "vision encoder+1"],
        "run_on": "always"
      }
    ],
    "inputs": [{"port": "producer.x", "kind": "external", "required": true}],
    "outputs": [{"port": "vision encoder+1.w", "alias": "produced"}]
  },
  "component_files": {
    "producer": "producer/model.onnx",
    "vision encoder+1": "vision encoder+1/model.onnx"
  }
}
)json";

// One component in one stage: everything about counting, capping, failing,
// and resetting is clearer with exactly one call per execution.
constexpr std::string_view kSingleComponentManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "producer",
        "role": "encoder",
        "run_on": "always",
        "inputs": [{"name": "x", "dtype": "FLOAT", "shape": [1, 4]}],
        "outputs": [{"name": "y", "dtype": "FLOAT", "shape": [1, 4]}]
      }
    ],
    "connections": [],
    "stages": [
      {
        "name": "pass",
        "kind": "single_pass",
        "components": ["producer"],
        "run_on": "always"
      }
    ],
    "inputs": [{"port": "producer.x", "kind": "external", "required": true}],
    "outputs": [{"port": "producer.y", "alias": "produced"}]
  },
  "component_files": {"producer": "model.onnx"}
}
)json";

// -- Pipeline helpers -------------------------------------------------------

[[nodiscard]] PipelineTelemetryOptions Telemetry(
    bool enabled,
    std::filesystem::path directory = {},
    std::size_t max_records = 256) {
  PipelineTelemetryOptions options;
  options.enabled = enabled;
  options.trace_directory = std::move(directory);
  options.max_trace_records = max_records;
  return options;
}

[[nodiscard]] Pipeline MakeTwoComponentPipeline(
    const std::shared_ptr<PrefixLog>& log,
    PipelineTelemetryOptions telemetry) {
  std::unordered_map<std::string, Model> models;
  models.emplace(
      "producer", Model(std::make_shared<TracingBackend>("x", "y", log)));
  models.emplace(
      "vision encoder+1",
      Model(std::make_shared<TracingBackend>("z", "w", log)));
  return Pipeline(
      PipelinePackage(
          {}, PipelineManifest::Parse(kTwoComponentManifest), std::move(models)),
      PipelineSchedulingOptions{},
      std::move(telemetry));
}

[[nodiscard]] Pipeline MakeSingleComponentPipeline(
    const std::shared_ptr<PrefixLog>& log,
    PipelineTelemetryOptions telemetry,
    TraceBehavior behavior = TraceBehavior::normal,
    const std::shared_ptr<std::optional<ErrorCode>>& failure = nullptr,
    const std::shared_ptr<Gate>& gate = nullptr) {
  std::unordered_map<std::string, Model> models;
  models.emplace(
      "producer",
      Model(std::make_shared<TracingBackend>(
          "x", "y", log, behavior, failure, gate)));
  return Pipeline(
      PipelinePackage(
          {},
          PipelineManifest::Parse(kSingleComponentManifest),
          std::move(models)),
      PipelineSchedulingOptions{},
      std::move(telemetry));
}

[[nodiscard]] NamedTensors PassInputs() {
  static constexpr std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
  NamedTensors inputs;
  inputs.emplace(
      "producer.x", Tensor::FromValues<float>({1, 4}, std::span(values)));
  return inputs;
}

[[nodiscard]] std::vector<PipelineTraceRecord> TracesFor(
    const Pipeline& pipeline,
    std::string_view component) {
  std::vector<PipelineTraceRecord> matching;
  for (const PipelineTraceRecord& record :
       pipeline.telemetry_snapshot().traces) {
    if (record.component == component) {
      matching.push_back(record);
    }
  }
  return matching;
}

// -- Checks -----------------------------------------------------------------

void RunTraceChecks(const std::filesystem::path& root) {
  {
    // A pipeline that collects counters only asks for no trace at all, which
    // is what keeps the counter-only configuration exactly as cheap as it was
    // before tracing existed.
    const auto log = std::make_shared<PrefixLog>();
    const Pipeline pipeline =
        MakeSingleComponentPipeline(log, Telemetry(true));
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("pass", PassInputs());

    const std::vector<std::filesystem::path> prefixes = log->Snapshot();
    const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
    Check(prefixes.size() == 1, "a counters-only pipeline still runs the call");
    Check(
        !prefixes.empty() && prefixes.front().empty(),
        "a counters-only pipeline hands the component an empty prefix");
    Check(
        snapshot.traces.empty() && snapshot.dropped_traces == 0 &&
            snapshot.failed_traces == 0,
        "a counters-only pipeline records no traces");
    Check(
        snapshot.components.at("producer").successful_calls == 1,
        "a counters-only pipeline still counts the call it measured");
  }

  {
    // Telemetry disabled is the other half of the same claim: no collector at
    // all means no prefix either.
    const auto log = std::make_shared<PrefixLog>();
    const Pipeline pipeline =
        MakeSingleComponentPipeline(log, Telemetry(false));
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("pass", PassInputs());

    const std::vector<std::filesystem::path> prefixes = log->Snapshot();
    Check(
        prefixes.size() == 1 && prefixes.front().empty(),
        "a pipeline without telemetry hands the component an empty prefix");
    Check(
        pipeline.telemetry_snapshot().traces.empty(),
        "a pipeline without telemetry reports no traces");
  }

  {
    // The shape of a prefix and of the record it produces, including the
    // sanitized component name a manifest is allowed to contain.
    const std::filesystem::path directory = root / "shape";
    const auto log = std::make_shared<PrefixLog>();
    const Pipeline pipeline =
        MakeTwoComponentPipeline(log, Telemetry(true, directory));
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("pass", PassInputs());

    const std::vector<std::filesystem::path> prefixes = log->Snapshot();
    Check(prefixes.size() == 2, "each component call is handed its own prefix");
    for (const std::filesystem::path& prefix : prefixes) {
      Check(
          !prefix.empty() && prefix.parent_path() == directory,
          "every prefix is inside the configured trace directory");
    }
    const bool sanitized = std::ranges::any_of(
        prefixes, [](const std::filesystem::path& prefix) {
          return prefix.filename().string().starts_with("vision_encoder_1.e");
        });
    const bool plain = std::ranges::any_of(
        prefixes, [](const std::filesystem::path& prefix) {
          return prefix.filename().string().starts_with("producer.e");
        });
    Check(sanitized, "a component name is sanitized into the file name");
    Check(plain, "an already portable component name is used unchanged");
    Check(
        prefixes.size() == 2 && prefixes[0] != prefixes[1],
        "two calls never share a prefix");

    const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
    Check(
        snapshot.traces.size() == 2,
        "one traced call publishes exactly one record");
    Check(
        snapshot.dropped_traces == 0 && snapshot.failed_traces == 0,
        "a trace that was found is neither dropped nor failed");
    for (const PipelineTraceRecord& record : snapshot.traces) {
      Check(record.epoch == 1, "a record carries the epoch it was made in");
      Check(record.trace_id > 0, "a record carries its trace identifier");
      Check(
          record.outcome == PipelineCallOutcome::success,
          "a successful call records a successful outcome");
      Check(!record.profiling_failed, "a discovered trace has not failed");
      Check(!record.path.empty(), "a discovered trace records its file");
      Check(
          record.path.is_absolute(),
          "a discovered trace records an absolute file");
      Check(
          std::filesystem::exists(record.path),
          "the recorded file is the one on disk");
      Check(record.size_bytes > 0, "a discovered trace records its size");
      Check(record.duration_ns > 0, "a record carries the call's duration");
    }
    const std::vector<PipelineTraceRecord> sanitized_records =
        TracesFor(pipeline, "vision encoder+1");
    Check(
        sanitized_records.size() == 1,
        "a record names its component exactly as the manifest does");
    Check(
        snapshot.traces.front().trace_id < snapshot.traces.back().trace_id,
        "trace identifiers increase with start order");
  }

  {
    // Concurrency: four calls are handed their prefixes and then all wait, so
    // every prefix is outstanding at the same moment and no two may collide.
    constexpr std::size_t kWorkers = 4;
    const std::filesystem::path directory = root / "concurrent";
    const auto log = std::make_shared<PrefixLog>();
    const auto gate = std::make_shared<Gate>();
    const Pipeline pipeline = MakeSingleComponentPipeline(
        log,
        Telemetry(true, directory),
        TraceBehavior::normal,
        nullptr,
        gate);

    std::vector<std::thread> workers;
    workers.reserve(kWorkers);
    for (std::size_t index = 0; index < kWorkers; ++index) {
      workers.emplace_back([&pipeline] {
        PipelineSession session = pipeline.CreateSession();
        (void)session.RunStage("pass", PassInputs());
      });
    }
    Check(
        WaitForEntries(*gate, kWorkers),
        "every concurrent call reaches its component");
    const std::vector<std::filesystem::path> outstanding = log->Snapshot();
    OpenGate(*gate);
    for (std::thread& worker : workers) {
      worker.join();
    }

    std::unordered_set<std::string> unique;
    for (const std::filesystem::path& prefix : outstanding) {
      unique.insert(prefix.string());
    }
    Check(
        outstanding.size() == kWorkers && unique.size() == kWorkers,
        "concurrent calls never share a prefix");
    const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
    Check(
        snapshot.traces.size() == kWorkers,
        "every concurrent call publishes its own record");
    Check(
        snapshot.failed_traces == 0,
        "concurrent calls each find their own file");
    Check(
        CountJsonFiles(directory) == kWorkers,
        "every concurrent call wrote its own file");
  }

  {
    // A call that did not finish is still traced, and its record says how it
    // ended -- classified by ErrorCode, exactly as the counters are.
    const std::array<std::pair<ErrorCode, PipelineCallOutcome>, 3> cases{{
        {ErrorCode::runtime_execution, PipelineCallOutcome::failure},
        {ErrorCode::cancelled, PipelineCallOutcome::cancelled},
        {ErrorCode::deadline_exceeded, PipelineCallOutcome::deadline_exceeded},
    }};
    for (const auto& [code, outcome] : cases) {
      const std::filesystem::path directory =
          root / ("outcome" + std::to_string(static_cast<int>(code)));
      const auto log = std::make_shared<PrefixLog>();
      const auto failure =
          std::make_shared<std::optional<ErrorCode>>(std::optional{code});
      const Pipeline pipeline = MakeSingleComponentPipeline(
          log, Telemetry(true, directory), TraceBehavior::normal, failure);
      PipelineSession session = pipeline.CreateSession();
      CheckThrowsCode(
          [&session] { (void)session.RunStage("pass", PassInputs()); },
          code,
          "a scripted component failure surfaces with its own code");

      const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
      Check(
          snapshot.traces.size() == 1,
          "a call that threw still publishes exactly one record");
      Check(
          !snapshot.traces.empty() && snapshot.traces.front().outcome == outcome,
          "a record classifies its outcome by ErrorCode");
      Check(
          !snapshot.traces.empty() &&
              !snapshot.traces.front().profiling_failed &&
              !snapshot.traces.front().path.empty(),
          "a trace written before the failure is still found");
    }
  }

  {
    // A pre-cancelled run never reaches the component, so there is nothing to
    // trace: the absence of a record is the claim, not a failed one.
    const std::filesystem::path directory = root / "precancelled";
    const auto log = std::make_shared<PrefixLog>();
    const Pipeline pipeline =
        MakeSingleComponentPipeline(log, Telemetry(true, directory));
    CancellationSource source;
    source.Cancel();
    PipelineRunOptions options;
    options.cancellation = source.token();
    PipelineSession session = pipeline.CreateSession();
    CheckThrowsCode(
        [&session, &options] {
          (void)session.RunStage("pass", PassInputs(), {}, options);
        },
        ErrorCode::cancelled,
        "an already cancelled run is refused");

    const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
    Check(
        log->Snapshot().empty(),
        "an already cancelled run never calls the component");
    Check(
        snapshot.traces.empty() && snapshot.failed_traces == 0,
        "a call that never happened is neither traced nor a trace failure");
    Check(
        CountJsonFiles(directory) == 0,
        "an already cancelled run writes no trace file");
  }

  {
    // The three ways discovery can fail. Each one is a record with no path
    // and a counted failure, and none of them changes the call's own outcome.
    const std::array<std::pair<TraceBehavior, const char*>, 3> cases{{
        {TraceBehavior::no_file, "nofile"},
        {TraceBehavior::two_files, "twofiles"},
        {TraceBehavior::empty_file, "emptyfile"},
    }};
    for (const auto& [behavior, name] : cases) {
      const std::filesystem::path directory = root / name;
      const auto log = std::make_shared<PrefixLog>();
      const Pipeline pipeline = MakeSingleComponentPipeline(
          log, Telemetry(true, directory), behavior);
      PipelineSession session = pipeline.CreateSession();
      (void)session.RunStage("pass", PassInputs());

      const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
      Check(
          snapshot.failed_traces == 1,
          "an unusable trace file is counted as a profiling failure");
      Check(
          snapshot.traces.size() == 1,
          "a profiling failure is still recorded when there is room");
      Check(
          !snapshot.traces.empty() && snapshot.traces.front().profiling_failed &&
              snapshot.traces.front().path.empty() &&
              snapshot.traces.front().size_bytes == 0,
          "a failed trace records no file and no size");
      Check(
          !snapshot.traces.empty() &&
              snapshot.traces.front().outcome == PipelineCallOutcome::success,
          "a profiling failure never changes the call's own outcome");
      Check(
          snapshot.components.at("producer").successful_calls == 1,
          "a profiling failure never fails the model call");
    }
  }

  {
    // Retention is a memory bound: the cap stops records, never files.
    const std::filesystem::path directory = root / "capped";
    const auto log = std::make_shared<PrefixLog>();
    const Pipeline pipeline =
        MakeSingleComponentPipeline(log, Telemetry(true, directory, 2));
    PipelineSession session = pipeline.CreateSession();
    for (int index = 0; index < 3; ++index) {
      (void)session.RunStage("pass", PassInputs());
    }

    const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
    Check(
        snapshot.traces.size() == 2,
        "a full epoch keeps exactly max_trace_records records");
    Check(snapshot.dropped_traces == 1, "records past the cap are counted");
    Check(
        snapshot.failed_traces == 0,
        "a dropped record is not a profiling failure");
    Check(
        CountJsonFiles(directory) == 3,
        "a dropped record still has its file on disk");
    Check(
        snapshot.traces.size() == 2 &&
            snapshot.traces[0].trace_id < snapshot.traces[1].trace_id,
        "the kept records are the first ones, in order");
  }

  {
    // A reset is a new epoch, not a cleanup: the records go and the files
    // stay, because this runtime never deletes one.
    const std::filesystem::path directory = root / "reset";
    const auto log = std::make_shared<PrefixLog>();
    Pipeline pipeline =
        MakeSingleComponentPipeline(log, Telemetry(true, directory));
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("pass", PassInputs());
    const std::size_t before = CountJsonFiles(directory);
    Check(before == 1, "the first run wrote its file");

    pipeline.ResetTelemetry();
    const PipelineTelemetrySnapshot snapshot = pipeline.telemetry_snapshot();
    Check(snapshot.epoch == 2, "a reset publishes the next epoch");
    Check(
        snapshot.traces.empty() && snapshot.dropped_traces == 0 &&
            snapshot.failed_traces == 0,
        "a reset clears this epoch's records and counters");
    Check(
        CountJsonFiles(directory) == before,
        "a reset never deletes a trace file");

    (void)session.RunStage("pass", PassInputs());
    const PipelineTelemetrySnapshot after = pipeline.telemetry_snapshot();
    Check(
        after.traces.size() == 1 && after.traces.front().epoch == 2,
        "the next call records into the new epoch");
    Check(
        !after.traces.empty() && after.traces.front().trace_id > 1,
        "trace identifiers keep increasing across a reset");
    Check(
        CountJsonFiles(directory) == 2,
        "the new epoch writes beside the files the old one left");
  }

  {
    // Configuration is checked while the Pipeline is built, so an unusable
    // trace directory can never become a per-run failure.
    const auto log = std::make_shared<PrefixLog>();
    CheckThrowsCode(
        [&log, &root] {
          (void)MakeSingleComponentPipeline(
              log, Telemetry(false, root / "rejected"));
        },
        ErrorCode::invalid_argument,
        "a trace directory without telemetry is rejected");
    CheckThrowsCode(
        [&log, &root] {
          (void)MakeSingleComponentPipeline(
              log, Telemetry(true, root / "rejected", 0));
        },
        ErrorCode::invalid_argument,
        "a zero retention cap is rejected while tracing");
    Check(
        !std::filesystem::exists(root / "rejected"),
        "a rejected configuration creates no directory");

    const std::filesystem::path file = root / "occupied";
    {
      std::ofstream stream(file, std::ios::binary | std::ios::trunc);
      stream << "not a directory";
    }
    CheckThrowsCode(
        [&log, &file] {
          (void)MakeSingleComponentPipeline(log, Telemetry(true, file));
        },
        ErrorCode::invalid_argument,
        "a trace directory that is a file is rejected");
  }

  {
    // The directory is created on the cold path, so a caller may name one
    // that does not exist yet.
    const std::filesystem::path directory = root / "created" / "nested";
    const auto log = std::make_shared<PrefixLog>();
    const Pipeline pipeline =
        MakeSingleComponentPipeline(log, Telemetry(true, directory));
    Check(
        std::filesystem::is_directory(directory),
        "building a tracing pipeline creates its trace directory");
    PipelineSession session = pipeline.CreateSession();
    (void)session.RunStage("pass", PassInputs());
    Check(
        CountJsonFiles(directory) == 1,
        "the created directory is the one traces are written into");
  }
}

}  // namespace

int main() {
  const std::filesystem::path root = ScratchRoot();
  std::error_code code;
  // A tree left behind by an earlier failed run would make file counts lie,
  // so the scratch directory always starts empty.
  std::filesystem::remove_all(root, code);
  std::filesystem::create_directories(root, code);
  if (code) {
    std::cerr << "UNEXPECTED: could not create the scratch directory: "
              << code.message() << '\n';
    return 1;
  }
  try {
    RunTraceChecks(root);
  } catch (const onnx_world_model::Error& error) {
    std::cerr << "UNEXPECTED: " << error.what() << '\n';
    ++failures;
  }
  // Removed whether the checks passed or failed, and only ever this one
  // directory, which nothing outside this test writes into.
  std::filesystem::remove_all(root, code);
  if (failures != 0) {
    std::cerr << failures << " trace check(s) failed\n";
    return 1;
  }
  std::cout << "pipeline trace tests passed\n";
  return 0;
}
