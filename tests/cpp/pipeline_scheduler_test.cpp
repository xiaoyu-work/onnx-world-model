/**
 * @agent-file
 * @agent-purpose: Standalone test executable for shared stage-aware admission scheduling: the unlimited default, a global ceiling, a per-stage-kind ceiling that never blocks a different kind, cancellation and deadlines while queued -- including a queued StageRun::Step or StageRun::Finish that must still close its handle and release its session's active-run slot -- permit recovery after a backend failure, which calls are executions and which are not, whether two Pipeline values share or separate their ceiling, and the PipelineSchedulingStats reading that every queue assertion here synchronizes on.
 * @agent-public-api: main
 * @agent-invariants: Registered with CTest as pipeline_scheduler_test; it counts failures through local Check and CheckThrowsCode helpers and returns a non-zero exit code when any check fails. It includes public headers only -- it never reaches into src/pipeline_scheduler.hpp -- so every claim is made through Pipeline, PipelineSession, StageRun, and Pipeline::scheduling_stats(). Every component is a stub ModelBackend gated by one shared Gate, and every PipelinePackage is built in memory from the embedded two-stage manifest, so the run needs no ONNX Runtime library, no ONNX model, and no filesystem access. The gate is a condition variable, never a sleep: a backend announces that it entered, records the concurrency peak, and waits for an explicit release, so "two entered at once" is a handshake rather than a race. "It is queued" is never inferred from a wait that timed out: WaitForQueued polls Pipeline::scheduling_stats().queued_executions until the queue holds exactly the expected number, which is the scheduler publishing its own state, and it is bounded by kAdmissionBudget so a scheduler that never enqueues -- or one that admits when it must not -- fails a check instead of hanging. Every test that cancels or expires a queued request therefore observes the insertion first, and asserts the queue drains back to zero afterwards; the deadlines those tests use are kQueuedDeadline, deliberately far longer than an insertion takes, so the watchdog can never claim a request that was never queued. Any call that might block on admission -- including the completed-Finish, completed-Step, and closed-handle probes that must *not* queue -- runs on a Worker thread under a finite WaitForWorker budget, and a budget that expires opens the gate to free the parked permit holder before joining and records the failure, so no probe can park this thread while the only permit is held. A Worker publishes its outcome and its finished flag under one mutex hold, and every reader takes that mutex, so no check races the thread that produced it. The two run-slot regressions keep the permit holder on the other stage kind, so the streaming session's own backend entries are counted separately, and they use Snapshot -- a state method that takes no permit -- as the assertion that the slot was released, because it succeeds while the ceiling is still saturated. Exact FIFO order within one kind is deliberately not asserted, because proving it needs an internal hook this file will not add; the queue's liveness -- every queued request is eventually admitted -- and its depth are asserted instead.
 * @agent-side-effects: Writes failure descriptions to stderr and starts short-lived worker threads that block inside a stub backend.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
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
using onnx_world_model::NamedTensors;
using onnx_world_model::Pipeline;
using onnx_world_model::PipelineManifest;
using onnx_world_model::PipelinePackage;
using onnx_world_model::PipelineRunOptions;
using onnx_world_model::PipelineSchedulingOptions;
using onnx_world_model::PipelineSession;
using onnx_world_model::StageEvent;
using onnx_world_model::StageEventKind;
using onnx_world_model::StageRun;
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

// True when `work` completed without throwing an Error, reported rather than
// propagated so a broken run-slot release fails a check instead of unwinding
// past worker threads that are still parked inside the gate.
template <typename Function>
[[nodiscard]] bool Succeeded(Function&& work) {
  try {
    work();
    return true;
  } catch (const onnx_world_model::Error& error) {
    std::cerr << "  (unexpected error: " << error.what() << ")\n";
    return false;
  }
}

// Long enough that ordinary scheduling noise cannot exhaust it, short enough
// that a scheduler that never admits fails the run quickly instead of hanging
// CTest.
constexpr std::chrono::seconds kAdmissionBudget{5};
// The same budget expressed the way every bounded wait below takes it.
constexpr auto kAdmissionBudgetMs =
    std::chrono::duration_cast<std::chrono::milliseconds>(kAdmissionBudget);
// How long a call that must return *without* being admitted is given to prove
// it did. It bounds a probe that is expected to return immediately, so it
// costs nothing when the runtime is correct and fails a check rather than
// parking this thread forever when it is not.
constexpr std::chrono::milliseconds kBlockedProbe{250};
// Long enough that a queue insertion is always observed before the deadline
// can fire, so a test that must see a request queued first is never racing
// its own timeout.
constexpr std::chrono::milliseconds kQueuedDeadline{1500};

// The one handshake every backend in this file goes through. A backend
// announces that it entered, updates the concurrency counters, and then waits
// for the test to hand it a release, so "how many were inside at once" is a
// recorded number rather than an inference from timing.
struct Gate {
  std::mutex mutex;
  std::condition_variable condition;
  std::size_t entered{0};
  std::size_t active{0};
  std::size_t peak{0};
  std::size_t releases{0};
  bool open{false};
  std::unordered_map<std::string, std::size_t> entered_by_component;
  std::unordered_map<std::string, std::size_t> peak_by_component;
  std::unordered_map<std::string, std::size_t> active_by_component;
};

// True only when `count` entries were observed within `budget`. The tests use
// it in both directions: to wait for progress, and -- with the short probe
// budget -- to assert that a request stayed queued.
[[nodiscard]] bool WaitForEntriesBounded(
    Gate& gate,
    std::size_t count,
    std::chrono::milliseconds budget) {
  std::unique_lock lock(gate.mutex);
  return gate.condition.wait_for(
      lock, budget, [&gate, count] { return gate.entered >= count; });
}

void WaitForEntries(Gate& gate, std::size_t count) {
  const bool reached = WaitForEntriesBounded(gate, count, kAdmissionBudgetMs);
  Check(reached, "An expected backend entry never happened");
}

// Lets `count` parked backend calls proceed, one release each.
void Release(Gate& gate, std::size_t count) {
  {
    std::scoped_lock lock(gate.mutex);
    gate.releases += count;
  }
  gate.condition.notify_all();
}

// Lets every current and future backend call run straight through.
void OpenGate(Gate& gate) {
  {
    std::scoped_lock lock(gate.mutex);
    gate.open = true;
  }
  gate.condition.notify_all();
}

// Closes the gate again so later calls park. Only safe while nothing is
// parked, which is why every caller does it between phases of a test.
void CloseGate(Gate& gate) {
  std::scoped_lock lock(gate.mutex);
  gate.open = false;
}

[[nodiscard]] std::size_t Entered(Gate& gate) {
  std::scoped_lock lock(gate.mutex);
  return gate.entered;
}

[[nodiscard]] std::size_t Peak(Gate& gate) {
  std::scoped_lock lock(gate.mutex);
  return gate.peak;
}

[[nodiscard]] std::size_t EnteredFor(
    Gate& gate,
    const std::string& component) {
  std::scoped_lock lock(gate.mutex);
  const auto found = gate.entered_by_component.find(component);
  return found == gate.entered_by_component.end() ? 0 : found->second;
}

[[nodiscard]] std::size_t PeakFor(Gate& gate, const std::string& component) {
  std::scoped_lock lock(gate.mutex);
  const auto found = gate.peak_by_component.find(component);
  return found == gate.peak_by_component.end() ? 0 : found->second;
}

// Adds one to every element of its input and parks inside the call until the
// test releases it, so a permit is provably held for as long as a component
// pass lasts. `fail_first` makes the very first pass throw instead, which is
// how "a backend failure still returns the permit" is asserted.
class GatedBackend final : public onnx_world_model::ModelBackend {
 public:
  GatedBackend(
      std::shared_ptr<Gate> gate,
      std::string component,
      std::string input,
      std::string output,
      std::vector<std::int64_t> shape,
      bool fail_first = false)
      : gate_(std::move(gate)),
        component_(std::move(component)),
        input_(std::move(input)),
        output_(std::move(output)),
        fail_first_(fail_first) {
    metadata_.inputs.push_back({
        .name = input_,
        .data_type = DataType::float32,
        .shape = shape,
    });
    metadata_.outputs.push_back({
        .name = output_,
        .data_type = DataType::float32,
        .shape = std::move(shape),
    });
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
    {
      std::unique_lock lock(gate_->mutex);
      ++gate_->entered;
      ++gate_->entered_by_component[component_];
      ++gate_->active;
      std::size_t& component_active = gate_->active_by_component[component_];
      ++component_active;
      gate_->peak = std::max(gate_->peak, gate_->active);
      std::size_t& component_peak = gate_->peak_by_component[component_];
      component_peak = std::max(component_peak, component_active);
      gate_->condition.notify_all();
      gate_->condition.wait(
          lock, [this] { return gate_->open || gate_->releases > 0; });
      if (!gate_->open) {
        --gate_->releases;
      }
      --gate_->active;
      --gate_->active_by_component[component_];
    }
    gate_->condition.notify_all();
    cancellation.ThrowIfCancellationRequested();
    if (fail_first_) {
      bool fail = false;
      {
        std::scoped_lock lock(gate_->mutex);
        fail = !failed_once_;
        failed_once_ = true;
      }
      if (fail) {
        throw onnx_world_model::Error(
            ErrorCode::runtime_execution,
            "Gated backend failed its first pass on purpose");
      }
    }
    Tensor result = inputs.at(input_);
    auto values = std::span(
        reinterpret_cast<float*>(result.mutable_bytes().data()),
        result.element_count());
    for (float& value : values) {
      value += 1.0F;
    }
    return {{output_, std::move(result)}};
  }

 private:
  ModelMetadata metadata_;
  std::shared_ptr<Gate> gate_;
  std::string component_;
  std::string input_;
  std::string output_;
  bool fail_first_;
  mutable bool failed_once_{false};
};

// Two independent one-component stages of two different kinds. Both execute
// exactly one component pass per execution, which is what makes "one
// execution is one permit" observable through the gate, and having two kinds
// is what makes a per-kind ceiling and cross-kind progress observable at all.
constexpr std::string_view kTwoKindManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "counter",
        "role": "dynamics",
        "run_on": "step",
        "inputs": [{"name": "state", "dtype": "FLOAT", "shape": [1]}],
        "outputs": [{"name": "next_state", "dtype": "FLOAT", "shape": [1]}],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT"
      },
      {
        "name": "producer",
        "role": "encoder",
        "run_on": "always",
        "inputs": [{"name": "x", "dtype": "FLOAT", "shape": [1, 4]}],
        "outputs": [{"name": "y", "dtype": "FLOAT", "shape": [1, 4]}]
      }
    ],
    "connections": [
      {
        "source": "counter.next_state",
        "target": "counter.state",
        "recurrent": true
      }
    ],
    "stages": [
      {
        "name": "transition",
        "kind": "state_transition",
        "components": ["counter"],
        "run_on": "step",
        "options": {"state_names": ["counter_state"]},
        "capabilities": ["loop_carried_state"]
      },
      {
        "name": "pass",
        "kind": "single_pass",
        "components": ["producer"],
        "run_on": "always"
      }
    ],
    "inputs": [
      {
        "port": "counter.state",
        "kind": "generated",
        "required": true,
        "semantic": "state.initial",
        "generator": {"kind": "zeros"}
      },
      {"port": "producer.x", "kind": "external", "required": true}
    ],
    "outputs": [
      {"state": "counter_state", "alias": "value"},
      {"port": "producer.y", "alias": "produced"}
    ],
    "states": [
      {
        "name": "counter_state",
        "kind": "recurrent",
        "input": "counter.state",
        "output": "counter.next_state",
        "lifetime": "session",
        "release_after": "transition"
      }
    ],
    "required_capabilities": ["loop_carried_state"]
  },
  "component_files": {
    "counter": "counter/model.onnx",
    "producer": "producer/model.onnx"
  }
}
)json";

[[nodiscard]] PipelinePackage MakePackage(
    const std::shared_ptr<Gate>& gate,
    bool fail_first_counter_pass = false) {
  std::unordered_map<std::string, Model> models;
  models.emplace(
      "counter",
      Model(std::make_shared<GatedBackend>(
          gate,
          "counter",
          "state",
          "next_state",
          std::vector<std::int64_t>{1},
          fail_first_counter_pass)));
  models.emplace(
      "producer",
      Model(std::make_shared<GatedBackend>(
          gate,
          "producer",
          "x",
          "y",
          std::vector<std::int64_t>{1, 4})));
  return PipelinePackage(
      {}, PipelineManifest::Parse(kTwoKindManifest), std::move(models));
}

[[nodiscard]] Pipeline MakePipeline(
    const std::shared_ptr<Gate>& gate,
    PipelineSchedulingOptions scheduling = {},
    bool fail_first_counter_pass = false) {
  return Pipeline(
      MakePackage(gate, fail_first_counter_pass), std::move(scheduling));
}

[[nodiscard]] PipelineSchedulingOptions GlobalLimit(std::size_t limit) {
  PipelineSchedulingOptions options;
  options.max_concurrent_executions = limit;
  return options;
}

[[nodiscard]] PipelineSchedulingOptions KindLimit(
    std::string kind,
    std::size_t limit) {
  PipelineSchedulingOptions options;
  options.max_concurrent_by_stage_kind.emplace(std::move(kind), limit);
  return options;
}

[[nodiscard]] PipelineRunOptions OptionsWith(const CancellationToken& token) {
  PipelineRunOptions options;
  options.cancellation = token;
  return options;
}

[[nodiscard]] NamedTensors PassInputs() {
  static constexpr std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
  NamedTensors inputs;
  inputs.emplace(
      "producer.x", Tensor::FromValues<float>({1, 4}, std::span(values)));
  return inputs;
}

// True once the pipeline's admission queue holds exactly `count` requests,
// false if it has not within kAdmissionBudget. This is the file's one
// synchronization on queue state, and it is a legitimate one: the scheduler
// publishes its own depth, so "the request is queued" is an observation of
// the runtime rather than an inference from how long something took. The wait
// is bounded, so a scheduler that never enqueues -- or one that admits when
// it must not -- fails a check instead of hanging CTest.
//
// A queue depth is not monotonic, so this looks for the exact count rather
// than a floor: every caller knows precisely how many requests it left
// waiting, and 0 is how "the queue drained again" is asserted.
[[nodiscard]] bool WaitForQueued(const Pipeline& pipeline, std::size_t count) {
  const auto deadline = std::chrono::steady_clock::now() + kAdmissionBudget;
  for (;;) {
    if (pipeline.scheduling_stats().queued_executions == count) {
      return true;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    // Yield first so a one-core machine still lets the queuing thread run,
    // then sleep so this poll cannot starve it either.
    std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
}

// Reads one stage kind's queue depth, which is what makes a per-kind ceiling
// claim about the right kind rather than about the queue as a whole.
[[nodiscard]] std::size_t QueuedFor(
    const Pipeline& pipeline,
    const std::string& kind) {
  const auto stats = pipeline.scheduling_stats();
  const auto found = stats.queued_by_stage_kind.find(kind);
  Check(
      found != stats.queued_by_stage_kind.end(),
      "Scheduling stats report every executable stage kind");
  return found == stats.queued_by_stage_kind.end() ? 0 : found->second;
}

// A worker that runs one stage on its own session and records how it ended,
// so a test can assert an outcome without owning the thread's lifetime. Every
// field except the thread is written once, under `mutex`, by FinishWorker, so
// a reader that woke on `condition` observes the outcome and the flag
// together rather than racing them.
struct Worker {
  std::thread thread;
  std::mutex mutex;
  std::condition_variable condition;
  bool finished{false};
  bool completed{false};
  std::optional<ErrorCode> error;
};

// Publishes the whole outcome and then the finished flag under one hold, so
// no reader can see "finished" without also seeing how it finished.
void FinishWorker(
    Worker& worker,
    bool completed,
    std::optional<ErrorCode> error) {
  {
    std::scoped_lock lock(worker.mutex);
    worker.completed = completed;
    worker.error = error;
    worker.finished = true;
  }
  worker.condition.notify_all();
}

[[nodiscard]] bool WaitForWorker(
    Worker& worker,
    std::chrono::milliseconds budget) {
  std::unique_lock lock(worker.mutex);
  return worker.condition.wait_for(
      lock, budget, [&worker] { return worker.finished; });
}

// Read under the same mutex the outcome was published under, so a check that
// runs before Join is as well defined as one that runs after it.
[[nodiscard]] bool Completed(Worker& worker) {
  std::scoped_lock lock(worker.mutex);
  return worker.completed;
}

[[nodiscard]] std::optional<ErrorCode> FailedWith(Worker& worker) {
  std::scoped_lock lock(worker.mutex);
  return worker.error;
}

void StartRunStage(
    Worker& worker,
    PipelineSession& session,
    std::string stage,
    NamedTensors inputs,
    PipelineRunOptions options) {
  worker.thread = std::thread(
      [&worker,
       &session,
       stage = std::move(stage),
       inputs = std::move(inputs),
       options = std::move(options)] {
        bool completed = false;
        std::optional<ErrorCode> error;
        try {
          (void)session.RunStage(stage, inputs, {}, options);
          completed = true;
        } catch (const onnx_world_model::Error& failure) {
          error = failure.code();
        }
        FinishWorker(worker, completed, error);
      });
}

void Join(Worker& worker) {
  if (worker.thread.joinable()) {
    worker.thread.join();
  }
}

// The same outcome capture for a call that is not RunStage, so a stage run
// that throws on a worker thread fails a check instead of terminating the
// process.
template <typename Function>
void StartCall(Worker& worker, Function work) {
  worker.thread = std::thread([&worker, work = std::move(work)] {
    bool completed = false;
    std::optional<ErrorCode> error;
    try {
      work();
      completed = true;
    } catch (const onnx_world_model::Error& failure) {
      error = failure.code();
    }
    FinishWorker(worker, completed, error);
  });
}

void TestTheDefaultPipelineAdmitsEveryExecutionAtOnce() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate);
  PipelineSession first = pipeline.CreateSession();
  PipelineSession second = pipeline.CreateSession();
  PipelineSession third = pipeline.CreateSession();

  Worker a;
  Worker b;
  Worker c;
  StartRunStage(a, first, "transition", {}, {});
  StartRunStage(b, second, "transition", {}, {});
  StartRunStage(c, third, "pass", PassInputs(), {});

  Check(
      WaitForEntriesBounded(*gate, 3, kAdmissionBudgetMs),
      "An unconfigured pipeline lets three sessions into the backend at once");
  Check(
      Peak(*gate) == 3,
      "All three executions were inside the backend simultaneously");
  // An unlimited kind is admitted without a permit at all, so the controller
  // has nothing to report even with three executions inside the backend.
  const onnx_world_model::PipelineSchedulingStats idle =
      pipeline.scheduling_stats();
  Check(
      idle.active_executions == 0 && idle.queued_executions == 0,
      "An unlimited pipeline takes no permits, so it reports zeros");
  Check(
      idle.active_by_stage_kind.size() == 6 &&
          idle.queued_by_stage_kind.size() == 6,
      "Both per-kind readings carry every executable stage kind");

  OpenGate(*gate);
  Join(a);
  Join(b);
  Join(c);
  Check(
      Completed(a) && Completed(b) && Completed(c),
      "Every unlimited execution finished");
}

void TestAGlobalCeilingIsNeverExceededAndStillAdmitsEveryone() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate, GlobalLimit(2));
  std::vector<PipelineSession> sessions;
  for (int index = 0; index < 4; ++index) {
    sessions.push_back(pipeline.CreateSession());
  }

  std::array<Worker, 4> workers;
  for (std::size_t index = 0; index < workers.size(); ++index) {
    StartRunStage(workers[index], sessions[index], "transition", {}, {});
  }

  WaitForEntries(*gate, 2);
  // The other two are provably in the queue rather than merely slow to
  // arrive, which is the whole claim a global ceiling makes.
  Check(
      WaitForQueued(pipeline, 2),
      "A global limit of two keeps the third and fourth executions queued");
  Check(
      Entered(*gate) == 2,
      "No queued execution reached the backend while the ceiling was full");
  Check(Peak(*gate) == 2, "Exactly two executions were inside at once");
  const onnx_world_model::PipelineSchedulingStats saturated =
      pipeline.scheduling_stats();
  Check(
      saturated.active_executions == 2,
      "The controller reports both permits as held");
  Check(
      saturated.active_by_stage_kind.at("state_transition") == 2,
      "A permit is counted against its own stage kind even when only the "
      "global limit constrains it");

  // Releasing one at a time proves the queue is work conserving: each freed
  // permit admits exactly one more waiter, and never more than the ceiling.
  Release(*gate, 1);
  Check(
      WaitForEntriesBounded(*gate, 3, kAdmissionBudgetMs),
      "Releasing one permit admits the next queued execution");
  Check(WaitForQueued(pipeline, 1), "Exactly one request is still queued");
  Check(Peak(*gate) <= 2, "The ceiling still held while the queue drained");

  OpenGate(*gate);
  for (Worker& worker : workers) {
    Join(worker);
  }
  Check(
      std::ranges::all_of(
          workers, [](Worker& worker) { return Completed(worker); }),
      "Every queued execution was eventually admitted and finished");
  Check(
      Entered(*gate) == 4,
      "All four executions ran exactly one component pass");
  Check(
      Peak(*gate) == 2,
      "The global ceiling of two was reached but never exceeded");
  Check(
      WaitForQueued(pipeline, 0) &&
          pipeline.scheduling_stats().active_executions == 0,
      "Every permit came back and the queue drained to empty");
}

void TestAPerKindCeilingDoesNotBlockADifferentKind() {
  auto gate = std::make_shared<Gate>();
  // Only single_pass is capped; state_transition and the global limit are
  // both unlimited, so a queued single_pass waiter must not hold the queue
  // head against an eligible state_transition request.
  const Pipeline pipeline = MakePipeline(gate, KindLimit("single_pass", 1));
  PipelineSession first = pipeline.CreateSession();
  PipelineSession second = pipeline.CreateSession();
  PipelineSession third = pipeline.CreateSession();

  Worker a;
  StartRunStage(a, first, "pass", PassInputs(), {});
  WaitForEntries(*gate, 1);
  Check(
      EnteredFor(*gate, "producer") == 1,
      "The first single_pass execution is inside the backend");

  Worker b;
  StartRunStage(b, second, "pass", PassInputs(), {});
  Check(
      WaitForQueued(pipeline, 1),
      "The second single_pass execution is queued behind the per-kind cap");
  Check(
      QueuedFor(pipeline, "single_pass") == 1,
      "The queued request is counted against the kind that is full");
  Check(
      pipeline.scheduling_stats().active_by_stage_kind.at("single_pass") == 1,
      "The capped kind holds its one permit");

  // The other kind arrives strictly after a same-kind waiter is queued, so
  // admitting it is exactly the no-head-of-line-blocking claim.
  Worker c;
  StartRunStage(c, third, "transition", {}, {});
  Check(
      WaitForEntriesBounded(*gate, 2, kAdmissionBudgetMs),
      "A different, uncapped stage kind enters while single_pass is full");
  Check(
      EnteredFor(*gate, "counter") == 1,
      "The execution that entered is the state_transition one");
  Check(
      EnteredFor(*gate, "producer") == 1,
      "The capped kind is still serialized at one execution");

  OpenGate(*gate);
  Join(a);
  Join(b);
  Join(c);
  Check(
      Completed(a) && Completed(b) && Completed(c),
      "Every execution finished once the gate opened");
  Check(
      PeakFor(*gate, "producer") == 1,
      "The per-kind ceiling of one was never exceeded");
  Check(
      WaitForQueued(pipeline, 0),
      "The per-kind queue drained once every execution finished");
}

void TestAFullKindAtTheQueueHeadDoesNotBlockAnotherKind() {
  auto gate = std::make_shared<Gate>();
  PipelineSchedulingOptions scheduling;
  scheduling.max_concurrent_executions = 2;
  scheduling.max_concurrent_by_stage_kind.emplace("single_pass", 1);
  const Pipeline pipeline = MakePipeline(gate, scheduling);
  PipelineSession first_pass = pipeline.CreateSession();
  PipelineSession queued_pass = pipeline.CreateSession();
  PipelineSession transition = pipeline.CreateSession();

  Worker holder;
  StartRunStage(holder, first_pass, "pass", PassInputs(), {});
  WaitForEntries(*gate, 1);

  Worker same_kind;
  StartRunStage(same_kind, queued_pass, "pass", PassInputs(), {});
  Check(
      WaitForQueued(pipeline, 1),
      "The second single-pass execution reached the admission queue");

  // Global capacity still has one slot, but the queue head cannot use it
  // because its kind is full. The transition behind it must be admitted.
  Worker other_kind;
  StartRunStage(other_kind, transition, "transition", {}, {});
  const bool transition_entered =
      WaitForEntriesBounded(*gate, 2, kAdmissionBudgetMs);
  Check(
      transition_entered && EnteredFor(*gate, "counter") == 1,
      "An eligible transition is admitted behind a full single-pass kind");
  const auto stats = pipeline.scheduling_stats();
  Check(
      stats.queued_executions == 1 &&
          stats.queued_by_stage_kind.at("single_pass") == 1,
      "The older same-kind waiter remains queued while the other kind runs");

  OpenGate(*gate);
  Join(holder);
  Join(same_kind);
  Join(other_kind);
  Check(
      Completed(holder) && Completed(same_kind) && Completed(other_kind),
      "Every mixed-kind execution finishes after the gate opens");
}

void TestAQueuedExecutionIsCancelledWithoutReachingTheBackend() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate, GlobalLimit(1));
  PipelineSession holder = pipeline.CreateSession();
  PipelineSession queued = pipeline.CreateSession();
  PipelineSession later = pipeline.CreateSession();

  Worker held;
  StartRunStage(held, holder, "transition", {}, {});
  WaitForEntries(*gate, 1);

  CancellationSource source;
  Worker cancelled;
  StartRunStage(
      cancelled, queued, "transition", {}, OptionsWith(source.token()));
  // Cancelling before the request is in the queue would prove nothing about
  // a queued waiter, so the insertion is observed first.
  Check(
      WaitForQueued(pipeline, 1),
      "The second execution is queued rather than running");

  source.Cancel();
  Join(cancelled);
  Check(
      FailedWith(cancelled) == ErrorCode::cancelled,
      "A queued execution that is cancelled reports ErrorCode::cancelled");
  Check(
      Entered(*gate) == 1,
      "A cancelled queued execution never reached the backend");
  Check(
      WaitForQueued(pipeline, 0),
      "The cancelled waiter left the queue rather than lingering in it");

  // The cancelled waiter must have surrendered its queue position rather than
  // leaving a permit or a ticket behind.
  Worker next;
  StartRunStage(next, later, "transition", {}, {});
  Release(*gate, 1);
  Check(
      WaitForEntriesBounded(*gate, 2, kAdmissionBudgetMs),
      "The next execution is admitted once the holder releases its permit");

  OpenGate(*gate);
  Join(held);
  Join(next);
  Check(
      Completed(held) && Completed(next),
      "The holding and following executions both finished");
}

void TestAQueuedExecutionStopsAtItsDeadline() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate, GlobalLimit(1));
  PipelineSession holder = pipeline.CreateSession();
  PipelineSession queued = pipeline.CreateSession();

  Worker held;
  StartRunStage(held, holder, "transition", {}, {});
  WaitForEntries(*gate, 1);

  // Nothing polls while a request waits for a permit, so only the shared
  // watchdog can claim this deadline. It is long enough that the queue
  // insertion below is always observed before it can fire, so the test proves
  // a *queued* request expires rather than one that expired on arrival.
  const CancellationSource source =
      CancellationSource::WithTimeout(kQueuedDeadline);
  Worker expired;
  StartRunStage(
      expired, queued, "transition", {}, OptionsWith(source.token()));
  Check(
      WaitForQueued(pipeline, 1),
      "The execution is in the queue before its deadline can fire");
  Join(expired);

  Check(
      FailedWith(expired) == ErrorCode::deadline_exceeded,
      "A queued execution whose deadline passes reports deadline_exceeded");
  Check(
      Entered(*gate) == 1,
      "An execution stopped by its deadline never reached the backend");
  Check(
      WaitForQueued(pipeline, 0),
      "The expired waiter released its queue position");

  OpenGate(*gate);
  Join(held);
  Check(Completed(held), "The permit holder finished normally");
}

void TestAFailedExecutionReturnsItsPermit() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate, GlobalLimit(1), true);
  PipelineSession session = pipeline.CreateSession();
  OpenGate(*gate);

  CheckThrowsCode(
      [&session] { (void)session.RunStage("transition"); },
      ErrorCode::runtime_execution,
      "The first execution fails inside the backend");

  // If the failing execution had leaked its permit, the ceiling of one would
  // make this call block forever rather than run.
  const NamedTensors outputs = session.RunStage("transition");
  Check(
      outputs.contains("value"),
      "The next execution is admitted after a backend failure");
  Check(
      Entered(*gate) == 2,
      "Exactly two component passes ran, the failing one and its successor");
}

void TestBeginStageAndAnIdleHandleHoldNoPermit() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate, GlobalLimit(1));
  PipelineSession streaming = pipeline.CreateSession();
  PipelineSession other = pipeline.CreateSession();

  // Beginning a run resolves the plan and claims the session's run slot; it
  // executes nothing, so it must not consume the pipeline's only permit.
  StageRun run = streaming.BeginStage("transition");
  Check(!run.done(), "The run was begun and is not finished");

  Worker outsider;
  StartRunStage(outsider, other, "pass", PassInputs(), {});
  Check(
      WaitForEntriesBounded(*gate, 1, kAdmissionBudgetMs),
      "An unstarted StageRun holds no permit, so another execution enters");
  Check(
      pipeline.scheduling_stats().queued_executions == 0,
      "Beginning a run queues nothing, because it is not an execution");

  // While that execution holds the only permit a Step must wait, because a
  // Step is an execution.
  std::optional<StageEventKind> observed;
  Worker stepper;
  StartCall(stepper, [&run, &observed] {
    const StageEvent event = run.Step();
    observed = event.kind;
  });
  Check(
      WaitForQueued(pipeline, 1),
      "A Step waits for a permit rather than running beside the ceiling");
  Check(
      QueuedFor(pipeline, "state_transition") == 1,
      "The queued Step is counted against its own stage kind");

  Release(*gate, 1);
  Join(outsider);
  Check(Completed(outsider), "The permit holder finished");
  Release(*gate, 1);
  Join(stepper);
  Check(
      observed.has_value() && *observed == StageEventKind::transition,
      "The Step ran once the permit was free");

  // The handle is idle again between steps, so the permit is back.
  Worker between;
  StartRunStage(between, other, "pass", PassInputs(), {});
  Check(
      WaitForEntriesBounded(*gate, 3, kAdmissionBudgetMs),
      "An idle StageRun between steps holds no permit");
  Release(*gate, 1);
  Join(between);
  Check(Completed(between), "The execution beside the idle handle finished");

  OpenGate(*gate);
  const NamedTensors outputs = run.Finish();
  Check(
      outputs.contains("value"),
      "Finishing the run still returns the stage's outputs");
}

void TestFinishHoldsOnePermitForItsWholeDrain() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate, GlobalLimit(1));
  PipelineSession streaming = pipeline.CreateSession();
  PipelineSession other = pipeline.CreateSession();

  StageRun run = streaming.BeginStage("transition");
  std::optional<bool> finished;
  Worker drain;
  StartCall(
      drain, [&run, &finished] { finished = run.Finish().contains("value"); });
  WaitForEntries(*gate, 1);

  Worker outsider;
  StartRunStage(outsider, other, "pass", PassInputs(), {});
  Check(
      WaitForQueued(pipeline, 1),
      "A drain in progress holds the only permit for its whole duration");

  Release(*gate, 1);
  Join(drain);
  Check(
      finished.has_value() && *finished,
      "The drain produced the stage's outputs");
  Check(
      WaitForEntriesBounded(*gate, 2, kAdmissionBudgetMs),
      "The finished drain returned its permit");

  OpenGate(*gate);
  Join(outsider);
  Check(Completed(outsider), "The waiting execution finished");
  Check(
      WaitForQueued(pipeline, 0) &&
          pipeline.scheduling_stats().active_executions == 0,
      "Nothing is admitted or queued once both executions finished");
}

void TestACompletedFinishNeedsNoAdmission() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate, GlobalLimit(1));
  PipelineSession streaming = pipeline.CreateSession();
  PipelineSession holder = pipeline.CreateSession();

  OpenGate(*gate);
  StageRun run = streaming.BeginStage("transition");
  const NamedTensors first_result = run.Finish();
  Check(first_result.contains("value"), "The first drain completed the run");
  Check(run.done(), "The run reported its completed event");

  // Nothing is parked at this point, so closing the gate only affects the
  // permit holder started next.
  CloseGate(*gate);
  Worker permit_holder;
  StartRunStage(permit_holder, holder, "transition", {}, {});
  WaitForEntries(*gate, 2);

  // A completed run has nothing left to execute, so asking for its cached
  // result must not queue behind the saturated ceiling. It is probed on its
  // own thread rather than called here, because a call that did queue would
  // park this thread while the only permit holder is itself parked, and the
  // test could never open the gate again.
  std::optional<bool> cached;
  Worker completed_finish;
  StartCall(completed_finish, [&run, &cached] {
    cached = run.Finish().contains("value");
  });
  const bool finish_returned = WaitForWorker(completed_finish, kBlockedProbe);
  if (!finish_returned) {
    // Release the parked holder first so the probe can drain and be joined.
    OpenGate(*gate);
  }
  Join(completed_finish);
  Check(
      finish_returned && cached.has_value() && *cached,
      "A completed Finish returns its cached result without admission");

  Worker completed_step;
  StartCall(completed_step, [&run] { (void)run.Step(); });
  const bool step_returned = WaitForWorker(completed_step, kBlockedProbe);
  if (!step_returned) {
    OpenGate(*gate);
  }
  Join(completed_step);
  Check(
      step_returned && FailedWith(completed_step) == ErrorCode::state,
      "A completed Step reports its state error without admission");

  OpenGate(*gate);
  Join(permit_holder);
  Check(Completed(permit_holder), "The permit holder finished");
}

void TestAClosedRunNeedsNoAdmission() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate, GlobalLimit(1));
  PipelineSession streaming = pipeline.CreateSession();
  PipelineSession holder = pipeline.CreateSession();

  Worker permit_holder;
  StartRunStage(permit_holder, holder, "pass", PassInputs(), {});
  WaitForEntries(*gate, 1);

  StageRun stepped = streaming.BeginStage("transition");
  stepped.Cancel();
  Worker stale_step;
  StartCall(stale_step, [&stepped] { (void)stepped.Step(); });
  const bool step_returned = WaitForWorker(stale_step, kBlockedProbe);
  if (!step_returned) {
    OpenGate(*gate);
  }
  Join(stale_step);
  Check(
      step_returned && FailedWith(stale_step) == ErrorCode::state,
      "A closed Step reports its state error without admission");

  StageRun drained = streaming.BeginStage("transition");
  drained.Cancel();
  Worker stale_finish;
  StartCall(stale_finish, [&drained] { (void)drained.Finish(); });
  const bool finish_returned = WaitForWorker(stale_finish, kBlockedProbe);
  if (!finish_returned) {
    OpenGate(*gate);
  }
  Join(stale_finish);
  Check(
      finish_returned && FailedWith(stale_finish) == ErrorCode::state,
      "A closed Finish reports its state error without admission");
  Check(
      EnteredFor(*gate, "counter") == 0,
      "Terminal handles never enter the streaming session's backend");
  Check(
      pipeline.scheduling_stats().queued_executions == 0,
      "Neither terminal handle ever took a queue ticket");

  OpenGate(*gate);
  Join(permit_holder);
  Check(Completed(permit_holder), "The permit holder finished normally");
}

// A Step that is stopped while it is still queued never takes the session
// lock, so the only thing that can release the session's single run slot is
// the admission-time failure path itself. If it did not, the handle would
// keep a slot for an execution that never happened and the session would be
// unusable for the rest of its life.
void TestAQueuedStepCancellationClosesTheRunAndFreesTheSession() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate, GlobalLimit(1));
  PipelineSession streaming = pipeline.CreateSession();
  PipelineSession holder = pipeline.CreateSession();

  // A different session takes the pipeline's only permit, and it takes it on
  // the other stage, so "counter" entries belong exclusively to the streaming
  // session under test.
  Worker permit_holder;
  StartRunStage(permit_holder, holder, "pass", PassInputs(), {});
  WaitForEntries(*gate, 1);

  // Beginning a run is not an execution, so this succeeds while the ceiling
  // is saturated and the run's slot is claimed before anything is queued.
  CancellationSource source;
  StageRun run =
      streaming.BeginStage("transition", {}, {}, OptionsWith(source.token()));
  Check(!run.done(), "The begun run is live before its first step");

  Worker stepper;
  StartCall(stepper, [&run] { (void)run.Step(); });
  // Cancelling before the Step is in the queue would test the wrong thing:
  // the token would already be cancelled when the Step polled it, and the
  // admission-time release path would never run. The scheduler's own queue
  // depth is what says the ticket exists.
  Check(
      WaitForQueued(pipeline, 1),
      "The Step is queued behind the saturated ceiling rather than running");

  source.Cancel();
  Join(stepper);
  Check(
      FailedWith(stepper) == ErrorCode::cancelled,
      "A Step cancelled while queued reports ErrorCode::cancelled");
  Check(
      run.done(),
      "A Step cancelled before admission closes its handle");
  Check(
      EnteredFor(*gate, "counter") == 0,
      "The cancelled Step never reached the streaming session's backend");
  Check(
      WaitForQueued(pipeline, 0),
      "The cancelled Step gave its queue position back");

  // Snapshot is a state method and takes no permit, so it succeeds here only
  // if the cancelled Step really released the session's active-run slot --
  // while the pipeline's only permit is still held by the other session. It
  // is checked without propagating, because the permit holder is still parked
  // inside the gate.
  Check(
      Succeeded([&streaming] { (void)streaming.Snapshot(); }),
      "The session accepts a Snapshot again once the queued Step released "
      "its run slot");

  OpenGate(*gate);
  Join(permit_holder);
  Check(Completed(permit_holder), "The permit holder finished normally");

  // The same session must accept a completely fresh run once capacity is
  // back, which it can only do if the cancelled run gave the slot up.
  StageRun fresh = streaming.BeginStage("transition");
  Check(!fresh.done(), "The session accepted a fresh run");

  // The stale handle is closed and no longer owns the slot, so cancelling it
  // must not hand the newer run's slot away.
  run.Cancel();
  CheckThrowsCode(
      [&streaming] { (void)streaming.Snapshot(); },
      ErrorCode::state,
      "A stale run's Cancel cannot release the newer run's slot");

  const NamedTensors outputs = fresh.Finish();
  Check(
      outputs.contains("value"),
      "The fresh run executes once capacity is released");
  Check(
      EnteredFor(*gate, "counter") == 1,
      "Exactly one component pass ran for the streaming session, the fresh "
      "one");
}

// The same claim for Finish, stopped by a deadline the shared watchdog claims
// rather than by an explicit cancel, because a queued Finish polls nothing.
void TestAQueuedFinishDeadlineClosesTheRunAndFreesTheSession() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate, GlobalLimit(1));
  PipelineSession streaming = pipeline.CreateSession();
  PipelineSession holder = pipeline.CreateSession();

  Worker permit_holder;
  StartRunStage(permit_holder, holder, "pass", PassInputs(), {});
  WaitForEntries(*gate, 1);

  // Long enough that the drain below is provably in the queue before the
  // deadline can fire, so this tests a queued Finish rather than one that
  // expired on arrival, and short enough that the test stays bounded.
  const CancellationSource source =
      CancellationSource::WithTimeout(kQueuedDeadline);
  StageRun run =
      streaming.BeginStage("transition", {}, {}, OptionsWith(source.token()));
  Check(!run.done(), "The begun run is live before its drain");

  Worker drain;
  StartCall(drain, [&run] { (void)run.Finish(); });
  Check(
      WaitForQueued(pipeline, 1),
      "The Finish is queued before its deadline can fire");
  Join(drain);
  Check(
      FailedWith(drain) == ErrorCode::deadline_exceeded,
      "A Finish whose deadline passes while queued reports "
      "deadline_exceeded");
  Check(run.done(), "A Finish stopped before admission closes its handle");
  Check(
      EnteredFor(*gate, "counter") == 0,
      "The expired Finish never reached the streaming session's backend");
  Check(
      WaitForQueued(pipeline, 0),
      "The expired Finish gave its queue position back");

  Check(
      Succeeded([&streaming] { (void)streaming.Snapshot(); }),
      "The session accepts a Snapshot again once the queued Finish released "
      "its run slot");

  OpenGate(*gate);
  Join(permit_holder);
  Check(Completed(permit_holder), "The permit holder finished normally");

  // A stale drained-out handle must not take the slot away from what comes
  // next either.
  StageRun fresh = streaming.BeginStage("transition");
  run.Cancel();
  CheckThrowsCode(
      [&streaming] { (void)streaming.Snapshot(); },
      ErrorCode::state,
      "A stale drained-out run's Cancel cannot release the newer run's slot");
  fresh.Cancel();

  // A whole RunStage, not just a StageRun, works on the same session.
  const NamedTensors outputs = streaming.RunStage("transition");
  Check(
      outputs.contains("value"),
      "The session runs a full stage after its queued Finish was stopped");
  Check(
      EnteredFor(*gate, "counter") == 1,
      "Exactly one component pass ran for the streaming session");
}

void TestAPipelineCopySharesItsCeilingAndASecondPipelineDoesNot() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate, GlobalLimit(1));
  // An implicit copy shares the package and the admission controller, so
  // sessions created from either side compete for the same one permit.
  const Pipeline copy = pipeline;
  PipelineSession original_session = pipeline.CreateSession();
  PipelineSession copied_session = copy.CreateSession();

  Worker first;
  StartRunStage(first, original_session, "transition", {}, {});
  WaitForEntries(*gate, 1);

  Worker second;
  StartRunStage(second, copied_session, "transition", {}, {});
  Check(
      WaitForQueued(pipeline, 1),
      "A copied Pipeline shares the original's admission ceiling");
  Check(
      copy.scheduling_stats().queued_executions == 1,
      "Both Pipeline values report the one shared controller");

  OpenGate(*gate);
  Join(first);
  Join(second);
  Check(
      Completed(first) && Completed(second),
      "Both sessions of the shared ceiling finished");

  // Two separately constructed Pipelines each get their own controller, even
  // over identical manifests, so their ceilings do not add up to one.
  auto independent_gate = std::make_shared<Gate>();
  const Pipeline left = MakePipeline(independent_gate, GlobalLimit(1));
  const Pipeline right = MakePipeline(independent_gate, GlobalLimit(1));
  PipelineSession left_session = left.CreateSession();
  PipelineSession right_session = right.CreateSession();

  Worker left_worker;
  Worker right_worker;
  StartRunStage(left_worker, left_session, "transition", {}, {});
  StartRunStage(right_worker, right_session, "transition", {}, {});
  Check(
      WaitForEntriesBounded(*independent_gate, 2, kAdmissionBudgetMs),
      "Independently constructed Pipelines have independent ceilings");
  Check(
      Peak(*independent_gate) == 2,
      "One execution per Pipeline ran at the same time");

  OpenGate(*independent_gate);
  Join(left_worker);
  Join(right_worker);
  Check(
      Completed(left_worker) && Completed(right_worker),
      "Both independent pipelines finished their execution");
}

void TestAForkedSessionSharesItsPipelineCeiling() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate, GlobalLimit(1));
  PipelineSession parent = pipeline.CreateSession();
  PipelineSession child = parent.Fork();

  Worker first;
  StartRunStage(first, parent, "transition", {}, {});
  WaitForEntries(*gate, 1);

  Worker second;
  StartRunStage(second, child, "transition", {}, {});
  Check(
      WaitForQueued(pipeline, 1),
      "A forked session competes for the same pipeline permits");

  OpenGate(*gate);
  Join(first);
  Join(second);
  Check(
      Completed(first) && Completed(second),
      "The parent and the fork both finished");
}

void TestAnUnsupportedStageKindKeyIsRejected() {
  auto gate = std::make_shared<Gate>();
  CheckThrowsCode(
      [&gate] {
        (void)MakePipeline(gate, KindLimit("continuous_batching", 1));
      },
      ErrorCode::invalid_argument,
      "An unknown stage-kind key is rejected at Pipeline construction");
  CheckThrowsCode(
      [&gate] { (void)MakePipeline(gate, KindLimit("", 1)); },
      ErrorCode::invalid_argument,
      "An empty stage-kind key is rejected at Pipeline construction");
  CheckThrowsCode(
      [&gate] { (void)MakePipeline(gate, KindLimit("Single_Pass", 1)); },
      ErrorCode::invalid_argument,
      "Stage-kind keys are matched exactly, not case-insensitively");

  // Every kind the runtime executes is accepted, including ones no stage in
  // this manifest declares.
  PipelineSchedulingOptions every;
  for (const char* kind : {
           "single_pass",
           "autoregressive",
           "iterative",
           "state_transition",
           "composite",
           "on_demand",
       }) {
    every.max_concurrent_by_stage_kind.emplace(kind, 1);
  }
  const Pipeline pipeline = MakePipeline(gate, std::move(every));
  PipelineSession session = pipeline.CreateSession();
  OpenGate(*gate);
  const NamedTensors outputs = session.RunStage("transition");
  Check(
      outputs.contains("value"),
      "A pipeline limited on every stage kind still executes");
}

void TestAZeroLimitIsUnlimited() {
  auto gate = std::make_shared<Gate>();
  PipelineSchedulingOptions explicit_zero = GlobalLimit(0);
  explicit_zero.max_concurrent_by_stage_kind.emplace("state_transition", 0);
  const Pipeline pipeline = MakePipeline(gate, std::move(explicit_zero));
  PipelineSession first = pipeline.CreateSession();
  PipelineSession second = pipeline.CreateSession();

  Worker a;
  Worker b;
  StartRunStage(a, first, "transition", {}, {});
  StartRunStage(b, second, "transition", {}, {});
  Check(
      WaitForEntriesBounded(*gate, 2, kAdmissionBudgetMs),
      "A limit of zero means unlimited, not blocked");

  OpenGate(*gate);
  Join(a);
  Join(b);
  Check(
      Completed(a) && Completed(b), "Both zero-limited executions finished");
}

void TestDirectStepStageIsAnExecution() {
  auto gate = std::make_shared<Gate>();
  const Pipeline pipeline = MakePipeline(gate, GlobalLimit(1));
  PipelineSession stepping = pipeline.CreateSession();
  PipelineSession other = pipeline.CreateSession();

  Worker holder;
  StartRunStage(holder, other, "pass", PassInputs(), {});
  WaitForEntries(*gate, 1);

  std::optional<bool> stepped;
  Worker worker;
  StartCall(worker, [&stepping, &stepped] {
    stepped = stepping.StepStage("transition").contains("value");
  });
  Check(
      WaitForQueued(pipeline, 1),
      "A direct StepStage waits for a permit like any other execution");

  OpenGate(*gate);
  Join(worker);
  Join(holder);
  Check(
      stepped.has_value() && *stepped,
      "The direct step ran once a permit was free");
  Check(Completed(holder), "The permit holder finished");
}

}  // namespace

int main() {
  try {
    TestTheDefaultPipelineAdmitsEveryExecutionAtOnce();
    TestAGlobalCeilingIsNeverExceededAndStillAdmitsEveryone();
    TestAPerKindCeilingDoesNotBlockADifferentKind();
    TestAFullKindAtTheQueueHeadDoesNotBlockAnotherKind();
    TestAQueuedExecutionIsCancelledWithoutReachingTheBackend();
    TestAQueuedExecutionStopsAtItsDeadline();
    TestAFailedExecutionReturnsItsPermit();
    TestBeginStageAndAnIdleHandleHoldNoPermit();
    TestFinishHoldsOnePermitForItsWholeDrain();
    TestACompletedFinishNeedsNoAdmission();
    TestAClosedRunNeedsNoAdmission();
    TestAQueuedStepCancellationClosesTheRunAndFreesTheSession();
    TestAQueuedFinishDeadlineClosesTheRunAndFreesTheSession();
    TestAPipelineCopySharesItsCeilingAndASecondPipelineDoesNot();
    TestAForkedSessionSharesItsPipelineCeiling();
    TestAnUnsupportedStageKindKeyIsRejected();
    TestAZeroLimitIsUnlimited();
    TestDirectStepStageIsAnExecution();
  } catch (const std::exception& error) {
    std::cerr << "EXCEPTION: " << error.what() << '\n';
    return 2;
  }

  if (failures != 0) {
    std::cerr << failures << " pipeline scheduler checks failed\n";
    return 1;
  }
  std::cout << "pipeline scheduler tests passed\n";
  return 0;
}
