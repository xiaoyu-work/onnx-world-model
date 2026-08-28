#pragma once

/**
 * @agent-file
 * @agent-purpose: Declares the Mobius pipeline contract: manifest value types, the validated PipelineManifest and PipelinePackage loaders, per-component placement and the conservative transfer plan it produces, the shareable Pipeline with its shared admission-scheduling limits and the PipelineSchedulingStats reading of them, the per-trajectory PipelineSession, its in-memory PipelineSessionSnapshot, its named in-memory checkpoints, and the incremental StageRun that reports each step of a stage as a StageEvent.
 * @agent-public-api: Endpoint, PipelineComponent, PipelineConnection, PipelineInputKind, PipelineInput, PipelineOutput, PipelineStage, PipelineState, PipelineAsset, PipelineManifest, ComponentPlacement, PipelinePlacementOptions, PipelineTransferKind, PipelineTransfer, PipelineTransferPlan, PipelinePackage, PipelineRunOptions, PipelineSchedulingOptions, PipelineSchedulingStats, Pipeline, PipelineSessionSnapshot, StageEventKind, StageEvent, StageRun, PipelineSession
 * @agent-invariants: Pipeline holds immutable component sessions through a shared_ptr and may be shared by callers, while PipelineSession is move-only and owns exactly one trajectory's mutable state; a manifest naming a capability outside PipelineManifest::SupportedCapabilities() is rejected during loading. PipelinePlacementOptions is load-time only: it appears on PipelinePackage::Load and Pipeline::Load and deliberately not on Pipeline(PipelinePackage, scheduling), whose sessions are already built, and an empty or unknown component key throws ErrorCode::invalid_argument right after the manifest is parsed and before any component model file is opened. A component's providers are its own list when it has one, otherwise the global order, otherwise its manifest preferences; the manifest preferences filter that choice unless allow_unpreferred_providers is set and that component supplied its own list. Provider options merge global-then-component per provider and per key, and component options for a provider that component does not run on throw ErrorCode::invalid_argument instead of being dropped. PipelineTransferPlan holds exactly one PipelineTransfer per manifest connection, recurrent edges included, in manifest order; it is the configured physical plan and is never rewritten when device_outputs_enabled is false, and nothing in this milestone executes from it. RunStage and StepStage preserve the storage of the tensors they are given and may return device-backed tensors, so a caller reading a result on the host calls Tensor::CopyToCpu() first. PipelineSessionSnapshot is an immutable copyable capture of one session's mutable execution state that only PipelineSession::Snapshot() can produce; it records the package it came from, so Restore and Fork accept it only for a session built on that same PipelinePackage instance and otherwise throw ErrorCode::state. Named checkpoints are in-memory transaction markers held beside that execution state, not inside it: a checkpoint name is never empty, Checkpoint captures the same fields Snapshot does, a snapshot never contains checkpoints, RestoreCheckpoint and DropCheckpoint throw ErrorCode::state for an unknown name instead of doing nothing, checkpoints outlive stage execution and Restore, Reset drops them all, and a forked session starts with an empty checkpoint namespace. BeginStage and RunStage share one StageRun state machine and produce identical results; RunStage drains it under one session-lock acquisition so ordinary concurrent calls retain whole-stage serialization. A StageRun is move-only, single-consumer, and synchronous -- Step() blocks until exactly one model or scheduler step finishes -- and it holds the session's only run slot until it completes, is cancelled, or is destroyed; while it holds that slot the session throws ErrorCode::state from BeginStage, RunStage, StepStage, Snapshot, Restore, Fork, Checkpoint, RestoreCheckpoint, DropCheckpoint, Reset, and ReleaseStage, while outputs(), state(), and HasCheckpoint() stay legal. PipelineRunOptions::cancellation carries an optional CancellationToken that every execution path checks at its own boundaries; StageRun::RequestCancellation signals an in-flight step without taking the session lock, while StageRun::Cancel takes it and only closes the handle, so the two are different operations and neither rolls anything back. PipelineSchedulingOptions is admission scheduling only and never batching: a Pipeline's limits are fixed at construction, shared by every copy of that Pipeline and by every session and StageRun it produces, and an unknown or empty per-kind key is rejected with ErrorCode::invalid_argument at construction. Exactly four calls are one execution and take exactly one permit for their whole duration -- RunStage, StepStage, StageRun::Step, and a StageRun::Finish that still has steps to drain -- while BeginStage, a completed Finish, an idle StageRun between Step calls, Cancel, RequestCancellation, and every query and state method take none. PipelineSchedulingStats is a detached value, not a view: Pipeline::scheduling_stats() reads the shared controller under its own lock and returns counts that never change afterwards, both per-kind maps always carry all six executable stage kinds, and the counts are permits rather than executions, so an unlimited stage kind -- which is admitted without one -- is reported as zero.
 * @agent-side-effects: none in this header; the declared Load functions read pipeline.json, component ONNX files, and assets from disk.
 */

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "onnx_world_model/model.hpp"

namespace onnx_world_model {

namespace detail {

//: The shared admission controller a Pipeline hands to every session it
//: creates. Its shape is an implementation detail of src/, so only the
//: pointer appears here.
class PipelineScheduler;

}  // namespace detail

struct Endpoint {
  std::string component;
  std::string port;

  [[nodiscard]] static Endpoint Parse(std::string_view value);
  [[nodiscard]] std::string qualified() const;

  bool operator==(const Endpoint&) const = default;
};

struct PipelineComponent {
  std::string name;
  std::string role;
  std::string run_on;
  ModelMetadata metadata;
  std::optional<std::string> presence;
  std::vector<std::string> capabilities;
  std::vector<std::string> preferred_execution_providers;
  std::optional<DataType> parameter_data_type;
  std::string source;
  std::string config_json{"{}"};
  std::string metadata_json{"{}"};
  std::unordered_map<std::string, std::vector<std::string>> input_dimension_symbols;
  std::unordered_map<std::string, std::vector<std::string>> output_dimension_symbols;
};

struct PipelineConnection {
  Endpoint source;
  Endpoint target;
  bool recurrent{false};
  std::optional<std::string> transform;
  std::vector<Endpoint> context;
  std::string parameters_json{"{}"};
};

enum class PipelineInputKind {
  external,
  generated,
  stateful,
  defaulted,
};

struct PipelineInput {
  Endpoint port;
  PipelineInputKind kind;
  std::string name;
  bool required{true};
  std::string semantic;
  std::string presence;
  std::string value_json;
  std::string generator_kind;
  std::string generator_json;
};

struct PipelineOutput {
  std::optional<Endpoint> port;
  std::optional<std::string> state;
  std::string name;
};

struct PipelineStage {
  std::string name;
  std::string kind;
  std::vector<std::string> components;
  std::string run_on;
  std::string options_json{"{}"};
  std::vector<std::string> capabilities;
};

struct PipelineState {
  std::string name;
  std::string kind;
  Endpoint input;
  Endpoint output;
  std::string lifetime;
  std::string release_after;
  std::optional<std::size_t> sequence_axis;
  std::string metadata_json{"{}"};
};

struct PipelineAsset {
  std::filesystem::path path;
  bool required{true};
};

class PipelineManifest {
 public:
  static PipelineManifest Parse(std::string_view document);
  static PipelineManifest Load(const std::filesystem::path& path);

  //: Pipeline capabilities this runtime implements. A manifest whose
  //: `required_capabilities` names anything outside this set is rejected.
  [[nodiscard]] static std::vector<std::string> SupportedCapabilities();

  [[nodiscard]] std::string_view schema_version() const noexcept;
  [[nodiscard]] std::string_view profile() const noexcept;
  [[nodiscard]] std::string_view profile_version() const noexcept;
  [[nodiscard]] std::string_view model_type() const noexcept;
  [[nodiscard]] std::string_view metadata_json() const noexcept;

  [[nodiscard]] const std::vector<PipelineComponent>& components() const noexcept;
  [[nodiscard]] const std::vector<PipelineConnection>& connections() const noexcept;
  [[nodiscard]] const std::vector<PipelineInput>& inputs() const noexcept;
  [[nodiscard]] const std::vector<PipelineOutput>& outputs() const noexcept;
  [[nodiscard]] const std::vector<PipelineStage>& stages() const noexcept;
  [[nodiscard]] const std::vector<PipelineState>& states() const noexcept;
  [[nodiscard]] const std::vector<PipelineAsset>& assets() const noexcept;
  [[nodiscard]] const std::vector<std::string>& required_capabilities()
      const noexcept;
  [[nodiscard]] const std::unordered_map<std::string, std::filesystem::path>&
  component_files() const noexcept;

  [[nodiscard]] const PipelineComponent& Component(std::string_view name) const;
  [[nodiscard]] const TensorSpec& Input(const Endpoint& endpoint) const;
  [[nodiscard]] const TensorSpec& Output(const Endpoint& endpoint) const;

 private:
  std::string schema_version_;
  std::string profile_;
  std::string profile_version_;
  std::string model_type_;
  std::string metadata_json_;
  std::vector<PipelineComponent> components_;
  std::vector<PipelineConnection> connections_;
  std::vector<PipelineInput> inputs_;
  std::vector<PipelineOutput> outputs_;
  std::vector<PipelineStage> stages_;
  std::vector<PipelineState> states_;
  std::vector<PipelineAsset> assets_;
  std::vector<std::string> required_capabilities_;
  std::unordered_map<std::string, std::filesystem::path> component_files_;
};

//: Where one component's ONNX Runtime session runs and how it is configured,
//: layered on top of the pipeline-wide RuntimeOptions. Every member is
//: optional in the sense that leaving it empty keeps the global value: an
//: empty `providers` inherits the global provider order or the component's
//: manifest preferences, an empty `provider_options` inherits only the global
//: options, and a disengaged optional leaves the global scalar in force.
//:
//: This is placement configuration only. It never warms a session up, loads
//: one lazily, offloads or evicts one, or arranges a peer-to-peer transfer;
//: all of those are deliberately outside this milestone.
struct ComponentPlacement {
  //: The execution-provider order for this component, most preferred first.
  //: `device_id` and every other device selector is a native provider option
  //: rather than a field of its own, so this runtime never invents a second
  //: spelling for something ONNX Runtime already names.
  std::vector<std::string> providers;
  //: Provider options for this component, merged over the global ones per
  //: provider and per key with the component winning. Supplying options for a
  //: provider this component does not end up running on throws
  //: ErrorCode::invalid_argument rather than being dropped.
  std::unordered_map<
      std::string,
      std::unordered_map<std::string, std::string>>
      provider_options;
  //: Overrides RuntimeOptions::graph_optimization for this component only.
  std::optional<GraphOptimizationLevel> graph_optimization;
  //: Overrides RuntimeOptions::intra_op_threads for this component only. 0 is
  //: ONNX Runtime's automatic choice; a negative value is rejected.
  std::optional<int> intra_op_threads;
  //: Overrides RuntimeOptions::inter_op_threads for this component only.
  std::optional<int> inter_op_threads;
};

//: Per-component placement for one package load. Keys are manifest component
//: names; an empty or unknown name throws ErrorCode::invalid_argument
//: immediately after the manifest is parsed and before any component model
//: file is opened, so a typo fails fast instead of silently placing nothing.
struct PipelinePlacementOptions {
  std::unordered_map<std::string, ComponentPlacement> components;
  //: Lets a component that explicitly names its own providers run on one its
  //: manifest does not prefer. It applies only to such a component: a
  //: component that inherits the global provider order or its own manifest
  //: preferences is still filtered by those preferences, which is the
  //: behavior every earlier release had.
  bool allow_unpreferred_providers{false};
};

//: How one manifest connection would have to move its tensor, given where
//: ONNX Runtime actually placed the two ports. Classification is conservative:
//: anything this runtime cannot prove is a pointer handoff is reported as
//: needing host involvement.
enum class PipelineTransferKind {
  //: Same device on both ends and no transform: the producer's buffer can be
  //: handed straight to the consumer.
  direct,
  //: CPU to a non-CPU device.
  upload,
  //: A non-CPU device to CPU.
  download,
  //: Two different non-CPU devices, which this runtime stages through the
  //: host because it has no peer-to-peer path.
  host_staged,
  //: A transform this runtime evaluates on the host sits between the ports,
  //: so the source is materialized there whatever the two devices are.
  host_transform,
  //: At least one endpoint's device is unknown, so nothing may be assumed.
  unknown,
};

//: One manifest connection classified for transfer. There is exactly one of
//: these per connection, recurrent edges included, in manifest order.
struct PipelineTransfer {
  Endpoint source;
  Endpoint target;
  //: True for a loop-carried edge, which is classified exactly like a forward
  //: one; it is reported so a caller can tell the two apart.
  bool recurrent{false};
  //: The connection's declared transform, if it has one.
  std::optional<std::string> transform;
  //: Where the producing output port lives, or nullopt when unknown.
  std::optional<TensorDevice> source_device;
  //: Where the consuming input port lives, or nullopt when unknown.
  std::optional<TensorDevice> target_device;
  PipelineTransferKind kind{PipelineTransferKind::unknown};
  //: True only for `direct`. A `reshape` counts as direct only when the
  //: declared source and target shapes are identical, because the current
  //: device-buffer binding requires the original shape to equal the shape of
  //: the view wrapped around it.
  bool direct_bind_eligible{false};
  //: Why this connection is not direct, in one sentence. Empty for `direct`.
  std::string reason;
};

//: The conservative transfer classification of a whole package: one entry per
//: manifest connection, in manifest order.
//:
//: This is the configured physical plan, not the effective one. It describes
//: where ONNX Runtime placed each port; it does not describe what the session
//: currently does, and nothing in this milestone rewrites execution from it.
//: In particular, when `device_outputs_enabled` is false every component
//: output is bound to CPU regardless of where the plan says the port lives,
//: so the effective behavior is CPU-bound even where the plan reports
//: `upload`, `download`, or `host_staged`. The plan is deliberately not
//: rewritten in that case, because it answers "how is this package placed",
//: which is what a caller needs before turning device outputs on.
struct PipelineTransferPlan {
  std::vector<PipelineTransfer> transfers;
  //: Mirrors RuntimeOptions::device_outputs for the sessions this plan was
  //: computed over.
  bool device_outputs_enabled{false};
};

class PipelinePackage {
 public:
  //: Builds a package over component sessions that already exist, which is
  //: how a custom backend or a test assembles one without touching disk.
  //: `device_outputs_enabled` records whether those sessions were configured
  //: to hand back device-resident outputs; it only annotates the computed
  //: transfer plan and changes no execution.
  PipelinePackage(
      std::filesystem::path root,
      PipelineManifest manifest,
      std::unordered_map<std::string, Model> components,
      bool device_outputs_enabled = false);

  //: Loads and validates the package under `directory`. `options` is the
  //: baseline every component starts from and `placement` layers per-component
  //: overrides on top of it. Both are defaulted, so an existing call keeps its
  //: behavior exactly.
  static PipelinePackage Load(
      const std::filesystem::path& directory,
      const RuntimeOptions& options = {},
      const PipelinePlacementOptions& placement = {});

  [[nodiscard]] const std::filesystem::path& root() const noexcept;
  [[nodiscard]] const PipelineManifest& manifest() const noexcept;
  [[nodiscard]] const Model& Component(std::string_view name) const;
  [[nodiscard]] std::unordered_map<std::string, std::vector<std::string>>
  execution_providers() const;
  //: The conservative transfer classification of every manifest connection,
  //: computed once while this package was built. It is a description, never a
  //: rewrite: execution reads none of it in this milestone.
  [[nodiscard]] const PipelineTransferPlan& transfer_plan() const noexcept;

 private:
  std::filesystem::path root_;
  PipelineManifest manifest_;
  std::unordered_map<std::string, Model> components_;
  PipelineTransferPlan transfer_plan_;
};

struct PipelineRunOptions {
  std::unordered_map<std::string, std::string> strings;
  std::unordered_map<std::string, std::int64_t> integers;
  std::unordered_map<std::string, double> numbers;
  //: Stops this call at its next safe boundary. The default token is never
  //: cancellable, so leaving it alone preserves the uncancellable behavior
  //: every earlier release had.
  CancellationToken cancellation;
};

class PipelineSession;

//: The shared admission limits one Pipeline applies to every execution that
//: runs through it. This is admission scheduling only: it decides how many
//: executions may be inside the runtime at once, and in what order queued
//: ones are let in. It never merges, splits, reorders, or batches the work
//: itself, and it never preempts an execution that was already admitted.
//:
//: A limit of 0 -- and a stage kind that is absent from
//: `max_concurrent_by_stage_kind` -- means unlimited, which is the default
//: and the behavior every earlier release had. An execution is admitted only
//: when there is room under both the non-zero global limit and the non-zero
//: limit of its own stage kind.
//:
//: Every key of `max_concurrent_by_stage_kind` must be one of the stage kinds
//: this runtime executes -- `single_pass`, `autoregressive`, `iterative`,
//: `state_transition`, `composite`, or `on_demand`. An empty or unknown key
//: throws ErrorCode::invalid_argument when the Pipeline is constructed rather
//: than being ignored.
struct PipelineSchedulingOptions {
  //: How many executions may run through this Pipeline at once, across every
  //: session and every stage kind. 0 means unlimited.
  std::size_t max_concurrent_executions{0};
  //: Per-stage-kind ceilings layered under the global one. A missing key, or
  //: a key mapped to 0, means that kind is unlimited.
  std::unordered_map<std::string, std::size_t> max_concurrent_by_stage_kind;
};

//: One instantaneous reading of a Pipeline's admission controller: how many
//: executions it has admitted and how many are waiting for a permit. It is a
//: plain value with no link back to the scheduler, so it never changes after
//: it is returned and reading it needs no lock.
//:
//: This is operational observability for admission only. It says nothing
//: about batching, throughput, latency, or how long anything took, and it is
//: not a profiler: the two counters are exactly the state the admission
//: queue is in at the moment it was read.
//:
//: Both maps always contain every stage kind this runtime executes --
//: `single_pass`, `autoregressive`, `iterative`, `state_transition`,
//: `composite`, and `on_demand` -- so a caller reads a kind directly rather
//: than checking for its key first; a kind with nothing active or queued
//: maps to 0. A queued execution whose stage kind is not one of those six is
//: still counted in `queued_executions`, so the per-kind counts sum to at
//: most the totals rather than exactly to them.
//:
//: These are permit counts, not execution counts. A stage kind constrained
//: by neither the global limit nor its own is admitted without taking a
//: permit at all -- that is the lock-free fast path every default-configured
//: Pipeline takes -- so it is never counted here. An unlimited Pipeline
//: therefore reports zeros no matter how much work is inside it, and a
//: Pipeline that caps only one kind counts only that kind's executions
//: unless it also has a global limit.
//:
//: The reading is a snapshot of a live system taken under the scheduler's
//: own lock, so every field is consistent with every other field at that one
//: moment, and any of them may already be stale by the time the caller looks
//: at it. Reading stats never admits, queues, blocks, or cancels anything.
struct PipelineSchedulingStats {
  //: Executions currently holding a permit, across every session and stage
  //: kind. It never exceeds a non-zero
  //: PipelineSchedulingOptions::max_concurrent_executions, and it stays 0 for
  //: an execution that needed no permit.
  std::size_t active_executions{0};
  //: Executions currently waiting for a permit. It is 0 for an unlimited
  //: pipeline, which never queues anything.
  std::size_t queued_executions{0};
  //: Active executions broken down by the stage kind they are running. A
  //: permit is counted against its own kind whether or not that kind has a
  //: limit of its own, so a purely global limit still produces a breakdown.
  std::unordered_map<std::string, std::size_t> active_by_stage_kind;
  //: Waiting executions broken down by the stage kind they want to run.
  std::unordered_map<std::string, std::size_t> queued_by_stage_kind;
};

class Pipeline {
 public:
  //: Builds a pipeline over `package` with its own admission controller.
  //: Copies of the returned Pipeline share that controller, so their sessions
  //: compete for the same permits; a separately constructed Pipeline over the
  //: same package gets an independent one.
  //:
  //: This overload takes no PipelinePlacementOptions on purpose. Placement
  //: decides how a component's ONNX Runtime session is built, and by the time
  //: a PipelinePackage exists every session in it has already been built, so
  //: accepting placement here could only be a silent no-op.
  explicit Pipeline(
      PipelinePackage package,
      PipelineSchedulingOptions scheduling = {});

  static Pipeline Load(
      const std::filesystem::path& directory,
      const RuntimeOptions& options = {},
      const PipelineSchedulingOptions& scheduling = {},
      const PipelinePlacementOptions& placement = {});

  [[nodiscard]] const PipelineManifest& manifest() const noexcept;
  [[nodiscard]] std::unordered_map<std::string, std::vector<std::string>>
  execution_providers() const;
  //: This package's conservative connection transfer plan. It is inspection
  //: only: nothing in this milestone executes from it.
  [[nodiscard]] const PipelineTransferPlan& transfer_plan() const noexcept;
  [[nodiscard]] PipelineSession CreateSession() const;

  //: Reads this pipeline's admission controller right now. Copies of one
  //: Pipeline share a controller, so they report the same numbers, and every
  //: session and StageRun created from any of them is counted here.
  //:
  //: An unlimited stage kind is admitted without a permit, so a pipeline with
  //: no limits at all always reports zeros. Two consecutive readings may
  //: differ with no call in between, because other threads keep running.
  [[nodiscard]] PipelineSchedulingStats scheduling_stats() const;

 private:
  std::shared_ptr<const PipelinePackage> package_;
  //: Held beside the package and shared the same way: an implicit Pipeline
  //: copy shares both, so one configured ceiling covers every session created
  //: from any copy.
  std::shared_ptr<detail::PipelineScheduler> scheduler_;

  friend class PipelineSession;
};

//: An immutable in-memory capture of one PipelineSession's mutable execution
//: state: its external, endpoint, recurrent-state, and guidance tensors, its
//: stage cursors, its scheduler histories, its position cursors, and its
//: random engine. Only PipelineSession::Snapshot() produces one, so a caller
//: cannot fabricate state a session never held. Copies are cheap because the
//: captured tensors share their storage copy-on-write, and a device-backed
//: tensor is never materialized to CPU by capturing or restoring it. This is
//: an in-process value only: it is not serialized to disk and cannot cross a
//: process boundary. A session's named checkpoints are control metadata, not
//: execution state, so they never appear inside a snapshot.
class PipelineSessionSnapshot {
 public:
  //: False only for a moved-from snapshot, which no session accepts.
  [[nodiscard]] bool valid() const noexcept;

 private:
  struct Impl;
  explicit PipelineSessionSnapshot(std::shared_ptr<const Impl> state);

  std::shared_ptr<const Impl> impl_;

  friend class PipelineSession;
};

//: What one StageRun::Step() call observed. A stage's kind decides which step
//: events it emits: `token` for autoregressive decoding, `iteration` for an
//: iterative scheduler step, and `transition` for the single pass of every
//: other stage kind. `completed` is the one terminal event that ends a run.
enum class StageEventKind {
  token,
  iteration,
  transition,
  completed,
};

//: One step of an incremental stage run. Events are synchronous results, not
//: notifications: StageRun::Step() returns exactly one of these after the
//: model or scheduler step it describes has finished.
struct StageEvent {
  StageEventKind kind{StageEventKind::completed};
  //: The stage this run drives. Every event of one run repeats it.
  std::string stage;
  //: Zero-based index of this step event within this StageRun. The terminal
  //: `completed` event carries the number of step events that preceded it, so
  //: it is a count of this run's steps rather than the stage's lifetime
  //: iteration cursor, which a resumed stage may have advanced already.
  std::size_t iteration{0};
  //: The tokens this step generated, present only on a `token` event and
  //: absent on every other kind. It is the int64 [batch, 1] tensor that was
  //: fed back into the stage, so a lane that has already emitted its
  //: end-of-sequence token reads as that token here.
  std::optional<Tensor> token_ids;
  //: The stage's public outputs as of this event. Storage is preserved
  //: exactly as RunStage preserves it, so a device-backed output stays on its
  //: device and a host reader calls Tensor::CopyToCpu() first.
  NamedTensors outputs;
  //: True only on the terminal `completed` event.
  bool finished{false};
};

//: One incremental execution of a single stage, produced only by
//: PipelineSession::BeginStage(). Step() advances the session by exactly one
//: model or scheduler step and returns what that step produced, so a caller
//: can stream tokens or diffusion steps without giving up the parity that
//: RunStage guarantees: Finish() returns exactly what RunStage would have
//: returned for the same arguments because both drain this state machine.
//:
//: A run is synchronous and single-consumer. Step() blocks until its step
//: finishes. It can be stopped, but only cooperatively: RequestCancellation()
//: signals the run and the work stops at the next boundary it checks --
//: before and after each component, transform, guidance pass, and sampling
//: step, and inside the ONNX Runtime call, which ONNX Runtime can only
//: interrupt between graph nodes. A deadline supplied through
//: PipelineRunOptions::cancellation is enforced at those same boundaries and,
//: independently, by the shared deadline watchdog, so it also stops a call
//: that is already blocked; a single long-running ONNX Runtime kernel is the
//: one thing that can still overrun it. Concurrent calls on one handle
//: serialize on the session lock and only one of them advances the run.
//:
//: A session has one run slot. While this run holds it the session throws
//: ErrorCode::state from every execution and state-mutating method --
//: BeginStage, RunStage, StepStage, Snapshot, Restore, Fork, Checkpoint,
//: RestoreCheckpoint, DropCheckpoint, Reset, and ReleaseStage -- because an
//: in-flight decode loop is not part of PipelineSessionSnapshot and capturing
//: the session mid-run would silently drop it. Reading outputs(), state(),
//: and HasCheckpoint() stays legal.
//:
//: The handle owns a share of the session's execution state, so it stays
//: valid even if the PipelineSession that produced it is moved or destroyed
//: first. Cancel() and the destructor release the run slot where the run
//: stopped; they never roll back what the run already applied.
class StageRun {
 public:
  StageRun(StageRun&&) noexcept;
  StageRun& operator=(StageRun&&) noexcept;
  ~StageRun();

  StageRun(const StageRun&) = delete;
  StageRun& operator=(const StageRun&) = delete;

  //: The stage this run drives. Throws ErrorCode::state on a moved-from
  //: handle, which owns no run at all.
  [[nodiscard]] std::string_view stage() const;
  //: True once the run has produced its terminal `completed` event, and also
  //: for a moved-from, cancelled, or failed handle, so this is the one query
  //: that is always answerable.
  [[nodiscard]] bool done() const noexcept;
  //: How many step events this run has returned so far, which is the
  //: `iteration` the next step event would carry. Throws ErrorCode::state on
  //: a moved-from handle.
  [[nodiscard]] std::size_t iteration() const;

  //: Advances the run by one step and returns that step's event. A run
  //: returns exactly one `completed` event: a stage with no work left reports
  //: it on the next Step() rather than folding it into the last step event,
  //: and Step() after it throws ErrorCode::state. A failing step releases the
  //: session's run slot, closes this handle, keeps whatever the run already
  //: applied to the session, and rethrows. One Step() is one execution: it
  //: takes one admission permit before it takes the session lock and returns
  //: it when the step ends, so a handle sitting idle between Step() calls
  //: holds no permit at all.
  [[nodiscard]] StageEvent Step();
  //: Drains the remaining steps under one session-lock acquisition and
  //: returns the same outputs RunStage returns for this stage. Calling it
  //: after the run completed returns that cached result again without
  //: re-running anything and without taking an admission permit; a drain that
  //: still has work takes exactly one permit and holds it for the whole
  //: drain.
  [[nodiscard]] NamedTensors Finish();
  //: Signals the work this run is doing to stop. Unlike Cancel() this never
  //: takes the session lock, so it is the one method a second thread can call
  //: while Step() or Finish() is executing: it cancels the run's own
  //: cancellation source, which the running step observes at its next
  //: boundary and which terminates an ONNX Runtime call already in flight.
  //: The interrupted Step() or Finish() then throws ErrorCode::cancelled,
  //: releases the run slot, and leaves everything the run already applied in
  //: place. Idempotent, safe on a moved-from or finished handle, and never a
  //: rollback.
  void RequestCancellation() noexcept;
  //: Abandons an unfinished run and releases the session's run slot, leaving
  //: every effect the run already applied in place. This is the local
  //: close-and-release operation, not a signal: it takes the session lock, so
  //: it waits for an in-flight Step() instead of interrupting it. Idempotent,
  //: and a no-op on a completed, cancelled, or moved-from handle.
  void Cancel() noexcept;

 private:
  struct Impl;
  explicit StageRun(std::unique_ptr<Impl> state) noexcept;

  std::unique_ptr<Impl> impl_;

  friend class PipelineSession;
};

class PipelineSession {
 public:
  PipelineSession(PipelineSession&&) noexcept;
  PipelineSession& operator=(PipelineSession&&) noexcept;
  ~PipelineSession();

  PipelineSession(const PipelineSession&) = delete;
  PipelineSession& operator=(const PipelineSession&) = delete;

  //: Executes `stage` to completion. This is one execution: it takes one
  //: admission permit from the Pipeline's scheduler before it takes the
  //: session lock and holds that permit for the whole stage. A queued call
  //: observes its own cancellation token and deadline while it waits, so it
  //: can fail with ErrorCode::cancelled or ErrorCode::deadline_exceeded
  //: without ever reaching a component.
  [[nodiscard]] NamedTensors RunStage(
      std::string_view stage,
      const NamedTensors& inputs = {},
      const NamedTensors& overrides = {},
      const PipelineRunOptions& options = {});
  //: Starts an incremental execution of `stage` and returns the handle that
  //: drives it. The stage kind, inputs, overrides, options, and every
  //: autoregressive decision -- sampling configuration, end-of-sequence
  //: tokens, the seeded random engine, and the token budget the prompt length
  //: allows -- are resolved once, here, so a streamed run cannot drift from
  //: the equivalent RunStage call. A session runs one stage at a time: this
  //: throws ErrorCode::state while another StageRun is still active.
  //: Beginning a run is not an execution and takes no admission permit; the
  //: Step() and Finish() calls that drive it each take their own.
  [[nodiscard]] StageRun BeginStage(
      std::string_view stage,
      const NamedTensors& inputs = {},
      const NamedTensors& overrides = {},
      const PipelineRunOptions& options = {});
  //: Executes exactly one pass of `stage` for a caller that drives its own
  //: loop. This is the low-level primitive underneath BeginStage, so it
  //: throws ErrorCode::state while a StageRun is active rather than
  //: interleaving with it. One call is one execution and takes one admission
  //: permit.
  [[nodiscard]] NamedTensors StepStage(
      std::string_view stage,
      const NamedTensors& inputs = {},
      const NamedTensors& overrides = {},
      const PipelineRunOptions& options = {});
  [[nodiscard]] NamedTensors outputs() const;
  [[nodiscard]] std::optional<Tensor> state(std::string_view name) const;
  void ReleaseStage(std::string_view stage);
  void Reset();

  //: Captures every mutable execution field of this session. The result is
  //: unaffected by anything the session does afterwards.
  [[nodiscard]] PipelineSessionSnapshot Snapshot() const;
  //: Replaces every mutable execution field with the captured one. The
  //: snapshot must come from a session on this session's PipelinePackage
  //: instance; otherwise this throws ErrorCode::state and changes nothing.
  void Restore(const PipelineSessionSnapshot& snapshot);
  //: Returns an independent session on the same immutable PipelinePackage,
  //: initialized from a snapshot of this one. Neither session observes the
  //: other's later runs, releases, or resets. The fork starts with an empty
  //: named-checkpoint namespace: execution state is inherited, checkpoint
  //: names are not.
  [[nodiscard]] PipelineSession Fork() const;

  //: Captures the same execution state Snapshot() captures and stores it in
  //: this session under `name`, replacing any checkpoint already stored under
  //: that name. Capture and store happen under one lock, so the checkpoint is
  //: exactly the state some other thread could have observed at that instant.
  //: An empty name throws ErrorCode::invalid_argument. Checkpoints are
  //: control metadata that lives beside the execution state rather than
  //: inside it, so a snapshot never carries them, they survive stage
  //: execution and Restore, and Reset() drops all of them.
  void Checkpoint(std::string_view name);
  //: Restores the checkpoint stored under `name` exactly as Restore() would
  //: restore the equivalent snapshot, leaving this session's checkpoints in
  //: place. An empty name throws ErrorCode::invalid_argument; an unknown one
  //: throws ErrorCode::state. Either way the session is left unchanged.
  void RestoreCheckpoint(std::string_view name);
  //: Drops the checkpoint stored under `name`. This is not a no-op for an
  //: unknown name: an empty name throws ErrorCode::invalid_argument and an
  //: unknown one throws ErrorCode::state.
  void DropCheckpoint(std::string_view name);
  //: Reports whether this session currently holds a checkpoint under `name`.
  //: An empty name throws ErrorCode::invalid_argument, so this is not
  //: noexcept.
  [[nodiscard]] bool HasCheckpoint(std::string_view name) const;

 private:
  struct Impl;
  explicit PipelineSession(
      std::shared_ptr<const PipelinePackage> package,
      std::shared_ptr<detail::PipelineScheduler> scheduler);

  //: Shared rather than unique so a StageRun can hold the same execution
  //: state: a run outlives a moved or destroyed session wrapper instead of
  //: dangling. The pointer stays private and the session itself stays
  //: move-only, so callers cannot alias one session's state.
  std::shared_ptr<Impl> impl_;

  friend class Pipeline;
  friend class StageRun;
};

}  // namespace onnx_world_model
