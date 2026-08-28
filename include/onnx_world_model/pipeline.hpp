#pragma once

/**
 * @agent-file
 * @agent-purpose: Declares the Mobius pipeline contract: manifest value types, the validated PipelineManifest and PipelinePackage loaders, the shareable Pipeline, the per-trajectory PipelineSession, its in-memory PipelineSessionSnapshot, its named in-memory checkpoints, and the incremental StageRun that reports each step of a stage as a StageEvent.
 * @agent-public-api: Endpoint, PipelineComponent, PipelineConnection, PipelineInputKind, PipelineInput, PipelineOutput, PipelineStage, PipelineState, PipelineAsset, PipelineManifest, PipelinePackage, PipelineRunOptions, Pipeline, PipelineSessionSnapshot, StageEventKind, StageEvent, StageRun, PipelineSession
 * @agent-invariants: Pipeline holds immutable component sessions through a shared_ptr and may be shared by callers, while PipelineSession is move-only and owns exactly one trajectory's mutable state; a manifest naming a capability outside PipelineManifest::SupportedCapabilities() is rejected during loading. RunStage and StepStage preserve the storage of the tensors they are given and may return device-backed tensors, so a caller reading a result on the host calls Tensor::CopyToCpu() first. PipelineSessionSnapshot is an immutable copyable capture of one session's mutable execution state that only PipelineSession::Snapshot() can produce; it records the package it came from, so Restore and Fork accept it only for a session built on that same PipelinePackage instance and otherwise throw ErrorCode::state. Named checkpoints are in-memory transaction markers held beside that execution state, not inside it: a checkpoint name is never empty, Checkpoint captures the same fields Snapshot does, a snapshot never contains checkpoints, RestoreCheckpoint and DropCheckpoint throw ErrorCode::state for an unknown name instead of doing nothing, checkpoints outlive stage execution and Restore, Reset drops them all, and a forked session starts with an empty checkpoint namespace. BeginStage and RunStage share one StageRun state machine and produce identical results; RunStage drains it under one session-lock acquisition so ordinary concurrent calls retain whole-stage serialization. A StageRun is move-only, single-consumer, and synchronous -- Step() blocks until exactly one model or scheduler step finishes -- and it holds the session's only run slot until it completes, is cancelled, or is destroyed; while it holds that slot the session throws ErrorCode::state from BeginStage, RunStage, StepStage, Snapshot, Restore, Fork, Checkpoint, RestoreCheckpoint, DropCheckpoint, Reset, and ReleaseStage, while outputs(), state(), and HasCheckpoint() stay legal. PipelineRunOptions::cancellation carries an optional CancellationToken that every execution path checks at its own boundaries; StageRun::RequestCancellation signals an in-flight step without taking the session lock, while StageRun::Cancel takes it and only closes the handle, so the two are different operations and neither rolls anything back.
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

class PipelinePackage {
 public:
  PipelinePackage(
      std::filesystem::path root,
      PipelineManifest manifest,
      std::unordered_map<std::string, Model> components);

  static PipelinePackage Load(
      const std::filesystem::path& directory,
      const RuntimeOptions& options = {});

  [[nodiscard]] const std::filesystem::path& root() const noexcept;
  [[nodiscard]] const PipelineManifest& manifest() const noexcept;
  [[nodiscard]] const Model& Component(std::string_view name) const;
  [[nodiscard]] std::unordered_map<std::string, std::vector<std::string>>
  execution_providers() const;

 private:
  std::filesystem::path root_;
  PipelineManifest manifest_;
  std::unordered_map<std::string, Model> components_;
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

class Pipeline {
 public:
  explicit Pipeline(PipelinePackage package);

  static Pipeline Load(
      const std::filesystem::path& directory,
      const RuntimeOptions& options = {});

  [[nodiscard]] const PipelineManifest& manifest() const noexcept;
  [[nodiscard]] std::unordered_map<std::string, std::vector<std::string>>
  execution_providers() const;
  [[nodiscard]] PipelineSession CreateSession() const;

 private:
  std::shared_ptr<const PipelinePackage> package_;

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
  //: applied to the session, and rethrows.
  [[nodiscard]] StageEvent Step();
  //: Drains the remaining steps under one session-lock acquisition and
  //: returns the same outputs RunStage returns for this stage. Calling it
  //: after the run completed returns that cached result again without
  //: re-running anything.
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
  [[nodiscard]] StageRun BeginStage(
      std::string_view stage,
      const NamedTensors& inputs = {},
      const NamedTensors& overrides = {},
      const PipelineRunOptions& options = {});
  //: Executes exactly one pass of `stage` for a caller that drives its own
  //: loop. This is the low-level primitive underneath BeginStage, so it
  //: throws ErrorCode::state while a StageRun is active rather than
  //: interleaving with it.
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
  explicit PipelineSession(std::shared_ptr<const PipelinePackage> package);

  //: Shared rather than unique so a StageRun can hold the same execution
  //: state: a run outlives a moved or destroyed session wrapper instead of
  //: dangling. The pointer stays private and the session itself stays
  //: move-only, so callers cannot alias one session's state.
  std::shared_ptr<Impl> impl_;

  friend class Pipeline;
  friend class StageRun;
};

}  // namespace onnx_world_model
