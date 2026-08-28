#pragma once

/**
 * @agent-file
 * @agent-purpose: Declares the Mobius pipeline contract: manifest value types, the validated PipelineManifest and PipelinePackage loaders, the shareable Pipeline, the per-trajectory PipelineSession, and its in-memory PipelineSessionSnapshot.
 * @agent-public-api: Endpoint, PipelineComponent, PipelineConnection, PipelineInputKind, PipelineInput, PipelineOutput, PipelineStage, PipelineState, PipelineAsset, PipelineManifest, PipelinePackage, PipelineRunOptions, Pipeline, PipelineSessionSnapshot, PipelineSession
 * @agent-invariants: Pipeline holds immutable component sessions through a shared_ptr and may be shared by callers, while PipelineSession is move-only and owns exactly one trajectory's mutable state; a manifest naming a capability outside PipelineManifest::SupportedCapabilities() is rejected during loading. RunStage and StepStage preserve the storage of the tensors they are given and may return device-backed tensors, so a caller reading a result on the host calls Tensor::CopyToCpu() first. PipelineSessionSnapshot is an immutable copyable capture of one session's mutable execution state that only PipelineSession::Snapshot() can produce; it records the package it came from, so Restore and Fork accept it only for a session built on that same PipelinePackage instance and otherwise throw ErrorCode::state.
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
//: process boundary.
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
  //: other's later runs, releases, or resets.
  [[nodiscard]] PipelineSession Fork() const;

 private:
  struct Impl;
  explicit PipelineSession(std::shared_ptr<const PipelinePackage> package);

  std::unique_ptr<Impl> impl_;

  friend class Pipeline;
};

}  // namespace onnx_world_model
