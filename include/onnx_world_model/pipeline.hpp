#pragma once

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

 private:
  struct Impl;
  explicit PipelineSession(std::shared_ptr<const PipelinePackage> package);

  std::unique_ptr<Impl> impl_;

  friend class Pipeline;
};

}  // namespace onnx_world_model
