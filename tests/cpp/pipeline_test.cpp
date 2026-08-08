#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "onnx_world_model/error.hpp"
#include "onnx_world_model/pipeline.hpp"

namespace {

int failures = 0;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    ++failures;
  }
}

template <typename Function>
void CheckThrows(Function&& function, const char* message) {
  try {
    function();
    Check(false, message);
  } catch (const onnx_world_model::Error&) {
  }
}

class IdentityBackend final : public onnx_world_model::ModelBackend {
 public:
  IdentityBackend() {
    metadata_.inputs.push_back({
        .name = "x",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {-1, 2},
    });
    metadata_.outputs.push_back({
        .name = "y",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {-1, 2},
    });
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    return {{"y", inputs.at("x")}};
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

constexpr std::string_view kManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "identity",
        "role": "dynamics",
        "run_on": "step",
        "inputs": [{"name": "x", "dtype": "FLOAT", "shape": ["batch", 2]}],
        "outputs": [{"name": "y", "dtype": "FLOAT", "shape": ["batch", 2]}],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT"
      }
    ],
    "connections": [],
    "stages": [
      {
        "name": "step",
        "kind": "single_pass",
        "components": ["identity"],
        "run_on": "step"
      }
    ],
    "inputs": [
      {
        "port": "identity.x",
        "kind": "external",
        "alias": "input",
        "semantic": "test.input",
        "required": true
      }
    ],
    "outputs": [
      {"port": "identity.y", "alias": "output"}
    ],
    "profile": {
      "name": "test-world",
      "version": "1.0"
    },
    "metadata": {
      "profile": "world-model",
      "model_type": "test"
    }
  },
  "component_files": {"identity": "model.onnx"}
}
)json";

constexpr std::string_view kProfilelessManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "identity",
        "role": "generic",
        "run_on": "always",
        "inputs": [{"name": "x", "dtype": "FLOAT", "shape": ["batch", 2]}],
        "outputs": [{"name": "y", "dtype": "FLOAT", "shape": ["batch", 2]}]
      }
    ],
    "connections": [],
    "stages": [
      {
        "name": "run",
        "kind": "single_pass",
        "components": ["identity"],
        "run_on": "always"
      }
    ],
    "inputs": [{"port": "identity.x", "kind": "external", "required": true}],
    "outputs": [{"port": "identity.y"}]
  },
  "component_files": {"identity": "model.onnx"}
}
)json";

}  // namespace

int main(int argument_count, char** arguments) {
  using onnx_world_model::Model;
  using onnx_world_model::PipelineManifest;
  using onnx_world_model::PipelinePackage;

  PipelineManifest manifest = PipelineManifest::Parse(kManifest);
  Check(manifest.profile() == "test-world", "profile");
  Check(manifest.profile_version() == "1.0", "profile version");
  Check(manifest.model_type() == "test", "model type");
  Check(manifest.components().size() == 1, "component count");
  Check(manifest.inputs()[0].name == "input", "external alias");
  Check(manifest.outputs()[0].name == "output", "output alias");

  PipelineManifest profileless =
      PipelineManifest::Parse(kProfilelessManifest);
  Check(profileless.profile().empty(), "optional profile");

  std::unordered_map<std::string, Model> models;
  models.emplace("identity", Model(std::make_shared<IdentityBackend>()));
  PipelinePackage package({}, manifest, std::move(models));
  Check(
      package.Component("identity").metadata().inputs[0].name == "x",
      "component lookup");

  std::string unsafe(kManifest);
  unsafe.replace(
      unsafe.find("model.onnx"),
      std::string("model.onnx").size(),
      "../model.onnx");
  CheckThrows(
      [&unsafe] { (void)PipelineManifest::Parse(unsafe); },
      "unsafe component path must fail");

  std::string missing_source(kManifest);
  const std::string input =
      R"json({
        "port": "identity.x",
        "kind": "external",
        "alias": "input",
        "semantic": "test.input",
        "required": true
      })json";
  missing_source.replace(
      missing_source.find(input),
      input.size(),
      "");
  CheckThrows(
      [&missing_source] { (void)PipelineManifest::Parse(missing_source); },
      "input without source must fail");

  std::string generated_without_recipe(kManifest);
  generated_without_recipe.replace(
      generated_without_recipe.find("\"external\""),
      std::string("\"external\"").size(),
      "\"generated\"");
  CheckThrows(
      [&generated_without_recipe] {
        (void)PipelineManifest::Parse(generated_without_recipe);
      },
      "generated input without recipe must fail");

  if (argument_count >= 2) {
    try {
      PipelineManifest loaded_manifest =
          PipelineManifest::Load(
              std::filesystem::path(arguments[1]) / "pipeline.json");
      Check(!loaded_manifest.profile().empty(), "loaded manifest profile");
    } catch (const std::exception& error) {
      std::cerr << "FAILED: manifest load: " << error.what() << '\n';
      ++failures;
    }
  }
  if (argument_count == 3) {
    try {
      onnx_world_model::RuntimeOptions options;
      options.ort_library_path = arguments[2];
      PipelinePackage loaded = PipelinePackage::Load(arguments[1], options);
      Check(
          !loaded.manifest().profile().empty(),
          "loaded package profile");
      const std::string component =
          loaded.manifest().components().front().name;
      Check(
          !loaded.Component(component).metadata().inputs.empty(),
          "loaded package model");
    } catch (const std::exception& error) {
      std::cerr << "FAILED: package load: " << error.what() << '\n';
      ++failures;
    }
  }

  if (failures == 0) {
    std::cout << "pipeline tests passed\n";
  }
  return failures == 0 ? 0 : 1;
}
