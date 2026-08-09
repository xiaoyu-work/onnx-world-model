#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
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

template <typename Function>
void CheckThrowsMessage(
    Function&& function,
    std::string_view expected,
    const char* message) {
  try {
    function();
    Check(false, message);
  } catch (const onnx_world_model::Error& error) {
    if (std::string_view(error.what()).find(expected) ==
        std::string_view::npos) {
      std::cerr << "FAILED: " << message << " (got: " << error.what() << ")\n";
      ++failures;
    }
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

class IncrementBackend final : public onnx_world_model::ModelBackend {
 public:
  IncrementBackend() {
    metadata_.inputs.push_back({
        .name = "state",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {1},
    });
    metadata_.outputs.push_back({
        .name = "next_state",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {1},
    });
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    onnx_world_model::Tensor result = inputs.at("state");
    auto value = std::span(
        reinterpret_cast<float*>(result.mutable_bytes().data()),
        result.element_count());
    value[0] += 1.0F;
    return {{"next_state", std::move(result)}};
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

class VelocityBackend final : public onnx_world_model::ModelBackend {
 public:
  VelocityBackend() {
    metadata_.inputs.push_back({
        .name = "state",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {1},
    });
    metadata_.outputs.push_back({
        .name = "next_state",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {1},
    });
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    (void)inputs;
    const std::array<float, 1> velocity{1.0F};
    return {
        {
            "next_state",
            onnx_world_model::Tensor::FromValues<float>(
                {1}, std::span(velocity)),
        },
    };
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

class NonlinearVelocityBackend final : public onnx_world_model::ModelBackend {
 public:
  NonlinearVelocityBackend() {
    metadata_.inputs.push_back({
        .name = "state",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {1},
    });
    metadata_.outputs.push_back({
        .name = "next_state",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {1},
    });
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    const std::array<float, 1> velocity{
        1.0F + 0.25F * inputs.at("state").values<float>()[0]};
    return {
        {
            "next_state",
            onnx_world_model::Tensor::FromValues<float>(
                {1}, std::span(velocity)),
        },
    };
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

class AutoregressiveBackend final : public onnx_world_model::ModelBackend {
 public:
  AutoregressiveBackend() {
    metadata_.inputs = {
        {
            .name = "tokens",
            .data_type = onnx_world_model::DataType::int64,
            .shape = {1, -1},
        },
        {
            .name = "state",
            .data_type = onnx_world_model::DataType::float32,
            .shape = {1},
        },
    };
    metadata_.outputs = {
        {
            .name = "logits",
            .data_type = onnx_world_model::DataType::float32,
            .shape = {1, -1, 4},
        },
        {
            .name = "next_state",
            .data_type = onnx_world_model::DataType::float32,
            .shape = {1},
        },
    };
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    const auto& tokens = inputs.at("tokens");
    const float state = inputs.at("state").values<float>()[0];
    const auto next_token =
        static_cast<std::size_t>(std::min(state + 1.0F, 3.0F));
    onnx_world_model::Tensor logits =
        onnx_world_model::Tensor::Zeros(
            onnx_world_model::DataType::float32,
            {1, tokens.shape()[1], 4});
    auto logits_values = std::span(
        reinterpret_cast<float*>(logits.mutable_bytes().data()),
        logits.element_count());
    logits_values[
        static_cast<std::size_t>((tokens.shape()[1] - 1) * 4) + next_token] =
        1.0F;
    const std::array<float, 1> next_state{state + 1.0F};
    return {
        {"logits", std::move(logits)},
        {
            "next_state",
            onnx_world_model::Tensor::FromValues<float>(
                {1}, std::span(next_state)),
        },
    };
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

class VideoVisionBackend final : public onnx_world_model::ModelBackend {
 public:
  VideoVisionBackend() {
    metadata_.inputs = {
        {
            .name = "pixel_values",
            .data_type = onnx_world_model::DataType::float32,
            .shape = {8, 1},
        },
        {
            .name = "grid_thw",
            .data_type = onnx_world_model::DataType::int64,
            .shape = {3},
        },
    };
    metadata_.outputs = {{
        .name = "image_features",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {4, 2},
    }};
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    const auto grid = inputs.at("grid_thw").values<std::int64_t>();
    Check(
        grid[0] == 2 && grid[1] == 2 && grid[2] == 4,
        "video vision grid");
    return {{
        "image_features",
        onnx_world_model::Tensor::Zeros(
            onnx_world_model::DataType::float32, {4, 2}),
    }};
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

class VideoEmbeddingBackend final : public onnx_world_model::ModelBackend {
 public:
  VideoEmbeddingBackend() {
    metadata_.inputs = {
        {
            .name = "input_ids",
            .data_type = onnx_world_model::DataType::int64,
            .shape = {1, 7},
        },
        {
            .name = "image_features",
            .data_type = onnx_world_model::DataType::float32,
            .shape = {-1, 2},
        },
        {
            .name = "video_features",
            .data_type = onnx_world_model::DataType::float32,
            .shape = {-1, 2},
        },
    };
    metadata_.outputs = {{
        .name = "inputs_embeds",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {1, 7, 2},
    }};
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    Check(
        inputs.at("image_features").shape()[0] == 0,
        "video route leaves image features empty");
    Check(
        inputs.at("video_features").shape()[0] == 4,
        "video route receives projected features");
    return {{
        "inputs_embeds",
        onnx_world_model::Tensor::Zeros(
            onnx_world_model::DataType::float32, {1, 7, 2}),
    }};
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

class VideoPositionBackend final : public onnx_world_model::ModelBackend {
 public:
  VideoPositionBackend() {
    metadata_.inputs = {
        {
            .name = "inputs_embeds",
            .data_type = onnx_world_model::DataType::float32,
            .shape = {1, 7, 2},
        },
        {
            .name = "position_ids",
            .data_type = onnx_world_model::DataType::int64,
            .shape = {3, 1, 7},
        },
    };
    metadata_.outputs = {{
        .name = "logits",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {1, 7, 1},
    }};
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    const auto positions =
        inputs.at("position_ids").values<std::int64_t>();
    constexpr std::array<std::int64_t, 21> expected{
        0, 1, 1, 3, 4, 4, 6,
        0, 1, 1, 3, 4, 4, 6,
        0, 1, 2, 3, 4, 5, 6,
    };
    Check(
        std::ranges::equal(positions, expected),
        "frame-level video mRoPE positions");
    return {{
        "logits",
        onnx_world_model::Tensor::Zeros(
            onnx_world_model::DataType::float32, {1, 7, 1}),
    }};
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

constexpr std::string_view kVideoRouteManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "reasoner_vision_encoder",
        "role": "encoder",
        "run_on": "prefill",
        "inputs": [
          {"name": "pixel_values", "dtype": "FLOAT", "shape": [8, 1]},
          {"name": "grid_thw", "dtype": "INT64", "shape": [3]}
        ],
        "outputs": [
          {"name": "image_features", "dtype": "FLOAT", "shape": [4, 2]}
        ],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT"
      },
      {
        "name": "reasoner_embedding",
        "role": "embedding",
        "run_on": "always",
        "inputs": [
          {"name": "input_ids", "dtype": "INT64", "shape": [1, 7]},
          {"name": "image_features", "dtype": "FLOAT", "shape": ["image", 2]},
          {"name": "video_features", "dtype": "FLOAT", "shape": ["video", 2]}
        ],
        "outputs": [
          {"name": "inputs_embeds", "dtype": "FLOAT", "shape": [1, 7, 2]}
        ],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT"
      },
      {
        "name": "reasoner_decoder",
        "role": "decoder",
        "run_on": "decode",
        "inputs": [
          {"name": "inputs_embeds", "dtype": "FLOAT", "shape": [1, 7, 2]},
          {"name": "position_ids", "dtype": "INT64", "shape": [3, 1, 7]}
        ],
        "outputs": [
          {"name": "logits", "dtype": "FLOAT", "shape": [1, 7, 1]}
        ],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT"
      }
    ],
    "connections": [
      {
        "source": "reasoner_vision_encoder.image_features",
        "target": "reasoner_embedding.image_features"
      },
      {
        "source": "reasoner_embedding.inputs_embeds",
        "target": "reasoner_decoder.inputs_embeds"
      }
    ],
    "stages": [
      {
        "name": "reasoner_prompt",
        "kind": "single_pass",
        "components": [
          "reasoner_vision_encoder",
          "reasoner_embedding"
        ],
        "run_on": "prefill"
      },
      {
        "name": "reasoner_decode",
        "kind": "autoregressive",
        "components": [
          "reasoner_embedding",
          "reasoner_decoder"
        ],
        "run_on": "decode",
        "options": {
          "tokenizer_asset": "tokenizer.json",
          "sampling": {"do_sample": false},
          "stop": {"kind": "token_ids", "eos_token_ids": [0]},
          "max_tokens": {"default": 1, "limit": 1},
          "state_names": []
        }
      }
    ],
    "inputs": [
      {
        "port": "reasoner_vision_encoder.pixel_values",
        "kind": "external",
        "semantic": "vision.pixel_values",
        "required": true
      },
      {
        "port": "reasoner_vision_encoder.grid_thw",
        "kind": "external",
        "semantic": "vision.grid_thw",
        "required": true
      },
      {
        "port": "reasoner_embedding.input_ids",
        "kind": "external",
        "semantic": "text.token_ids",
        "required": true
      },
      {
        "port": "reasoner_embedding.video_features",
        "kind": "external",
        "semantic": "vision.video_features",
        "required": false,
        "presence": "video_understanding"
      },
      {
        "port": "reasoner_decoder.position_ids",
        "kind": "generated",
        "semantic": "position.multimodal",
        "generator": {
          "kind": "multimodal_position_ids",
          "parameters": {
            "source": "reasoner_embedding.input_ids",
            "axes": 3
          }
        }
      }
    ],
    "outputs": [{"port": "reasoner_decoder.logits"}],
    "profile": {"name": "video-test", "version": "1.0"},
    "required_capabilities": ["position_program"],
    "assets": [{"path": "tokenizer.json"}],
    "metadata": {
      "vision_understanding": {
        "encoder": "reasoner_vision_encoder",
        "tokens": {"image": 19, "video": 18},
        "routing": {
          "image": "reasoner_embedding.image_features",
          "video": "reasoner_embedding.video_features"
        },
        "preprocessing": {
          "patchify": {"merge_size": 2}
        }
      }
    }
  },
  "component_files": {
    "reasoner_decoder": "reasoner_decoder/model.onnx",
    "reasoner_embedding": "reasoner_embedding/model.onnx",
    "reasoner_vision_encoder": "reasoner_vision_encoder/model.onnx"
  }
}
)json";

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

constexpr std::string_view kStateManifest = R"json(
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
      }
    ],
    "inputs": [
      {
        "port": "counter.state",
        "kind": "generated",
        "required": true,
        "semantic": "state.initial",
        "generator": {"kind": "zeros"}
      }
    ],
    "outputs": [{"state": "counter_state", "alias": "value"}],
    "profile": {"name": "counter-world", "version": "1.0"},
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
  "component_files": {"counter": "model.onnx"}
}
)json";

constexpr std::string_view kIterativeManifest = R"json(
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
        "name": "iterate",
        "kind": "iterative",
        "components": ["counter"],
        "run_on": "step",
        "options": {
          "scheduler": {
            "kind": "FlowMatchEulerDiscreteScheduler",
            "config_asset": "scheduler.json"
          },
          "default_steps": 3,
          "timestep": {},
          "state_inputs": ["counter.state"]
        },
        "capabilities": ["loop_carried_state"]
      }
    ],
    "inputs": [
      {
        "port": "counter.state",
        "kind": "generated",
        "required": true,
        "semantic": "state.initial",
        "generator": {"kind": "zeros"}
      }
    ],
    "outputs": [{"state": "counter_state", "alias": "value"}],
    "profile": {"name": "counter-world", "version": "1.0"},
    "states": [
      {
        "name": "counter_state",
        "kind": "recurrent",
        "input": "counter.state",
        "output": "counter.next_state",
        "lifetime": "request",
        "release_after": "iterate"
      }
    ],
    "assets": [{"path": "scheduler.json"}],
    "required_capabilities": ["loop_carried_state"]
  },
  "component_files": {"counter": "model.onnx"}
}
)json";

constexpr std::string_view kAutoregressiveManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "decoder",
        "role": "decoder",
        "run_on": "decode",
        "inputs": [
          {"name": "tokens", "dtype": "INT64", "shape": [1, "sequence"]},
          {"name": "state", "dtype": "FLOAT", "shape": [1]}
        ],
        "outputs": [
          {"name": "logits", "dtype": "FLOAT", "shape": [1, "sequence", 4]},
          {"name": "next_state", "dtype": "FLOAT", "shape": [1]}
        ],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT"
      }
    ],
    "connections": [
      {
        "source": "decoder.next_state",
        "target": "decoder.state",
        "recurrent": true
      }
    ],
    "stages": [
      {
        "name": "decode",
        "kind": "autoregressive",
        "components": ["decoder"],
        "run_on": "decode",
        "options": {
          "tokenizer_asset": "tokenizer.json",
          "sampling": {"do_sample": false},
          "stop": {"kind": "token_ids", "eos_token_ids": [3]},
          "max_tokens": {"default": 4, "limit": 4},
          "state_names": ["cache"]
        },
        "capabilities": ["loop_carried_state"]
      }
    ],
    "inputs": [
      {
        "port": "decoder.state",
        "kind": "generated",
        "required": true,
        "semantic": "kv_cache.initial",
        "generator": {"kind": "zeros"}
      },
      {
        "port": "decoder.tokens",
        "kind": "external",
        "required": true,
        "semantic": "text.token_ids"
      }
    ],
    "outputs": [{"port": "decoder.logits"}],
    "profile": {"name": "decoder-world", "version": "1.0"},
    "states": [
      {
        "name": "cache",
        "kind": "kv_cache",
        "input": "decoder.state",
        "output": "decoder.next_state",
        "lifetime": "sequence",
        "release_after": "decode"
      }
    ],
    "assets": [{"path": "tokenizer.json"}],
    "required_capabilities": ["loop_carried_state"]
  },
  "component_files": {"decoder": "model.onnx"}
}
)json";

class GuidedVelocityBackend final : public onnx_world_model::ModelBackend {
 public:
  struct Pass {
    std::int64_t prompt{0};
    std::int64_t und_len{0};
    std::vector<std::int64_t> text_indexes;
    std::vector<std::int64_t> vision_sequence_indexes;
    std::vector<std::int64_t> vision_mse_loss_indexes;
  };

  GuidedVelocityBackend() {
    metadata_.inputs = {
        {
            .name = "input_ids",
            .data_type = onnx_world_model::DataType::int64,
            .shape = {-1},
        },
        {
            .name = "text_indexes",
            .data_type = onnx_world_model::DataType::int64,
            .shape = {-1},
        },
        {
            .name = "und_len",
            .data_type = onnx_world_model::DataType::int64,
            .shape = {1},
        },
        {
            .name = "vision_tokens",
            .data_type = onnx_world_model::DataType::float32,
            .shape = {-1, 2},
        },
        {
            .name = "vision_sequence_indexes",
            .data_type = onnx_world_model::DataType::int64,
            .shape = {-1},
        },
        {
            .name = "vision_timestep_token_indexes",
            .data_type = onnx_world_model::DataType::int64,
            .shape = {-1},
        },
        {
            .name = "vision_mse_loss_indexes",
            .data_type = onnx_world_model::DataType::int64,
            .shape = {-1},
        },
        {
            .name = "vision_timesteps",
            .data_type = onnx_world_model::DataType::float32,
            .shape = {-1},
        },
    };
    metadata_.outputs = {{
        .name = "vision_pred",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {-1, 2},
    }};
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    ++calls;
    const auto prompt = inputs.at("input_ids").values<std::int64_t>();
    const auto indexes =
        inputs.at("vision_timestep_token_indexes").values<std::int64_t>();
    Check(
        inputs.at("vision_timesteps").element_count() == indexes.size(),
        "one timestep per noisy vision token");
    Check(
        inputs.at("vision_tokens").shape()[0] == 3,
        "the generator always sees every vision token");
    const auto values = [](const onnx_world_model::Tensor& tensor) {
      const auto span = tensor.values<std::int64_t>();
      return std::vector<std::int64_t>(span.begin(), span.end());
    };
    passes.push_back(Pass{
        .prompt = prompt[0],
        .und_len = inputs.at("und_len").values<std::int64_t>()[0],
        .text_indexes = values(inputs.at("text_indexes")),
        .vision_sequence_indexes = values(inputs.at("vision_sequence_indexes")),
        .vision_mse_loss_indexes = values(inputs.at("vision_mse_loss_indexes")),
    });
    // Predict a constant velocity that identifies the prompt, so the test can
    // verify the classifier-free combination exactly.
    onnx_world_model::Tensor prediction = onnx_world_model::Tensor::Zeros(
        onnx_world_model::DataType::float32,
        {static_cast<std::int64_t>(indexes.size()), 2});
    auto span = std::span(
        reinterpret_cast<float*>(prediction.mutable_bytes().data()),
        prediction.element_count());
    std::ranges::fill(span, static_cast<float>(prompt[0]));
    return {{"vision_pred", std::move(prediction)}};
  }

  mutable int calls{0};
  mutable std::vector<Pass> passes;

 private:
  onnx_world_model::ModelMetadata metadata_;
};

class ConditioningEncoderBackend final
    : public onnx_world_model::ModelBackend {
 public:
  ConditioningEncoderBackend() {
    metadata_.inputs = {{
        .name = "sample",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {1, 3, 1, 2, 2},
    }};
    metadata_.outputs = {{
        .name = "latent",
        .data_type = onnx_world_model::DataType::float32,
        .shape = {1, 2, 1, 1, 1},
    }};
  }

  [[nodiscard]] const onnx_world_model::ModelMetadata& metadata()
      const noexcept override {
    return metadata_;
  }

  [[nodiscard]] onnx_world_model::NamedTensors Run(
      const onnx_world_model::NamedTensors& inputs) const override {
    (void)inputs;
    const std::array<float, 2> latent{10.0F, 10.0F};
    return {{
        "latent",
        onnx_world_model::Tensor::FromValues<float>(
            {1, 2, 1, 1, 1}, std::span(latent)),
    }};
  }

 private:
  onnx_world_model::ModelMetadata metadata_;
};

constexpr std::string_view kGuidanceManifest = R"json(
{
  "format": "mobius-pipeline",
  "schema_version": "1.1",
  "manifest": {
    "schema_version": "1.1",
    "components": [
      {
        "name": "generator",
        "role": "dynamics",
        "run_on": "step",
        "inputs": [
          {"name": "input_ids", "dtype": "INT64", "shape": ["text"]},
          {"name": "text_indexes", "dtype": "INT64", "shape": ["text"]},
          {"name": "und_len", "dtype": "INT64", "shape": [1]},
          {"name": "vision_tokens", "dtype": "FLOAT", "shape": ["vision", 2]},
          {
            "name": "vision_sequence_indexes",
            "dtype": "INT64",
            "shape": ["vision"]
          },
          {
            "name": "vision_timestep_token_indexes",
            "dtype": "INT64",
            "shape": ["noisy"]
          },
          {
            "name": "vision_mse_loss_indexes",
            "dtype": "INT64",
            "shape": ["noisy"]
          },
          {"name": "vision_timesteps", "dtype": "FLOAT", "shape": ["noisy"]}
        ],
        "outputs": [
          {"name": "vision_pred", "dtype": "FLOAT", "shape": ["noisy", 2]}
        ],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT"
      },
      {
        "name": "video_encoder",
        "role": "encoder",
        "run_on": "on_demand",
        "inputs": [
          {"name": "sample", "dtype": "FLOAT", "shape": [1, 3, 1, 2, 2]}
        ],
        "outputs": [
          {"name": "latent", "dtype": "FLOAT", "shape": [1, 2, 1, 1, 1]}
        ],
        "preferred_execution_providers": ["cpu"],
        "parameter_dtype": "FLOAT"
      }
    ],
    "connections": [
      {
        "source": "generator.vision_pred",
        "target": "generator.vision_tokens",
        "recurrent": true,
        "transform": "scheduler_step",
        "parameters": {
          "scheduler_asset": "scheduler.json",
          "stage": "world_generation",
          "state": "vision_state",
          "timestep_input": "generator.vision_timesteps"
        }
      }
    ],
    "stages": [
      {
        "name": "world_generation",
        "kind": "iterative",
        "components": ["generator"],
        "run_on": "step",
        "capabilities": [
          "loop_carried_state",
          "classifier_free_guidance",
          "conditioned_diffusion"
        ],
        "options": {
          "scheduler": {
            "kind": "FlowMatchEulerDiscreteScheduler",
            "config_asset": "scheduler.json",
            "mode_overrides": {
              "image_to_video": {"guidance_scale": 3.0}
            }
          },
          "guidance": {
            "kind": "classifier_free",
            "conditioning_input": "generator.input_ids",
            "scale_option": "guidance_scale",
            "default_scale": 1.0,
            "combine": "unconditional + scale * (conditional - unconditional)"
          },
          "conditioning": {
            "vision": {
              "encoder_stage": "encode_video",
              "encoder_input": "video_encoder.sample",
              "encoder_output": "video_encoder.latent",
              "state": "vision_state",
              "conditioned_latent_frames_option":
                "vision_conditioned_latent_frames",
              "default_conditioned_latent_frames": [],
              "preprocessing": {
                "resize": "stretch_to_target",
                "resample": "bicubic",
                "convert_rgb": true,
                "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
              },
              "packing": {
                "spatial_patch_size": 1,
                "temporal_patch_size": 1,
                "input_layout": "BCTHW",
                "output_layout": "NC",
                "channel_order": "patch_height_patch_width_channel"
              }
            }
          },
          "default_steps": 1,
          "timestep": {},
          "state_inputs": ["generator.vision_tokens"]
        }
      },
      {
        "name": "encode_video",
        "kind": "single_pass",
        "components": ["video_encoder"],
        "run_on": "on_demand"
      }
    ],
    "inputs": [
      {
        "port": "generator.input_ids",
        "kind": "external",
        "alias": "generator_input_ids",
        "semantic": "text.token_ids",
        "required": true
      },
      {
        "port": "generator.vision_tokens",
        "kind": "external",
        "alias": "initial_vision_tokens",
        "semantic": "diffusion.initial_vision_latent",
        "required": true
      },
      {
        "port": "generator.vision_timestep_token_indexes",
        "kind": "generated",
        "semantic": "packing.vision_timestep_token_indexes",
        "generator": {
          "kind": "packed_sequence_layout",
          "parameters": {
            "modality": "vision",
            "source": "generator.vision_tokens",
            "index_kind": "vision_timestep_token_indexes"
          }
        }
      },
      {
        "port": "generator.text_indexes",
        "kind": "generated",
        "semantic": "packing.text_indexes",
        "generator": {
          "kind": "packed_sequence_layout",
          "parameters": {
            "modality": "text",
            "source": "generator.input_ids",
            "index_kind": "text_indexes"
          }
        }
      },
      {
        "port": "generator.und_len",
        "kind": "generated",
        "semantic": "packing.und_len",
        "generator": {
          "kind": "packed_sequence_layout",
          "parameters": {
            "modality": "text",
            "source": "generator.input_ids",
            "understanding_prefix": true,
            "index_kind": "und_len"
          }
        }
      },
      {
        "port": "generator.vision_sequence_indexes",
        "kind": "generated",
        "semantic": "packing.vision_sequence_indexes",
        "generator": {
          "kind": "packed_sequence_layout",
          "parameters": {
            "modality": "vision",
            "source": "generator.vision_tokens",
            "index_kind": "vision_sequence_indexes"
          }
        }
      },
      {
        "port": "generator.vision_mse_loss_indexes",
        "kind": "generated",
        "semantic": "packing.vision_mse_loss_indexes",
        "generator": {
          "kind": "packed_sequence_layout",
          "parameters": {
            "modality": "vision",
            "source": "generator.vision_tokens",
            "index_kind": "vision_mse_loss_indexes"
          }
        }
      },
      {
        "port": "generator.vision_timesteps",
        "kind": "generated",
        "semantic": "diffusion.vision.timesteps",
        "generator": {
          "kind": "scheduler_timesteps",
          "parameters": {"stage": "world_generation", "modality": "vision"}
        }
      },
      {
        "port": "video_encoder.sample",
        "kind": "external",
        "alias": "video_encoder_sample",
        "semantic": "video.frames",
        "required": true
      }
    ],
    "outputs": [
      {"state": "vision_state", "alias": "vision_latent"},
      {"port": "generator.vision_pred", "alias": "vision_velocity"},
      {"port": "video_encoder.latent", "alias": "encoded_video_latent"}
    ],
    "profile": {"name": "guided-world", "version": "1.0"},
    "states": [
      {
        "name": "vision_state",
        "kind": "diffusion_latent",
        "input": "generator.vision_tokens",
        "output": "generator.vision_pred",
        "lifetime": "request",
        "release_after": "world_generation",
        "sequence_axis": 0
      }
    ],
    "assets": [{"path": "scheduler.json"}],
    "required_capabilities": [
      "iterative_scheduler",
      "loop_carried_state",
      "packed_sequence_program"
    ]
  },
  "component_files": {
    "generator": "generator/model.onnx",
    "video_encoder": "video_encoder/model.onnx"
  }
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
  onnx_world_model::Pipeline pipeline(std::move(package));
  onnx_world_model::PipelineSession pipeline_session =
      pipeline.CreateSession();
  const std::array<float, 2> input_values{3.0F, 4.0F};
  auto stage_outputs = pipeline_session.RunStage(
      "step",
      {
          {
              "input",
              onnx_world_model::Tensor::FromValues<float>(
                  {1, 2}, std::span(input_values)),
          },
      });
  Check(
      stage_outputs.at("output").values<float>()[1] == 4.0F,
      "single-pass pipeline execution");

  PipelineManifest video_manifest =
      PipelineManifest::Parse(kVideoRouteManifest);
  std::unordered_map<std::string, Model> video_models;
  video_models.emplace(
      "reasoner_vision_encoder",
      Model(std::make_shared<VideoVisionBackend>()));
  video_models.emplace(
      "reasoner_embedding",
      Model(std::make_shared<VideoEmbeddingBackend>()));
  video_models.emplace(
      "reasoner_decoder",
      Model(std::make_shared<VideoPositionBackend>()));
  onnx_world_model::Pipeline video_pipeline(
      PipelinePackage({}, video_manifest, std::move(video_models)));
  auto video_session = video_pipeline.CreateSession();
  const std::array<std::int64_t, 3> video_grid{2, 2, 4};
  const std::array<std::int64_t, 7> video_ids{
      100, 18, 18, 101, 18, 18, 102};
  onnx_world_model::PipelineRunOptions video_options;
  video_options.strings.emplace("vision_modality", "video");
  (void)video_session.RunStage(
      "reasoner_prompt",
      {
          {
              "vision.pixel_values",
              onnx_world_model::Tensor::Zeros(
                  onnx_world_model::DataType::float32, {8, 1}),
          },
          {
              "vision.grid_thw",
              onnx_world_model::Tensor::FromValues<std::int64_t>(
                  {3}, std::span(video_grid)),
          },
          {
              "text.token_ids",
              onnx_world_model::Tensor::FromValues<std::int64_t>(
                  {1, 7}, std::span(video_ids)),
          },
      },
      {
          {
              "reasoner_embedding.image_features",
              onnx_world_model::Tensor::Zeros(
                  onnx_world_model::DataType::float32, {0, 2}),
          },
      },
      video_options);
  auto video_outputs = video_session.RunStage(
      "reasoner_decode",
      {},
      {
          {
              "reasoner_embedding.image_features",
              onnx_world_model::Tensor::Zeros(
                  onnx_world_model::DataType::float32, {0, 2}),
          },
          {
              "reasoner_embedding.video_features",
              onnx_world_model::Tensor::Zeros(
                  onnx_world_model::DataType::float32, {0, 2}),
          },
      });
  Check(
      video_outputs.at("logits").shape() ==
          std::vector<std::int64_t>({1, 7, 1}),
      "video understanding stage output");

  PipelineManifest state_manifest =
      PipelineManifest::Parse(kStateManifest);
  std::unordered_map<std::string, Model> state_models;
  state_models.emplace(
      "counter", Model(std::make_shared<IncrementBackend>()));
  onnx_world_model::Pipeline state_pipeline(
      PipelinePackage({}, state_manifest, std::move(state_models)));
  auto state_session = state_pipeline.CreateSession();
  auto first_state = state_session.RunStage("transition");
  auto second_state = state_session.RunStage("transition");
  Check(
      first_state.at("value").values<float>()[0] == 1.0F,
      "generated initial state");
  Check(
      second_state.at("value").values<float>()[0] == 2.0F,
      "recurrent state update");
  state_session.ReleaseStage("transition");
  Check(!state_session.state("counter_state").has_value(), "state release");

  PipelineManifest iterative_manifest =
      PipelineManifest::Parse(kIterativeManifest);
  std::unordered_map<std::string, Model> iterative_models;
  iterative_models.emplace(
      "counter", Model(std::make_shared<IncrementBackend>()));
  onnx_world_model::Pipeline iterative_pipeline(
      PipelinePackage({}, iterative_manifest, std::move(iterative_models)));
  auto iterative_session = iterative_pipeline.CreateSession();
  auto iterative_output = iterative_session.RunStage("iterate");
  Check(
      iterative_output.at("value").values<float>()[0] == 3.0F,
      "iterative stage loop");
  iterative_session.ReleaseStage("iterate");
  auto restarted_output = iterative_session.RunStage("iterate");
  Check(
      restarted_output.at("value").values<float>()[0] == 3.0F,
      "iterative cursor reset");

  std::string scheduled_document(kIterativeManifest);
  const std::string recurrent = R"json("recurrent": true)json";
  scheduled_document.replace(
      scheduled_document.find(recurrent),
      recurrent.size(),
      R"json("recurrent": true,
        "transform": "scheduler_step",
        "parameters": {
          "scheduler_asset": "scheduler.json",
          "stage": "iterate",
          "state": "counter_state"
        })json");
  const std::string capabilities = R"json(["loop_carried_state"])json";
  scheduled_document.replace(
      scheduled_document.rfind(capabilities),
      capabilities.size(),
      R"json(["iterative_scheduler", "loop_carried_state"])json");
  PipelineManifest scheduled_manifest =
      PipelineManifest::Parse(scheduled_document);
  const auto scheduler_root =
      std::filesystem::temp_directory_path() /
      "onnx-world-model-scheduler-test";
  std::filesystem::remove_all(scheduler_root);
  std::filesystem::create_directories(scheduler_root);
  {
    std::ofstream scheduler(scheduler_root / "scheduler.json");
    scheduler << R"json({
      "_class_name":"FlowMatchEulerDiscreteScheduler",
      "num_train_timesteps":1000,
      "shift":1.0
    })json";
  }
  std::unordered_map<std::string, Model> scheduled_models;
  scheduled_models.emplace(
      "counter", Model(std::make_shared<VelocityBackend>()));
  onnx_world_model::Pipeline scheduled_pipeline(PipelinePackage(
      scheduler_root, scheduled_manifest, std::move(scheduled_models)));
  auto scheduled_session = scheduled_pipeline.CreateSession();
  auto scheduled_output = scheduled_session.RunStage("iterate");
  Check(
      std::abs(scheduled_output.at("value").values<float>()[0] + 1.0F) <
          1e-5F,
      "flow-match scheduler steps");

  std::string unipc_document = scheduled_document;
  const std::string flow_scheduler = "FlowMatchEulerDiscreteScheduler";
  unipc_document.replace(
      unipc_document.find(flow_scheduler),
      flow_scheduler.size(),
      "UniPCMultistepScheduler");
  PipelineManifest unipc_manifest =
      PipelineManifest::Parse(unipc_document);
  {
    std::ofstream scheduler(scheduler_root / "scheduler.json");
    scheduler << R"json({
      "_class_name":"UniPCMultistepScheduler",
      "num_train_timesteps":1000,
      "flow_shift":1.0,
      "use_flow_sigmas":true,
      "use_karras_sigmas":false,
      "prediction_type":"flow_prediction",
      "predict_x0":true,
      "solver_order":2,
      "solver_type":"bh2",
      "lower_order_final":true,
      "thresholding":false,
      "final_sigmas_type":"zero"
    })json";
  }
  std::unordered_map<std::string, Model> unipc_models;
  unipc_models.emplace(
      "counter", Model(std::make_shared<NonlinearVelocityBackend>()));
  onnx_world_model::Pipeline unipc_pipeline(PipelinePackage(
      scheduler_root, unipc_manifest, std::move(unipc_models)));
  auto unipc_session = unipc_pipeline.CreateSession();
  auto unipc_output = unipc_session.RunStage("iterate");
  Check(
      std::abs(
          unipc_output.at("value").values<float>()[0] -
          (-0.913395524F)) < 1e-4F,
      "UniPC scheduler steps");
  std::filesystem::remove_all(scheduler_root);

  // Classifier-free guidance over a conditioned diffusion stage.
  PipelineManifest guidance_manifest =
      PipelineManifest::Parse(kGuidanceManifest);
  const auto guidance_root =
      std::filesystem::temp_directory_path() /
      "onnx-world-model-guidance-test";
  std::filesystem::remove_all(guidance_root);
  std::filesystem::create_directories(guidance_root);
  {
    std::ofstream scheduler(guidance_root / "scheduler.json");
    scheduler << R"json({
      "_class_name":"FlowMatchEulerDiscreteScheduler",
      "num_train_timesteps":1000,
      "shift":1.0
    })json";
  }
  auto guided_backend = std::make_shared<GuidedVelocityBackend>();
  std::unordered_map<std::string, Model> guided_models;
  guided_models.emplace("generator", Model(guided_backend));
  guided_models.emplace(
      "video_encoder",
      Model(std::make_shared<ConditioningEncoderBackend>()));
  onnx_world_model::Pipeline guided_pipeline(PipelinePackage(
      guidance_root, guidance_manifest, std::move(guided_models)));
  auto guided_session = guided_pipeline.CreateSession();
  auto encoded = guided_session.RunStage(
      "encode_video",
      {
          {
              "video.frames",
              onnx_world_model::Tensor::Zeros(
                  onnx_world_model::DataType::float32, {1, 3, 1, 2, 2}),
          },
      });
  Check(
      encoded.at("encoded_video_latent").values<float>()[0] == 10.0F,
      "conditioning encoder stage output");

  // Latent frame 0 carries the encoded conditioning frame; frames 1 and 2 are
  // noisy and are the only rows the timestep/loss index sets name. The two
  // prompts have different lengths, so every packed index must be rebuilt for
  // the unconditional pass.
  const std::array<float, 6> conditioned_latent{
      10.0F, 10.0F, 20.0F, 20.0F, 30.0F, 30.0F};
  const std::array<std::int64_t, 3> conditional_prompt{7, 7, 7};
  const std::array<std::int64_t, 1> unconditional_prompt{1};
  const std::array<std::int64_t, 2> noisy_tokens{1, 2};
  onnx_world_model::PipelineRunOptions guided_options;
  guided_options.strings.emplace("mode", "image_to_video");
  auto guided_output = guided_session.RunStage(
      "world_generation",
      {
          {
              "text.token_ids",
              onnx_world_model::Tensor::FromValues<std::int64_t>(
                  {3}, std::span(conditional_prompt)),
          },
          {
              "diffusion.initial_vision_latent",
              onnx_world_model::Tensor::FromValues<float>(
                  {3, 2}, std::span(conditioned_latent)),
          },
          {
              "unconditional:generator.input_ids",
              onnx_world_model::Tensor::FromValues<std::int64_t>(
                  {1}, std::span(unconditional_prompt)),
          },
      },
      {
          {
              "generator.vision_timestep_token_indexes",
              onnx_world_model::Tensor::FromValues<std::int64_t>(
                  {2}, std::span(noisy_tokens)),
          },
      },
      guided_options);
  Check(guided_backend->calls == 2, "guided step runs cond and uncond passes");
  Check(
      guided_backend->passes.size() == 2 &&
          guided_backend->passes[0].prompt == 7 &&
          guided_backend->passes[1].prompt == 1,
      "guided passes use the conditional and unconditional prompts");
  const auto& conditional_pass = guided_backend->passes[0];
  const auto& unconditional_pass = guided_backend->passes[1];
  // The conditional prompt is three tokens long, so vision starts at row 3.
  Check(
      conditional_pass.und_len == 3 &&
          conditional_pass.text_indexes ==
              std::vector<std::int64_t>({0, 1, 2}) &&
          conditional_pass.vision_sequence_indexes ==
              std::vector<std::int64_t>({3, 4, 5}) &&
          conditional_pass.vision_mse_loss_indexes ==
              std::vector<std::int64_t>({4, 5}),
      "conditional pass packs the joint sequence behind its prompt");
  // The unconditional prompt is one token long, so every joint index shifts.
  Check(
      unconditional_pass.und_len == 1 &&
          unconditional_pass.text_indexes ==
              std::vector<std::int64_t>({0}) &&
          unconditional_pass.vision_sequence_indexes ==
              std::vector<std::int64_t>({1, 2, 3}) &&
          unconditional_pass.vision_mse_loss_indexes ==
              std::vector<std::int64_t>({2, 3}),
      "unconditional pass rebuilds the joint layout for its own prompt");
  const auto guided_velocity =
      guided_output.at("vision_velocity").values<float>();
  Check(
      guided_velocity.size() == 4 &&
          std::ranges::all_of(
              guided_velocity,
              [](float value) { return std::abs(value - 19.0F) < 1e-5F; }),
      "classifier-free velocity combination");
  const auto guided_latent =
      guided_output.at("vision_latent").values<float>();
  Check(
      guided_latent.size() == 6 && guided_latent[0] == 10.0F &&
          guided_latent[1] == 10.0F,
      "conditioned latent frame is preserved by the scheduler");
  Check(
      std::abs(guided_latent[2] - 1.0F) < 1e-5F &&
          std::abs(guided_latent[4] - 11.0F) < 1e-5F,
      "noisy latent frames follow the guided velocity");

  auto unguided_backend = std::make_shared<GuidedVelocityBackend>();
  std::unordered_map<std::string, Model> unguided_models;
  unguided_models.emplace("generator", Model(unguided_backend));
  unguided_models.emplace(
      "video_encoder",
      Model(std::make_shared<ConditioningEncoderBackend>()));
  onnx_world_model::Pipeline unguided_pipeline(PipelinePackage(
      guidance_root, guidance_manifest, std::move(unguided_models)));
  auto unguided_session = unguided_pipeline.CreateSession();
  onnx_world_model::PipelineRunOptions unguided_options;
  unguided_options.strings.emplace("mode", "image_to_video");
  unguided_options.numbers.emplace("guidance_scale", 1.0);
  auto unguided_output = unguided_session.RunStage(
      "world_generation",
      {
          {
              "text.token_ids",
              onnx_world_model::Tensor::FromValues<std::int64_t>(
                  {3}, std::span(conditional_prompt)),
          },
          {
              "diffusion.initial_vision_latent",
              onnx_world_model::Tensor::FromValues<float>(
                  {3, 2}, std::span(conditioned_latent)),
          },
      },
      {
          {
              "generator.vision_timestep_token_indexes",
              onnx_world_model::Tensor::FromValues<std::int64_t>(
                  {2}, std::span(noisy_tokens)),
          },
      },
      unguided_options);
  Check(
      unguided_backend->calls == 1,
      "guidance scale 1 skips the unconditional pass");
  const auto unguided_latent =
      unguided_output.at("vision_latent").values<float>();
  Check(
      unguided_latent[0] == 10.0F &&
          std::abs(unguided_latent[2] - 13.0F) < 1e-5F,
      "unguided step uses the conditional velocity only");

  auto missing_session = guided_pipeline.CreateSession();
  CheckThrows(
      [&missing_session,
       &conditional_prompt,
       &conditioned_latent,
       &noisy_tokens] {
        onnx_world_model::PipelineRunOptions options;
        options.strings.emplace("mode", "image_to_video");
        (void)missing_session.RunStage(
            "world_generation",
            {
                {
                    "text.token_ids",
                    onnx_world_model::Tensor::FromValues<std::int64_t>(
                        {3}, std::span(conditional_prompt)),
                },
                {
                    "diffusion.initial_vision_latent",
                    onnx_world_model::Tensor::FromValues<float>(
                        {3, 2}, std::span(conditioned_latent)),
                },
            },
            {
                {
                    "generator.vision_timestep_token_indexes",
                    onnx_world_model::Tensor::FromValues<std::int64_t>(
                        {2}, std::span(noisy_tokens)),
                },
            },
            options);
      },
      "guidance without an unconditional value must fail");
  std::filesystem::remove_all(guidance_root);

  std::string unknown_guidance(kGuidanceManifest);
  const std::string guidance_kind = R"json("kind": "classifier_free")json";
  unknown_guidance.replace(
      unknown_guidance.find(guidance_kind),
      guidance_kind.size(),
      R"json("kind": "distilled")json");
  CheckThrowsMessage(
      [&unknown_guidance] {
        (void)PipelineManifest::Parse(unknown_guidance);
      },
      "unsupported kind",
      "unknown guidance kind must fail");

  std::string unguided_capability(kGuidanceManifest);
  const std::string guidance_capability =
      R"json("classifier_free_guidance",)json";
  unguided_capability.replace(
      unguided_capability.find(guidance_capability),
      guidance_capability.size(),
      "");
  CheckThrowsMessage(
      [&unguided_capability] {
        (void)PipelineManifest::Parse(unguided_capability);
      },
      "requires stage capability 'classifier_free_guidance'",
      "guidance without the stage capability must fail");

  // The runtime advertises the capabilities it implements, and a stage may
  // not claim one without the option that describes it.
  const auto supported = PipelineManifest::SupportedCapabilities();
  for (const std::string_view capability :
       {"classifier_free_guidance",
        "conditioned_diffusion",
        "loop_carried_state",
        "iterative_scheduler",
        "packed_sequence_program"}) {
    Check(
        std::ranges::find(supported, capability) != supported.end(),
        "runtime advertises an implemented capability");
  }

  std::string unimplemented(kGuidanceManifest);
  const std::string required_list = R"json("iterative_scheduler",)json";
  unimplemented.replace(
      unimplemented.rfind(required_list),
      required_list.size(),
      R"json("iterative_scheduler", "streaming",)json");
  CheckThrowsMessage(
      [&unimplemented] { (void)PipelineManifest::Parse(unimplemented); },
      "which this runtime does not implement",
      "unimplemented required capability must fail");

  // A capability a component or stage declares stays honored even though the
  // runtime has no program of its own for it.
  std::string provided_capability(unimplemented);
  const std::string stage_capabilities =
      R"json("loop_carried_state",
          "classifier_free_guidance",)json";
  provided_capability.replace(
      provided_capability.find(stage_capabilities),
      stage_capabilities.size(),
      R"json("loop_carried_state",
          "streaming",
          "classifier_free_guidance",)json");
  (void)PipelineManifest::Parse(provided_capability);
  std::string component_capability(unimplemented);
  const std::string component_dtype = R"json("parameter_dtype": "FLOAT")json";
  component_capability.replace(
      component_capability.find(component_dtype),
      component_dtype.size(),
      R"json("parameter_dtype": "FLOAT",
        "capabilities": ["streaming"])json");
  (void)PipelineManifest::Parse(component_capability);

  std::string unknown_preprocessing(kGuidanceManifest);
  const std::string resample = R"json("resample": "bicubic")json";
  unknown_preprocessing.replace(
      unknown_preprocessing.find(resample),
      resample.size(),
      R"json("resample": "spline")json");
  CheckThrowsMessage(
      [&unknown_preprocessing] {
        (void)PipelineManifest::Parse(unknown_preprocessing);
      },
      "field 'resample' has unsupported value",
      "unknown conditioning resample must fail");

  std::string unknown_preprocessing_field(kGuidanceManifest);
  unknown_preprocessing_field.replace(
      unknown_preprocessing_field.find(resample),
      resample.size(),
      R"json("sharpen": true)json");
  CheckThrowsMessage(
      [&unknown_preprocessing_field] {
        (void)PipelineManifest::Parse(unknown_preprocessing_field);
      },
      "has unknown field 'sharpen'",
      "unknown conditioning preprocessing field must fail");

  std::string zero_std(kGuidanceManifest);
  const std::string normalize_std = R"json("std": [0.5, 0.5, 0.5])json";
  zero_std.replace(
      zero_std.find(normalize_std),
      normalize_std.size(),
      R"json("std": [0.5, 0.0, 0.5])json");
  CheckThrowsMessage(
      [&zero_std] { (void)PipelineManifest::Parse(zero_std); },
      "must be non-zero",
      "zero conditioning normalization std must fail");

  std::string claimed_capability(kGuidanceManifest);
  const std::size_t conditioning_begin =
      claimed_capability.find(R"json("conditioning": {)json");
  const std::size_t conditioning_end =
      claimed_capability.find(R"json("default_steps")json");
  claimed_capability.erase(
      conditioning_begin, conditioning_end - conditioning_begin);
  CheckThrowsMessage(
      [&claimed_capability] {
        (void)PipelineManifest::Parse(claimed_capability);
      },
      "declares no 'conditioning' option",
      "advertised capability without its option must fail");

  // Scheduler modes are selected explicitly; an absent mode uses the stage
  // default instead of silently adopting an image-to-video recipe.
  std::string mode_document(kIterativeManifest);
  const std::string scheduler_asset =
      R"json("config_asset": "scheduler.json")json";
  mode_document.replace(
      mode_document.find(scheduler_asset),
      scheduler_asset.size(),
      R"json("config_asset": "scheduler.json",
            "mode_overrides": {
              "image_to_video": {"num_inference_steps": 1}
            })json");
  PipelineManifest mode_manifest = PipelineManifest::Parse(mode_document);
  std::unordered_map<std::string, Model> mode_models;
  mode_models.emplace("counter", Model(std::make_shared<IncrementBackend>()));
  onnx_world_model::Pipeline mode_pipeline(
      PipelinePackage({}, mode_manifest, std::move(mode_models)));
  auto default_mode_session = mode_pipeline.CreateSession();
  auto default_mode_output = default_mode_session.RunStage("iterate");
  Check(
      default_mode_output.at("value").values<float>()[0] == 3.0F,
      "absent scheduler mode keeps the declared default step count");
  auto explicit_mode_session = mode_pipeline.CreateSession();
  onnx_world_model::PipelineRunOptions mode_options;
  mode_options.strings.emplace("mode", "image_to_video");
  auto explicit_mode_output =
      explicit_mode_session.RunStage("iterate", {}, {}, mode_options);
  Check(
      explicit_mode_output.at("value").values<float>()[0] == 1.0F,
      "explicit scheduler mode selects its override");
  auto unknown_mode_session = mode_pipeline.CreateSession();
  CheckThrows(
      [&unknown_mode_session] {
        onnx_world_model::PipelineRunOptions options;
        options.strings.emplace("mode", "video_to_video");
        (void)unknown_mode_session.RunStage("iterate", {}, {}, options);
      },
      "unknown scheduler mode must fail");

  PipelineManifest autoregressive_manifest =
      PipelineManifest::Parse(kAutoregressiveManifest);
  std::unordered_map<std::string, Model> autoregressive_models;
  autoregressive_models.emplace(
      "decoder", Model(std::make_shared<AutoregressiveBackend>()));
  onnx_world_model::Pipeline autoregressive_pipeline(PipelinePackage(
      {}, autoregressive_manifest, std::move(autoregressive_models)));
  auto autoregressive_session = autoregressive_pipeline.CreateSession();
  const std::array<std::int64_t, 2> prompt{10, 11};
  auto autoregressive_output = autoregressive_session.RunStage(
      "decode",
      {
          {
              "text.token_ids",
              onnx_world_model::Tensor::FromValues<std::int64_t>(
                  {1, 2}, std::span(prompt)),
          },
      });
  const auto generated =
      autoregressive_output.at("generated_token_ids")
          .values<std::int64_t>();
  Check(generated.size() == 3, "autoregressive stop length");
  Check(generated[0] == 1 && generated[2] == 3, "greedy token loop");
  auto sampled_session = autoregressive_pipeline.CreateSession();
  onnx_world_model::PipelineRunOptions sampling_options;
  sampling_options.integers = {
      {"do_sample", 1},
      {"top_k", 1},
      {"seed", 7},
  };
  auto sampled_output = sampled_session.RunStage(
      "decode",
      {
          {
              "text.token_ids",
              onnx_world_model::Tensor::FromValues<std::int64_t>(
                  {1, 2}, std::span(prompt)),
          },
      },
      {},
      sampling_options);
  Check(
      sampled_output.at("generated_token_ids")
              .values<std::int64_t>()
              .back() == 3,
      "top-k sampling loop");

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
      if (component == "model") {
        onnx_world_model::Pipeline loaded_pipeline(std::move(loaded));
        auto loaded_session = loaded_pipeline.CreateSession();
        const std::array<float, 4> observation{};
        const std::array<float, 2> action{};
        const std::array<float, 3> state{};
        auto outputs = loaded_session.RunStage(
            "step",
            {
                {
                    "observation",
                    onnx_world_model::Tensor::FromValues<float>(
                        {1, 4}, std::span(observation)),
                },
                {
                    "action",
                    onnx_world_model::Tensor::FromValues<float>(
                        {1, 2}, std::span(action)),
                },
                {
                    "state",
                    onnx_world_model::Tensor::FromValues<float>(
                        {1, 3}, std::span(state)),
                },
            });
        Check(
            outputs.at("next_state").values<float>()[0] == 0.1F,
            "loaded package execution");
      }
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
