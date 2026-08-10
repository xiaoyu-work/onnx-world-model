# Cosmos3 Edge validation

The runtime was exercised against an actual 23.35 GB
`cosmos3-edge-f32-export` package on ONNX Runtime CPU.

| Check | Result |
|---|---|
| Parse and validate Mobius schema 1.1 | Passed |
| Load all six ONNX component sessions | Passed, approximately 25–30 seconds |
| Text-only Reasoner control (`2 + 2`) | Passed, generated `4` |
| Vision encoder, embedding, and Reasoner execution | Passed without runtime or non-finite tensor errors |
| One-step Generator and action state | Passed |
| UniPC scheduler and packed state update | Passed |
| Wan VAE decoder micro-run | Passed, produced `[1, 3, 5, 32, 32]` |
| Cosmos3 Edge image-description semantic parity | **Not established** |

The image path executes, but a natural cat-image prompt did not produce a
correct description. The same decoder succeeds on text-only inference.

Mobius currently documents its Cosmos3 Edge visual implementation as L1
graph-build only. Projector pixel-shuffle ordering and numerical parity have
not been verified against an authoritative NVIDIA implementation. The runtime
is therefore proven to load and execute this package, but visual semantic
correctness is not yet established.

The tested export uses a fixed vision encoder input of normalized NCHW
`[1, 3, 256, 256]` and produces 64 feature rows. Its reasoner input therefore
contains 64 image-placeholder tokens. These values describe that test artifact
only; newer exports may use variable-resolution packed vision inputs.

## Image-to-video

Image-to-video orchestration is covered by unit tests, by an end-to-end
guided/conditioned test on real ONNX Runtime sessions with tiny hand-built
components, and by a full `f16` Edge export on CUDA. Reproducing the official
Edge example (the published input frame and prompt JSON, 121 frames at 480x832
and 24 fps, 35 steps) on one H200 takes about 30 seconds:

| Stage | Provider | Seconds |
|---|---|---|
| Load six sessions | CUDA | 8.5 |
| Conditioning encode | CUDA | 0.6 |
| Reasoner and 35 guided denoising steps | CUDA | 22.0 |
| VAE decode, `decode_latent_chunk=6` | CUDA | 6.9 |
| VAE decode, one pass | CPU | 583.3 |

The decoded clip's intensity statistics track the published reference
(mean 113.6 / std 69.9 versus 114.2 / 68.5), and its first frame matches the
conditioning image, so conditioning demonstrably takes effect.

Two runtime constraints apply to this export:

- The reasoner decoder session aborts inside ONNX Runtime's extended fusions,
  so the package must be loaded with `graph_optimization="basic"`.
- One full-resolution VAE decode makes the `avg_shortcut` broadcasts in
  `up_blocks` exceed what ONNX Runtime's CUDA kernels can index, which surfaces
  as `cudaErrorInvalidValue` on an `Expand` node. Chunked decoding avoids it;
  chunk plus overlap must stay at or below eight latent frames. Raising the
  overlap from 2 to 4 changes the result by about 1 dB PSNR, and chunk seams
  show no frame-to-frame discontinuity above normal motion.

Do not export with `--ep cuda`. That writes
`preferred_execution_providers: ["cuda"]` for every component, which drops the
CPU fallback and makes session creation fail on nodes the CUDA provider does
not implement.

`tools/image_to_video_smoke.py` runs a standalone check:

```bash
# Host preprocessing only; works with any exported pipeline.json.
python tools/image_to_video_smoke.py output/cosmos3-edge frame.png --dry-run

# Full conditioning, guided denoising, and decode.
python tools/image_to_video_smoke.py output/cosmos3-edge frame.png \
    --prompt "A robot picks up the red block." --steps 2 --frames 5
```

`tests/python/test_image_to_video_smoke.py` runs the dry check on every test
run and the full check when `ONNX_WORLD_MODEL_EDGE_PACKAGE` points at an
exported package.
