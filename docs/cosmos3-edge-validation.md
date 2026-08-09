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

Image-to-video orchestration is covered by unit tests and by an end-to-end
guided/conditioned test on real ONNX Runtime sessions with tiny hand-built
components. It has not yet been replayed against a full Edge export, because
that requires a package whose manifest declares the `guidance`,
`conditioning`, `generation_recipes`, and `generator_prompt` contracts.
`tools/image_to_video_smoke.py` runs that check once such a package exists:

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
