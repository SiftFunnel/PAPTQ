# Research code for mixed-precision QDQ quantization

This directory is a sanitized, local release candidate of the quantization method
used in the paper. It contains code only: no images, dataset archives, trained
weights, reconstructions, private filesystem paths, or attack implementations.
It has **not** been published or uploaded.

## Contents

| File | Purpose |
| --- | --- |
| `split_data.py` | Reproducible index-only split helpers and overlap audit |
| `public_data.py` | User-supplied FaceScrub/Palm/Iris NPZ loaders |
| `resnet50_mp.py` | Quantizable ResNet-50 architecture and activation QDQ |
| `transformer_quant_models.py` | Quantizable ViT-B/16 and Swin-B architectures |
| `public_activation_quantization.py` | Transformer activation-calibration correction |
| `quantization_core.py` | Shared Fisher, allocation, calibration and refinement code |
| `run_quantization.py` | Portable entry point with user-provided paths |

The method evaluates quantize-dequantize (QDQ) values in floating-point
arithmetic. It does not pack low-bit weights or use native integer kernels;
the bit-width reports do not imply measured runtime or storage savings.

## Data contract

Provide your own uint8 NPZ file in the directory passed as `--dataset_root`:

- FaceScrub: `facescrub_color1.npz` with `actor_images`, `actor_labels`,
  `actress_images`, and `actress_labels`. The historical loader concatenates
  the two groups and applies a seed-624 global shuffle followed by an 80/20
  index split. This is not a class-stratified split.
- Palm: `palm.npz` with `train_data`, `train_labels`, `test_data`, `test_labels`.
- Iris-Thousand: `iris_thousand.npz` with the same target-train/test keys.
  The historical target split has 150 labels (0--149), 2,250 train rows and
  750 test rows. An optional auxiliary split is neither loaded nor used by
  the quantization method.

`split_data.py` includes a deterministic classwise function for **new**
data. It cannot reconstruct the original Palm/Iris raw-image indices from a
pre-split NPZ. It rejects exact target train/test image overlap on export;
this stricter rule means newly generated splits may not equal historical ones.

The FP32 checkpoint must match `--arch`, `--nc`, `--nz`, and `--img_size`.
The loader uses safe PyTorch loading and requires a strict state-dictionary
match. Convert checkpoints containing arbitrary pickled objects to a plain
state dictionary in a trusted environment before using this release.

## Environment

Use a compatible PyTorch/torchvision installation, then install
`requirements.txt`. No dependency is downloaded automatically by this code.

## Example: Iris-Thousand ResNet-50

Run from this folder after supplying your own dataset and FP32 weights:

```bash
python run_quantization.py \
  --arch resnet50 --dataset iris_thousand \
  --dataset_root /path/to/your/data \
  --ckpt_path /path/to/your/fp32_resnet50.pth \
  --save_dir outputs/iris_resnet50 --save_name type9_w4a4.pth \
  --nc 1 --nz 150 --img_size 224 \
  --bits 6,5,4,3 --ref_bit 4 \
  --target_avg_w_bit 4 --target_avg_a_bit 4 \
  --batch_size 16 --test_batch_size 16 \
  --scale_classes_per_batch 10 --scale_samples_per_class 2 \
  --scale_epochs 100 --label_smoothing 0
```

For ViT-B/16 or Swin-B, change `--arch` to `vit` or `swin`, use the matching
FP32 checkpoint, and explicitly set `--batch_size 4`,
`--test_batch_size 8`, `--scale_classes_per_batch 4`, and
`--scale_samples_per_class 2`. The adapter selects
the architecture's pre-head feature tap. The transformer implementation uses
the same numerical Fisher/allocation/refinement core while preserving its
architecture-specific activation QDQ and original-image activation weighting.
The example is illustrative, not an archived formal preset. For exact
historical comparison, supply the recorded epoch count, learning-rate
schedule, bit targets, and checkpoint/data provenance.

Use `--dry_run` to stop after sensitivity estimation. This still loads the
supplied dataset and model; it does not run without them. All paths are local.

## Local checks

From this directory, run `python -m unittest test_public_package -v`.
The fixture tests do not use private data or trained weights.

## Interpretation and limitations

The original experiments select the best hard-quantized Stage-3 epoch using
the target test split. Consequently, reported test accuracy is **not** a
selection-free blind-test estimate. Keep this behavior only when reproducing
the historical protocol; a new study should reserve a separate validation
set and untouched test set. The release candidate does not include the
paper's model-inversion attack pipelines or their private evaluators.

The package provides source code and import/fixture checks; it has not been
validated end-to-end against every private checkpoint and dataset after
sanitization. Do not claim exact numerical reproduction from this package
until the full experiments have been rerun. A license should be chosen before
public release.
