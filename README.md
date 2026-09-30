# Mixed-Precision Quantize-Dequantize (QDQ) Quantization

This repository provides the quantization implementation accompanying our
research on post-training, mixed-precision image classifiers. It supports a
quantizable ResNet-50 for FaceScrub, Palm, and Iris-Thousand, and quantizable
ViT-B/16 and Swin-B models for Iris-Thousand. Supply your own dataset and
matching FP32 checkpoint; neither is distributed here.

The pipeline estimates layer sensitivity, assigns weight and activation bit
widths under a budget, calibrates activations, and refines quantization
parameters. Quantization is simulated with floating-point quantize-dequantize
(QDQ) operations. Reported bit widths do not imply packed low-bit storage,
native integer inference, or measured speedups.

## Files

| File                                | Purpose                                                      |
| ----------------------------------- | ------------------------------------------------------------ |
| `run_quantization.py`               | Command-line entry point and model/dataset adapters          |
| `quantization_core.py`              | Sensitivity estimation, bit allocation, calibration, and refinement |
| `resnet50_mp.py`                    | Quantizable ResNet-50 architecture                           |
| `transformer_quant_models.py`       | Quantizable ViT-B/16 and Swin-B architectures                |
| `public_activation_quantization.py` | Transformer activation-calibration support                   |
| `public_data.py`                    | Loaders for user-supplied NPZ datasets                       |
| `split_data.py`                     | Split helpers and exact-image overlap checks                 |
| `test_public_package.py`            | CPU fixture tests without datasets or checkpoints            |

## Setup

Use Python 3.9 or newer and a compatible PyTorch/torchvision installation.
From this directory, install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

The code does not download datasets or checkpoints. The FP32 checkpoint must
match the selected architecture, input channels, image size, and class count.
The loader accepts a plain state dictionary or a dictionary containing
`model` or `state_dict`, loads it with `weights_only=True`, and checks model
keys strictly. Convert other checkpoint formats in a trusted environment.

## Dataset format

Pass the directory containing the appropriate NPZ file as `--dataset_root`.
Images must be `uint8`; labels must be integer class IDs compatible with the
checkpoint's classifier head. Grayscale images should have shape
`N x H x W` and color images `N x H x W x 3`. Images are not resized by the
loader, so their spatial size and channel count must match the model
configuration.

| `--dataset`     | NPZ file               | Required arrays                                              |
| --------------- | ---------------------- | ------------------------------------------------------------ |
| `facescrub`     | `facescrub_color1.npz` | `actor_images`, `actor_labels`, `actress_images`, `actress_labels` |
| `palm`          | `palm.npz`             | `train_data`, `train_labels`, `test_data`, `test_labels`     |
| `iris_thousand` | `iris_thousand.npz`    | `train_data`, `train_labels`, `test_data`, `test_labels`     |

The FaceScrub loader concatenates actor and actress arrays, applies a
seed-624 global shuffle, and uses the first 80% for training and the
remainder for testing; this is not a class-stratified split. Palm and
Iris-Thousand loaders use the supplied target train/test arrays as-is. The
historical Iris-Thousand target split has 150 classes (IDs 0-149), 2,250
training images, and 750 test images. Auxiliary images are not used by the
quantization pipeline.

`split_data.py` provides a deterministic classwise split for **new** datasets.
It does not reconstruct the historical Palm or Iris-Thousand splits from raw
images. Its NPZ exporter rejects exact image overlap between target train and
test splits.

## Run quantization

The following is an illustrative Iris-Thousand ResNet-50 invocation. Replace
the dataset and checkpoint paths with your own files, and verify that the
hyperparameters match the experiment you intend to reproduce:

```bash
python run_quantization.py \
  --arch resnet50 --dataset iris_thousand \
  --dataset_root /path/to/data \
  --ckpt_path /path/to/fp32_resnet50.pth \
  --save_dir outputs/iris_resnet50 --save_name quantized_model.pth \
  --nc 1 --nz 150 --img_size 224 \
  --bits 6,5,4,3 --ref_bit 4 \
  --target_avg_w_bit 4 --target_avg_a_bit 4 \
  --batch_size 16 --test_batch_size 16 \
  --scale_classes_per_batch 10 --scale_samples_per_class 2 \
  --scale_epochs 100 --label_smoothing 0
```

For Iris-Thousand ViT-B/16 or Swin-B, set `--arch vit` or `--arch swin` and
provide the corresponding FP32 checkpoint. These models require
`--nc 1 --nz 150 --img_size 224`; choose architecture-appropriate batch
settings and the learning-rate schedule specified by your experiment. The
transformer adapter uses the same sensitivity/allocation/refinement core
while retaining architecture-specific activation QDQ behavior.

Use `--dry_run` to stop after sensitivity estimation. It still requires the
dataset and FP32 checkpoint. Run `python run_quantization.py --help` for all
available options.

## Checks and reproducibility

Run `python -m unittest test_public_package -v` for CPU fixture tests. These
tests use synthetic inputs and verify the split helpers, imports, and command
parser; they do not validate the paper's reported numerical results.

The historical pipeline uses the target test split to select the best
hard-quantized Stage-3 epoch. Its resulting test accuracy is therefore not
an untouched, selection-free estimate. For a new study, use a separate
validation set for model selection and preserve an untouched test set.

Datasets, trained checkpoints, and model-inversion attack pipelines and their
independent evaluators are not included. Reproducing reported results requires
the matching data splits, preprocessing, FP32 checkpoints, hyperparameters,
and evaluation protocol. The supplied code and synthetic checks alone do not
establish numerical reproduction of the paper's results.
