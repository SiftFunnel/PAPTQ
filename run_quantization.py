"""Portable entry point for the paper's QDQ mixed-precision quantization.

The numerical stages are kept in quantization_core.py.
This adapter replaces private checkpoint/dataset contracts with user-provided
paths. It does not download data or weights and never invokes attack code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import quantization_core as core  # noqa: E402
from public_data import FaceScrub, IRIS_thousand, Palm  # noqa: E402
from public_activation_quantization import install_local_activation_quantizers  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = core.build_parser()
    parser.description = "Mixed-precision quantize-dequantize (QDQ) pipeline"
    parser.add_argument("--arch", choices=("resnet50", "vit", "swin"), required=True)
    return parser


def build_loaders(args):
    classes = {"facescrub": FaceScrub, "face": FaceScrub,
               "palm": Palm, "iris_thousand": IRIS_thousand}
    if args.dataset not in classes:
        raise ValueError(f"Choose a supported dataset: {sorted(classes)}")
    transform = core.build_transform(args.nc, args.dataset)
    dataset_cls = classes[args.dataset]
    train_set = dataset_cls(args.dataset_root, train=True, transform=transform)
    test_set = dataset_cls(args.dataset_root, train=False, transform=transform)
    if min(len(train_set), len(test_set)) == 0:
        raise ValueError("train and test splits must be nonempty")
    train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    return train, test, train_set


def _extract_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("FP32 checkpoint must be a state dictionary or contain model/state_dict")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or not state:
        raise TypeError("FP32 checkpoint has no model state dictionary")
    return {key.removeprefix("_module.").removeprefix("module."): value
            for key, value in state.items()}


def load_source(args, device):
    if args.arch == "resnet50":
        from resnet50_mp import resnet50_ptq
        model = resnet50_ptq(in_channels=args.nc, num_classes=args.nz,
                             img_size=args.img_size)
    else:
        from transformer_quant_models import vit_ptq, swin_b_ptq
        if args.dataset != "iris_thousand" or (args.nc, args.nz, args.img_size) != (1, 150, 224):
            raise ValueError("Published ViT/Swin configuration expects Iris-Thousand 1x224x224, 150 classes")
        model = (vit_ptq(model_size="base", img_size=224, patch_size=16,
                         in_chans=1, num_classes=150, default_act_bits=8)
                 if args.arch == "vit" else
                 swin_b_ptq(num_classes=150, img_size=224, in_chans=1,
                            default_act_bits=8))
        install_local_activation_quantizers(model)
        disable_drop_path(model)
    try:
        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError("Could not safely load the supplied FP32 checkpoint") from error
    model.load_state_dict(_extract_state(checkpoint), strict=True)
    if hasattr(model, "set_fp32_train_mode"):
        model.set_fp32_train_mode()
    if args.arch != "resnet50":
        disable_drop_path(model)
    return model.to(device).eval()


def disable_drop_path(model) -> int:
    changed = 0
    for module in model.modules():
        if type(module).__name__ == "DropPath" and hasattr(module, "drop_prob"):
            if float(module.drop_prob) != 0.0:
                module.drop_prob = 0.0
                changed += 1
    return changed


def transformer_activation_numels(model, loader, device, max_batches=1):
    """Count Swin-window activation elements per original image, not per window."""
    sizes, handles = {}, []
    context = {"batch": None}
    def make_hook(name):
        def hook(_module, _inputs, output):
            if name in sizes:
                return
            tensor = output
            if isinstance(output, (tuple, list)):
                tensor = next((x for x in output if torch.is_tensor(x)), None)
            if torch.is_tensor(tensor):
                batch = int(context["batch"] or 1)
                if tensor.numel() % batch:
                    raise RuntimeError(f"activation {name} is not divisible by batch size")
                sizes[name] = int(tensor.numel() // batch)
        return hook
    for name, module in model.named_modules():
        if hasattr(module, "bits") and "ActQuant" in type(module).__name__:
            handles.append(module.register_forward_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for index, (images, _) in enumerate(loader):
                if index >= int(max_batches):
                    break
                context["batch"] = int(images.shape[0])
                model(images.to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return sizes


def main() -> None:
    args = build_parser().parse_args()
    if not Path(args.ckpt_path).is_file():
        raise FileNotFoundError("Pass an existing --ckpt_path; weights are not distributed")
    output = Path(args.save_dir).expanduser() / args.save_name
    if not args.dry_run and output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing checkpoint: {output}")
    if args.arch != "resnet50":
        if args.dataset != "iris_thousand":
            raise ValueError("ViT/Swin published setup is Iris-Thousand only")
        if (args.nc, args.nz, args.img_size) != (1, 150, 224):
            raise ValueError("ViT/Swin require --nc 1 --nz 150 --img_size 224")
        args.target_feature_idx = 86 if args.arch == "vit" else 173
        original_fisher = core.compute_weight_fisher_tensors
        def deterministic_fisher(model, loader, device, max_batches=50):
            disable_drop_path(model)
            result = original_fisher(model, loader, device, max_batches=max_batches)
            model.zero_grad(set_to_none=True)
            model.eval()
            return result
        core.compute_weight_fisher_tensors = deterministic_fisher
        core.estimate_activation_numels = transformer_activation_numels
    core.build_loaders = build_loaders
    core.load_resnet50_ptq = load_source
    core.run_quantization_pipeline(args)


if __name__ == "__main__":
    main()
