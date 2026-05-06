#!/usr/bin/env python3
"""
Precompute frozen-backbone features and save to disk.

Supported datasets: cifar10, cifar100, tiny_imagenet, stl10, eurosat

Outputs a single .pt file containing:
  feat_train: [N_train, D] float16 (or float32 if --fp32)
  lab_train:  [N_train] int64
  feat_test:  [N_test,  D]
  lab_test:   [N_test]  int64
  meta: dict with arch/dataset/input_size/num_classes/etc.

IMPORTANT:
- Uses deterministic "eval-style" transform (no random augmentation),
  so it is compatible with loading once and slicing by index.

Example usage:
  python precompute_features.py --dataset cifar10   --arch mobilenet_v2   --out_path ./cache/cifar10_mnv2.pt
  python precompute_features.py --dataset cifar100  --arch mobilenet_v2   --out_path ./cache/cifar100_mnv2.pt
  python precompute_features.py --dataset cifar100  --arch efficientnet_b0 --out_path ./cache/cifar100_enb0.pt
  python precompute_features.py --dataset tiny_imagenet --arch mobilenet_v2 --out_path ./cache/tinyimagenet_mnv2.pt
  python precompute_features.py --dataset stl10     --arch mobilenet_v2   --out_path ./cache/stl10_mnv2.pt
  python precompute_features.py --dataset eurosat   --arch mobilenet_v2   --out_path ./cache/eurosat_mnv2.pt
"""

import argparse
import os
import time
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from torchvision.models import mobilenet_v2, efficientnet_b0
try:
    from torchvision.models import MobileNet_V2_Weights, EfficientNet_B0_Weights
except Exception:
    MobileNet_V2_Weights = None
    EfficientNet_B0_Weights = None


# -----------------------------------------------------------------------
# Dataset metadata
# -----------------------------------------------------------------------

DATASET_INFO = {
    "cifar10": {
        "num_classes": 10,
        "mean": (0.485, 0.456, 0.406),
        "std":  (0.229, 0.224, 0.225),
    },
    "cifar100": {
        "num_classes": 100,
        "mean": (0.485, 0.456, 0.406),
        "std":  (0.229, 0.224, 0.225),
    },
    "tiny_imagenet": {
        "num_classes": 200,
        "mean": (0.485, 0.456, 0.406),
        "std":  (0.229, 0.224, 0.225),
    },
    "stl10": {
        "num_classes": 10,
        "mean": (0.485, 0.456, 0.406),
        "std":  (0.229, 0.224, 0.225),
    },
    "eurosat": {
        "num_classes": 10,
        "mean": (0.485, 0.456, 0.406),
        "std":  (0.229, 0.224, 0.225),
    },
}


# -----------------------------------------------------------------------
# Dataset loaders
# -----------------------------------------------------------------------

def _build_datasets(
    dataset: str,
    data_root: str,
    input_size: int,
) -> Tuple[Dataset, Dataset]:
    """Return (train_ds, test_ds) with deterministic eval-style transforms."""

    info = DATASET_INFO[dataset]
    mean, std = info["mean"], info["std"]

    transform = T.Compose([
        T.Resize(input_size + 32),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    root = os.path.join(data_root, dataset)
    os.makedirs(root, exist_ok=True)

    if dataset == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(
            root=data_root, train=True,  download=True, transform=transform)
        test_ds  = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform)

    elif dataset == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(
            root=data_root, train=True,  download=True, transform=transform)
        test_ds  = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=transform)

    elif dataset == "stl10":
        # STL-10: 'train' split has 5000 labelled images, 'test' has 8000.
        train_ds = torchvision.datasets.STL10(
            root=root, split="train", download=True, transform=transform)
        test_ds  = torchvision.datasets.STL10(
            root=root, split="test",  download=True, transform=transform)

    elif dataset == "tiny_imagenet":
        # Tiny ImageNet is not in torchvision; use ImageFolder after
        # downloading and extracting from:
        #   http://cs231n.stanford.edu/tiny-imagenet-200.zip
        # Expected directory layout:
        #   <data_root>/tiny_imagenet/train/<class_dir>/...
        #   <data_root>/tiny_imagenet/val/<class_dir>/...   (re-organised)
        train_dir = os.path.join(root, "train")
        val_dir   = os.path.join(root, "val")
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise FileNotFoundError(
                f"Tiny ImageNet not found at {root}.\n"
                f"Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
                f"and extract so that {train_dir}/<class>/ and {val_dir}/<class>/ exist.\n"
                f"The val/ folder must be re-organised into per-class subdirectories "
                f"(use scripts/reorganise_tiny_imagenet_val.py if needed)."
            )
        train_ds = torchvision.datasets.ImageFolder(train_dir, transform=transform)
        test_ds  = torchvision.datasets.ImageFolder(val_dir,   transform=transform)

    elif dataset == "eurosat":
        # EuroSAT: 27,000 Sentinel-2 satellite image patches, 10 classes.
        # Auto-downloads via torchvision.  No built-in train/test split,
        # so we use a deterministic 80/20 split.
        full_ds = torchvision.datasets.EuroSAT(
            root=root, download=True, transform=transform)
        n = len(full_ds)
        n_train = int(0.8 * n)
        n_test = n - n_train
        train_ds, test_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_test],
            generator=torch.Generator().manual_seed(42))

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. "
                         f"Choose from: {list(DATASET_INFO.keys())}")

    return train_ds, test_ds


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

def _get_mobilenet_v2_pretrained():
    if MobileNet_V2_Weights is not None:
        return mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    return mobilenet_v2(pretrained=True)


def _get_efficientnet_b0_pretrained():
    if EfficientNet_B0_Weights is not None:
        return efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    return efficientnet_b0(pretrained=True)


class FrozenFeatureExtractor(torch.nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        arch = arch.lower()
        self.arch = arch

        if arch == "mobilenet_v2":
            base = _get_mobilenet_v2_pretrained()
            self.backbone = base.features
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.feat_dim = int(base.last_channel)           # 1280
        elif arch == "efficientnet_b0":
            base = _get_efficientnet_b0_pretrained()
            self.backbone = base.features
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.feat_dim = int(base.classifier[-1].in_features)  # 1280
        else:
            raise ValueError("arch must be one of: mobilenet_v2, efficientnet_b0")

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        f = self.pool(f)
        return torch.flatten(f, 1)


# -----------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------

@torch.no_grad()
def extract_all_features(
    ds,
    model: FrozenFeatureExtractor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    feats, labs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = model.forward_features(x)
        else:
            z = model.forward_features(x)
        feats.append(z.detach().cpu().to(dtype=out_dtype))
        labs.append(y.detach().cpu().to(dtype=torch.long))

    return torch.cat(feats, dim=0), torch.cat(labs, dim=0)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Precompute frozen-backbone features for a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=list(DATASET_INFO.keys()),
        help="Dataset to extract features from.",
    )
    ap.add_argument(
        "--arch", type=str, default="mobilenet_v2",
        choices=["mobilenet_v2", "efficientnet_b0"],
    )
    ap.add_argument("--input_size",   type=int,  default=224)
    ap.add_argument("--batch_size",   type=int,  default=512)
    ap.add_argument("--num_workers",  type=int,  default=4)
    ap.add_argument("--data_root",    type=str,  default="./data",
                    help="Root directory where torchvision will download / look for datasets.")
    ap.add_argument("--no_amp",       action="store_true")
    ap.add_argument("--fp32",         action="store_true",
                    help="Store features as float32 instead of float16.")
    ap.add_argument(
        "--out_path", type=str, default="",
        help=(
            "Path to write the .pt cache file.  "
            "If empty, defaults to "
            "<data_root>/feature_cache/<dataset>_<arch>_<input_size>.pt"
        ),
    )
    args = ap.parse_args()

    # Default output path.
    out_path = args.out_path.strip()
    if not out_path:
        cache_dir = os.path.join(args.data_root, "feature_cache")
        out_path  = os.path.join(
            cache_dir, f"{args.dataset}_{args.arch}_{args.input_size}.pt"
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dtype = torch.float32 if args.fp32 else torch.float16
    amp       = not args.no_amp
    info      = DATASET_INFO[args.dataset]

    print(f"[CONFIG] dataset={args.dataset}  arch={args.arch}  "
          f"input_size={args.input_size}  device={device}  "
          f"dtype={'fp32' if args.fp32 else 'fp16'}  amp={amp}")
    print(f"[OUTPUT] {out_path}")

    print("[LOAD] Building datasets...")
    train_ds, test_ds = _build_datasets(args.dataset, args.data_root, args.input_size)
    print(f"  train={len(train_ds)}  test={len(test_ds)}  "
          f"num_classes={info['num_classes']}")

    model = FrozenFeatureExtractor(args.arch).to(device)
    model.eval()
    print(f"[MODEL] {args.arch}  feat_dim={model.feat_dim}")

    t0 = time.time()
    print("[EXTRACT] train split...")
    feat_train, lab_train = extract_all_features(
        train_ds, model, device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=amp,
        out_dtype=out_dtype,
    )

    print("[EXTRACT] test split...")
    feat_test, lab_test = extract_all_features(
        test_ds, model, device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=amp,
        out_dtype=out_dtype,
    )
    t1 = time.time()

    blob = {
        "feat_train": feat_train.contiguous(),
        "lab_train":  lab_train.contiguous(),
        "feat_test":  feat_test.contiguous(),
        "lab_test":   lab_test.contiguous(),
        "meta": {
            "dataset":     args.dataset,
            "arch":        args.arch,
            "input_size":  args.input_size,
            "num_classes": info["num_classes"],
            "mean":        info["mean"],
            "std":         info["std"],
            "dtype":       str(out_dtype),
            "amp":         bool(amp),
            "n_train":     int(feat_train.shape[0]),
            "n_test":      int(feat_test.shape[0]),
            "feat_dim":    int(feat_train.shape[1]),
            "created":     time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    print(f"[SAVE] {out_path}")
    torch.save(blob, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(
        f"[DONE] train={feat_train.shape}  test={feat_test.shape}  "
        f"feat_dim={feat_train.shape[1]}  time={t1-t0:.1f}s  "
        f"file={size_mb:.1f} MB"
    )


if __name__ == "__main__":
    main()