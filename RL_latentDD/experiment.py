# experiment.py

"""Experiment configuration utilities for structured, reproducible runs.

The goal of this module is to keep `main.py` focused on orchestration while
capturing every choice (data, backbone, RL knobs) in auditable data classes.
This encourages research-grade hygiene for benchmarking and ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torchvision
import torchvision.transforms as T

from config import DEFAULT_BACKBONE, DEFAULT_DATASET
from utils import ensure_dir


@dataclass
class DataConfig:
    name: str = DEFAULT_DATASET
    data_root: str = "./data"
    custom_train_dir: Optional[str] = None
    custom_test_dir: Optional[str] = None
    val_fraction: float = 0.1
    max_train_points: int = 20_000
    override_image_size: Optional[int] = None

    def validate(self) -> None:
        if self.name == "custom":
            if not self.custom_train_dir or not self.custom_test_dir:
                raise ValueError("custom dataset requires --custom-train-dir and --custom-test-dir")
        if not 0.0 < self.val_fraction < 0.5:
            raise ValueError("val_fraction should be in (0, 0.5) to preserve training pool size")


@dataclass
class BackboneConfig:
    name: str = DEFAULT_BACKBONE
    latent_dim: int = 128

    def requires_imagenet_norm(self) -> bool:
        return self.name in {"resnet18", "vit_b_16"}


@dataclass
class LoaderConfig:
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ExperimentMetadata:
    seed: int
    num_classes: int
    image_size: int
    backbone: BackboneConfig
    data: DataConfig
    loader: LoaderConfig

    def as_dict(self) -> Dict:
        payload = asdict(self)
        payload["backbone"] = asdict(self.backbone)
        payload["data"] = asdict(self.data)
        payload["loader"] = asdict(self.loader)
        return payload


def build_transforms(backbone: BackboneConfig, image_size: int) -> Tuple[T.Compose, int]:
    if backbone.requires_imagenet_norm():
        target_size = 224
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        target_size = image_size
        normalize = None

    ops = [T.Resize((target_size, target_size)), T.ToTensor()]
    if normalize:
        ops.append(normalize)
    return T.Compose(ops), target_size


def load_dataset(config: DataConfig, transform: T.Compose):
    config.validate()
    if config.name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(root=config.data_root, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(root=config.data_root, train=False, download=True, transform=transform)
    elif config.name == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(root=config.data_root, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR100(root=config.data_root, train=False, download=True, transform=transform)
    else:
        train_ds = torchvision.datasets.ImageFolder(config.custom_train_dir, transform=transform)
        test_ds = torchvision.datasets.ImageFolder(config.custom_test_dir, transform=transform)

    num_classes = len(train_ds.classes) if hasattr(train_ds, "classes") else 0
    if num_classes == 0:
        raise ValueError("Could not infer class cardinality; ensure dataset exposes .classes")
    return train_ds, test_ds, num_classes


def backbone_name_choices() -> Iterable[str]:
    return ["conv", "resnet18", "vit_b_16"]


def dataset_name_choices() -> Iterable[str]:
    return ["cifar10", "cifar100", "custom"]


def prepare_output_dir(base_dir: str | Path, dataset: str, backbone: str, run_name: str | None = None) -> Path:
    """Create a unique directory for a single experiment run.

    A timestamped directory avoids overwriting previous runs unless a custom
    run_name is provided (e.g., when coordinating sweeps programmatically).
    """

    suffix = run_name or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    timestamped = ensure_dir(Path(base_dir) / dataset / backbone / suffix)
    return timestamped
