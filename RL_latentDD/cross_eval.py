# cross_eval.py
#
# Train external classifiers (Conv / ResNet / ViT) on distilled synthetic images
# (decoded prototypes) and evaluate on real test set.
#
# Adds calibration baselines:
#   - REAL IPC baseline: train on real images, IPC matched to prototypes
#   - AE RECON baseline: train on reconstructions D(E(x_real)) with same labels

from typing import Dict, List, Tuple
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torchvision.models as tv_models


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    """
    Apply ImageNet mean/std normalization to a batch of images in [0,1].
    images: [N, 3, H, W]
    """
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


class ConvSmallClassifier(nn.Module):
    """
    Simple conv classifier with global average pooling so it works for any
    image size (e.g., 32x32 or 224x224).
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def build_eval_model(model_name: str, num_classes: int, image_size: int) -> nn.Module:
    """
    Build a classifier to be trained *from scratch* on small synthetic sets.

    model_name:
        - "conv_small"
        - "resnet18"
        - "resnet50"
        - "vit_b_16"
    """
    if model_name == "conv_small":
        return ConvSmallClassifier(num_classes=num_classes)

    if model_name == "resnet18":
        m = tv_models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if model_name == "resnet50":
        m = tv_models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if model_name == "vit_b_16":
        m = tv_models.vit_b_16(weights=None, image_size=image_size)
        in_features = m.heads.head.in_features
        m.heads.head = nn.Linear(in_features, num_classes)
        return m

    raise ValueError(f"Unknown eval model name: {model_name}")


def train_on_synth_and_eval(
    model_name: str,
    num_classes: int,
    image_size: int,
    train_imgs: torch.Tensor,
    train_labels: torch.Tensor,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    batch_size: int = 128,
) -> float:
    """
    Train an evaluation classifier from scratch on train_imgs/train_labels and measure
    test accuracy on the real test_loader.

    train_imgs MUST already match the preprocessing used by test_loader
    (e.g., ImageNet normalization if your test_loader uses it).
    """
    train_dataset = TensorDataset(train_imgs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = build_eval_model(model_name, num_classes, image_size).to(device)

    criterion = nn.CrossEntropyLoss()

    # Keep it simple: SGD for all (we'll tune later if calibration shows recipe is too harsh)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(epochs * 0.5), int(epochs * 0.75)],
        gamma=0.1,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs

        scheduler.step()

        if epoch % max(1, epochs // 5) == 0:
            avg_loss = total_loss / max(1, total_samples)
            print(f"[Cross-Eval/{model_name}] Epoch {epoch}/{epochs} - train loss: {avg_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += x.size(0)

    return float(correct / max(1, total))


# =========================
# Calibration helpers
# =========================

def sample_real_ipc_tensors(
    dataset,
    num_classes: int,
    ipc: int,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample ipc images per class from a torch Dataset (already transformed).
    Returns:
      imgs: [C*ipc, 3, H, W] (CPU)
      labels: [C*ipc] (CPU, long)
    """
    rng = random.Random(seed)
    idx_by_class: List[List[int]] = [[] for _ in range(num_classes)]

    # Build index list per class
    for i in range(len(dataset)):
        _, y = dataset[i]
        yi = int(y)
        if 0 <= yi < num_classes:
            idx_by_class[yi].append(i)

    # Sample ipc per class
    chosen_indices: List[int] = []
    chosen_labels: List[int] = []
    for c in range(num_classes):
        if len(idx_by_class[c]) < ipc:
            raise RuntimeError(f"Not enough samples for class {c}: have {len(idx_by_class[c])}, need {ipc}")
        picks = rng.sample(idx_by_class[c], ipc)
        chosen_indices.extend(picks)
        chosen_labels.extend([c] * ipc)

    # Materialize tensors
    imgs = []
    for i in chosen_indices:
        x, _ = dataset[i]
        imgs.append(x)
    imgs_t = torch.stack(imgs, dim=0).cpu()
    labels_t = torch.tensor(chosen_labels, dtype=torch.long).cpu()
    return imgs_t, labels_t


def reconstruct_images(
    ae_model: nn.Module,
    imgs: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """
    Compute reconstructions x_hat = D(E(x)) for a batch of images.
    imgs: [N, 3, H, W] on CPU (preprocessed exactly as AE expects).
    Returns:
      recons: [N, 3, H, W] on CPU in [0,1] (decoder output)
    """
    ae_model.eval()
    ds = TensorDataset(imgs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    outs = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device, non_blocking=True)
            z = ae_model.encode(x)
            xhat = ae_model.decode(z)
            outs.append(xhat.detach().cpu())

    return torch.cat(outs, dim=0)
