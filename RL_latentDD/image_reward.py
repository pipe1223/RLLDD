# image_reward.py

from typing import Callable

import torch
from torch import nn

from latent_utils import train_latent_classifier


def make_decoded_latent_reward_fn(
    ae_model: nn.Module,
    val_z: torch.Tensor,
    val_y: torch.Tensor,
    num_classes: int,
    device: torch.device,
    train_latent_classifier_fn: Callable = train_latent_classifier,
    classifier_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> Callable[[torch.Tensor, torch.Tensor], float]:
    """
    Build a reward function that:

      1) takes decoded prototype images + labels,
      2) re-encodes them with a frozen AE backbone (Conv / ResNet / ViT),
      3) trains a fresh latent linear classifier on those latents,
      4) evaluates it on (val_z, val_y),
      5) returns validation accuracy as a scalar reward.

    This is exactly "decode prototypes + frozen backbone head" in latent space.
    If you run the AE with a ResNet backbone (--backbone resnet18), this uses a frozen ResNet head.
    """

    ae_model = ae_model.to(device)
    ae_model.eval()

    # Keep validation latents / labels on CPU; the helper will move them to device.
    val_z_cpu = val_z.detach().cpu()
    val_y_cpu = val_y.detach().cpu()

    def reward_fn(proto_imgs: torch.Tensor, proto_labels: torch.Tensor) -> float:
        """
        Args:
            proto_imgs: [N, C, H, W] decoded prototype images in [0, 1].
            proto_labels: [N] long, class indices.

        Returns:
            float: validation accuracy of latent classifier trained on re-encoded prototypes.
        """
        # 1) Re-encode images via the frozen AE encoder (this is your "head": Conv/ResNet/ViT).
        with torch.no_grad():
            z_train = ae_model.encode(proto_imgs.to(device))

        # 2) Move training latents / labels to CPU for train_latent_classifier.
        z_train_cpu = z_train.detach().cpu()
        y_train_cpu = proto_labels.detach().cpu()

        # 3) Train a fresh latent linear classifier and evaluate on val_z / val_y.
        acc = train_latent_classifier_fn(
            train_z=z_train_cpu,
            train_y=y_train_cpu,
            val_z=val_z_cpu,
            val_y=val_y_cpu,
            n_classes=num_classes,
            device=device,
            epochs=classifier_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
        )
        return float(acc)

    return reward_fn
