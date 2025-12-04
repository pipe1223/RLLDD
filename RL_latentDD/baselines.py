# baselines.py

import numpy as np
import torch
from sklearn.cluster import KMeans

from latent_utils import train_latent_classifier


def random_subset_indices(train_y: torch.Tensor, budget_per_class: int, num_classes: int) -> np.ndarray:
    y_np = train_y.numpy()
    idx_list = []
    for c in range(num_classes):
        idx_c = np.where(y_np == c)[0]
        chosen = np.random.choice(idx_c, size=budget_per_class, replace=False)
        idx_list.extend(chosen.tolist())
    return np.array(idx_list, dtype=np.int64)


def evaluate_random_latent_subset(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    budget_per_class: int,
    num_classes: int,
    device: torch.device,
    classifier_epochs: int,
):
    idx = random_subset_indices(train_y, budget_per_class, num_classes)
    subset_z = train_z[idx]
    subset_y = train_y[idx]
    print(f"Random latent subset size: {subset_z.size(0)}")

    acc = train_latent_classifier(
        train_z=subset_z,
        train_y=subset_y,
        val_z=test_z,
        val_y=test_y,
        n_classes=num_classes,
        device=device,
        epochs=classifier_epochs,
        batch_size=128,
        lr=1e-2,
        weight_decay=1e-4,
    )
    return acc, idx


def evaluate_kmeans_centroids(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    budget_per_class: int,
    num_classes: int,
    device: torch.device,
    classifier_epochs: int,
    kmeans_seed: int = 0,
):
    z_np = train_z.numpy()
    y_np = train_y.numpy()
    centroids_list = []
    labels_list = []

    for c in range(num_classes):
        idx_c = np.where(y_np == c)[0]
        data_c = z_np[idx_c]
        if len(idx_c) < budget_per_class:
            raise ValueError(f"Not enough samples for class {c} to do k-means.")
        km = KMeans(
            n_clusters=budget_per_class,
            n_init=10,
            random_state=kmeans_seed,
        )
        km.fit(data_c)
        centroids_list.append(km.cluster_centers_)
        labels_list.append(np.full(budget_per_class, c, dtype=np.int64))

    centroids = torch.from_numpy(np.concatenate(centroids_list, axis=0)).float()
    labels = torch.from_numpy(np.concatenate(labels_list, axis=0))

    print(f"k-means centroids dataset size: {centroids.size(0)}")

    acc = train_latent_classifier(
        train_z=centroids,
        train_y=labels,
        val_z=test_z,
        val_y=test_y,
        n_classes=num_classes,
        device=device,
        epochs=classifier_epochs,
        batch_size=128,
        lr=1e-2,
        weight_decay=1e-4,
    )
    return acc
