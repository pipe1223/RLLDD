# latent_utils.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import ConvAEClassifier, LatentLinearClassifier, LatentPrototypeModel


def train_ae_classifier(
    model: ConvAEClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    alpha_cls: float = 1.0,
    momentum: float = 0.9,
    weight_decay: float =5e-4,
    step_size: int = 30,
    gamma: float =0.1,
):
    model.to(device)
    
    #------- Adam optimizer -----
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # ----- SGD optimizer -----
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # ----- StepLR scheduler -----
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )
    
    recon_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_recon = 0.0
        total_cls = 0.0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            x_recon, logits, _ = model(x)

            recon_loss = recon_criterion(x_recon, x)
            cls_loss = cls_criterion(logits, y)
            loss = recon_loss + alpha_cls * cls_loss

            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_recon += float(recon_loss.item()) * bs
            total_cls += float(cls_loss.item()) * bs
            total_samples += bs

        avg_recon = total_recon / total_samples
        avg_cls = total_cls / total_samples
        # Step the LR scheduler once per epoch
        scheduler.step()
                
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                _, logits_val, _ = model(x_val)
                preds = logits_val.argmax(dim=1)
                correct += int((preds == y_val).sum().item())
                total += y_val.size(0)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"[AE+CLS Epoch {epoch:03d}] "
            f"Recon={avg_recon:.4f} | Cls={avg_cls:.4f} | Val cls acc={val_acc:.4f}"
        )


def extract_latents(
    model: ConvAEClassifier,
    dataset,
    batch_size: int,
    device: torch.device,
):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    all_z = []
    all_y = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model.encode(x)
            all_z.append(z.cpu())
            all_y.append(y.cpu())

    latents = torch.cat(all_z, dim=0)
    labels = torch.cat(all_y, dim=0)
    return latents, labels


def train_latent_classifier(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    val_z: torch.Tensor,
    val_y: torch.Tensor,
    n_classes: int,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> float:
    latent_dim = train_z.size(1)
    model = LatentLinearClassifier(latent_dim, n_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    n_train = train_z.size(0)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        train_z_epoch = train_z[perm]
        train_y_epoch = train_y[perm]

        for i in range(0, n_train, batch_size):
            xb = train_z_epoch[i:i+batch_size].to(device)
            yb = train_y_epoch[i:i+batch_size].to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_z = val_z.to(device)
        val_y = val_y.to(device)
        logits = model(val_z)
        preds = logits.argmax(dim=1)
        acc = (preds == val_y).float().mean().item()

    return acc


def init_prototypes_from_data(
    model: LatentPrototypeModel,
    train_z: torch.Tensor,
    train_y: torch.Tensor,
):
    num_classes = model.num_classes
    m_per_class = model.m_per_class
    latent_dim = model.latent_dim

    z_np = train_z.numpy()
    y_np = train_y.numpy()

    with torch.no_grad():
        for c in range(num_classes):
            idx_c = np.where(y_np == c)[0]
            if len(idx_c) < m_per_class:
                raise ValueError(f"Not enough samples for class {c} to init prototypes.")
            chosen = np.random.choice(idx_c, size=m_per_class, replace=False)
            model.protos.data[c] = torch.from_numpy(z_np[chosen]).view(m_per_class, latent_dim)

    print(f"Initialized prototypes from real latent data (m_per_class={m_per_class}).")


def compute_diversity_loss(model: LatentPrototypeModel) -> torch.Tensor:
    protos = model.protos  # [C, M, D]
    C, M, D = protos.shape
    protos_norm = F.normalize(protos, dim=2)

    div_loss = 0.0
    for c in range(C):
        p = protos_norm[c]
        sim = p @ p.t()
        mask = torch.eye(M, device=sim.device).bool()
        off_diag = sim[~mask]
        div_loss = div_loss + off_diag.mean()

    div_loss = div_loss / C
    return div_loss


def train_prototypes_baseline(
    model: LatentPrototypeModel,
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    val_z: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    n_steps: int = 500,
    batch_size_real: int = 256,
    lambda_real: float = 1.0,
    lambda_div: float = 0.5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    print_every: int = 100,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    N = train_z.size(0)
    val_z_device = val_z.to(device)
    val_y_device = val_y.to(device)

    for step in range(1, n_steps + 1):
        model.train()

        synth_z, synth_y = model.get_synthetic_dataset()
        logits_synth = model(synth_z)
        loss_synth = criterion(logits_synth, synth_y)

        idx = torch.randint(0, N, (batch_size_real,))
        z_real = train_z[idx].to(device)
        y_real = train_y[idx].to(device)
        logits_real = model(z_real)
        loss_real = criterion(logits_real, y_real)

        div_loss = compute_diversity_loss(model)

        loss = loss_synth + lambda_real * loss_real + lambda_div * div_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_z_device)
                val_pred = val_logits.argmax(dim=1)
                val_acc = (val_pred == val_y_device).float().mean().item()

            print(
                f"[Baseline Proto step {step:04d}] "
                f"L_syn={loss_synth.item():.4f} "
                f"L_real={loss_real.item():.4f} "
                f"L_div={div_loss.item():.4f} "
                f"Val acc (proto-cls)={val_acc:.4f}"
            )


def evaluate_distilled_prototypes(
    model: LatentPrototypeModel,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    device: torch.device,
    classifier_epochs: int,
) -> float:
    model.eval()
    with torch.no_grad():
        synth_z, synth_y = model.get_synthetic_dataset()
        synth_z = synth_z.detach().cpu()
        synth_y = synth_y.detach().cpu()

    print(f"Synthetic prototypes dataset size: {synth_z.size(0)}")

    acc = train_latent_classifier(
        train_z=synth_z,
        train_y=synth_y,
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
