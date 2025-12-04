# main.py

import argparse
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

from config import (
    SEED,
    AE_LATENT_DIM,
    AE_EPOCHS,
    DEFAULT_DATASET,
    DEFAULT_BACKBONE,
    DEFAULT_IMAGE_SIZE,
    M_PER_CLASS,
    PROTO_STEPS_BASE,
    PROTO_STEPS_RL_EPISODE,
    BATCH_SIZE_REAL_BASE,
    BATCH_SIZE_REAL_RL,
    SEL_BUDGET_PER_CLASS,
    SEL_EPISODES,
    SEL_GAMMA,
    SEL_LR,
    CLASSIFIER_EPOCHS_RL_REWARD,
    PROTO_RL_EPISODES,
    PROTO_RL_GAMMA,
    PROTO_RL_LR,
    CRITIC_WEIGHT,
    MAX_TRAIN_POINTS,
    CLASSIFIER_EPOCHS_EVAL,
    PRINT_EVERY_SEL,
    PRINT_EVERY_PROTO_RL,
    RESULTS_DIR,
)
from utils import set_seed, get_device, save_experiment_report
from models import (
    ConvAEClassifier,
    LatentPrototypeModel,
    ResNetAEClassifier,
    ViTAEClassifier,
)
from latent_utils import (
    train_ae_classifier,
    extract_latents,
    train_latent_classifier,
    init_prototypes_from_data,
    train_prototypes_baseline,
    evaluate_distilled_prototypes,
)
from rl_envs import LatentSelectionEnv, ProtoUpdateEnv
from rl_algos import (
    PolicyNet,
    ActorCriticNet,
    train_policy_reinforce,
    run_greedy_episode,
    train_actor_critic_proto,
    run_greedy_proto_episode_ac,
)
from baselines import evaluate_random_latent_subset, evaluate_kmeans_centroids
from viz import show_and_save_grid, visualize_tsne_all, plot_proto_action_usage


def parse_args():
    parser = argparse.ArgumentParser(description="Latent DD with flexible backbones and datasets")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "custom"], default=DEFAULT_DATASET)
    parser.add_argument("--backbone", choices=["conv", "resnet18", "vit_b_16"], default=DEFAULT_BACKBONE)
    parser.add_argument("--data-root", default="./data", help="Root directory for torchvision datasets")
    parser.add_argument("--custom-train-dir", help="Path to custom training images (ImageFolder compatible)")
    parser.add_argument("--custom-test-dir", help="Path to custom test images (ImageFolder compatible)")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction for AE training")
    parser.add_argument("--max-train-points", type=int, default=MAX_TRAIN_POINTS, help="Cap latent pool size")
    parser.add_argument("--override-image-size", type=int, default=None, help="Force resize for custom datasets")
    return parser.parse_args()


def build_transforms(backbone: str, image_size: int):
    if backbone in {"resnet18", "vit_b_16"}:
        target_size = 224
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        target_size = image_size
        normalize = None

    ops = [T.Resize((target_size, target_size)), T.ToTensor()]
    if normalize:
        ops.append(normalize)
    return T.Compose(ops), target_size


def load_dataset(
    name: str,
    data_root: str,
    transform,
    custom_train: Optional[str],
    custom_test: Optional[str],
):
    if name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        num_classes = 10
    elif name == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
        num_classes = 100
    else:
        if not custom_train or not custom_test:
            raise ValueError("custom dataset requires --custom-train-dir and --custom-test-dir")
        train_ds = torchvision.datasets.ImageFolder(custom_train, transform=transform)
        test_ds = torchvision.datasets.ImageFolder(custom_test, transform=transform)
        num_classes = len(train_ds.classes)
    return train_ds, test_ds, num_classes


def build_backbone(name: str, latent_dim: int, num_classes: int, img_size: int):
    if name == "resnet18":
        return ResNetAEClassifier(latent_dim=latent_dim, num_classes=num_classes, img_size=img_size)
    if name == "vit_b_16":
        return ViTAEClassifier(latent_dim=max(latent_dim, 256), num_classes=num_classes, img_size=img_size)
    return ConvAEClassifier(latent_dim=latent_dim, num_classes=num_classes)


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    print("Using device:", device)

    transform, resolved_size = build_transforms(args.backbone, args.override_image_size or DEFAULT_IMAGE_SIZE)
    full_train_dataset, test_dataset, num_classes = load_dataset(
        args.dataset, args.data_root, transform, args.custom_train_dir, args.custom_test_dir
    )

    # Train/val split for AE+classifier
    val_size = max(1, int(len(full_train_dataset) * args.val_fraction))
    train_size = len(full_train_dataset) - val_size
    g = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # --- Train AE + classifier ---
    ae_cls = build_backbone(args.backbone, AE_LATENT_DIM, num_classes, resolved_size)
    print("Training AE+classifier...")
    train_ae_classifier(
        model=ae_cls,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=AE_EPOCHS,
        lr=1e-3,
        alpha_cls=1.0,
    )

    # --- Extract latents ---
    print("Extracting latents for full train set...")
    train_latents_all, train_labels_all = extract_latents(ae_cls, full_train_dataset, batch_size=256, device=device)

    print("Extracting latents for test set...")
    test_latents, test_labels = extract_latents(ae_cls, test_dataset, batch_size=256, device=device)

    num_train_lat = train_latents_all.size(0)
    val_lat_size = 5000
    pure_train_size = num_train_lat - val_lat_size

    train_z_all = train_latents_all[:pure_train_size]
    train_y_all = train_labels_all[:pure_train_size]
    val_z = train_latents_all[pure_train_size:]
    val_y = train_labels_all[pure_train_size:]

    print(f"Train latents (all): {train_z_all.shape}, Val latents: {val_z.shape}, Test latents: {test_latents.shape}")

    # Subsample latent pool
    max_train_points = args.max_train_points
    if train_z_all.size(0) > max_train_points:
        pool_indices = np.random.choice(train_z_all.size(0), size=max_train_points, replace=False)
        train_z_pool = train_z_all[pool_indices]
        train_y_pool = train_y_all[pool_indices]
        print(f"Subsampled train latents for pool: {train_z_pool.shape}")
    else:
        pool_indices = np.arange(train_z_all.size(0))
        train_z_pool = train_z_all
        train_y_pool = train_y_all

    latent_dim = train_z_pool.size(1)

    # --- Full-latent baseline (upper bound) ---
    print("\nTraining full-latent linear classifier (upper bound, pool)...")
    full_latent_acc = train_latent_classifier(
        train_z=train_z_pool,
        train_y=train_y_pool,
        val_z=test_latents,
        val_y=test_labels,
        n_classes=num_classes,
        device=device,
        epochs=CLASSIFIER_EPOCHS_EVAL,
        batch_size=128,
        lr=1e-2,
        weight_decay=1e-4,
    )
    print(f"Full-latent classifier test acc: {full_latent_acc:.4f}")

    # --- Random subset baseline ---
    print("\nEvaluating random latent subset baseline...")
    random_acc, random_indices_pool = evaluate_random_latent_subset(
        train_z=train_z_pool,
        train_y=train_y_pool,
        test_z=test_latents,
        test_y=test_labels,
        budget_per_class=SEL_BUDGET_PER_CLASS,
        num_classes=num_classes,
        device=device,
        classifier_epochs=CLASSIFIER_EPOCHS_EVAL,
    )
    print(f"Random subset test acc: {random_acc:.4f}")

    # --- k-means centroids baseline ---
    print("\nEvaluating k-means centroids baseline...")
    kmeans_acc = evaluate_kmeans_centroids(
        train_z=train_z_pool,
        train_y=train_y_pool,
        test_z=test_latents,
        test_y=test_labels,
        budget_per_class=SEL_BUDGET_PER_CLASS,
        num_classes=num_classes,
        device=device,
        classifier_epochs=CLASSIFIER_EPOCHS_EVAL,
        kmeans_seed=SEED,
    )
    print(f"k-means centroids test acc: {kmeans_acc:.4f}")

    # --- RL selection (REINFORCE on subset) ---
    print("\nSetting up RL selection environment...")
    sel_env = LatentSelectionEnv(
        train_z=train_z_pool,
        train_y=train_y_pool,
        val_z=val_z,
        val_y=val_y,
        num_classes=num_classes,
        budget_per_class=SEL_BUDGET_PER_CLASS,
        device=device,
        classifier_epochs=CLASSIFIER_EPOCHS_RL_REWARD,
        train_latent_classifier_fn=train_latent_classifier,
    )
    sel_policy = PolicyNet(state_dim=sel_env.state_dim, n_actions=2, hidden_dim=128)

    print("\nTraining RL policy for latent subset selection (REINFORCE)...")
    _ = train_policy_reinforce(
        env=sel_env,
        policy=sel_policy,
        device=device,
        n_episodes=SEL_EPISODES,
        gamma=SEL_GAMMA,
        lr=SEL_LR,
        print_every=PRINT_EVERY_SEL,
        desc="RL-Select",
    )

    print("\nRunning greedy RL-selection episode...")
    sel_eval_env = LatentSelectionEnv(
        train_z=train_z_pool,
        train_y=train_y_pool,
        val_z=val_z,
        val_y=val_y,
        num_classes=num_classes,
        budget_per_class=SEL_BUDGET_PER_CLASS,
        device=device,
        classifier_epochs=CLASSIFIER_EPOCHS_RL_REWARD,
        train_latent_classifier_fn=train_latent_classifier,
    )
    rl_indices_pool = np.array(
        run_greedy_episode(sel_eval_env, sel_policy, device, is_proto_env=False),
        dtype=np.int64,
    )
    print(f"RL selected {len(rl_indices_pool)} points in pool.")

    rl_subset_z = train_z_pool[rl_indices_pool]
    rl_subset_y = train_y_pool[rl_indices_pool]
    rl_sel_acc = train_latent_classifier(
        train_z=rl_subset_z,
        train_y=rl_subset_y,
        val_z=test_latents,
        val_y=test_labels,
        n_classes=num_classes,
        device=device,
        epochs=CLASSIFIER_EPOCHS_EVAL,
        batch_size=128,
        lr=1e-2,
        weight_decay=1e-4,
    )
    print(f"RL-selected latent subset test acc: {rl_sel_acc:.4f}")

    # --- Baseline prototypes (no RL) ---
    print("\nTraining baseline prototypes (fixed loss weights)...")
    proto_base = LatentPrototypeModel(
        num_classes=num_classes,
        latent_dim=latent_dim,
        m_per_class=M_PER_CLASS,
    )
    init_prototypes_from_data(proto_base, train_z_pool, train_y_pool)
    train_prototypes_baseline(
        model=proto_base,
        train_z=train_z_pool,
        train_y=train_y_pool,
        val_z=val_z,
        val_y=val_y,
        device=device,
        n_steps=PROTO_STEPS_BASE,
        batch_size_real=BATCH_SIZE_REAL_BASE,
        lambda_real=1.0,
        lambda_div=0.5,
        lr=1e-3,
        weight_decay=1e-4,
        print_every=100,
    )

    base_proto_acc = evaluate_distilled_prototypes(
        model=proto_base,
        test_z=test_latents,
        test_y=test_labels,
        num_classes=num_classes,
        device=device,
        classifier_epochs=CLASSIFIER_EPOCHS_EVAL,
    )
    print(f"\nBaseline distilled prototypes test acc: {base_proto_acc:.4f}")

    # --- Actor–Critic RL for prototype shaping (fine-tuning baseline) ---
    print("\nSetting up Actor–Critic RL environment for prototype shaping (fine-tune baseline)...")
    proto_base_state = proto_base.state_dict()

    proto_env = ProtoUpdateEnv(
        real_z=train_z_pool,
        real_y=train_y_pool,
        val_z=val_z,
        val_y=val_y,
        num_classes=num_classes,
        latent_dim=latent_dim,
        m_per_class=M_PER_CLASS,
        steps_per_episode=PROTO_STEPS_RL_EPISODE,
        device=device,
        lr=5e-4,                      # smaller LR for safer fine-tuning
        weight_decay=1e-4,
        batch_size_real=BATCH_SIZE_REAL_RL,
        base_model_state=proto_base_state,  # start from baseline
        from_scratch=False,                 # fine-tuning mode
        init_noise_std=0.01,                # small noise for training episodes
    )
    ac_proto = ActorCriticNet(state_dim=proto_env.state_dim, n_actions=3, hidden_dim=64)

    print("\nTraining Actor–Critic policy for prototype shaping (fine-tuning baseline)...")
    _, action_counts = train_actor_critic_proto(
        env=proto_env,
        ac_net=ac_proto,
        device=device,
        n_episodes=PROTO_RL_EPISODES,
        gamma=PROTO_RL_GAMMA,
        lr=PROTO_RL_LR,
        critic_weight=CRITIC_WEIGHT,
        print_every=PRINT_EVERY_PROTO_RL,
    )

    # Plot RL action usage over prototype steps
    plot_proto_action_usage(action_counts, filename="proto_rl_actions_over_steps.png")

    print("\nRunning greedy prototype-shaping episode with learned Actor–Critic policy (fine-tune)...")
    proto_eval_env = ProtoUpdateEnv(
        real_z=train_z_pool,
        real_y=train_y_pool,
        val_z=val_z,
        val_y=val_y,
        num_classes=num_classes,
        latent_dim=latent_dim,
        m_per_class=M_PER_CLASS,
        steps_per_episode=PROTO_STEPS_RL_EPISODE,
        device=device,
        lr=5e-4,
        weight_decay=1e-4,
        batch_size_real=BATCH_SIZE_REAL_RL,
        base_model_state=proto_base_state,
        from_scratch=False,
        init_noise_std=0.0,   # no noise at eval
    )
    proto_rl_model = run_greedy_proto_episode_ac(proto_eval_env, ac_proto, device)

    rl_proto_acc = evaluate_distilled_prototypes(
        model=proto_rl_model,
        test_z=test_latents,
        test_y=test_labels,
        num_classes=num_classes,
        device=device,
        classifier_epochs=CLASSIFIER_EPOCHS_EVAL,
    )
    print(f"\nRL-shaped distilled prototypes (Actor–Critic, fine-tune) test acc: {rl_proto_acc:.4f}")

    # --- Visualizations ---
    ae_cls.eval()

    # Baseline prototypes decoded
    with torch.no_grad():
        base_protos_flat = proto_base.protos.detach().view(-1, latent_dim).to(device)
        base_proto_imgs = ae_cls.decode(base_protos_flat)
    show_and_save_grid(
        images=base_proto_imgs.cpu(),
        filename="prototypes_baseline_decoded.png",
        title="Decoded prototypes (baseline, fixed loss weights)",
        nrow=M_PER_CLASS,
    )

    # RL-shaped prototypes decoded
    with torch.no_grad():
        rl_protos_flat = proto_rl_model.protos.detach().view(-1, latent_dim).to(device)
        rl_proto_imgs = ae_cls.decode(rl_protos_flat)
    show_and_save_grid(
        images=rl_proto_imgs.cpu(),
        filename="prototypes_rl_shaped_decoded.png",
        title="Decoded prototypes (RL-shaped loss schedule, Actor–Critic, fine-tune)",
        nrow=M_PER_CLASS,
    )

    # RL-selected real images vs random subset
    rl_indices_full = pool_indices[rl_indices_pool]
    rl_imgs = torch.stack([full_train_dataset[i][0] for i in rl_indices_full], dim=0)
    show_and_save_grid(
        images=rl_imgs,
        filename="rl_selected_real_images.png",
        title="RL-selected real training images",
        nrow=SEL_BUDGET_PER_CLASS,
    )

    rand_indices_full = pool_indices[random_indices_pool]
    rand_imgs = torch.stack([full_train_dataset[i][0] for i in rand_indices_full], dim=0)
    show_and_save_grid(
        images=rand_imgs,
        filename="random_real_images_subset.png",
        title="Random real training images (same budget)",
        nrow=SEL_BUDGET_PER_CLASS,
    )

    # t-SNE: real + baseline protos + RL protos + RL-selected
    visualize_tsne_all(
        train_z_pool=train_z_pool,
        train_y_pool=train_y_pool,
        proto_base=proto_base,
        proto_rl=proto_rl_model,
        rl_indices_pool=rl_indices_pool,
        num_samples_per_class=200,
        filename="tsne_all_rl_dd.png",
    )

    # --- Summary ---
    print("\n==================== SUMMARY (Latent + RL + DD, Actor–Critic fine-tune) ====================")
    print(f"Full-latent classifier test acc (upper bound): {full_latent_acc:.4f}")
    print(f"Random latent subset test acc:                {random_acc:.4f}")
    print(f"k-means centroids test acc:                  {kmeans_acc:.4f}")
    print(f"RL-selected latent subset test acc:          {rl_sel_acc:.4f}")
    print(f"Baseline prototypes (no RL) test acc:        {base_proto_acc:.4f}")
    print(f"RL-shaped prototypes (Actor–Critic, fine-tune) test acc: {rl_proto_acc:.4f}")
    print("Saved images:")
    print("  prototypes_baseline_decoded.png")
    print("  prototypes_rl_shaped_decoded.png")
    print("  rl_selected_real_images.png")
    print("  random_real_images_subset.png")
    print("  proto_rl_actions_over_steps.png")
    print("t-SNE plot:")
    print("  tsne_all_rl_dd.png")

    experiment_config = {
        "seed": SEED,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "image_size": resolved_size,
        "ae": {"latent_dim": AE_LATENT_DIM, "epochs": AE_EPOCHS},
        "prototypes": {
            "m_per_class": M_PER_CLASS,
            "baseline_steps": PROTO_STEPS_BASE,
            "rl_steps_per_episode": PROTO_STEPS_RL_EPISODE,
            "lambda_real": 1.0,
            "lambda_div": 0.5,
        },
        "selection": {
            "budget_per_class": SEL_BUDGET_PER_CLASS,
            "episodes": SEL_EPISODES,
            "gamma": SEL_GAMMA,
            "lr": SEL_LR,
            "reward_epochs": CLASSIFIER_EPOCHS_RL_REWARD,
        },
        "proto_rl": {
            "episodes": PROTO_RL_EPISODES,
            "gamma": PROTO_RL_GAMMA,
            "lr": PROTO_RL_LR,
            "critic_weight": CRITIC_WEIGHT,
        },
        "max_train_points": max_train_points,
        "classifier_epochs_eval": CLASSIFIER_EPOCHS_EVAL,
        "train_pool_size": int(train_z_pool.size(0)),
        "num_classes": num_classes,
    }

    experiment_metrics = {
        "full_latent_acc": float(full_latent_acc),
        "random_subset_acc": float(random_acc),
        "kmeans_acc": float(kmeans_acc),
        "rl_selected_acc": float(rl_sel_acc),
        "baseline_proto_acc": float(base_proto_acc),
        "rl_proto_acc": float(rl_proto_acc),
    }

    report_path = save_experiment_report(
        output_dir=RESULTS_DIR,
        config=experiment_config,
        metrics=experiment_metrics,
        notes="Single-run summary generated from main.py",
    )
    print(f"\nStructured experiment report saved to: {report_path}")


if __name__ == "__main__":
    main()
