# main.py
# -*- coding: cp1252 -*-

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from config import (
    SEED,
    AE_LATENT_DIM,
    AE_EPOCHS,
    DEFAULT_DATASET,
    DEFAULT_BACKBONE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_AE_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    M_PER_CLASS,
    PROTO_STEPS_BASE,
    PROTO_STEPS_RL_EPISODE,
    BATCH_SIZE_REAL_BASE,
    BATCH_SIZE_REAL_RL,
    BASE_LAMBDA_REAL,
    BASE_LAMBDA_DIV,
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
    CROSS_EVAL_EPOCHS,
    PRINT_EVERY_SEL,
    PRINT_EVERY_PROTO_RL,
    RESULTS_DIR,
)
from experiment import (
    BackboneConfig,
    DataConfig,
    ExperimentMetadata,
    LoaderConfig,
    backbone_name_choices,
    dataset_name_choices,
    build_transforms,
    load_dataset,
    prepare_output_dir,
)
from utils import ensure_dir, set_seed, get_device, save_experiment_report
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
from image_reward import make_decoded_latent_reward_fn
from cross_eval import (
    normalize_imagenet,
    train_on_synth_and_eval,
    sample_real_ipc_tensors,
    reconstruct_images,
)

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
    parser.add_argument("--dataset", choices=list(dataset_name_choices()), default=DEFAULT_DATASET)
    parser.add_argument("--backbone", choices=list(backbone_name_choices()), default=DEFAULT_BACKBONE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--latent-dim", type=int, default=AE_LATENT_DIM, help="Latent dimensionality for AE")
    parser.add_argument("--ae-epochs", type=int, default=AE_EPOCHS, help="Number of epochs for AE pretraining")
    parser.add_argument("--data-root", default="./data", help="Root directory for torchvision datasets")
    parser.add_argument("--custom-train-dir", help="Path to custom training images (ImageFolder compatible)")
    parser.add_argument("--custom-test-dir", help="Path to custom test images (ImageFolder compatible)")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction for AE training")
    parser.add_argument("--max-train-points", type=int, default=MAX_TRAIN_POINTS, help="Cap latent pool size")
    parser.add_argument("--override-image-size", type=int, default=None, help="Force resize for custom datasets")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name for results folder")
    parser.add_argument("--m-per-class", type=int, default=M_PER_CLASS, help="Number of prototypes per class")
    parser.add_argument("--sel-budget-per-class", type=int, default=SEL_BUDGET_PER_CLASS)
    parser.add_argument("--proto-steps-base", type=int, default=PROTO_STEPS_BASE)
    parser.add_argument("--proto-steps-rl-episode", type=int, default=PROTO_STEPS_RL_EPISODE)
    parser.add_argument("--sel-episodes", type=int, default=SEL_EPISODES)
    parser.add_argument("--sel-gamma", type=float, default=SEL_GAMMA)
    parser.add_argument("--sel-lr", type=float, default=SEL_LR)
    parser.add_argument("--proto-rl-episodes", type=int, default=PROTO_RL_EPISODES)
    parser.add_argument("--proto-rl-gamma", type=float, default=PROTO_RL_GAMMA)
    parser.add_argument("--proto-rl-lr", type=float, default=PROTO_RL_LR)
    parser.add_argument("--critic-weight", type=float, default=CRITIC_WEIGHT)
    parser.add_argument(
        "--proto-reward-mode",
        type=str,
        default="latent",
        choices=["latent", "decoded_ae_head"],
        help="Reward type for prototype RL: "
             "'latent' = val acc of proto classifier in latent space (default); "
             "'decoded_ae_head' = decode prototypes and use frozen AE head (Conv/ResNet/ViT) "
             "by re-encoding and training a latent linear classifier.",
    )
    parser.add_argument(
        "--enable-cross-eval",
        action="store_true",
        help="If set, train external eval models (Conv / ResNet / ViT) on decoded prototypes "
             "and report test accuracy to measure cross-architecture generalization.",
    )
    parser.add_argument(
        "--cross-eval-epochs",
        type=int,
        default=CROSS_EVAL_EPOCHS,
        help="Training epochs for external eval models on distilled data.",
    )
    
    parser.add_argument(
        "--ae-batch-size",
        type=int,
        default=DEFAULT_AE_BATCH_SIZE,
        help="Batch size for AE pretraining",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help="Where to store structured reports/artifacts",
    )
    return parser.parse_args()



def build_backbone(name: str, latent_dim: int, num_classes: int, img_size: int):
    if name == "resnet18":
        return ResNetAEClassifier(latent_dim=latent_dim, num_classes=num_classes, img_size=img_size)
    if name == "vit_b_16":
        return ViTAEClassifier(latent_dim=max(latent_dim, 256), num_classes=num_classes, img_size=img_size)
    return ConvAEClassifier(latent_dim=latent_dim, num_classes=num_classes)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    sel_budget_per_class = args.sel_budget_per_class
    m_per_class = args.m_per_class
    proto_steps_base = args.proto_steps_base
    proto_steps_rl_episode = args.proto_steps_rl_episode
    sel_episodes = args.sel_episodes
    sel_gamma = args.sel_gamma
    sel_lr = args.sel_lr
    proto_rl_episodes = args.proto_rl_episodes
    proto_rl_gamma = args.proto_rl_gamma
    proto_rl_lr = args.proto_rl_lr
    critic_weight = args.critic_weight
    ae_epochs = args.ae_epochs
    latent_dim_cfg = args.latent_dim
    proto_reward_mode = args.proto_reward_mode
    enable_cross_eval = args.enable_cross_eval
    cross_eval_epochs = args.cross_eval_epochs

    data_cfg = DataConfig(
        name=args.dataset,
        data_root=args.data_root,
        custom_train_dir=args.custom_train_dir,
        custom_test_dir=args.custom_test_dir,
        val_fraction=args.val_fraction,
        max_train_points=args.max_train_points,
        override_image_size=args.override_image_size,
    )
    backbone_cfg = BackboneConfig(name=args.backbone, latent_dim=latent_dim_cfg)
    loader_cfg = LoaderConfig(batch_size=args.ae_batch_size, num_workers=args.num_workers)

    transform, resolved_size = build_transforms(backbone_cfg, args.override_image_size or DEFAULT_IMAGE_SIZE)
    full_train_dataset, test_dataset, num_classes = load_dataset(data_cfg, transform)

    val_size = max(1, int(len(full_train_dataset) * data_cfg.val_fraction))
    train_size = len(full_train_dataset) - val_size
    if train_size <= num_classes:
        raise ValueError("Training split too small; reduce val_fraction or use a larger dataset.")
    g = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=g)

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=True,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
    )
    # Test loader (for external cross-evaluation on real images)
    test_loader = DataLoader(
        test_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
    )

    metadata = ExperimentMetadata(
        seed=args.seed,
        num_classes=num_classes,
        image_size=resolved_size,
        backbone=backbone_cfg,
        data=data_cfg,
        loader=loader_cfg,
    )
    print("\n[CONFIG]", metadata.as_dict())

    output_dir = prepare_output_dir(args.results_dir, args.dataset, args.backbone, run_name=args.run_name)
    ensure_dir(output_dir)

    ae_cls = build_backbone(args.backbone, latent_dim_cfg, num_classes, resolved_size)
    print("Training AE+classifier...")
    train_ae_classifier(
        model=ae_cls,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=ae_epochs,
        lr=1e-3,
        alpha_cls=1.0,
    )

    # --- Extract latents ---
    print("Extracting latents for full train set...")
    train_latents_all, train_labels_all = extract_latents(
        ae_cls, full_train_dataset, batch_size=loader_cfg.batch_size, device=device
    )

    print("Extracting latents for test set...")
    test_latents, test_labels = extract_latents(
        ae_cls, test_dataset, batch_size=loader_cfg.batch_size, device=device
    )

    num_train_lat = train_latents_all.size(0)
    adaptive_val = max(1000, min(5000, num_train_lat // 10))
    pure_train_size = num_train_lat - adaptive_val

    train_z_all = train_latents_all[:pure_train_size]
    train_y_all = train_labels_all[:pure_train_size]
    val_z = train_latents_all[pure_train_size:]
    val_y = train_labels_all[pure_train_size:]

    print(f"Train latents (all): {train_z_all.shape}, Val latents: {val_z.shape}, Test latents: {test_latents.shape}")

    # Subsample latent pool
    max_train_points = data_cfg.max_train_points
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
    if sel_budget_per_class * num_classes > train_z_pool.size(0):
        raise ValueError("Selection budget exceeds available latent pool; reduce SEL_BUDGET_PER_CLASS or max_train_points.")

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
        budget_per_class=sel_budget_per_class,
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
        budget_per_class=sel_budget_per_class,
        num_classes=num_classes,
        device=device,
        classifier_epochs=CLASSIFIER_EPOCHS_EVAL,
        kmeans_seed=args.seed,
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
        budget_per_class=sel_budget_per_class,
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
        n_episodes=sel_episodes,
        gamma=sel_gamma,
        lr=sel_lr,
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
        budget_per_class=sel_budget_per_class,
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
        m_per_class=m_per_class,
    )
    init_prototypes_from_data(proto_base, train_z_pool, train_y_pool)
    train_prototypes_baseline(
        model=proto_base,
        train_z=train_z_pool,
        train_y=train_y_pool,
        val_z=val_z,
        val_y=val_y,
        device=device,
        n_steps=proto_steps_base,
        batch_size_real=BATCH_SIZE_REAL_BASE,
        lambda_real=BASE_LAMBDA_REAL,
        lambda_div=BASE_LAMBDA_DIV,
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
    
    
    # --- Build optional image-space reward for prototype RL ---
    decode_latents_fn = None
    proto_image_reward_fn = None

    if proto_reward_mode == "decoded_ae_head":
        print("\n[Proto RL] Using decoded prototypes + frozen AE head reward (decode -> AE.encode -> latent classifier).")

        # Decode function: latent -> image, always on the correct device
        decode_latents_fn = lambda z: ae_cls.decode(z.to(device))

        # Reward function: image -> AE.encode -> latent classifier -> val acc
        proto_image_reward_fn = make_decoded_latent_reward_fn(
            ae_model=ae_cls,
            val_z=val_z,
            val_y=val_y,
            num_classes=num_classes,
            device=device,
            classifier_epochs=CLASSIFIER_EPOCHS_RL_REWARD,
        )
    else:
        print("\n[Proto RL] Using latent proto-classifier reward (no decode in the reward).")

    
    
    
    
    # --- Actor-Critic RL for prototype shaping (fine-tuning baseline) ---
    print("\nSetting up Actor-Critic RL environment for prototype shaping (fine-tune baseline)...")
    proto_base_state = proto_base.state_dict()

    proto_env = ProtoUpdateEnv(
        real_z=train_z_pool,
        real_y=train_y_pool,
        val_z=val_z,
        val_y=val_y,
        num_classes=num_classes,
        latent_dim=latent_dim,
        m_per_class=m_per_class,
        steps_per_episode=proto_steps_rl_episode,
        device=device,
        lr=5e-4,
        weight_decay=1e-4,
        batch_size_real=BATCH_SIZE_REAL_RL,
        base_model_state=proto_base_state,
        from_scratch=False,
        init_noise_std=0.01,
        decode_latents_fn=decode_latents_fn,
        image_reward_fn=proto_image_reward_fn,
    )

    ac_proto = ActorCriticNet(state_dim=proto_env.state_dim, n_actions=3, hidden_dim=64)

    print("\nTraining Actor–Critic policy for prototype shaping (fine-tuning baseline)...")
    _, action_counts = train_actor_critic_proto(
        env=proto_env,
        ac_net=ac_proto,
        device=device,
        n_episodes=proto_rl_episodes,
        gamma=proto_rl_gamma,
        lr=proto_rl_lr,
        critic_weight=critic_weight,
        print_every=PRINT_EVERY_PROTO_RL,
    )

    plot_proto_action_usage(action_counts, filename=str(Path(output_dir) / "proto_rl_actions_over_steps.png"))

    print("\nRunning greedy prototype-shaping episode with learned Actor–Critic policy (fine-tune)...")
    proto_eval_env = ProtoUpdateEnv(
        real_z=train_z_pool,
        real_y=train_y_pool,
        val_z=val_z,
        val_y=val_y,
        num_classes=num_classes,
        latent_dim=latent_dim,
        m_per_class=m_per_class,
        steps_per_episode=proto_steps_rl_episode,
        device=device,
        lr=5e-4,
        weight_decay=1e-4,
        batch_size_real=BATCH_SIZE_REAL_RL,
        base_model_state=proto_base_state,
        from_scratch=False,
        init_noise_std=0.0,
        decode_latents_fn=decode_latents_fn,
        image_reward_fn=proto_image_reward_fn,
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
    print(f"\nRL-shaped distilled prototypes (Actor?Critic, fine-tune) test acc: {rl_proto_acc:.4f}")

    # --- Cross-architecture evaluation of decoded prototypes (optional) ---
    cross_eval_metrics = {}

    if enable_cross_eval:
        print("\n[Cross-Eval] Evaluating decoded prototypes on external models "
              "(Conv / ResNet18 / ResNet50 / ViT-B/16)...")

        # Decode prototypes to images (CPU tensors)
        with torch.no_grad():
            base_protos_flat = proto_base.protos.detach().view(-1, latent_dim).to(device)
            base_proto_imgs = ae_cls.decode(base_protos_flat).cpu()  # [C*M, 3, H, W]

            rl_protos_flat = proto_rl_model.protos.detach().view(-1, latent_dim).to(device)
            rl_proto_imgs = ae_cls.decode(rl_protos_flat).cpu()

        # Labels for both prototype sets
        proto_labels = torch.arange(num_classes).repeat_interleave(m_per_class)

        # Match preprocessing with the AE backbone's transform.
        # If we used ResNet / ViT as backbone, ImageNet normalization was applied
        # to real images; do the same for decoded prototypes.
        if backbone_cfg.requires_imagenet_norm():
            print("[Cross-Eval] Applying ImageNet normalization to decoded prototypes.")
            base_proto_imgs_proc = normalize_imagenet(base_proto_imgs)
            rl_proto_imgs_proc = normalize_imagenet(rl_proto_imgs)
        else:
            base_proto_imgs_proc = base_proto_imgs
            rl_proto_imgs_proc = rl_proto_imgs

        # Eval model portfolio: a small conv net + ResNet18 + ResNet50 + ViT-B/16
        eval_model_names = ["conv_small", "resnet18", "resnet50", "vit_b_16"]
        
        ipc = m_per_class  # match prototype IPC by default
        print(f"\n[Cross-Eval/Calib] Sampling REAL {ipc} IPC per class from the training split...")

        real_ipc_imgs, real_ipc_labels = sample_real_ipc_tensors(
            dataset=train_dataset,  # uses the same preprocessing as training split
            num_classes=num_classes,
            ipc=ipc,
            seed=seed,
        )

        print("[Cross-Eval/Calib] Computing AE reconstructions for the REAL IPC subset...")
        real_ipc_recon = reconstruct_images(
            ae_model=ae_cls,
            imgs=real_ipc_imgs,
            device=device,
            batch_size=128,
        )

        # Preprocess reconstructions to match test preprocessing
        if backbone_cfg.requires_imagenet_norm():
            real_ipc_recon_proc = normalize_imagenet(real_ipc_recon)
        else:
            real_ipc_recon_proc = real_ipc_recon
        
        
        for model_name in eval_model_names:
            print(f"\n[Cross-Eval] Training {model_name} on BASELINE prototypes...")
            
            # ---- Calibration: REAL IPC baseline (same IPC as prototypes) ----
            if model_name in ["conv_small", "resnet18"]:
                print(f"\n[Cross-Eval/Calib] Training {model_name} on REAL {ipc} IPC...")
                real_acc = train_on_synth_and_eval(
                    model_name=model_name,
                    num_classes=num_classes,
                    image_size=resolved_size,
                    train_imgs=real_ipc_imgs,          # already preprocessed by dataset transform
                    train_labels=real_ipc_labels,
                    test_loader=test_loader,
                    device=device,
                    epochs=cross_eval_epochs,
                )
                print(f"[Cross-Eval/Calib] {model_name} REAL {ipc} IPC - test acc: {real_acc:.4f}")
                cross_eval_metrics[f"real_{ipc}ipc_{model_name}_acc"] = float(real_acc)

                # ---- Calibration: AE reconstruction baseline ----
                print(f"[Cross-Eval/Calib] Training {model_name} on AE reconstructions of REAL {ipc} IPC...")
                recon_acc = train_on_synth_and_eval(
                    model_name=model_name,
                    num_classes=num_classes,
                    image_size=resolved_size,
                    train_imgs=real_ipc_recon_proc,     # recon images normalized if needed
                    train_labels=real_ipc_labels,
                    test_loader=test_loader,
                    device=device,
                    epochs=cross_eval_epochs,
                )
                print(f"[Cross-Eval/Calib] {model_name} AE-RECON {ipc} IPC - test acc: {recon_acc:.4f}")
                cross_eval_metrics[f"recon_{ipc}ipc_{model_name}_acc"] = float(recon_acc)
            
            base_acc = train_on_synth_and_eval(
                model_name=model_name,
                num_classes=num_classes,
                image_size=resolved_size,
                train_imgs=base_proto_imgs_proc,
                train_labels=proto_labels,
                test_loader=test_loader,
                device=device,
                epochs=cross_eval_epochs,
            )
            print(f"[Cross-Eval] {model_name} on baseline prototypes - test acc: {base_acc:.4f}")

            print(f"[Cross-Eval] Training {model_name} on RL-SHAPED prototypes...")
            rl_acc = train_on_synth_and_eval(
                model_name=model_name,
                num_classes=num_classes,
                image_size=resolved_size,
                train_imgs=rl_proto_imgs_proc,
                train_labels=proto_labels,
                test_loader=test_loader,
                device=device,
                epochs=cross_eval_epochs,
            )
            print(f"[Cross-Eval] {model_name} on RL-shaped prototypes - test acc: {rl_acc:.4f}")

            cross_eval_metrics[f"baseline_proto_{model_name}_acc"] = float(base_acc)
            cross_eval_metrics[f"rl_proto_{model_name}_acc"] = float(rl_acc)

    else:
        print("\n[Cross-Eval] Skipped (use --enable-cross-eval to evaluate on external models).")
    
    # --- Visualizations ---
    ae_cls.eval()

    with torch.no_grad():
        base_protos_flat = proto_base.protos.detach().view(-1, latent_dim).to(device)
        base_proto_imgs = ae_cls.decode(base_protos_flat)
    show_and_save_grid(
        images=base_proto_imgs.cpu(),
        filename=str(Path(output_dir) / "prototypes_baseline_decoded.png"),
        title="Decoded prototypes (baseline, fixed loss weights)",
        nrow=m_per_class,
    )

    with torch.no_grad():
        rl_protos_flat = proto_rl_model.protos.detach().view(-1, latent_dim).to(device)
        rl_proto_imgs = ae_cls.decode(rl_protos_flat)
    show_and_save_grid(
        images=rl_proto_imgs.cpu(),
        filename=str(Path(output_dir) / "prototypes_rl_shaped_decoded.png"),
        title="Decoded prototypes (RL-shaped loss schedule, Actor–Critic, fine-tune)",
        nrow=m_per_class,
    )

    rl_indices_full = pool_indices[rl_indices_pool]
    rl_imgs = torch.stack([full_train_dataset[i][0] for i in rl_indices_full], dim=0)
    show_and_save_grid(
        images=rl_imgs,
        filename=str(Path(output_dir) / "rl_selected_real_images.png"),
        title="RL-selected real training images",
        nrow=sel_budget_per_class,
    )

    rand_indices_full = pool_indices[random_indices_pool]
    rand_imgs = torch.stack([full_train_dataset[i][0] for i in rand_indices_full], dim=0)
    show_and_save_grid(
        images=rand_imgs,
        filename=str(Path(output_dir) / "random_real_images_subset.png"),
        title="Random real training images (same budget)",
        nrow=sel_budget_per_class,
    )

    visualize_tsne_all(
        train_z_pool=train_z_pool,
        train_y_pool=train_y_pool,
        proto_base=proto_base,
        proto_rl=proto_rl_model,
        rl_indices_pool=rl_indices_pool,
        num_samples_per_class=200,
        filename=str(Path(output_dir) / "tsne_all_rl_dd.png"),
    )

    print("\n==================== SUMMARY (Latent + RL + DD, Actor–Critic fine-tune) ====================")
    print(f"Full-latent classifier test acc (upper bound): {full_latent_acc:.4f}")
    print(f"Random latent subset test acc:                {random_acc:.4f}")
    print(f"k-means centroids test acc:                  {kmeans_acc:.4f}")
    print(f"RL-selected latent subset test acc:          {rl_sel_acc:.4f}")
    print(f"Baseline prototypes (no RL) test acc:        {base_proto_acc:.4f}")
    print(f"RL-shaped prototypes (Actor-Critic, fine-tune) test acc: {rl_proto_acc:.4f}")
    
    if cross_eval_metrics:
        print("\n[Cross-Eval] External model accuracies on decoded prototypes:")
        for k, v in cross_eval_metrics.items():
            print(f"  {k}: {v:.4f}")
    
    experiment_config = metadata.as_dict()
    experiment_config.update(
        {
            "ae": {"latent_dim": latent_dim_cfg, "epochs": ae_epochs},
            "prototypes": {
                "m_per_class": m_per_class,
                "baseline_steps": proto_steps_base,
                "rl_steps_per_episode": proto_steps_rl_episode,
                "lambda_real": BASE_LAMBDA_REAL,
                "lambda_div": BASE_LAMBDA_DIV,
            },
            "selection": {
                "budget_per_class": sel_budget_per_class,
                "episodes": sel_episodes,
                "gamma": sel_gamma,
                "lr": sel_lr,
                "reward_epochs": CLASSIFIER_EPOCHS_RL_REWARD,
            },
            "proto_rl": {
                "episodes": proto_rl_episodes,
                "gamma": proto_rl_gamma,
                "lr": proto_rl_lr,
                "critic_weight": critic_weight,
            },
            "classifier_epochs_eval": CLASSIFIER_EPOCHS_EVAL,
            "train_pool_size": int(train_z_pool.size(0)),
        }
    )

    experiment_metrics = {
        "full_latent_acc": float(full_latent_acc),
        "random_subset_acc": float(random_acc),
        "kmeans_acc": float(kmeans_acc),
        "rl_selected_acc": float(rl_sel_acc),
        "baseline_proto_acc": float(base_proto_acc),
        "rl_proto_acc": float(rl_proto_acc),
    }
    experiment_metrics.update(cross_eval_metrics)

    distillation_artifacts_path = Path(output_dir) / "distillation_artifacts.pt"
    torch.save(
        {
            "rl_selected_indices_full": rl_indices_full,
            "rl_selected_latents": rl_subset_z.cpu(),
            "rl_selected_labels": rl_subset_y.cpu(),
            "baseline_proto_state": proto_base.state_dict(),
            "rl_proto_state": proto_rl_model.state_dict(),
        },
        distillation_artifacts_path,
    )

    print(f"Saved distilled data bundle to {distillation_artifacts_path}")

    report_path = save_experiment_report(
        output_dir=output_dir,
        config=experiment_config,
        metrics={**experiment_metrics, "distillation_artifacts": str(distillation_artifacts_path)},
        notes="Single-run summary generated from main.py",
    )
    print(f"\nStructured experiment report saved to: {report_path}")


if __name__ == "__main__":
    main()
