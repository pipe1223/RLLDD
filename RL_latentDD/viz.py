# viz.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.manifold import TSNE


def show_and_save_grid(images: torch.Tensor, filename: str, title: str = None, nrow: int = 10):
    grid = make_grid(images, nrow=nrow, padding=2)
    npimg = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(nrow, max(1, images.size(0) // nrow)))
    plt.imshow(np.clip(npimg, 0, 1))
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()
    print(f"Saved image grid to {filename}")


def visualize_tsne_all(
    train_z_pool: torch.Tensor,
    train_y_pool: torch.Tensor,
    proto_base,
    proto_rl,
    rl_indices_pool: np.ndarray,
    num_samples_per_class: int = 200,
    filename: str = "tsne_all_rl_dd.png",
):
    num_classes = proto_base.num_classes
    m_pc = proto_base.m_per_class
    latent_dim = proto_base.latent_dim

    z_np = train_z_pool.numpy()
    y_np = train_y_pool.numpy()
    
    vis_indices = []
    for c in range(num_classes):
        idx_c = np.where(y_np == c)[0]
        np.random.shuffle(idx_c)
        idx_c = idx_c[:num_samples_per_class]
        vis_indices.extend(idx_c.tolist())
    vis_indices = np.array(vis_indices, dtype=np.int64)

    real_z = z_np[vis_indices]
    real_y = y_np[vis_indices]

    with torch.no_grad():
        base_protos = proto_base.protos.detach().cpu().view(-1, latent_dim).numpy()
        rl_protos = proto_rl.protos.detach().cpu().view(-1, latent_dim).numpy()

    rl_z = z_np[rl_indices_pool]
    rl_y = y_np[rl_indices_pool]

    data = np.concatenate([real_z, base_protos, rl_protos, rl_z], axis=0)
    print("Running t-SNE on real + baseline protos + RL protos + RL-selected latents...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, init="pca")
    emb = tsne.fit_transform(data)

    n_real = real_z.shape[0]
    n_base = base_protos.shape[0]
    n_rlp = rl_protos.shape[0]
    n_rl = rl_z.shape[0]

    emb_real = emb[:n_real]
    emb_base = emb[n_real:n_real + n_base]
    emb_rlp = emb[n_real + n_base:n_real + n_base + n_rlp]
    emb_rl = emb[n_real + n_base + n_rlp:]

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    # Real
    for c in range(num_classes):
        mask = real_y == c
        plt.scatter(
            emb_real[mask, 0],
            emb_real[mask, 1],
            s=5,
            alpha=0.25,
            color=cmap(c),
            label=f"Real class {c}" if c == 0 else None,
        )

    # Baseline prototypes (X)
    for c in range(num_classes):
        start = c * m_pc
        end = start + m_pc
        plt.scatter(
            emb_base[start:end, 0],
            emb_base[start:end, 1],
            s=80,
            marker="X",
            edgecolors="black",
            linewidths=1.0,
            color=cmap(c),
            label=f"Baseline proto class {c}" if c == 0 else None,
        )

    # RL-shaped prototypes (D)
    for c in range(num_classes):
        start = c * m_pc
        end = start + m_pc
        plt.scatter(
            emb_rlp[start:end, 0],
            emb_rlp[start:end, 1],
            s=80,
            marker="D",
            edgecolors="black",
            linewidths=1.0,
            color=cmap(c),
            label=f"RL proto class {c}" if c == 0 else None,
        )

    # RL-selected latents (^)
    for c in range(num_classes):
        mask = rl_y == c
        plt.scatter(
            emb_rl[mask, 0],
            emb_rl[mask, 1],
            s=40,
            marker="^",
            edgecolors="black",
            linewidths=0.5,
            color=cmap(c),
            label=f"RL-selected class {c}" if c == 0 else None,
        )

    plt.title("t-SNE: real (dots), baseline protos (X), RL protos (D), RL-selected (^)")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()
    print(f"Saved t-SNE plot to {filename}")


def plot_proto_action_usage(action_counts: np.ndarray, filename: str = "proto_rl_actions_over_steps.png"):
    steps, n_actions = action_counts.shape
    totals = action_counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    probs = action_counts / totals

    x = np.arange(steps)

    plt.figure(figsize=(8, 4))
    labels = ["Balanced", "Diversity-heavy", "Real-heavy"]
    for a in range(n_actions):
        plt.plot(x, probs[:, a], label=f"{labels[a]} (action {a})")

    plt.xlabel("Prototype training step index")
    plt.ylabel("Action usage probability")
    plt.title("Prototype-RL (Actor–Critic): action usage over steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()
    print(f"Saved prototype action-usage plot to {filename}")
