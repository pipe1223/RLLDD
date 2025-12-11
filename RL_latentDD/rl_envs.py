# rl_envs.py

import numpy as np
import torch
import torch.nn as nn

from models import LatentPrototypeModel
from latent_utils import init_prototypes_from_data, compute_diversity_loss
from config import BASE_LAMBDA_REAL, BASE_LAMBDA_DIV


class LatentSelectionEnv:
    """
    RL env to select a subset of latent points.

    State = [z, one_hot(y), normalized class counts, progress]
    Actions: 0=skip, 1=select (if class budget not exceeded).
    Reward at end = val accuracy of classifier trained on selected subset.
    """
    def __init__(
        self,
        train_z: torch.Tensor,
        train_y: torch.Tensor,
        val_z: torch.Tensor,
        val_y: torch.Tensor,
        num_classes: int,
        budget_per_class: int,
        device: torch.device,
        classifier_epochs: int = 15,
        train_latent_classifier_fn=None,
    ):
        assert train_latent_classifier_fn is not None, "Must pass train_latent_classifier_fn"
        self.train_z = train_z
        self.train_y = train_y
        self.val_z = val_z
        self.val_y = val_y
        self.num_classes = num_classes
        self.budget_per_class = budget_per_class
        self.device = device
        self.classifier_epochs = classifier_epochs
        self.train_latent_classifier_fn = train_latent_classifier_fn

        self.N = train_z.size(0)
        self.latent_dim = train_z.size(1)
        self.state_dim = self.latent_dim + num_classes + num_classes + 1

        self.reset()

    def reset(self):
        self.t = 0
        self.selected_indices = []
        self.class_counts = np.zeros(self.num_classes, dtype=np.int32)
        self.done = False
        return self._get_state()

    def _get_state(self):
        if self.t >= self.N:
            return np.zeros(self.state_dim, dtype=np.float32)

        z = self.train_z[self.t].numpy()
        y = int(self.train_y[self.t].item())

        y_one_hot = np.zeros(self.num_classes, dtype=np.float32)
        y_one_hot[y] = 1.0

        counts_norm = self.class_counts / max(1, self.budget_per_class)
        progress = np.array([self.t / max(1, self.N - 1)], dtype=np.float32)

        state = np.concatenate([z, y_one_hot, counts_norm, progress], axis=0)
        return state.astype(np.float32)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Step called on finished episode")

        idx = self.t
        y = int(self.train_y[idx].item())

        if action == 1:
            if self.class_counts[y] < self.budget_per_class:
                self.selected_indices.append(idx)
                self.class_counts[y] += 1

        self.t += 1

        if self.t >= self.N or np.all(self.class_counts >= self.budget_per_class):
            self.done = True
            reward = self._compute_reward()
            next_state = np.zeros(self.state_dim, dtype=np.float32)
            return next_state, reward, True, {}
        else:
            reward = 0.0
            next_state = self._get_state()
            return next_state, reward, False, {}

    def _compute_reward(self) -> float:
        if len(self.selected_indices) < self.num_classes:
            return 0.0

        subset_z = self.train_z[self.selected_indices]
        subset_y = self.train_y[self.selected_indices]

        acc = self.train_latent_classifier_fn(
            train_z=subset_z,
            train_y=subset_y,
            val_z=self.val_z,
            val_y=self.val_y,
            n_classes=self.num_classes,
            device=self.device,
            epochs=self.classifier_epochs,
            batch_size=128,
            lr=1e-2,
            weight_decay=1e-4,
        )
        return acc

    def get_selected_indices(self):
        return self.selected_indices


class ProtoUpdateEnv:
    """
    RL env where the agent controls how prototypes are updated.

    State (3 dims):
      [step_frac, val_acc, diversity]

    Actions (3 discrete modes):
      0: Balanced
         ?_real = BASE_LAMBDA_REAL
         ?_div  = BASE_LAMBDA_DIV
      1: Diversity-heavy
         ?_real = BASE_LAMBDA_REAL
         ?_div  = BASE_LAMBDA_DIV * 3
      2: Real-heavy
         ?_real = BASE_LAMBDA_REAL * 2
         ?_div  = BASE_LAMBDA_DIV * 0.2

    Can either start from scratch or fine-tune a given baseline prototype model.

    Optionally, you can provide:
      - decode_latents_fn: maps latent vectors -> images (e.g. AE.decode),
      - image_reward_fn: takes decoded images + labels and returns a scalar reward.

    If image_reward_fn is provided, the FINAL episode reward is computed in image space
    using that function; otherwise we fall back to latent proto-classifier val accuracy.
    """

    def __init__(
        self,
        real_z: torch.Tensor,
        real_y: torch.Tensor,
        val_z: torch.Tensor,
        val_y: torch.Tensor,
        num_classes: int,
        latent_dim: int,
        m_per_class: int,
        steps_per_episode: int,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size_real: int = 128,
        base_model_state: dict = None,
        from_scratch: bool = True,
        init_noise_std: float = 0.0,
        decode_latents_fn=None,
        image_reward_fn=None,
    ):
        self.real_z = real_z
        self.real_y = real_y
        self.val_z = val_z
        self.val_y = val_y

        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.m_per_class = m_per_class

        self.steps_per_episode = steps_per_episode
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size_real = batch_size_real
        self.N = real_z.size(0)

        self.state_dim = 3
        self.criterion = nn.CrossEntropyLoss()

        self.base_model_state = base_model_state
        self.from_scratch = from_scratch
        self.init_noise_std = init_noise_std

        self.model = None
        self.optimizer = None
        self.step_idx = 0
        self.done = False

        self.val_z_device = self.val_z.to(self.device)
        self.val_y_device = self.val_y.to(self.device)

        # Optional image-space reward components
        self.decode_latents_fn = decode_latents_fn
        self.image_reward_fn = image_reward_fn

    def reset(self):
        """
        If from_scratch=True or base_model_state is None:
            - initialize prototypes from real data.
        Else:
            - load baseline prototype weights and optionally add small noise.
        """
        self.step_idx = 0
        self.done = False

        self.model = LatentPrototypeModel(
            num_classes=self.num_classes,
            latent_dim=self.latent_dim,
            m_per_class=self.m_per_class,
        ).to(self.device)

        if (self.base_model_state is not None) and (not self.from_scratch):
            self.model.load_state_dict(self.base_model_state)
            if self.init_noise_std > 0.0:
                with torch.no_grad():
                    self.model.protos.add_(
                        self.init_noise_std * torch.randn_like(self.model.protos)
                    )
        else:
            init_prototypes_from_data(self.model, self.real_z, self.real_y)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return self._compute_state()

    def _compute_state(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.val_z_device)
            preds = logits.argmax(dim=1)
            val_acc = (preds == self.val_y_device).float().mean().item()
            div_loss_val = compute_diversity_loss(self.model).item()
        step_frac = self.step_idx / max(1, self.steps_per_episode - 1)
        return np.array([step_frac, val_acc, div_loss_val], dtype=np.float32)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Step called on finished episode")

        if action == 0:
            lambda_real = BASE_LAMBDA_REAL
            lambda_div = BASE_LAMBDA_DIV
        elif action == 1:
            lambda_real = BASE_LAMBDA_REAL
            lambda_div = BASE_LAMBDA_DIV * 3.0
        elif action == 2:
            lambda_real = BASE_LAMBDA_REAL * 2.0
            lambda_div = BASE_LAMBDA_DIV * 0.2
        else:
            raise ValueError("Invalid action")

        self.model.train()

        # Synthetic latent batch from prototypes
        synth_z, synth_y = self.model.get_synthetic_dataset()
        logits_synth = self.model(synth_z)
        loss_synth = self.criterion(logits_synth, synth_y)

        # Real latent batch
        idx = torch.randint(0, self.N, (self.batch_size_real,))
        z_real = self.real_z[idx].to(self.device)
        y_real = self.real_y[idx].to(self.device)
        logits_real = self.model(z_real)
        loss_real = self.criterion(logits_real, y_real)

        # Diversity regularizer
        div_loss = compute_diversity_loss(self.model)

        loss = loss_synth + lambda_real * loss_real + lambda_div * div_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_idx += 1

        if self.step_idx >= self.steps_per_episode:
            self.done = True

            # --- Reward computation ---
            if (self.image_reward_fn is None) or (self.decode_latents_fn is None):
                # Default: latent proto-classifier val accuracy
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(self.val_z_device)
                    preds = logits.argmax(dim=1)
                    final_val_acc = (preds == self.val_y_device).float().mean().item()
                reward = final_val_acc
            else:
                # Image-space reward: decode prototypes, then call external reward fn
                self.model.eval()
                with torch.no_grad():
                    synth_z, synth_y = self.model.get_synthetic_dataset()
                    synth_z = synth_z.to(self.device)
                    synth_y = synth_y.to(self.device)
                    proto_imgs = self.decode_latents_fn(synth_z)
                reward = float(self.image_reward_fn(proto_imgs, synth_y))

            next_state = np.zeros(self.state_dim, dtype=np.float32)
            done = True
        else:
            reward = 0.0
            next_state = self._compute_state()
            done = False

        return next_state, reward, done, {}

    def get_model(self):
        return self.model
