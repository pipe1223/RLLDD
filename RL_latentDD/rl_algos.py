# rl_algos.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """Plain policy (used for selection RL)."""
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ActorCriticNet(nn.Module):
    """Actor-Critic network for prototype shaping."""
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        h = self.shared(state)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value.squeeze(-1)


def train_policy_reinforce(
    env,
    policy: PolicyNet,
    device: torch.device,
    n_episodes: int,
    gamma: float,
    lr: float,
    print_every: int,
    desc: str = "RL",
):
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline = None
    reward_history = []

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        done = False

        log_probs = []
        rewards = []

        while not done:
            s = torch.from_numpy(state).float().to(device)
            logits = policy(s.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(int(action.item()))
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        R = returns[0].item()
        reward_history.append(R)

        if baseline is None:
            baseline = R
        else:
            baseline = 0.9 * baseline + 0.1 * R

        advantage = returns - baseline

        policy_loss = []
        for log_prob, adv in zip(log_probs, advantage):
            policy_loss.append(-log_prob * adv)
        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if ep % print_every == 0 or ep == 1:
            print(f"[{desc} Episode {ep:03d}] Reward={R:.4f}  Baseline={baseline:.4f}")

    return reward_history


def run_greedy_episode(env, policy: PolicyNet, device: torch.device, is_proto_env: bool = False):
    state = env.reset()
    done = False

    while not done:
        s = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            logits = policy(s.unsqueeze(0))
            action = torch.argmax(logits, dim=1)
        next_state, _, done, _ = env.step(int(action.item()))
        state = next_state

    if is_proto_env:
        return env.get_model()
    else:
        return env.get_selected_indices()


def train_actor_critic_proto(
    env,
    ac_net: ActorCriticNet,
    device: torch.device,
    n_episodes: int,
    gamma: float,
    lr: float,
    critic_weight: float,
    print_every: int,
):
    ac_net.to(device)
    optimizer = torch.optim.Adam(ac_net.parameters(), lr=lr)

    steps_per_episode = env.steps_per_episode
    n_actions = 3

    action_counts = np.zeros((steps_per_episode, n_actions), dtype=np.int64)
    reward_history = []

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        done = False

        log_probs = []
        values = []
        rewards = []
        actions_taken = []

        while not done:
            s = torch.from_numpy(state).float().to(device).unsqueeze(0)
            logits, value = ac_net(s)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(int(action.item()))

            log_probs.append(log_prob.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(reward)
            actions_taken.append(int(action.item()))

            state = next_state

        for step_idx, a in enumerate(actions_taken):
            if step_idx < steps_per_episode:
                action_counts[step_idx, a] += 1

        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        values_tensor = torch.stack(values)
        log_probs_tensor = torch.stack(log_probs)

        advantages = returns - values_tensor.detach()

        actor_loss = -(log_probs_tensor * advantages).sum()
        critic_loss = F.mse_loss(values_tensor, returns)
        loss = actor_loss + critic_weight * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_R = returns[0].item()
        reward_history.append(final_R)

        if ep % print_every == 0 or ep == 1:
            state_dbg = env._compute_state()
            _, last_val = ac_net(torch.from_numpy(state_dbg).float().to(device).unsqueeze(0))
            print(
                f"[RL-Proto (AC) Episode {ep:03d}] "
                f"Reward={final_R:.4f} | Critic last V={last_val.item():.4f}"
            )

    return reward_history, action_counts


def run_greedy_proto_episode_ac(env, ac_net: ActorCriticNet, device: torch.device):
    state = env.reset()
    done = False

    while not done:
        s = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = ac_net(s)
            action = torch.argmax(logits, dim=1)
        next_state, _, done, _ = env.step(int(action.item()))
        state = next_state

    return env.get_model()
