import os
import sys
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snake_game_custom_wrapper_cnn import SnakeEnv


def _to_tensor(state: np.ndarray) -> torch.Tensor:
    """Convert observation to a float tensor with shape (1, C, H, W)."""
    return torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


class CNNPolicy(nn.Module):
    """Policy network with convolutional layers."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            out_dim = self.features(torch.zeros(1, 3, 84, 84)).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        logits = self.fc(self.features(x))
        return torch.softmax(logits, dim=-1)


class CNNValue(nn.Module):
    """Value network with convolutional layers."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            out_dim = self.features(torch.zeros(1, 3, 84, 84)).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        value = self.fc(self.features(x))
        return value.squeeze(-1)


class PPOAgent:
    """PPO agent encapsulating policy, value and update logic."""

    def __init__(
        self,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        k_epochs: int = 4,
    ) -> None:
        self.policy = CNNPolicy(action_dim)
        self.value = CNNValue()
        self.opt_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.opt_value = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.memory: Deque[Tuple[np.ndarray, int, float, float, bool, np.ndarray]] = (
            deque()
        )

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        state_t = _to_tensor(state)
        with torch.no_grad():
            probs = self.policy(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        done: bool,
        next_state: np.ndarray,
    ) -> None:
        self.memory.append((state, action, reward, log_prob, done, next_state))

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    def update(self) -> None:
        if not self.memory:
            return
        states, actions, rewards, old_log_probs, dones, next_states = zip(*self.memory)
        self.memory.clear()
        states_t = torch.cat([_to_tensor(s) for s in states])
        next_states_t = torch.cat([_to_tensor(ns) for ns in next_states])
        actions_t = torch.tensor(actions)
        old_log_probs_t = torch.tensor(old_log_probs)
        rewards_t = torch.tensor(rewards)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        values = self.value(states_t).detach()
        next_value = self.value(next_states_t[-1:]).detach()
        adv, returns = self._compute_gae(rewards_t, values, dones_t, next_value.item())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.detach()
        returns = returns.detach()
        for _ in range(self.k_epochs):
            probs = self.policy(states_t)
            new_log_probs = torch.log(
                probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            )
            ratio = (new_log_probs - old_log_probs_t).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.nn.functional.mse_loss(self.value(states_t), returns)
            self.opt_policy.zero_grad()
            policy_loss.backward()
            self.opt_policy.step()
            self.opt_value.zero_grad()
            value_loss.backward()
            self.opt_value.step()

    def save(self, path: str) -> None:
        torch.save(
            {"policy": self.policy.state_dict(), "value": self.value.state_dict()}, path
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value.load_state_dict(checkpoint["value"])


def train(model_path: str) -> None:
    env = SnakeEnv()
    action_dim = env.action_space.n
    agent = PPOAgent(action_dim)
    episodes = 10000
    max_steps = 1000
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        for _ in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, log_prob, done, next_state)
            state = next_state
            episode_reward += reward
            if done:
                break
        agent.update()
        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")
        if (episode + 1) % 1000 == 0:
            agent.save(f"main/myRLTest/model/ppo_snake_{episode + 1}.pth")
    agent.save(model_path)
    env.close()


if __name__ == "__main__":
    MODEL_PATH = "main/myRLTest/model/ppo_snake_final.pth"
    train(MODEL_PATH)
