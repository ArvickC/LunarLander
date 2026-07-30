import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def collect_rollout(envs, actor, critic, states, rollout_length, num_envs, device):
    all_states, all_actions, all_old_log_probs, all_values, all_rewards, all_dones = \
        [], [], [], [], [], []
    episode_rewards_completed = []
    current_episode_rewards = np.zeros(num_envs, dtype=np.float32)

    for _ in range(rollout_length):
        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = actor(states_t)  # (N, action_dim)
            values = critic(states_t)  # (N,)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()  # (N,)
        log_probs = dist.log_prob(actions)  # (N,)

        # envs.step wants a plain numpy array of N integers, not a tensor.
        next_states, rewards, terminateds, truncateds, _ = envs.step(actions.cpu().numpy())
        dones = np.logical_or(terminateds, truncateds)

        all_states.append(states)
        all_actions.append(actions.cpu().numpy())
        all_old_log_probs.append(log_probs.cpu().numpy())
        all_values.append(values.cpu().numpy())
        all_rewards.append(rewards)
        all_dones.append(dones)

        current_episode_rewards += rewards

        for i in range(num_envs):
            if dones[i]:
                episode_rewards_completed.append(current_episode_rewards[i])
                current_episode_rewards[i] = 0.0

        states = next_states

    with torch.no_grad():
        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        last_values = critic(states_t).cpu().numpy()  # shape (N,)

    rollout = {
        "states": np.array(all_states, dtype=np.float32),
        "actions": np.array(all_actions),
        "old_log_probs": np.array(all_old_log_probs, dtype=np.float32),
        "values": np.array(all_values, dtype=np.float32),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "dones": np.array(all_dones, dtype=np.float32),
        "last_values": last_values,
    }
    return rollout, states, episode_rewards_completed


def compute_gae(rollout, gamma=0.99, gae_lambda=0.95):
    rewards = rollout["rewards"]  # (T, N)
    values = rollout["values"]  # (T, N)
    dones = rollout["dones"]  # (T, N)
    last_values = rollout["last_values"]  # (N,)

    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    next_value = last_values  # shape (N,)
    running_advantage = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]  # (N,)
        delta = rewards[t] + gamma * next_value * mask - values[t]
        running_advantage = delta + gamma * gae_lambda * mask * running_advantage
        advantages[t] = running_advantage
        next_value = values[t]

    returns = advantages + values  # (T, N)
    return advantages, returns

def ppo_update(actor, critic, actor_optimizer, critic_optimizer, rollout,
               advantages, returns, clip_eps=0.2, epochs=10, minibatch_size=64,
               entropy_coef=0.01, device="cpu"):
    T, N = rollout["rewards"].shape
    states = torch.as_tensor(rollout["states"].reshape(T * N, -1), device=device)
    actions = torch.as_tensor(rollout["actions"].reshape(T * N), device=device)
    old_log_probs = torch.as_tensor(rollout["old_log_probs"].reshape(T * N), device=device)
    advantages_t = torch.as_tensor(advantages.reshape(T * N), device=device)
    returns_t = torch.as_tensor(returns.reshape(T * N), device=device)

    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    n = len(states)
    total_actor_loss, total_critic_loss, total_entropy = 0.0, 0.0, 0.0
    num_updates = 0

    for _ in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, minibatch_size):
            batch_idx = indices[start:start + minibatch_size]

            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages_t[batch_idx]
            batch_returns = returns_t[batch_idx]

            probs = actor(batch_states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            actor_loss = -torch.min(unclipped, clipped).mean() - entropy_coef * entropy

            values_pred = critic(batch_states)
            critic_loss = (values_pred - batch_returns).pow(2).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            num_updates += 1

    return total_actor_loss / num_updates, total_critic_loss / num_updates, total_entropy / num_updates


def train(total_timesteps=500_000, num_envs=8, rollout_length=256, gamma=0.99, gae_lambda=0.95,
          clip_eps=0.2, epochs=10, minibatch_size=64, actor_lr=3e-4, critic_lr=1e-3,
          entropy_coef=0.01, device_override = "cpu", run_name=None, vector_mode = "sync"):
    device = torch.device(device_override)
    print(f"Using device: {device}")

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/ppo_parallel_{run_name}")

    envs = gym.make_vec(
        "LunarLander-v3", num_envs=num_envs, vectorization_mode=vector_mode,
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n

    actor = ActorNetwork(state_dim, action_dim).to(device)
    critic = CriticNetwork(state_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    states, _ = envs.reset()
    all_episode_rewards = []
    timesteps = 0
    iteration = 0

    while timesteps < total_timesteps:
        iteration += 1

        rollout, states, completed_rewards = collect_rollout(
            envs, actor, critic, states, rollout_length, num_envs, device
        )
        all_episode_rewards.extend(completed_rewards)

        timesteps += rollout_length * num_envs

        advantages, returns = compute_gae(rollout, gamma, gae_lambda)

        actor_loss, critic_loss, entropy = ppo_update(
            actor, critic, actor_optimizer, critic_optimizer, rollout,
            advantages, returns, clip_eps, epochs, minibatch_size, entropy_coef, device
        )

        writer.add_scalar("train/actor_loss", actor_loss, timesteps)
        writer.add_scalar("train/critic_loss", critic_loss, timesteps)
        writer.add_scalar("train/entropy", entropy, timesteps)
        if completed_rewards:
            writer.add_scalar("reward/episode", np.mean(completed_rewards), timesteps)
        if len(all_episode_rewards) >= 10:
            avg_recent = np.mean(all_episode_rewards[-10:])
            writer.add_scalar("reward/avg_10", avg_recent, timesteps)

        if iteration % 5 == 0 and all_episode_rewards:
            avg_recent = np.mean(all_episode_rewards[-10:])
            print(f"Timesteps {timesteps:8d} | episodes {len(all_episode_rewards):5d} | "
                  f"avg reward (last 10): {avg_recent:8.1f}")

        if len(all_episode_rewards) >= 200 and np.mean(all_episode_rewards[-200:]) >= 250:
            print(f"\nSolved after {timesteps} timesteps!")
            break

    envs.close()
    writer.close()
    torch.save(actor.state_dict(), "ppo_vec_actor.pt")
    torch.save(critic.state_dict(), "ppo_vec_critic.pt")
    print("Saved actor and critic weights.")
    print("View training curves with: tensorboard --logdir=runs")
    return all_episode_rewards

if __name__ == "__main__":
    train(minibatch_size=2048, num_envs=64, device_override='cpu', run_name="ppo_64envs_sync", vector_mode='sync')