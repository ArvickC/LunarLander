import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, time


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

def collect_rollout(env, actor, critic, state, rollout_length, device):
    states, actions, old_log_probs, values, rewards, dones = [], [], [], [], [], []
    episode_rewards_completed = []
    current_episode_reward = 0.0

    for _ in range(rollout_length):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = actor(state_t)
            value = critic(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(state)
        actions.append(action.item())
        old_log_probs.append(dist.log_prob(action).item())
        values.append(value.item())
        rewards.append(reward)
        dones.append(done)

        current_episode_reward += reward
        state = next_state

        if done:
            episode_rewards_completed.append(current_episode_reward)
            current_episode_reward = 0.0
            state, _ = env.reset()

    with torch.no_grad():
        next_state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        last_value = critic(next_state_t).item()

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions),
        "old_log_probs": np.array(old_log_probs, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
        "last_value": last_value,
    }, state, episode_rewards_completed

def compute_gae(rollout, gamma=0.99, gae_lambda=0.95):
    rewards = rollout["rewards"]
    values = rollout["values"]
    dones = rollout["dones"]
    last_value = rollout["last_value"]

    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    next_value = last_value
    running_advantage = 0.0

    for t in reversed(range(n)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        running_advantage = delta + gamma * gae_lambda * mask * running_advantage
        advantages[t] = running_advantage
        next_value = values[t]

    returns = advantages + values
    return advantages, returns

def ppo_update(actor, critic, actor_optimizer, critic_optimizer, rollout,
               advantages, returns, clip_eps = 0.2, epochs=10, minibatch_size = 64,
               entropy_coef=0.01, device='cpu'):
    states = torch.as_tensor(rollout["states"], device=device)
    actions = torch.as_tensor(rollout["actions"], device=device)
    old_log_probs = torch.as_tensor(rollout["old_log_probs"], device=device)
    advantages_t = torch.as_tensor(advantages, device=device)
    returns_t = torch.as_tensor(returns, device=device)

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
            actor_loss = -torch.min(unclipped, clipped).mean() - entropy * entropy_coef

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

def train(total_timesteps=500_000, rollout_length = 2048, gamma=0.99, gae_lambda=0.95,
          clip_eps = 0.2, epochs=10, minibatch_size = 64, actor_lr=3e-4, critic_lr=1e-3,
          entropy_coef=0.01, run_name = None):
    device = torch.device('cpu')
    print(f"Using device: {device}")

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f'runs/{run_name}')

    env = gym.make('LunarLander-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = ActorNetwork(state_dim, action_dim).to(device)
    critic = CriticNetwork(state_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    state, _ = env.reset()
    all_episode_rewards = []
    timesteps = 0
    iteration = 0

    while timesteps < total_timesteps:
        iteration += 1

        rollout, state, completed_rewards = collect_rollout(
            env, actor, critic, state, rollout_length, device
        )
        all_episode_rewards.extend(completed_rewards)
        timesteps += rollout_length

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

    env.close()
    writer.close()
    torch.save(actor.state_dict(), "models/ppo_actor.pt")
    torch.save(critic.state_dict(), "models/ppo_critic.pt")
    print("Saved actor and critic weights.")
    print("View training curves with: tensorboard --logdir=runs")
    return all_episode_rewards

if __name__ == "__main__":
    train(run_name="ppo_" + datetime.now().strftime("%Y%m%d-%H%M%S"))