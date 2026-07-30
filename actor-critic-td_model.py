import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train(n_episodes = 2000, gamma = 0.99, actor_lr=3e-4, critic_lr = 1e-3,
          entropy_coef = 0.01, log_every = 20, run_name = None):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using {device} device")

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = ActorNetwork(state_dim, action_dim).to(device)
    critic = CriticNetwork(state_dim).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    episode_rewards = []

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        td_errors = []

        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)

            probs = actor(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=device)

            next_value = torch.tensor(0.0, device=device)
            if not terminated:
                with torch.no_grad():
                    next_value = critic(next_state_t)

            td_target = reward + gamma * next_value
            value = critic(state_t)
            td_error = td_target - value

            critic_loss = td_error.pow(2)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            advantage = td_error.detach()
            actor_loss = -log_prob * advantage - entropy_coef * entropy
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            td_errors.append(td_error.item())
            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

        writer.add_scalar("reward/episode", total_reward, episode)
        if len(episode_rewards) >= 100:
            writer.add_scalar("reward/avg_100", np.mean(episode_rewards[-100:]), episode)
        writer.add_scalar("train/avg_td_error", np.mean(np.abs(td_errors)), episode)
        writer.add_scalar("train/episode_length", len(td_errors), episode)

        if episode % log_every == 0:
            avg_reward = np.mean(episode_rewards[-log_every:])
            print(f"Episode {episode:5d} | avg reward (last {log_every}): {avg_reward:8.1f}")

        if len(episode_rewards) >= 250 and np.mean(episode_rewards[-250:]) >= 250:
            print(f"\nSolved at episode {episode}!")
            break

    env.close()
    writer.close()
    torch.save(actor.state_dict(), "actor_critic_td_actor_2.pt")
    torch.save(critic.state_dict(), "actor_critic_td_critic_2.pt")
    print("Saved actor and critic weights.")
    print("View training curves with: tensorboard --logdir=runs")
    return episode_rewards

if __name__ == "__main__":
    train(n_episodes=5000, run_name="actor_critic_td_" + datetime.now().strftime("%Y%m%d-%H%M%S"))