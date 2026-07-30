import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)

def run_episode(env, policy, device):
    states, actions, rewards, log_probs, entropies = [], [], [], [], []
    state, _ = env.reset()
    done = False

    while not done:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        probs = policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_state, reward, term, trunc, _ = env.step(action.item())
        done = term or trunc

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        state = next_state

    return states, actions, rewards, log_probs, entropies

def calculate_returns(rewards, gamma=0.99):
    out = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        out.insert(0, G)
    return out

def train(
        n_episodes: int = 2000,
        gamma: float = 0.99, lr=3e-4, log_every: int = 20,
        entropy_coef: float = 0.01
):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    env = gym.make('LunarLander-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim).to(device)
    try:
        policy.load_state_dict(torch.load("reinforce_policy.pth", map_location=device))  # Load existing model if available
        print("Loaded existing model from reinforce_policy.pth")
    except FileNotFoundError:
        print("No existing model found, starting from scratch")
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []

    baseline_mean = 0.0
    baseline_sq_mean = 1.0
    baseline_momentum = 0.01

    for episode in range(1, n_episodes+1):
        states, actions, rewards, log_probs, entropies = run_episode(env, policy, device)

        returns = calculate_returns(rewards, gamma)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)

        G0 = returns[0]
        baseline_mean = (1 - baseline_momentum) * baseline_mean + baseline_momentum * G0
        baseline_sq_mean = (1 - baseline_momentum) * baseline_sq_mean + baseline_momentum * G0 * G0
        baseline_std = max((baseline_sq_mean - baseline_mean ** 2) ** 0.5, 1e-3)\

        advantages_tensor = (returns_tensor - baseline_mean) / baseline_std

        log_probs_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies).sum()

        loss = -(log_probs_tensor * advantages_tensor).sum() - entropy_coef * entropy_tensor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        if episode % log_every == 0:
            avg_reward = np.mean(episode_rewards[-log_every:])
            print(f"Episode: {episode:5d} | avg reward (last {log_every}): {avg_reward:8.1f}")

        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 200:
            print(f"Solved at episode {episode}!")
            break

    env.close()
    torch.save(policy.state_dict(), "./reinforce_policy.pth")
    print(f"Saved model to ./reinforce_policy.pth")
    return episode_rewards

if __name__ == "__main__":
    train(n_episodes=4000)