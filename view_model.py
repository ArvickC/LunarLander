from REINFORCE_model import PolicyNetwork
import time
import gymnasium as gym
import torch


def watch_trained(num_episodes=3, checkpoint="ppo_actor.pt"):
    env = gym.make("LunarLander-v3", render_mode="human")
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    policy.load_state_dict(torch.load("./models/" + checkpoint, map_location="cpu"))
    policy.eval()

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32)
            probs = policy(state_t)
            action = torch.argmax(probs).item()  # greedy: take the best action
            state, reward, term, trunc, info = env.step(action)
            total_reward += reward
            done = term or trunc
        print(f"Episode {ep + 1}: total reward = {total_reward:.1f}")
    env.close()

if __name__ == "__main__":
    watch_trained()