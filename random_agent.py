import gymnasium as gym

NUM_EPISODES = 3

env = gym.make('LunarLander-v3', render_mode='human')
for ep in range(NUM_EPISODES):
    state, _ = env.reset(seed=42)
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        state, reward, term, trunc, info = env.step(action)
        total_reward += reward
        done = term or trunc
    print(f"Episode {ep + 1}: Total Reward = {total_reward}")
env.close()

