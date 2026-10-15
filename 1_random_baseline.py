import gymnasium as gym

# Initialize environment with screen rendering enabled
env = gym.make("LunarLander-v3", render_mode="human")

episodes = 3
for episode in range(1, episodes + 1):
    state, info = env.reset()
    done = False
    truncated = False
    total_reward = 0.0

    print(f"\n--- Episode {episode} Starting ---")
    while not (done or truncated):
        # Sample completely random action: 0, 1, 2, or 3
        action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Display raw 8-dimensional observation vector
        print(f"X: {obs[0]:.2f} | Y: {obs[1]:.2f} | Vx: {obs[2]:.2f} | Vy: {obs[3]:.2f} | "
              f"Angle: {obs[4]:.2f} | AngVel: {obs[5]:.2f} | Legs: ({int(obs[6])}, {int(obs[7])})", end="\r")

    print(f"\nEpisode {episode} Finished | Cumulative Score: {total_reward:.2f}")

env.close()