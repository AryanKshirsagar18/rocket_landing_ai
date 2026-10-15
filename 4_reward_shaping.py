import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

class BrokenHoverWrapper(gym.RewardWrapper):
    """
    Induces reward hacking: gives a large static bonus for being
    upright and slow, causing the lander to hover indefinitely.
    """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # Unpack state: obs[4] is tilt (rad), obs[2] is Vx, obs[3] is Vy
        state = self.env.unwrapped.lander.position  # or from step observation
        # Using the standard 8-element observation from the environment
        obs = self.env.unwrapped.state if hasattr(self.env.unwrapped, "state") else None
        
        # Access observation directly via step return in custom wrapper:
        return reward

class CustomRewardWrapper(gym.Wrapper):
    """
    Explicit Gym wrapper allowing access to observation and step returns.
    """
    def __init__(self, env, mode="corrected"):
        super().__init__(env)
        self.mode = mode

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        tilt = abs(obs[4])
        speed = np.sqrt(obs[2]**2 + obs[3]**2)
        
        if self.mode == "hacked":
            # REWARD HACK: +1.0 for every step spent upright and stable
            if tilt < 0.1 and speed < 0.2:
                reward += 1.0
        elif self.mode == "corrected":
            # CORRECTED: Frame penalty to discourage idling + shaping on upright alignment
            frame_fee = -0.1
            upright_incentive = -0.5 * tilt
            reward += (frame_fee + upright_incentive)

        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    # Test training the corrected reward configuration
    print("Training lander with corrected reward wrapper...")
    base_env = gym.make("LunarLander-v3")
    shaped_env = CustomRewardWrapper(base_env, mode="corrected")

    shaped_model = PPO("MlpPolicy", shaped_env, verbose=1, learning_rate=3e-4)
    shaped_model.learn(total_timesteps=300_000)

    shaped_model.save("models/ppo_lunarlander_corrected.zip")
    print("Corrected model training complete and saved.")
    shaped_env.close()