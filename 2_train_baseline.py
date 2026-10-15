import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

os.makedirs("models", exist_ok=True)

# Training environment (vectorized single-instance, no rendering for maximum FPS)
train_env = gym.make("LunarLander-v3")

# PPO policy definition with an MLP (64x64 hidden layers)
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

print("Starting PPO training on LunarLander-v3...")
# 300,000 steps provides a strong policy; 500,000 to 1,000,000 achieves >200 score
model.learn(total_timesteps=350_000)

model_path = os.path.join("models", "ppo_lunarlander_baseline.zip")
model.save(model_path)
print(f"Model successfully saved to {model_path}")

train_env.close()