import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

model_path = os.path.join("models", "ppo_lunarlander_corrected.zip")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Run 2_train_baseline.py first.")

# Wrap environment to capture video to './recordings'
eval_env = gym.make("LunarLander-v3", render_mode="rgb_array")
eval_env = RecordVideo(
    eval_env,
    video_folder="recordings",
    episode_trigger=lambda episode_id: True,
    name_prefix="corrected-lander"
)

model = PPO.load(model_path, env=eval_env)

num_episodes = 10
scores = []

for ep in range(1, num_episodes + 1):
    obs, info = eval_env.reset()
    done = False
    truncated = False
    episode_reward = 0.0

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        episode_reward += reward

    scores.append(episode_reward)
    print(f"Evaluation Run {ep}: Score = {episode_reward:.2f}")

eval_env.close()

avg_score = sum(scores) / len(scores)
print("\n" + "="*40)
print(f"Mean Score over {num_episodes} attempts: {avg_score:.2f}")
if avg_score >= 200:
    print("Benchmark Status: SOLVED (Score >= 200)")
else:
    print("Benchmark Status: Sub-optimal (< 200). Consider additional training steps.")
print("="*40)