# Lunar Lander AI 🚀

An autonomous AI agent trained to perform a soft landing using Reinforcement Learning (RL). This project utilizes the Proximal Policy Optimization (PPO) algorithm to solve the `LunarLander-v3` environment, leveraging Stable-Baselines3 and Gymnasium's Box2D physics engine.

## Overview

The agent learns entirely through trial and error. At each physics step (50 frames per second), the neural network evaluates an 8-dimensional continuous observation space:
* Horizontal and vertical coordinates
* Linear and angular velocities
* Lander orientation angle (tilt)
* Left and right leg ground contact flags

Based on this telemetry, the policy network outputs one of four discrete actions: idle, fire main engine, fire left thruster, or fire right thruster.

## Project Structure

* `1_random_baseline.py`: Runs an untrained agent taking random actions to verify the physics engine and rendering pipeline.
* `2_train_baseline.py`: Initializes and trains the PPO model using an MLP policy on the default reward structure.
* `3_evaluate_and_render.py`: Loads the trained model checkpoint, evaluates it over multiple episodes, and records `.mp4` gameplay footage of the agent.
* `4_reward_shaping.py`: Implements custom Gymnasium wrappers to alter the reward function. It demonstrates both "reward hacking" (where the agent learns to hover indefinitely to farm points) and potential-based corrections.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/rocket_landing_ai.git](https://github.com/YOUR_USERNAME/rocket_landing_ai.git)
   cd rocket_landing_ai
