import numpy as np
import gymnasium as gym
from sailing_env import SailingEnv
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    # Create the environment with rendering
    env = SailingEnv(render_mode="human")
    obs, info = env.reset()

    # Create PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=1000000)

    # Test the trained agent
    obs = env.reset()[0]
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            print("Episode finished!")
            break
    env.close()
