import gymnasium as gym
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 1. Recreate the environment
def make_env():
    # render_mode="human" opens the visual window
    return gym.make("Ant-v5", render_mode="human")

env = DummyVecEnv([make_env])

# 2. Load the SAVED normalization stats
# Check the path! Make sure this file exists in your models/ folder
stats_path = "saved_models/vecnorm_trial_14.pkl"
env = VecNormalize.load(stats_path, env)

# IMPORTANT: Testing settings
env.training = False 
env.norm_reward = False 

# 3. Load the trained brain
model = SAC.load("saved_models/ant_trial_14")

print("Visualizing the Ant... Press Ctrl+C in the terminal to stop.")

try:
    obs = env.reset()
    while True: # Run infinitely so you can watch it
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        # Add a tiny sleep so the simulation doesn't run too fast for the human eye
        time.sleep(0.01) 
        
        if dones:
            print("Ant flipped/reset!")
            obs = env.reset()
            
except KeyboardInterrupt:
    print("\nClosing viewer...")
finally:
    # This helps prevent the error you saw
    env.close()