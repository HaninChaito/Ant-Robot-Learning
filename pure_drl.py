import gymnasium as gym
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# 1. Create directory for logs and models
log_dir = "./logs/ant_v5/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("./models/", exist_ok=True)

# 2. Define the environment creation function
def make_env():
    # We use Ant-v5 as per the latest MuJoCo standards
    env = gym.make("Ant-v5", render_mode=None)
    # CRITICAL: Monitor allows Stable Baselines to record rewards/lengths
    env = Monitor(env) 
    return env

# 3. Wrap the environment
# We use DummyVecEnv because SAC requires a vectorized environment 
# and VecNormalize to scale the 105-dimensional observation space
env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# 4. Initialize the SAC Agent
# Using default hyperparameters for the "Pure DRL" stage
model = SAC(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir,
    device="auto" # Will use GPU if you have one
)

# 5. Train the agent
# 500k steps is recommended for the Ant to find a stable gait
print("--- Starting Training Stage 1: Pure DRL ---")
model.learn(
    total_timesteps=500000, 
    tb_log_name="SAC_Ant_Baseline",
)

# 6. Save the Model and the Normalization Stats
# IMPORTANT: You need BOTH to run the agent later
model.save("models/ant_v5_sac_pure")
env.save("models/vec_normalize_v5_pure.pkl")

print("--- Training Finished! ---")
print(f"Model saved to models/ant_v5_sac_pure")
print(f"Stats saved to models/vec_normalize_v5_pure.pkl")