import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import os

# 1. The Parameters from Trial 14 (Must match exactly for the environment)
best_params = {
    'forward_reward_weight': 0.6603575972652962, 
    'ctrl_cost_weight': 0.9864905644087003, 
    'contact_cost_weight': 0.0006816298147368613
}

# 2. Recreate the environment with the same weights
def make_env():
    env = gym.make("Ant-v5", 
                   forward_reward_weight=best_params['forward_reward_weight'],
                   ctrl_cost_weight=best_params['ctrl_cost_weight'],
                   contact_cost_weight=best_params['contact_cost_weight'],
                   render_mode=None)
    return Monitor(env)

env = DummyVecEnv([make_env])

# 3. Load the Normalization stats from Trial 14
# Check your folder to make sure the name is correct!
vec_path = "saved_models/vecnorm_trial_14.pkl"
env = VecNormalize.load(vec_path, env)

# IMPORTANT: Since we are CONTINUING training, we want these to be True
env.training = True
env.norm_reward = True

# 4. Load the SAC Model from Trial 14
# This automatically loads the learning_rate, batch_size, and gamma used in that trial
model_path = "saved_models/ant_trial_14"
model = SAC.load(model_path, env=env, train_freq=4, gradient_steps=4)

model.tensorboard_log = "./logs/ant_tuning/" 

# 5. Continue Training
# We want to reach 1,000,000 total. 
# Since we already did 200,000, we need 800,000 more.
print("Continuing training from Trial 14...")
model.learn(
    total_timesteps=600000, 
    tb_log_name="Phase2_Continued_Expert",
    # CRITICAL: This keeps the x-axis (steps) moving forward from 200,000
    reset_num_timesteps=False 
)
# 6. Save the Final Phase 2 Champion
os.makedirs("models/phase2", exist_ok=True)
model.save("models/phase2/ant_tuned_expert_final")
env.save("models/phase2/vecnorm_tuned_expert_final.pkl")

print("Continued Training Finished! Expert Model Saved.")