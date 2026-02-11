

# 737 Winning Parameters
# BEST_PARAMS = {
#     'fwd_w': 0.7495489928847656, 
#     'ctrl_w': 0.9776447922328251, 
#     'cont_w': 0.0011162399103064964,
#     'lr': 0.0004019262125726062, 
#     'batch': 128, 
#     'gamma': 0.9752735767988601
# }

import gymnasium as gym
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Exact Weights used in Trial 737 (Environment)
BEST_PARAMS_ENV = {
    'fwd_w': 0.7495489928847656, 
    'ctrl_w': 0.9776447922328251, 
    'cont_w': 0.0011162399103064964
}

def make_env():
    env = gym.make("Ant-v5", 
                   forward_reward_weight=BEST_PARAMS_ENV['fwd_w'],
                   ctrl_cost_weight=BEST_PARAMS_ENV['ctrl_w'],
                   contact_cost_weight=BEST_PARAMS_ENV['cont_w'], 
                   render_mode=None)
    return Monitor(env)

if __name__ == "__main__":
    # 1. Paths to your Trial 737 files
    model_load_path = "refined_models/ant_trial_737.zip"
    stats_load_path = "refined_models/vecnorm_trial_737.pkl"

    # 2. Recreate and Load Environment Stats
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(stats_load_path, env)
    
    # IMPORTANT: We want to continue updating stats during training
    env.training = True
    env.norm_reward = True

    # 3. Load the SAC Model
    # This automatically loads the LR, Batch Size, Gamma, etc. from Trial 737
    model = SAC.load(model_load_path, env=env)

    # 4. Sync TensorBoard to your existing logs
    model.tensorboard_log = "./logs/refined_tuning/"

    # 5. Continue Training
    # We want 1M total. We already did 300k. So we need 700k more.
    print("--- Resuming Training of Trial 737 (Champion) ---")
    model.learn(
        total_timesteps=600000, 
        tb_log_name="trial_737_Continued",
        reset_num_timesteps=False, # <--- THE MAGIC LINE for the curve
    )

    # 6. Save the Final Champion
    os.makedirs("models/final", exist_ok=True)
    model.save("models/final/ant_champion_737_final")
    env.save("models/final/vecnorm_champion_737_final.pkl")
    print("Final Champion Saved in models/final/")