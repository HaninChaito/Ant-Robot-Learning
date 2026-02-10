import gymnasium as gym
import os
import sys
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# 1. BEST PARAMETERS FROM TRIAL 14
# These are the "Stable but Shuffling" parameters that won Phase 2
BEST_PARAMS = {
    'forward_reward_weight': 0.6603575972652962, 
    'ctrl_cost_weight': 0.9864905644087003, 
    'contact_cost_weight': 0.0006816298147368613,
    'learning_rate': 0.0004909216375597309, 
    'batch_size': 128, 
    'gamma': 0.9660289085016437
}

def train_expert(seed_value):
    print(f"\n=========================================")
    print(f"STARTING TRAINING: EXPERT SEED {seed_value}")
    print(f"=========================================\n")

    # 2. Create Directory for models
    os.makedirs("models/ensemble", exist_ok=True)
    os.makedirs("logs/ensemble", exist_ok=True)

    # 3. Define Environment
    def make_env():
        env = gym.make(
            "Ant-v5", 
            forward_reward_weight=BEST_PARAMS['forward_reward_weight'],
            ctrl_cost_weight=BEST_PARAMS['ctrl_cost_weight'],
            contact_cost_weight=BEST_PARAMS['contact_cost_weight'],
            render_mode=None
        )
        return Monitor(env)

    # 4. Wrap and Normalize
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # 5. Initialize SAC with Trial 14 Hyperparameters
    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=BEST_PARAMS['learning_rate'],
        batch_size=BEST_PARAMS['batch_size'],
        gamma=BEST_PARAMS['gamma'],
        seed=seed_value,
        verbose=0,
        tensorboard_log="./logs/ant_tuning/"
    )

    # 6. Train for 600,000 steps 
    # (Since Trial 14 was already good at 200k, 600k will make it a master)
    model.learn(
        total_timesteps=600000, 
        tb_log_name=f"Expert_Seed_{seed_value}",
    )

    # 7. Save Model and Stats
    model_name = f"ant_expert_s{seed_value}"
    model.save(f"models/ensemble/{model_name}")
    env.save(f"models/ensemble/vecnorm_s{seed_value}.pkl")
    
    print(f"\nSuccessfully saved model: {model_name}")

# ==========================================
# THE MAIN BLOCK: HOW TO RUN IN 2 TERMINALS
# ==========================================
if __name__ == "__main__":
    # Check if a seed was provided in the command line
    # Usage: python train_experts.py 100
    if len(sys.argv) > 1:
        chosen_seed = int(sys.argv[1])
        train_expert(seed_value=chosen_seed)
    else:
        print("ERROR: You must provide a seed number.")
        print("Example: python train_experts.py 100")