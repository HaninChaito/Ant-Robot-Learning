import gymnasium as gym
import os, sys
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

BEST_PARAMS = {
    'fwd_w': 0.7495, 'ctrl_w': 0.9776, 'cont_w': 0.0011,
    'lr': 0.0004, 'batch': 128, 'gamma': 0.9752
}

def train_expert(seed_val):
    def make_env():
        return Monitor(gym.make("Ant-v5", forward_reward_weight=BEST_PARAMS['fwd_w'],
                               ctrl_cost_weight=BEST_PARAMS['ctrl_w'],
                               contact_cost_weight=BEST_PARAMS['cont_w']))
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    

    model = SAC("MlpPolicy", env, learning_rate=BEST_PARAMS['lr'], 
                batch_size=BEST_PARAMS['batch'], gamma=BEST_PARAMS['gamma'],
                seed=seed_val, verbose=0)
    
    model.tensorboard_log = "./logs/refined_tuning/"

    print(f"Training Expert Seed {seed_val}...")
    model.learn(total_timesteps=800000,tb_log_name=f"expert_s{seed_val}") # 800k is enough for ensemble stability
    
    os.makedirs("models/ensemble", exist_ok=True)
    model.save(f"models/ensemble/ant_expert_s{seed_val}")
    env.save(f"models/ensemble/vecnorm_s{seed_val}.pkl")
    print(f"Expert Seed {seed_val} Finished!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_expert(int(sys.argv[1]))
    else:
        print("Usage: python phase3_train_experts.py <seed>")