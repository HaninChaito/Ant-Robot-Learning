import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==========================================
# 1. SETTINGS & PATHS
# ==========================================
NUM_EVAL_EPISODES = 10  # Run each method 10 times for a fair average

PATHS = {
    "baseline": {
        "model": "models/Pure_DRL/ant_v5_sac_pure",
        "stats": "models/Pure_DRL/vec_normalize_v5_pure.pkl"
    },
    "tuned": {
        "model": "models/final/ant_champion_737_final",
        "stats": "models/final/vecnorm_champion_737_final.pkl"
    },
    "ensemble": {
        "models": [
            "models/final/ant_champion_737_final",
            "models/ensemble/ant_expert_s100",
            "models/ensemble/ant_expert_s200"
        ],
        "stats": "models/final/vecnorm_champion_737_final.pkl"
    }
}

# ==========================================
# 2. EVALUATION FUNCTION
# ==========================================
def evaluate_method(method_name):
    print(f"Evaluating {method_name.upper()}...")
    
    # Setup Env
    def make_env():
        return gym.make("Ant-v5", render_mode=None)
    
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(PATHS[method_name]["stats"], env)
    env.training = False
    env.norm_reward = False

    # Load Model(s)
    if method_name == "ensemble":
        models = [SAC.load(m) for m in PATHS["ensemble"]["models"]]
    else:
        model = SAC.load(PATHS[method_name]["model"])

    results = []

    for i in range(NUM_EVAL_EPISODES):
        obs = env.reset()
        done = False
        total_reward = 0
        total_steps = 0
        total_torque = 0
        y_deviations = []
        
        while not done:
            # Action Selection
            if method_name == "ensemble":
                actions = [m.predict(obs, deterministic=True)[0] for m in models]
                action = np.mean(actions, axis=0)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)
            
            # Physics Metrics
            total_reward += reward[0]
            total_steps += 1
            total_torque += np.sum(np.abs(action)) # Energy used
            
            # Get positions from info (Ant-v5 specific)
            y_pos = info[0].get('y_position', 0)
            y_deviations.append(y_pos)
            final_x = info[0].get('x_position', 0)

        # Calculations for this episode
        # COT = Energy / Distance
        cot = total_torque / (final_x + 1e-6) 
        # Y-Error = How much it drifted from the straight line (MSE)
        y_mse = np.mean(np.square(y_deviations))
        
        results.append({
            "reward": total_reward,
            "length": total_steps,
            "cot": cot,
            "y_error": y_mse,
            "velocity": final_x / (total_steps * 0.05) # 0.05 is dt per step
        })

    env.close()
    return pd.DataFrame(results).mean()

# ==========================================
# 3. RUN ALL & SHOW COMPARISON
# ==========================================
if __name__ == "__main__":
    baseline_stats = evaluate_method("baseline")
    tuned_stats = evaluate_method("tuned")
    ensemble_stats = evaluate_method("ensemble")

    # Combine into a final table
    df_final = pd.DataFrame({
        "Baseline": baseline_stats,
        "Tuned (737)": tuned_stats,
        "Ensemble": ensemble_stats
    }).T

    print("\n" + "="*50)
    print("FINAL PROJECT COMPARISON")
    print("="*50)
    print(df_final.round(4))
    
    # Save to Excel/CSV for your report
    df_final.to_csv("project_comparison_results.csv")
    print("\nResults saved to 'project_comparison_results.csv'")