import os
import optuna
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import torch

print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

def objective(trial):
    # --- 1. HYPERPARAMETER SEARCH SPACE ---
    # Environment Weights
    fwd_w = trial.suggest_float("forward_reward_weight", 0.5, 2.0)
    ctrl_w = trial.suggest_float("ctrl_cost_weight", 0.01, 1.0)
    contact_w = trial.suggest_float("contact_cost_weight", 1e-4, 1e-2, log=True)
    
    # RL Hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)

    # --- 2. ENVIRONMENT SETUP ---
    def make_env():
        env = gym.make(
            "Ant-v5",
            forward_reward_weight=fwd_w,
            ctrl_cost_weight=ctrl_w,
            contact_cost_weight=contact_w,
            render_mode=None
        )
        return Monitor(env)

    # Setup Training Env
    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # --- 3. MODEL SETUP ---
    # We use a unique log folder for each trial to keep TensorBoard clean
    model = SAC(
        "MlpPolicy", 
        train_env, 
        learning_rate=lr, 
        batch_size=batch_size, 
        gamma=gamma,
        seed=0, 
        verbose=0,
        tensorboard_log="./logs/ant_tuning/"
    )

    try:
        # --- 4. TRAINING ---
        model.learn(total_timesteps=200_000, tb_log_name=f"trial_{trial.number}")

        # --- 5. SAVING (Crucial: Save BEFORE Evaluation) ---
        os.makedirs("saved_models", exist_ok=True)
        model_path = f"saved_models/ant_trial_{trial.number}"
        vec_path = f"saved_models/vecnorm_trial_{trial.number}.pkl"
        
        model.save(model_path)
        train_env.save(vec_path)

        # --- 6. RIGOROUS EVALUATION ---
        eval_env = DummyVecEnv([make_env])
        # Load the EXACT stats from the training we just finished
        eval_env = VecNormalize.load(vec_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False 
        
        # Load the model we just saved into the eval environment
        eval_model = SAC.load(model_path, env=eval_env)
        
        mean_reward, _ = evaluate_policy(
            eval_model,
            eval_env,
            n_eval_episodes=5,
            deterministic=True
        )

        eval_env.close()
        return mean_reward

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return -5000 # Give a low score so Optuna avoids these params
    
    finally:
       # Ensure we always free up memory
       train_env.close()

if __name__ == "__main__":
    # Create the database and study
    storage = "sqlite:///ant_tuning.db"
    study = optuna.create_study(
        study_name="ant_joint_optimization",
        storage=storage,
        direction="maximize",
        load_if_exists=True
    )

    # Resilient loop: ensures we reach 30 successful trials
    trials_needed = 15
    while len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])) < trials_needed:
        try:
            current_completed = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
            print(f"Status: {current_completed}/{trials_needed} trials finished. Starting next...")
            study.optimize(objective, n_trials=1)
        except Exception as e:
            print(f"Critical Error: {e}. Restarting loop...")
            continue

    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Reward: {study.best_value}")
    print("Best Params:", study.best_params)