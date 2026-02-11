import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


USE_ENSEMBLE = True 
ADD_NOISE = False # Set to True to prove robustness during presentation
NOISE_LEVEL = 0.15 # 15% noise

MODEL_PATHS = [
    "models/final/ant_champion_737_final", # Champion 1
    #"models/Pure_DRL/ant_v5_sac_pure",                 # Champion 2 (Phase 1)
    "models/ensemble/ant_expert_s100",        # Expert 2
    "models/ensemble/ant_expert_s200"         # Expert 3
]

STATS_PATH = "models/final/vecnorm_champion_737_final.pkl"
#STATS_PATH = "models/Pure_DRL/vec_normalize_v5_pure.pkl" # If using Phase 1 model


BEST_PARAMS = {
    'forward_reward_weight': 0.7495489928847656, 
    'ctrl_cost_weight': 0.9776447922328251, 
    'contact_cost_weight': 0.0011162399103064964
}


def make_env():
    env = gym.make("Ant-v5", 
                   forward_reward_weight=BEST_PARAMS['forward_reward_weight'],
                   ctrl_cost_weight=BEST_PARAMS['ctrl_cost_weight'],
                   contact_cost_weight=BEST_PARAMS['contact_cost_weight'],
                   render_mode="human")
    return env

env = DummyVecEnv([make_env])

env = VecNormalize.load(STATS_PATH, env)
env.training = False
env.norm_reward = False


def setup_camera(env):
    try:
        base_env = env.venv.envs[0].unwrapped

        viewer = base_env.mujoco_renderer._get_viewer(render_mode="human")

        import mujoco
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        
        viewer.cam.trackbodyid = 1 
        
        viewer.cam.distance = 30.0
        viewer.cam.elevation = -20
        
        print("Camera successfully locked on Ant.")
    except Exception as e:
        print(f"Automatic camera setup failed: {e}")
        print("Manual Fix: Double-click the Ant and press 'T' in the window.")


if USE_ENSEMBLE:
    models = [SAC.load(p) for p in MODEL_PATHS]
    print(f"Loaded {len(models)} models for Ensemble.")
else:
    model = SAC.load(MODEL_PATHS[0])
    print(f"Loaded single model: {MODEL_PATHS[0]}")


obs = env.reset()
setup_camera(env) 

print("Running visualization... Close window to stop.")

try:
    while True:
        if USE_ENSEMBLE:
            # Get actions from all 3 brains
            actions = [m.predict(obs, deterministic=True)[0] for m in models]
            # Average the actions (Ensemble Logic)
            final_action = np.mean(actions, axis=0)
        else:
            final_action, _ = model.predict(obs, deterministic=True)

        # Step environment
        if ADD_NOISE:
            noise = np.random.normal(0, NOISE_LEVEL, size=final_action.shape)
            final_action += noise
        obs, reward, done, info = env.step(final_action)
        
        # Slow down simulation to match human eyes
        time.sleep(0.01)

        if done:
            obs = env.reset()
            setup_camera(env) # Re-lock camera after reset

except KeyboardInterrupt:
    print("Stopping...")
finally:
    env.close()