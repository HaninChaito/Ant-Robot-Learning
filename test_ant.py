import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==========================================
# CONFIGURATION
# ==========================================
# Set this to True if you want to test your 3-model Ensemble
# Set this to False if you want to test just one model (Phase 1 or 2)
USE_ENSEMBLE = False 

# Paths to your saved experts
MODEL_PATHS = [
    "models/phase2/ant_tuned_expert_final", # Champion 1
    #"models/ant_v5_sac_pure",                 # Champion 2 (Phase 1)
    "models/ensemble/ant_expert_s100",        # Expert 2
    "models/ensemble/ant_expert_s200"         # Expert 3
]

# Path to your normalization stats (from Phase 2)
STATS_PATH = "models/phase2/vecnorm_tuned_expert_final.pkl"
#STATS_PATH = "models/vec_normalize_v5_pure.pkl" # If using Phase 1 model

# Environment reward weights (Use your BEST parameters here)
BEST_PARAMS = {
    'forward_reward_weight': 0.6603575972652962, 
    'ctrl_cost_weight': 0.9864905644087003, 
    'contact_cost_weight': 0.0006816298147368613
}

# ==========================================
# ENVIRONMENT SETUP
# ==========================================
def make_env():
    # render_mode="human" is required for the window
    env = gym.make("Ant-v5", 
                   forward_reward_weight=BEST_PARAMS['forward_reward_weight'],
                   ctrl_cost_weight=BEST_PARAMS['ctrl_cost_weight'],
                   contact_cost_weight=BEST_PARAMS['contact_cost_weight'],
                   render_mode="human")
    return env

env = DummyVecEnv([make_env])

# Load and Sync Normalization
env = VecNormalize.load(STATS_PATH, env)
env.training = False
env.norm_reward = False

# ==========================================
# CAMERA TRACKING (PROGRAMMATIC)
# ==========================================
def setup_camera(env):
    try:
        # 1. Reach the base gymnasium environment
        # Path: VecNormalize -> DummyVecEnv -> Ant-v5
        base_env = env.venv.envs[0].unwrapped

        # 2. Access the viewer
        # In Gymnasium v29+, the viewer is created via the mujoco_renderer
        viewer = base_env.mujoco_renderer._get_viewer(render_mode="human")

        # 3. Set camera properties
        import mujoco
        # type 2 is mjCAMERA_TRACKING
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        
        # trackbodyid 1 is the torso in the standard Ant model
        viewer.cam.trackbodyid = 1 
        
        # Camera distance and angles
        viewer.cam.distance = 30.0
        viewer.cam.elevation = -20
        
        print("Camera successfully locked on Ant.")
    except Exception as e:
        print(f"Automatic camera setup failed: {e}")
        print("Manual Fix: Double-click the Ant and press 'T' in the window.")

# ==========================================
# LOAD MODELS
# ==========================================
if USE_ENSEMBLE:
    models = [SAC.load(p) for p in MODEL_PATHS]
    print(f"Loaded {len(models)} models for Ensemble.")
else:
    model = SAC.load(MODEL_PATHS[0])
    print(f"Loaded single model: {MODEL_PATHS[0]}")

# ==========================================
# MAIN LOOP
# ==========================================
obs = env.reset()
setup_camera(env) # Run once after reset

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