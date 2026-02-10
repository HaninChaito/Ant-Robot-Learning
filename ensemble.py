import gymnasium as gym
import numpy as np
import mujoco
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==========================================
# 1. SETUP CAMERA FUNCTION
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
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -20
        
        print("Camera successfully locked on Ant.")
    except Exception as e:
        print(f"Automatic camera setup failed: {e}")
        print("Manual Fix: Double-click the Ant and press 'T' in the window.")

# ==========================================
# 2. CONFIGURATION & PATHS
# ==========================================
BEST_PARAMS = {
    'forward_reward_weight': 0.6603575972652962, 
    'ctrl_cost_weight': 0.9864905644087003, 
    'contact_cost_weight': 0.0006816298147368613
}

MODEL_PATHS = [
    "models/phase2/ant_tuned_expert_final", # Your Champion (Seed 42)
    "models/ensemble/ant_expert_s100",        # Expert Seed 100
    "models/ensemble/ant_expert_s200"         # Expert Seed 200
]

# Use the normalization stats from your champion
STATS_PATH = "models/phase2/vecnorm_tuned_expert_final.pkl"

# ==========================================
# 3. INITIALIZE ENVIRONMENT & MODELS
# ==========================================
def make_env():
    return gym.make("Ant-v5", 
                   forward_reward_weight=BEST_PARAMS['forward_reward_weight'],
                   ctrl_cost_weight=BEST_PARAMS['ctrl_cost_weight'],
                   contact_cost_weight=BEST_PARAMS['contact_cost_weight'],
                   render_mode="human")

env = DummyVecEnv([make_env])
env = VecNormalize.load(STATS_PATH, env)
env.training = False
env.norm_reward = False

print("Loading 3 Expert Models...")
models = [SAC.load(path) for path in MODEL_PATHS]

# ==========================================
# 4. MAIN INFERENCE LOOP
# ==========================================
obs = env.reset()

# Important: Run a few steps so the window "wakes up" before setting camera
for _ in range(5):
    env.step([env.action_space.sample()])

setup_camera(env)

print("Running Ensemble Inference. Press Ctrl+C to stop.")

try:
    while True:
        # Get actions from all 3 experts
        actions = []
        for m in models:
            action, _ = m.predict(obs, deterministic=True)
            actions.append(action)
        
        # --- ENSEMBLE LOGIC: MEAN ACTION ---
        ensemble_action = np.mean(actions, axis=0)
        
        # Step the environment
        noise = np.random.normal(0, 0.15, size=ensemble_action.shape)
        obs, reward, done, info = env.step(ensemble_action + noise)
        
        # Add sleep to make it look realistic (100 FPS)
        time.sleep(0.01)

        if done:

            obs = env.reset()
            setup_camera(env)

except KeyboardInterrupt:
    print("Closing...")
finally:
    env.close()