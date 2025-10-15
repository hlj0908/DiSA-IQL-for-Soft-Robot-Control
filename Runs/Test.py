import os
import numpy as np
import torch
import random
import d3rlpy
import configparser
import pandas as pd
import csv
from SnakeRobot.SnakeEnv_lessA_sector import SnakeEnv

# Seed
SEED = 3407

D3_MODEL_PATH = 'd3rlpy_logs/CQL_20250905102958/model_58800.d3'
N_EPISODES = 500
DEVICE = "cpu"
CONFIG_INI = "../configs/config2.ini"


def set_seed(seed=3407):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def robust_load_algo(d3_path=None, device="cpu"):
    if d3_path is not None:
        print(f"Loading learnable from .d3: {d3_path}")
        algo = d3rlpy.load_learnable(d3_path, device=device)
        return algo
    raise ValueError("need to provide d3_path")


def get_action_from_algo(algo, obs: np.ndarray):
    return algo.predict(np.asarray(obs, dtype=np.float32)[None])[0]


def step_env_handle(env, action):
    obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    return np.asarray(obs, dtype=np.float32), float(reward), done, dict(info)


def evaluate(algo, env, n_episodes=50,
             metrics_path="eval_metrics.csv",
             traj_path="trajectories.csv"):
    success = 0
    rewards, steps_list = [], []
    distances, deflections, velocities = [], [], []
    all_rows = []   # Store the trajectory data of all steps

    for ep in range(n_episodes):
        reset_out = env.reset(episode=ep)
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        obs = np.asarray(obs, dtype=np.float32)

        done, ep_r, step = False, 0.0, 0
        coords = []

        while not done and step < getattr(env, "episode_length", 150):
            action = get_action_from_algo(algo, obs)
            obs, r, done, info = step_env_handle(env, action)
            ep_r += r
            step += 1
            coords.append((env.X, env.Y))

        # episode metrics
        avg_dis = np.mean(env.dis) if hasattr(env, "dis") and len(env.dis) > 0 else 0.0
        avg_def = np.mean(env.ang_dis) if hasattr(env, "ang_dis") and len(env.ang_dis) > 0 else 0.0
        avg_vel = np.sqrt(env.X_d**2 + env.Y_d**2) if hasattr(env, "X_d") else 0.0
        achieved = bool(info.get("achieve_goal", 0))
        succ_rate_ep = 1.0 if achieved else 0.0

        rewards.append(ep_r)
        steps_list.append(step)
        distances.append(avg_dis)
        deflections.append(avg_def)
        velocities.append(avg_vel)
        if achieved:
            success += 1

        # Save the data of each step + episode metrics
        for step_idx, (x, y) in enumerate(coords):
            all_rows.append({
                "episode": ep + 1,
                "step_idx": step_idx,
                "x": x,
                "y": y,
                "Reward": ep_r,
                "Steps": step,
                "AvgDistance": avg_dis,
                "AvgDeflection": avg_def,
                "AvgVelocity": avg_vel,
                "SuccessRate": succ_rate_ep,
                "Goal_x": env.goal_x,
                "Goal_y": env.goal_y,
                "AchieveGoal": achieved
            })

        print(f"Episode {ep+1}/{n_episodes} -> reward {ep_r:.3f}, steps={step}, achieve_goal={achieved}")

    # Save all the episode data to a CSV file
    df = pd.DataFrame(all_rows)
    df.to_csv(traj_path, index=False)
    print(f"All trajectories + per-episode metrics saved to {traj_path}")

    # Calculate and save the overall average indicator
    succ_rate = success / n_episodes
    avg_r = float(np.mean(rewards))
    avg_steps = float(np.mean(steps_list))
    avg_dis = float(np.mean(distances)) if distances else 0.0
    avg_def = float(np.mean(deflections)) if deflections else 0.0
    avg_vel = float(np.mean(velocities)) if velocities else 0.0

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SuccessRate", "AvgReward", "AvgSteps", "AvgDistance", "AvgDeflection", "AvgVelocity"])
        writer.writerow([succ_rate, avg_r, avg_steps, avg_dis, avg_def, avg_vel])
    print(f"Metrics saved to {metrics_path}")

    return succ_rate, avg_r, avg_steps, avg_dis, avg_def, avg_vel


if __name__ == "__main__":
    set_seed(SEED)

    if not os.path.exists(CONFIG_INI):
        raise FileNotFoundError(f"The config file cannot be found: {CONFIG_INI}")

    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_INI)
    cfg['ENV_CONFIG']['draw'] = 'False'
    env = SnakeEnv(cfg["ENV_CONFIG"])

    algo = robust_load_algo(d3_path=D3_MODEL_PATH, device=DEVICE)

    evaluate(algo, env, n_episodes=N_EPISODES,
             metrics_path="./result/cql_eval_metrics_h.csv",
             traj_path="./result/cql_trajectories_h.csv")


