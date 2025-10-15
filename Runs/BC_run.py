import numpy as np
import configparser
from SnakeRobot.SnakeEnv_lessA_sector import SnakeEnv
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import BCConfig
from d3rlpy.metrics import EnvironmentEvaluator
import random
import torch

def set_seed(seed=3407):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 3407
set_seed(SEED)

def load_and_filter_data():
    # 加载环境
    cfg = configparser.ConfigParser()
    cfg.read('../configs/config.ini')
    env = SnakeEnv(cfg['ENV_CONFIG'], seed=SEED)

    # datasets
    data = np.load("../datasets/50000_hr.npz", allow_pickle=True)
    observations = data["observations"]
    actions = data["actions"]
    rewards = data["rewards"]
    terminals = data["dones"]

    if np.max(np.abs(actions[:, :2])) > 1.0 + 1e-6:
        actions[:, :2] = np.clip(actions[:, :2] / 8.0, -1.0, 1.0)

    return env, observations, actions, rewards, terminals


def train_bc():
    # env
    env, observations, actions, rewards, terminals = load_and_filter_data()

    # Create an MDP dataset
    dataset = MDPDataset(
        observations,
        actions,
        rewards,
        terminals
    )

    # Create BC configuration
    bc_config = BCConfig(
        batch_size=256,
        learning_rate=1e-4,
        policy_type="deterministic",
    )

    bc = bc_config.create(device="cpu")
    evaluator = EnvironmentEvaluator(env)

    # Training
    bc.fit(
        dataset,
        n_steps=60000,
        n_steps_per_epoch=150,
        evaluators={"env": evaluator},
    )


if __name__ == "__main__":
    train_bc()
