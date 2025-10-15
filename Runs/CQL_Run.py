import numpy as np
import configparser
from SnakeRobot.SnakeEnv_lessA_sector import SnakeEnv
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQLConfig
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


cfg = configparser.ConfigParser()
cfg.read('../configs/config.ini')
env = SnakeEnv(cfg['ENV_CONFIG'], seed=SEED)

data = np.load("../datasets/50000_hr.npz", allow_pickle=True)
observations = data["observations"]
actions = data["actions"]
rewards = data["rewards"]
terminals = data["dones"].astype(bool)
rewards=rewards/50

# Ensure the action range [-1, 1] (environment act_limit = 8, data should be scaled accordingly)
if np.max(np.abs(actions)) > 1.0 + 1e-6:
    actions = np.clip(actions / 8.0, -1.0, 1.0)

dataset = MDPDataset(observations, actions, rewards, terminals)

# Create CQL
cql = CQLConfig(
    actor_learning_rate=1e-4,
    critic_learning_rate=1e-4,
    temp_learning_rate=1e-4,
    alpha_learning_rate=1e-4,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    n_critics=2,
    initial_temperature=1.0,
    initial_alpha=1.0,
    alpha_threshold=0.0,
    conservative_weight=0.5,  # 5.0
    n_action_samples=50,
    soft_q_backup=False,
    max_q_backup=True,
).create(device="cpu")

# evaluator
evaluator = EnvironmentEvaluator(env)

# Training
cql.fit(
    dataset,
    n_steps=60000,
    n_steps_per_epoch=150,
    evaluators={"env": evaluator},
)
