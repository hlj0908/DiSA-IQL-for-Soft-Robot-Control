import numpy as np
import random
import torch
import configparser
from SnakeRobot.SnakeEnv_lessA_sector import SnakeEnv
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import EnvironmentEvaluator
from d3rlpy.preprocessing import MinMaxRewardScaler
from d3rlpy.algos import RobustIQLConfig

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

# load dataset
data = np.load("../datasets/50000_hr.npz", allow_pickle=True)
observations = data["observations"]
actions = data["actions"]
rewards = data["rewards"]
terminals = data["dones"].astype(bool)

if np.max(np.abs(actions[:, :2])) > 1.0 + 1e-6:
    actions[:, :2] = np.clip(actions[:, :2] / 8.0, -1.0, 1.0)

dataset = MDPDataset(observations, actions, rewards, terminals)

# initialize Robust IQL
iql = RobustIQLConfig(
    batch_size=256,
    actor_learning_rate=1e-4,
    critic_learning_rate=1e-4,
    tau=0.005,
    n_critics=2,
    expectile=0.7,
    weight_temp=3,
    max_weight=100.0,
    gamma=0.99,
    use_robust=True,
    robust_alpha=0.1,
    uncertainty_set="KL",  # ["Wasserstein", "KL", "Chi2", "TV"]
    decay_schedule=True,
    reward_scaler=MinMaxRewardScaler(minimum=-1.0, maximum=1.0),
).create(device="cpu")

evaluator = EnvironmentEvaluator(env)

iql.fit(
    dataset,
    n_steps=60000,
    n_steps_per_epoch=150,
    evaluators={"env": evaluator},
)

