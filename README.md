# 🐍 DiSA-IQL: Offline Reinforcement Learning for Robust Soft Robot Control under Distribution Shifts
---

## Overview

**DiSA-IQL** (Distribution-aware Implicit Q-Learning) is an **offline reinforcement learning** framework designed for **robust soft robot control** under **distribution shifts**.  
This project extends the standard [Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169) algorithm with **robustness modulation** and **distribution-sensitive adaptation** (DiSA), enabling stable policy learning in scenarios where **training and evaluation distributions differ**.

The framework is implemented on top of the **`d3rlpy`** library and includes:
- A **custom soft snake robot environment** (`SnakeRobot/`)
- A **large-scale offline dataset generator**
- Modified algorithm core: `RobustIQLImpl` (inheriting from `IQLImpl`)
- Unified evaluation for **in-distribution** and **out-of-distribution (OOD)** tasks.

---
## Dataset Details

The offline datasets used in this work are collected from large-scale simulations of the **soft snake robot** under different environmental distributions.  
Each dataset follows the **D4RL-style MDP format**, containing arrays of `observations`, `actions`, `rewards`, and `dones`.

| Dataset File | Description                                                                                                                                                                                                         | Episodes | Distribution Type |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|-------------------|
| `50000_r.npz` | Data collected from **in-distribution (ID)** regions using the **BER (Back-stepping Experience Replay)** model, where training and evaluation environments share similar goal and field configurations.             | 50,000 | In-distribution |
| `50000_hr.npz` | Data collected from **out-of-distribution (OOD)** regions using the **BER (Back-stepping Experience Replay)** model. It only covers the regular half of the goal space, while the other half remains unseen and uncollected to enable evaluation of the model’s generalization capability. | 50,000 | Out-of-distribution (OOD) |

### Format Specification

Each `.npz` dataset includes:
- `observations`: state vectors from the soft snake simulator  
- `actions`: normalized control inputs (clipped to \([-1,1]\))  
- `rewards`: environment rewards after MinMax normalization  
- `dones`: episode termination flags (`True` for terminal states)

---
## Installation

### Clone Repository
```bash
git clone https://github.com/hlj0908/DiSA-IQL-for-Soft-Robot-Control.git
cd DiSA-IQL-SnakeRobot
```
### Create Environment
```bash
python -m venv venv
source venv/bin/activate    # (Linux/Mac)
venv\Scripts\activate       # (Windows)
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start
### Train DiSA-IQL
```bash
python Runs/RobustIQL_Run.py
```
### Test model
```bash
python Runs/Test.py
```

---
## Citation
```bash
@misc{he2025disaiqlofflinereinforcementlearning,
      title={DiSA-IQL: Offline Reinforcement Learning for Robust Soft Robot Control under Distribution Shifts}, 
      author={Linjin He and Xinda Qi and Dong Chen and Zhaojian Li and Xiaobo Tan},
      year={2025},
      eprint={2510.00358},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.00358}, 
}
```

---
## Reference
**Based on:** [d3rlpy](https://github.com/takuseno/d3rlpy)  
