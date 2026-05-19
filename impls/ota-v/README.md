# Option-aware Temporally Abstracted Value for Offline Goal-Conditioned Reinforcement Learning

### [Paper](https://arxiv.org/abs/2505.12737) | [Project page](https://ota-v.github.io)


## 🧩 Overview

This repository contains the official implementation of **"Option-aware Temporally Abstracted Value for Offline Goal-Conditioned Reinforcement Learning"**. In this work, we propose an **Option-aware Temporally Abstracted (OTA) value function** to address long-horizon tasks in offline goal-conditioned reinforcement learning (GCRL).

For detailed implementation, please check [ota.py](agents/ota.py).

## 📦 Installation

### 1. 🐳 Docker **(Recommended)**

We provide a **Docker environment** to make setup and execution easier.
Please see the [Docker guideline](docker/README.md) for detailed instructions.

### 2. PyPI

The environment can be set up using Python 3.9+ with `jax>=0.4.2` and `ogbench>=1.1.5`.

```bash
pip install -r requirements.txt
```

## 🎯 Optimal trajectories

To collect optimal trajectories, we followed the data generation procedure used in OGBench.
You can find the detailed process here: [OGBench dataset generation](https://github.com/seohongpark/ogbench?tab=readme-ov-file#reproducing-datasets-1)
- **Maze** environments: expert policy from OGBench
- **Cube** environments: scripted policy from OGBench

The collected trajectories are available in the [optimal_trajs](./optimal_trajs) directory:
- 📁 **Data**: [optimal_trajs/data](./optimal_trajs/data)
- 📈 **Visualizations**: [optimal_trajs/figure](./optimal_trajs/figure)
- 💻 **Notebook**: [optimal_trajs/visualize_maze.ipynb](./optimal_trajs/figure/visualize_maze.ipynb)
  - Plots the observation coordinates directly on the maze layout

These trajectories are used to **plot values** and **measure orderconsistency ratio** during evaluation.

## 🚀 Reproducing Experiments

Run the following command to generate **value plots** and evaluate the **order consistency ratio**:
```bash
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/ota.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.abstraction_factor=5
```

To reproduce all experiments, use:
```bash
bash hyperparameters.sh
```
This script runs and compares three settings: `HIQL`, `OTA` (ours), and `OTA (carefully tuned)`.

📊 You can check the experimental logs here: [wandb logs](https://wandb.ai/chwoong/OTA)

💡 Tip: For OTA, keeping the same hyperparameters as HIQL and only adjusting the `abstraction_factor` is often enough to achieve strong performance. For additional performance improvement, you can carefully tune the hyperparameters — for example, using the `OTA (carefully tuned)` configuration.

## Citation

```
@inproceedings{ota2025,
  title={Option-aware Temporally Abstracted Value for Offline Goal-Conditioned Reinforcement Learning},
  author={Ahn, Hongjoon and Choi, Heewoong and Han, Jisu and Moon, Taesup},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
}
```


## Acknowledgments

This code is based on [**OGBench**](https://github.com/seohongpark/ogbench).
