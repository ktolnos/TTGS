# TEST-TIME GRAPH SEARCH FOR GOAL-CONDITIONED REINFORCEMENT LEARNING

## Overview
This is the implementation of TTGS method.

Most directories mirror the upstream [OGBench](https://github.com/seohongpark/ogbench)
layout for datasets, MuJoCo assets, and environment wrappers. TTGS-specific
components live under `impls/`:

- `impls/main.py` is the primary launcher for offline training and evaluation.
- `impls/ttgs.py`, `impls/whole_task_eval.py`, and `impls/eval_utils.py` provide
  the test-time graph search routines, whole-task evaluation loop, plotting, and
  WandB logging helpers introduced by TTGS.
- `impls/agents/` extends the OGBench agent library. In particular,
  `gciql.py`, `hiql.py`, and `qrl.py` expose value-to-distance conversions used
  by TTGS to turn learned value estimates into expected steps-to-goal signals.
## Requirements
* Python 3.10
* MuJoCo 3.1.6
* JAX >= 0.4.31 (CUDA 12 build)
## Installation

Clone the repository and choose one of the environment managers below. Both flows
install MuJoCo 3.1.6 and the Python dependencies listed in
`impls/requirements.txt`. Ensure your GPU driver stack is compatible with the
CUDA 12 JAX wheels if you plan to use accelerated training.

### Conda

```bash
conda create -n ttgs python=3.10 -y
conda activate ttgs
pip install --upgrade pip
pip install mujoco==3.1.6 -r impls/requirements.txt
```

### uv

```bash
uv venv --python 3.10 ttgs
source ttgs/bin/activate
uv pip install --upgrade pip
uv pip install mujoco==3.1.6 -r impls/requirements.txt
```

If you are running headless (e.g., on a server), export
`MUJOCO_GL=egl` before launching experiments to enable off-screen MuJoCo
rendering.

## Quick Start (HIQL + TTGS on humanoidmaze-giant-stitch-v0)
```bash
cd impls
python main.py --subsample_ablt=random_points --random_size=4000 --tau=24 --threshold=48 --env_name=humanoidmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
```

## Acknowledgments
This codebase is inspired by or partly uses code from the following repositories:
- [OGBench](https://github.com/seohongpark/ogbench) for the dataset structure and the state-based and pixel-based environments.
