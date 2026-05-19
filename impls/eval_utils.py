from __future__ import annotations

import pdb
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import whole_task_eval as ttgs_eval_mod

# Local imports
from utils.evaluation import evaluate as eval_standard


@dataclass
class EvalParams:
    agent: object
    env: object
    seed: int = 0
    train_data: Optional[dict] = None
    task_id: Optional[int] = None
    num_tasks: Optional[int] = None
    num_eval_episodes: int = 50
    num_video_episodes: int = 0
    video_frame_skip: int = 3
    eval_temperature: float = 0.0
    eval_gaussian: Optional[float] = None
    config: Optional[dict] = None
    # Which eval implementation to use: 'standard' or 'ttgs'
    mode: str = "standard"
    add_dataset_trajectories: bool = False


def _get_task_infos(env) -> List[dict]:
    return (
        env.unwrapped.task_infos
        if hasattr(env.unwrapped, "task_infos")
        else env.task_infos
    )


def _filter_task_metrics(stats: Dict[str, float]) -> Dict[str, float]:
    """Keep only success/returns metrics when present."""
    out = {}
    for k, v in stats.items():
        if (
            k.endswith("success")
            or k.endswith("returns")
            or ("success" in k)
            or ("returns" in k)
            or ("timing." in k)
            or ("flops" in k)
            or ("replan" in k)
            or ("inf_path" in k)
        ):
            out[k] = float(v)
    return out


def run_taskwise_evaluation(
    params: EvalParams,
) -> Tuple[Dict[str, float], Dict[str, float], List[np.ndarray]]:
    """Runs evaluation across tasks and aggregates overall metrics."""
    subsample_time = None
    graph_build_time = None
    task_infos = _get_task_infos(params.env)
    n_tasks = params.num_tasks if params.num_tasks is not None else len(task_infos)

    task_metrics: Dict[str, float] = {}
    overall_lists = defaultdict(list)
    renders: List[np.ndarray] = []
    ttgs_graph_prepared = False
    ttgs_graph_task_name: Optional[str] = None

    for task_id in range(1, n_tasks + 1):
        task_name = task_infos[task_id - 1]["task_name"]
        if params.task_id is not None and task_id != params.task_id:
            continue

        if params.mode == "standard":
            stats, trajs, cur_renders = eval_standard(
                agent=params.agent,
                env=params.env,
                seed=params.seed,
                task_id=task_id,
                config=params.config or {},
                num_eval_episodes=params.num_eval_episodes,
                num_video_episodes=params.num_video_episodes,
                video_frame_skip=params.video_frame_skip,
                eval_temperature=params.eval_temperature,
                eval_gaussian=params.eval_gaussian,
            )
        elif params.mode == "ttgs":
            if not ttgs_graph_prepared:
                subsample_start = time.time()
                subsampled_obs = params.agent.get_subsampled_observations(
                    params.train_data
                )
                subsample_time = time.time() - subsample_start

                graph_build_start = time.time()
                additional_trajectories = (
                    params.train_data if params.add_dataset_trajectories else None
                )
                params.agent.build_graph(subsampled_obs, additional_trajectories)
                graph_build_time = time.time() - graph_build_start
                ttgs_graph_prepared = True
                ttgs_graph_task_name = task_name
                print(
                    f"[TTGS] subsample_time={subsample_time:.2f}s, graph_build_time={graph_build_time:.2f}s"
                )
            else:
                reused_from = ttgs_graph_task_name or "previous task"
                print(
                    f"[TTGS] Reusing pre-built graph from {reused_from}, skipping rebuild.",
                    flush=True,
                )
            stats, trajs, cur_renders = ttgs_eval_mod.evaluate(
                agent=params.agent,
                env=params.env,
                seed=params.seed,
                task_id=task_id,
                num_eval_episodes=params.num_eval_episodes,
                num_video_episodes=params.num_video_episodes,
                video_frame_skip=params.video_frame_skip,
            )
            print(f"ttgs task {task_id} success rate: {stats['success']:.2f}")
        else:
            raise ValueError(f"Unknown eval mode: {params.mode}")

        # Record filtered metrics and accumulate overall
        filtered = _filter_task_metrics(stats)
        task_filtered = filtered
        if params.mode == "ttgs":
            task_filtered = {k: v for k, v in filtered.items() if "success" in k}
        prefix = "evaluation" if params.mode == "standard" else "ttgs_evaluation"
        for k, v in task_filtered.items():
            task_metrics[f"{prefix}/{task_name}_{k}"] = v
            if k.endswith("success") or ("success" in k):
                overall_lists["success"].append(v)
            if k.endswith("returns") or ("returns" in k):
                overall_lists["returns"].append(v)
            if params.mode != "ttgs" and ("timing." in k or "flops" in k):
                overall_lists[k].append(v)
        if params.mode == "ttgs":
            for key, val in filtered.items():
                if "timing." in key or "replan" in key or "inf_path" in key:
                    overall_lists[key].append(val)

        renders.extend(cur_renders)

    overall: Dict[str, float] = {}
    if len(overall_lists["success"]) > 0:
        overall["overall_success"] = float(np.mean(overall_lists["success"]))
    if len(overall_lists["returns"]) > 0:
        overall["overall_returns"] = float(np.mean(overall_lists["returns"]))
    for key, values in overall_lists.items():
        if key in {"success", "returns"}:
            continue
        if values:
            metric_name = key.replace(".", "_")
            overall[f"overall_{metric_name}"] = float(np.mean(values))

    if params.mode == "ttgs":
        if subsample_time is not None:
            overall["overall_timing_subsample"] = float(subsample_time)
        if graph_build_time is not None:
            overall["overall_timing_graph_build"] = float(graph_build_time)
        overall["overall_ttgs_clustering_time"] = float(
            getattr(params.agent, "last_clustering_time", 0.0)
        )
        overall["overall_ttgs_cluster_centers"] = float(
            getattr(params.agent, "last_cluster_center_count", 0)
        )

    return task_metrics, overall, renders


def plot_success_comparison(
    base_success: float,
    ttgs_success: float,
    title: str,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    methods = ["Original Agent", "With TTGS"]
    success_rates = [base_success, ttgs_success]

    bars = ax.bar(methods, success_rates, alpha=0.8, edgecolor="black", linewidth=2)
    ax.set_ylabel("Success Rate", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    if base_success > 0:
        improvement = (ttgs_success - base_success) / base_success * 100.0
        ax.text(
            0.5,
            0.95,
            f"Improvement: {improvement:+.1f}%",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
        )

    plt.tight_layout()
    return fig
