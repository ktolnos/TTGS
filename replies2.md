# Reply to Reviewer 1VSa (Follow-up)

We thank Reviewer 1VSa for the thorough follow-up.

> Narrow evaluation scope vs. framing...

We agree and will narrow the framing throughout the paper. In the **abstract**, we will qualify that the primary gains are on long-horizon locomotion. In the **introduction**, we will state that TTGS is most impactful when (1) the task requires long-horizon stitching, (2) the base policy is locally reliable, and (3) the dataset covers intermediate states. We will note that these conditions hold for navigation but not for the tested manipulation tasks. In the **conclusion**, we will replace broad language with a precise summary: TTGS provides substantial navigation improvements and defaults to base policy behavior when its conditions are not met.

> Conditions under which TTGS is effective... Is there a quantitative metric that could predict in advance whether TTGS will provide gains on a new task?

We identify two complementary predictors and provide new experimental evidence for the first.

**1. Dataset coverage: largest hop ratio (rollout-free).** TTGS decomposes a long-horizon task into short hops through dataset states. We propose a rollout-free diagnostic: after running Dijkstra on the TTGS graph, compute the *largest hop ratio*

$$\frac{\max\bigl(d(s_0, p_0),\; d(p_0, p_1),\; \ldots,\; d(p_{L-1}, p_L),\; d(p_L, g)\bigr)}{d(s_0, g)}$$

A low ratio means every hop is small relative to the total task, which is exactly the regime where TTGS helps. A high ratio means at least one hop spans a large fraction of the distance, indicating insufficient coverage.

We computed this metric across all evaluation tasks using GCIQL (8 seeds), sorted by hop ratio:

| Dataset | Hop Ratio | GCIQL | +TTGS | $\Delta$ (pp) |
| --- | --- | --- | --- | --- |
| pointmaze-giant-stitch | 0.041 | 0.0 | 98.0 | **+98.0** |
| antmaze-giant-stitch | 0.080 | 0.0 | 32.7 | **+32.7** |
| humanoidmaze-giant-stitch | 0.112 | 0.2 | 0.2 | 0.0 |
| humanoidmaze-medium-stitch | 0.140 | 14.0 | 20.2 | +6.2 |
| antmaze-medium-stitch | 0.207 | 29.5 | 53.0 | +23.5 |
| pointmaze-medium-stitch | 0.217 | 18.2 | 44.0 | +25.8 |
| scene-play | 0.458 | 50 | 52 | +2 |
| cube-triple-play | 0.470 | 4 | 4 | 0 |
| puzzle-4x6-play | 0.620 | 10 | 10 | 0 |

Manipulation tasks have hop ratios 4-6$\times$ higher than navigation tasks, confirming that the dataset lacks intermediate waypoints. Among locomotion tasks, giant mazes have the lowest ratios and the largest TTGS gains. This metric requires only the offline dataset and value function, without any rollouts.

**2. Base policy local competence (requires rollouts).** Even with good coverage, TTGS needs the base policy to execute short hops reliably. Figure 2c shows navigation policies achieve >80% short-range success but <10% long-range, which is ideal for TTGS. For manipulation, even short-range performance is low, leaving TTGS without a reliable executor. The case of humanoidmaze-giant-stitch (low hop ratio, zero improvement) illustrates this: although coverage is good, the humanoid embodiment makes short-hop execution unreliable for GCIQL, unlike pointmaze and antmaze where the simpler embodiment allows reliable local control.

However, estimating local competence requires rollouts, at which point it is simpler to evaluate TTGS directly. We therefore recommend the following procedure for a new task: (1) compute the hop ratio as a rollout-free preliminary check; if it exceeds ~0.4, TTGS is unlikely to provide gains; (2) if it is low, run TTGS directly, since the method defaults safely to base policy behavior when its conditions are not met.

We will add both diagnostics and the table to the paper. We hope these concrete predictors and the planned framing revisions fully address the reviewer's remaining concerns.
