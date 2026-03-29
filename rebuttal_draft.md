# Rebuttal to Reviewer K7zo (Score: 4, Weak Accept)

We thank Reviewer K7zo for the thoughtful review and the recognition of TTGS's soundness, presentation, and originality.

> Value-derived distances are generally asymmetric and need not satisfy the triangle inequality. In the graph construction, do edge weights ever violate these properties in practice, and if so, how does this affect the validity of shortest-path planning?

This is a very good point, and we acknowledge that value-derived distances are not formal metrics (Section 3.1). To quantify this, we ran new experiments on triangle inequality violations across three base learners. We sample 1M triplets globally ("all") and separately 1M triplets where all pairwise distances fall under $\tau$ ("under $\tau$"), over 8 seeds:

| Base Learner | Dataset | Violation Rate (all) | Violation Rate (under $\tau$) | Mean Positive Violation (all) | Mean Positive Violation (under $\tau$) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| HIQL | humanoidmaze-giant-stitch | 0.2% | 13.4% | 585.7 | 3.1 |
| HIQL | antmaze-giant-stitch | 1.3% | 9.1% | 587.8 | 1.2 |
| QRL | humanoidmaze-giant-stitch | 0.7% | 0.3% | 0.001 | 0.002 |
| QRL | antmaze-giant-stitch | 0.7% | 0.2% | 0.001 | 0.002 |
| GCIQL | humanoidmaze-giant-stitch | 0.4% | 8.3% | 575.6 | 1.2 |
| GCIQL | antmaze-giant-stitch | 5.6% | 8.2% | 574.9 | 1.2 |

While violations do occur within the trust region, their magnitude is negligible: mean violations under $\tau$ are 0.002-3.1 steps, compared to 575-588 for unconstrained distances. This means that even when the triangle inequality is violated locally, the errors are small enough that they do not meaningfully distort shortest-path planning. This is consistent with TTGS's strong empirical performance across all base learners despite their different violation profiles. We will add this analysis to the paper.

> TTGS fails when intermediate states are absent from the offline dataset, particularly on manipulation tasks. Could the authors provide a more detailed discussion of potential solutions to this limitation?

We agree this is a fundamental constraint of any method planning over fixed data. We note two nuances: (1) sometimes intermediate states exist in the dataset, but the value function fails to recognize them as reachable due to long-horizon estimation errors. Improving value learning is complementary to TTGS. (2) When coverage gaps are detected (e.g., paths with too few intermediate waypoints), generative models trained on the offline data could synthesize additional states. Additionally, our manual analysis of 100 failed navigation episodes shows that the dominant failure mode is the base policy failing to reach a close subgoal, not missing graph coverage. This suggests that in practice, coverage is less often the bottleneck than policy execution quality. We will expand this discussion.

> The results suggest non-trivial sensitivity to hyperparameters across environments. Could the authors provide any heuristics or guidelines for setting these parameters in a new environment?

We achieved our strong performance without extensive hyperparameter sweeps. For most experiments, we tested 1-2 settings per $(\tau, T)$ pair; SAW used a single configuration across all tasks with no tuning. Our heuristic (detailed in Appendix H): set $\tau$ by visually inspecting the shortest paths on the graph. If paths are excessively fragmented into many short hops, $\tau$ is too small; if paths contain "blank" segments indicative of unreliable long edges, $\tau$ is too large. Then set $T = 2\tau$. We will highlight this procedure in the main text.

> TTGS relies on an additional penalty function to balance two competing failure modes... This reveals that the usability of the graph remains sensitive to errors in the distance estimates, which are largely driven by value function noise.

The soft penalty is specifically designed for this: rather than removing noisy edges (which fragments the graph), it assigns them a high cost, preserving connectivity while steering the planner toward reliable short hops. The ablation in Figure 4a confirms that this outperforms both no-penalty and hard-threshold variants. Empirically, TTGS improves all five base learners we tested despite their substantially different value function characteristics, suggesting that the soft penalty is effective across a range of noise levels. We agree that improving value function quality remains a valuable complementary direction.

We believe we have addressed all the raised concerns. If any remain, we would be happy to discuss them further. We kindly ask the reviewer to consider updating their score in light of these responses.

---

# Rebuttal to Reviewer Zg7J (Score: 3, Weak Reject)

We thank Reviewer Zg7J for the detailed feedback. We address each concern below.

> The problem setting is narrow. TTGS assumes locally accurate value functions, good dataset coverage along solution paths, and a known goal-reaching metric. These sidestep the central GCRL challenges: sparse/unknown rewards, exploration for coverage, unknown metrics, and subgoal selection under non-uniform coverage.

We believe this characterization conflates the offline GCRL setting, which is considered in our work, with general GCRL. We address each listed challenge:

- **Sparse/unknown rewards:** In offline GCRL, the dataset contains only trajectories of states and actions with no reward labels. All base learners we evaluate construct their training signal via hindsight relabeling, a self-supervised procedure that does not require any external reward. This is not an assumption specific to TTGS; it is the standard offline GCRL formulation shared by all prior work we compare against (Park et al., 2025; Wang et al., 2023; Zhou & Kao, 2025; Ahn et al., 2025).
- **Exploration for coverage:** Exploration is not available by definition in the offline setting. The learner must work with whatever data is provided. We evaluate across three data regimes (navigate, stitch, explore) with varying coverage quality.  
- **Unknown metrics:** TTGS does not require a known metric. It derives distances from the learned value function, which is trained without access to any ground-truth metric.

The absence of online interaction is a constraint that makes the problem harder, not easier.

Regarding locally accurate value functions: we do not assume access to particularly good value functions. We take five different base learners as-is, with their published hyperparameters, and show consistent gains across all of them. The fact that TTGS improves agents as different as HIQL, QRL, GCIQL, SAW, and OTA suggests it does not depend on any specific quality of the value function beyond what these standard methods already provide. If locally accurate value functions were a strong assumption, we would expect TTGS to fail on at least some of these learners.

That said, we agree the contribution is practically focused: given standard offline GCRL agents that already exist, TTGS provides a simple, training-free way to dramatically improve their long-horizon performance. We believe this is a valuable finding because it changes how practitioners should approach long-horizon offline GCRL.

> The formulation in Section 2 defines the MDP without rewards, then introduces $\|s-g\|$ without specifying what the norm operates on... The reward presumably uses privileged ground-truth state.

The norm $\|s-g\| < \epsilon$ is the benchmark's evaluation criterion, not an input to TTGS. TTGS only needs a distance signal derived from $V(s,g)$ at test time and doesn't have access to privileged information. The base learners train without privileged information, too, using hindsight relabeling from the offline dataset. Since no online rollouts occur, no goal-reaching check is needed during training. We will clarify this.

> The related work should engage with the broader planning literature.

We agree and will position TTGS relative to classical planning, motion planning, and task-and-motion planning in the revision. TTGS shares the graph-search structure with these fields, but differs in a key aspect: edge costs are obtained from learned, noisy value functions rather than known dynamics or collision checkers. This motivates TTGS's soft penalty, which has no direct analogue in settings where feasibility can be verified exactly.

> The manipulation results belong in the main paper. / Baseline methods are introduced only by acronym.

We will move manipulation results to the main text and discuss both locomotion and manipulation results more thoroughly, including clarifying in the abstract that the primary gains are in locomotion. We have chosen HIQL, QRL, GCIQL because they were included in the OGBench evaluation, and SAW, and OTA for their strong performance. We will add brief descriptions of each baseline method, including full names, and clarify why we chose them.

> Several prominent terms are vague: "latent capabilities," "locally consistent geometric structure," "global errors."

We will tighten these definitions in the revision. By "locally consistent geometric structure" we mean that value-derived distances are approximately correct for nearby state pairs (within the trust region $\tau$), even when they are unreliable at longer horizons. We will make this precise.

We believe we have addressed all the raised concerns with new experiments (which we will include in the revised paper) and planned revisions. If any remain, we would be happy to discuss them further. We kindly ask the reviewer to consider updating their score in light of these responses.

---

# Rebuttal to Reviewer 1VSa (Score: 4, Weak Accept)

We thank Reviewer 1VSa for the thorough and constructive review.

> The main limitation is the narrow range of tasks in the evaluation... The framing overstates the method's scope.

Manipulation results are in Appendix C and Figure 5. We will move them to the main text and clarify in the abstract that the primary gains are in long-horizon locomotion. We note that TTGS does not degrade performance on manipulation tasks; it defaults to base policy behavior when the graph is sparse or disconnected.

> Graph-based planning over dataset states has prior precedent. Earlier work, such as SoRB (Eysenbach et al., 2019\) and related methods, has explored this direction.

We agree and do not claim the invention of graph-based planning. Instead, our contribution lies in the techniques needed for successful graph-based planning for offline GCRL. SoRB operates in the online RL setting with a growing replay buffer and trains a specialized distance function using distributional Q-learning with ensembles to obtain reliable distance estimates. It then applies a hard distance threshold to prune long edges. TTGS, by contrast, requires no gradient-based training and works with any frozen offline GCRL agent and its existing value function. This immediate applicability to a wide range of base learners (five in our experiments, with consistent gains across all of them) without requiring specialized training is, in our view, the key conceptual difference. TTGS also replaces SoRB's hard threshold with a soft penalty that preserves graph connectivity, which our ablations show is critical to combat noise (Figure 4a, Appendix E).

> The comparison with GAS and CompDiffuser is asymmetric, as these methods require additional training. Including other test-time-only planning methods or explicitly noting this difference would improve the comparison.

Thank you for raising this point. We acknowledge this difference in Appendix A and note that TTGS operates under strictly more restrictive constraints, requiring no additional training, no generative models, and no online interaction, yet achieves comparable performance. To the best of our knowledge, TTGS is the first method that operates purely offline and at test time in the GCRL setting, so no direct baseline in this category exists. We will make this distinction more prominent. Exploring alternative test-time-only planning paradigms (e.g., tree search with learned world models, RRT-based planning) under the same constraints is an exciting direction for future work.

> Computational scaling is not empirically validated... practical performance on large offline datasets is not shown.

We provide new scaling experiments (HIQL, 8 seeds):

| M | Success Rate | Build Time (s) | Path Time (s) |
| :---- | :---- | :---- | :---- |
| **humanoidmaze-giant-stitch** | | | |
| 125 | 10.4 ±10.0 | 7.4 ±2.0 | 0.06 |
| 250 | 13.3 ±11.6 | 5.9 ±1.1 | 0.07 |
| 500 | 42.4 ±22.9 | 5.0 ±1.5 | 0.14 |
| 1000 | 60.0 ±19.6 | 6.9 ±1.7 | 0.24 |
| 2000 | 72.8 ±13.0 | 10.7 ±1.7 | 0.44 |
| 4000 | 76.7 ±18.1 | 26.2 ±1.5 | 0.84 |
| 8000 | 78.7 ±12.8 | 99.8 ±8.7 | 1.31 |
| **antmaze-giant-stitch** | | | |
| 125 | 29.2 ±8.3 | 9.3 ±4.2 | 0.05 |
| 250 | 29.9 ±12.4 | 8.2 ±4.4 | 0.06 |
| 500 | 51.6 ±12.1 | 3.7 ±0.4 | 0.10 |
| 1000 | 53.6 ±16.8 | 5.8 ±1.4 | 0.20 |
| 2000 | 70.9 ±10.6 | 10.2 ±1.3 | 0.39 |
| 4000 | 76.5 ±5.2 | 25.5 ±0.9 | 0.69 |
| 8000 | 81.1 ±2.5 | 88.6 ±2.5 | 1.13 |

Performance increases with number of samples. Build time grows quadratically but remains under 2 minutes even at $M{=}8000$. Path computation stays under 1.5s.

> Graph construction depends on access to the full offline dataset at test time.

No, access to the full offline dataset is not required. We merely require access to a small subset of the original data (we use M=4000, which is 0.4% of the dataset). When the policy is deployed by the same user/organization that trained it, the data is immediately available. Otherwise, requesting a small sample or collecting one through limited interaction is feasible. We will discuss this more explicitly in the limitations.

> When the method fails, what are the primary failure modes?

This is a very interesting question. We conducted a new manual analysis of 100 failed episodes on navigation tasks. The dominant failure mode is the base policy failing to reach a close subgoal, while the planned path itself is valid. Path-level failures (e.g., paths through walls due to graph artifacts) are rare. This confirms that TTGS's bottleneck is policy execution, not planning quality, and that improving low-level control is complementary to our approach. We will add this analysis.

We believe we have addressed all raised concerns with new experiments (which we will include in the revised paper) and planned revisions. If any remain, we would be happy to discuss them further. We kindly ask the reviewer to consider updating their score in light of these responses.

---

# Rebuttal to Reviewer VfQS (Score: 3, Weak Reject)

We thank Reviewer VfQS for the constructive review.

> How does TTGS compare to a simple training-free waypoint planner such as a kNN landmark graph over the sampled states and using A\*/Dijkstra \+ the same subgoal execution rule? Instead of the value-derived distance and soft penalty, use a purely geometric distance and standard sparse graph.

We ran this experiment with HIQL, $M{=}4000$, 8 seeds:

| Method | k | Success Rate | Disconnect Ratio |
| :---- | :---- | :---- | :---- |
| **humanoidmaze-giant-stitch** | | | |
| HIQL | -- | 3.2 ±0.5 | -- |
| HIQL + TTGS | -- | 76.7 ±18.1 | 0.00 |
| HIQL + L2-kNN | 2 | 3.2 ±1.4 | 1.00 |
| HIQL + L2-kNN | 5 | 28.1 ±20.6 | 0.64 |
| HIQL + L2-kNN | 10 | 59.4 ±15.3 | 0.00 |
| HIQL + L2-kNN | 20 | 18.8 ±13.3 | 0.00 |
| HIQL + L2-kNN | 50 | 3.2 ±3.9 | 0.00 |
| **antmaze-giant-stitch** | | | |
| HIQL | -- | 1.6 ±0.9 | -- |
| HIQL + TTGS | -- | 76.5 ±5.2 | 0.00 |
| HIQL + L2-kNN | 2 | 1.6 ±1.1 | 1.00 |
| HIQL + L2-kNN | 5 | 1.6 ±1.1 | 1.00 |
| HIQL + L2-kNN | 10 | 29.2 ±23.9 | 0.60 |
| HIQL + L2-kNN | 20 | 66.6 ±3.6 | 0.00 |
| HIQL + L2-kNN | 50 | 9.6 ±7.0 | 0.00 |

This experiment is informative in several ways. First, waypoint planning with geometric distances does help substantially at the right $k$, confirming that graph-based subgoal decomposition is a powerful idea. However, L2-kNN is fragile: performance is highly sensitive to $k$, with graph disconnections at low $k$ and wormhole-like shortcuts at high $k$. The optimal $k$ also differs across environments ($k{=}10$ for humanoidmaze, $k{=}20$ for antmaze). TTGS addresses exactly this fragility through its soft penalty and full-graph construction, achieving 76.7% and 76.5% without needing to select $k$. Second, the L2 baseline computes distances between body positions, which requires privileged knowledge of which observation dimensions correspond to the agent's position. In visual domains this information is simply absent. Value-derived distances, which TTGS uses by default, require no such privileged knowledge and perform comparably to L2 on state-based tasks (Table 1) while being the only option for pixel-based tasks.

> For each baseline method (HIQL, QRL, etc.) how are the subgoals sampled during training, and what is the distribution of goal distances?

All base learners are trained with their original published hyperparameters and are exposed to long-range goals. HIQL and GCIQL mix future-trajectory goals (50%, geometric sampling with mean ${\sim}100$ steps) and random dataset goals (30%). QRL uses 100% random goals. All methods are thus trained with long-range goals in mind. TTGS provides complementary gains on top of these already-tuned baselines.

> Are there environments in which the baselines are already mostly successful where a poor choice of $\tau$ and $T$ would result in worse performance?

Large $\tau$ and $T$ cause the path to degenerate to (start, goal), recovering base learner performance. Very small values force excessive waypoints, but we did not observe complete degradation in any experiment. The heuristic $T = 2\tau$ works well: $\tau$ approximates the inter-subgoal distance, and $T = 2\tau$ selects the next-to-closest subgoal, enabling smooth transitions. We will add this discussion.

> Are there quantitative predictors that can be used to predict when TTGS would help or not?

We conducted a new manual analysis of 100 failed navigation episodes. The dominant cause of failure is the base policy failing to reach a close subgoal, while the planned path is geometrically valid. This suggests a simple predictor: TTGS helps most when the base policy is reliable at short range but fails at long horizons. For manipulation, we analyze graph structure in Appendix C and show that sparse coverage and out-of-distribution goals limit the graph's connectivity. We will discuss both in the main text.

> Section 2 typos.

Thank you, we will fix these.

We believe we have addressed all raised concerns with new experiments (which we will include in the revised paper) and planned revisions. If any remain, we would be happy to discuss them further. We kindly ask the reviewer to consider updating their score in light of these responses.  