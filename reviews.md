Official Review of Submission26883 by Reviewer K7zo
Official Reviewby Reviewer K7zo11 Mar 2026, 14:16 (modified: 24 Mar 2026, 10:27)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer K7zoRevisions
Summary:
Rather than designing complex auxiliary training mechanisms to overcome the noise in value functions, TTGS constructs a graph over offline dataset states, reinterprets existing goal-conditioned value functions as distance signals, and performs shortest-path search at test time. This dynamically provides a sequence of reachable intermediate subgoals to a frozen GCRL policy, substantially improving long-horizon goal-reaching capability without any retraining.

Strengths And Weaknesses:
Strengths
Soundness: The authors reinterpret goal-conditioned values as distance signals, construct a graph over offline data, and combine soft penalties with adaptive subgoal selection for test-time shortest-path planning. Extensive experiments spanning 5 base learners, multiple maze scales, and diverse observation types (state-based and pixel-based) provide strong support for the method's generality. Ablation studies further confirm that both the soft penalty and adaptive subgoal selection are critical to the observed performance gains.

Presentation: The paper is clearly written overall, with an intuitive presentation of problem motivation, method decomposition, and experimental structure. The appendices are thorough, covering complete result tables, runtime analysis, hyperparameter settings, and value-to-distance mappings, providing a reasonable degree of reproducibility.

Significance and Originality: Prior graph search methods typically require online interaction or specialized distance function training. TTGS restricts the entire pipeline to the test phase, which constitutes a meaningful departure from existing paradigms. The paper treats existing goal-conditioned value functions as local geometric signals and, under a fully test-time, training-free setting, unlocks the latent potential of pretrained policies through soft-penalized graph search and adaptive subgoal selection.

Weaknesses
The paper uses value functions as distances. However, value-derived distances generally do not satisfy the formal properties of a metric. They are not necessarily symmetric and may not obey the triangle inequality. This means the core edge weights in TTGS lack rigorous metric-theoretic guarantees, which is especially concerning given that value estimates tend to become noisier over longer horizons.

The method is highly sensitive to offline data coverage. TTGS retrieves and plans exclusively over dataset states, so whenever intermediate states between the start and goal are absent, no viable path exists in the graph. The paper observes this on manipulation tasks, where evaluation goals frequently fall outside the main training distribution. This is a fundamental methodological limitation, not merely an incidental experimental observation.

The paper introduces a soft penalty to address the problem of spurious shortcuts. However, using a hard threshold instead risks disconnecting the graph entirely. In other words, TTGS relies on an additional penalty function to balance two competing failure modes: over-trusting long edges versus fragmenting the graph. This reveals that the usability of the graph remains sensitive to errors in the distance estimates, which are largely driven by value function noise.

Experimental results indicate that TTGS exhibits notable sensitivity to its hyperparameters, which may limit the method's out-of-the-box applicability in new environments.

Soundness: 3: good
Presentation: 3: good
Significance: 3: good
Originality: 3: good
Key Questions For Authors:
Questions
Value-derived distances are generally asymmetric and need not satisfy the triangle inequality. In the graph construction, do edge weights ever violate these properties in practice, and if so, how does this affect the validity of shortest-path planning?

The paper acknowledges that TTGS fails when intermediate states are absent from the offline dataset, particularly on manipulation tasks. Could the authors provide a more detailed discussion of potential solutions to this limitation, even at a theoretical level?

The results suggest non-trivial sensitivity to hyperparameters across environments. Could the authors provide any heuristics or guidelines for setting these parameters in a new environment？

Limitations:
Yes.

Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Official Review of Submission26883 by Reviewer Zg7J
Official Reviewby Reviewer Zg7J10 Mar 2026, 12:37 (modified: 24 Mar 2026, 10:27)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer Zg7JRevisions
Summary:
This paper proposes Test-Time Graph Search (TTGS), a planner for offline goal-conditioned reinforcement learning (GCRL). The key observation is that pretrained goal-conditioned value functions, while unreliable for long-horizon planning, encode locally consistent distance estimates that can support graph-based search. TTGS samples states from the offline dataset as graph vertices, assigns edge weights by inverting the learned value function into step-distance estimates, and runs Dijkstra's algorithm to produce a sequence of subgoals for frozen low-level policies. An adaptive subgoal selection scheme feeds the farthest reachable waypoint to the policy at each step. On OGBench locomotion tasks, TTGS dramatically improves success rates.

Strengths And Weaknesses:
Strengths:

Simple, clearly explained method with only three hyperparameters and no retraining. Immediately practical.
Impressive navigation results. Gains like HIQL going from 0% to 81% on pointmaze-giant-stitch, consistent across five base learners, demonstrate this is a general-purpose wrapper.
Useful practical insight: complex auxiliary training (generative planners, distributional ensembles) may be unnecessary for many OGBench navigation tasks.
Thorough ablations.
Weaknesses:

The problem setting is narrow. TTGS assumes locally accurate value functions, good dataset coverage along solution paths, and a known goal-reaching metric. These sidestep the central GCRL challenges: sparse/unknown rewards, exploration for coverage, unknown metrics, and subgoal selection under non-uniform coverage. Substantial prior work on graph-based planning for goal-conditioned agents has tackled these harder problems — discovering graph structure and skills jointly, hierarchical goal-conditioned learning across abstraction levels, skill discovery without reward signals, and planning under poor coverage. While the offline setting is distinct, TTGS operates where the hardest parts are already solved. The contribution reduces to showing Dijkstra suffices in this regime — practically useful but conceptually limited.
The formulation in Section 2 defines the MDP without rewards, then introduces ||s-g|| without specifying what the norm operates on. This is fine for low-dimensional state, but the paper evaluates on visual domains where pixel-space Euclidean distance is meaningless. The reward presumably uses privileged ground-truth state — this should be stated explicitly, as it affects generality claims.
The related work should engage with the broader planning literature. TTGS is a classical planning algorithm (graph + shortest-path search) applied on top of RL. Classical planning, motion planning, and task-and-motion planning all deal with similar graph-based search and the paper should position itself relative to these.
The manipulation results belong in the main paper. The method working well on navigation but not manipulation is a key finding about its scope, not an extraneous detail.
Baseline methods are introduced only by acronym without describing what they do or why they were chosen.
Minor: Several prominent terms are vague: "latent capabilities," "locally consistent geometric structure," "global errors." The central claim about value functions' local reliability deserves a precise definition.
Soundness: 3: good
Presentation: 2: fair
Significance: 3: good
Originality: 2: fair
Key Questions For Authors:
In visual domains, does TTGS require ground-truth low-dimensional state for the goal-reaching reward? How does this affect generality? How does TTGS perform with substantially weaker value functions where even local estimates are unreliable?

Limitations:
Yes

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Official Review of Submission26883 by Reviewer 1VSa
Official Reviewby Reviewer 1VSa04 Mar 2026, 10:55 (modified: 24 Mar 2026, 10:27)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 1VSaRevisions
Summary:
TTGS is a wrapper method for Offline-GCRL that improves long-horizon task performance without requiring policy retraining. The approach constructs a graph from the offline dataset's states and uses the learned value function to assign edge costs. A soft-penalty distance metric is employed to penalize transitions unlikely to be executable by the base policy, and the resulting path is computed via Dijkstra's algorithm. The path yields a sequence of subgoals, which are sequentially provided to the frozen policy. Three key design choices are central to the method: (1) value-function-based edge weights combined with soft-penalty weighting rather than hard feasibility thresholds, (2) adaptive waypoint selection that advances to the next subgoal once proximity constraints are satisfied, and (3) zero-retraining applicability across any offline goal-conditioned RL agent.

The paper evaluates TTGS on OGBench across five distinct base learners (HIQL, QRL, GCIQL, SAW, OTA). Notable performance improvements are observed on long-horizon navigation tasks, including pointmaze-giant-stitch (0.0% to 80.9% for HIQL), humanoidmaze-giant-stitch (4.4% to 78.1%), and similar gains on antmaze variants. Computational overhead remains modest across all experimental conditions.

Strengths And Weaknesses:
Strengths
The empirical results on long-horizon trajectory stitching are strong. Improvements from near-zero to over 80% success rates indicate a marked shift in agent capability on these tasks. This scale of improvement is uncommon and merits attention.
Testing across five distinct base learners yields consistent improvements across all algorithms, suggesting that the method offers complementary benefits rather than exploiting architecture-specific properties of any single algorithm.
The soft-penalty formulation for edge costs is well-motivated. Hard feasibility cutoffs can discard useful information, while soft penalties allow the planner to consider less certain edges when necessary. Ablation studies provide supporting evidence for this choice.
The zero-retraining requirement is genuinely valuable for deployment scenarios and has practical significance.
The writing is clear, the problem formulation is direct, and the method description is detailed enough that re-implementation from the paper seems feasible.
Weaknesses
The main limitation is the narrow range of tasks in the evaluation. Although OGBench includes manipulation tasks, the strongest results are limited to maze-based navigation problems rather than general offline goal-conditioned RL. This distinction should be made more explicit.
Graph construction depends on access to the full offline dataset at test time. While this is feasible in research, deployment scenarios rarely include the training dataset with the policy. This practical constraint should be discussed in more detail, as it limits applicability.
The method’s effectiveness relies on value function calibration. When the value function provides accurate distance estimates, the approach works as intended. However, if the value function is poorly calibrated, especially outside the training distribution, distance estimates can become unreliable, and the planner may select infeasible transitions. Soft penalties help, but do not fully address this issue. Additional analysis of failure modes in value function calibration would strengthen the paper.
Computational scaling is not empirically validated. While the theoretical complexity is O(N^2) for pairwise distances or O(Nk) for k-NN approaches, practical performance on large offline datasets is not shown. Since modern datasets can be very large, empirical scaling plots relating graph construction time to dataset size would clarify this issue.
The comparison with GAS and CompDiffuser is asymmetric, as these methods require additional training. Including other test-time-only planning methods or explicitly noting this difference would improve the comparison.
The framing overstates the method’s scope. The title suggests broad applicability, but the approach requires navigation-style goals and dataset availability at test time. This is a meaningful but narrower domain than the framing implies.
Ablation studies cover subgoal selection and soft penalties but do not span a sufficiently wide experimental range. Additional ablations could address how sparse the graph can be before performance degrades, whether simpler distance heuristics can replace value-function-based costs, and where TTGS fails and why. More detailed failure-case analysis would strengthen the contribution.
Graph-based planning over dataset states has prior precedent. Earlier work, such as SoRB (Eysenbach et al., 2019) and related methods, has explored this direction. The main novelty here is in the engineering: soft penalties, adaptive subgoal selection, and broad empirical evaluation. Making an existing idea work well is valuable, but represents incremental rather than conceptual novelty.
Soundness: 3: good
Presentation: 3: good
Significance: 3: good
Originality: 2: fair
Key Questions For Authors:
Can you provide results on non-maze OGBench tasks, particularly manipulation environments? The current evidence strongly suggests navigation, even though the paper frames contributions more broadly. Clarifying whether the method generalizes to manipulation would be important.
What is the empirical computational cost as the dataset size increases? At what N do the O(N^2) or O(Nk) costs become practically problematic? Are there recommended subsampling strategies?
How does TTGS compare to other test-time planning approaches, such as tree search with learned world models or RRT-based planning using the value function as a heuristic?
When the method fails, what are the primary failure modes? Does failure stem from inaccurate distance estimation, insufficient state-space coverage in the graph, or the inability of the base policy to execute the proposed subgoals?
Limitations:
The paper would benefit from a more explicit discussion of: (a) the method being fundamentally suited to navigation-like problems rather than general goal-conditioned RL, (b) the need for the full offline dataset at test time, (c) empirical scaling with large datasets, and (d) sensitivity to value function calibration. The current limitations section is brief and does not fully address these points.

Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Official Review of Submission26883 by Reviewer VfQS
Official Reviewby Reviewer VfQS21 Feb 2026, 02:39 (modified: 24 Mar 2026, 10:27)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer VfQSRevisions
Summary:
TTGS introduces a framework to apply planning to offline goal-conditioned reinforcement learning methods. A graph over a sampled subset of states from an offline dataset are used to create a graph, with edge weights derived from value functions or known heuristics. To mitigate issues due to unreliable long jumps in the state-space, a soft penalty is applied to distance estimates over a threshold. At test time, the start and goal state are mapped to vertices on the graph by distance. A graph planning algorithm like Dijkstra is used to find a path of waypoints from the start to goal. Subgoals/waypoints are then selected from the planned path and used to condition the policy to guide the agent towards them. Empirically, TTGS improves the baseline performance over all tested GCRL methods.

Strengths And Weaknesses:
Strengths

The overall method and framing was easy to follow
The gains shown across the OGBench environments are quite substantial, with a low barrier to adoption as further training is not needed
Ablation helps justify that the additions of subgoal choice and penalty to avoid the wormhole phenomenon matters
I appreciate the care given to explaining the limitations and failure modes
Weaknesses

Missing a baseline for planning over states/landmarks, so its hard to isolate how much of the gains are from a TTGS-specific design or the fact that any waypoint planning helps (see the questions section below for more details)
Scalability concerns when it comes to requiring many states for sufficient coverage, as both compute and space will grow quite large
Its unclear how the baselines were trained with respect to their conditioned subgoals and distance away, and how much uplift TTGS actually provides if these were trained with long-range goals in mind
The ablation over 
 an 
 suggest that these choice can have a pretty large impact over the performance, and its not clear how these should be chosen other than broad sweeps over a range of values
Misc

Section 2 in the first sentence you have "i.e," but it should be "i.e.,"
Section 2 you define the expectation over over trajectories 
 but write a sampling distribution over states 
, then immediately in the following text use 
. The notation here is mismatched
Soundness: 3: good
Presentation: 3: good
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
For each baseline method (HIQL, QRL, etc.) how are the subgoals sampled during training, and what is the distribution of goal distances? I'd like to know whether TTGS's gains persist when the baseline is retrained with a goal-sampling scheme that emphasizes longer horizon goals
How does TTGS compare to a simple training-free waypoint planner such as a kNN landmark graph over the sampled states and using A*/Dijkstra + the same subgoal execution rule? Instead of the value-derived distance and soft penalty, use a purely geometric distance and standard sparce graph. This would clarify whether gains from TTGS are mostly from the graph search and waypoints versus value-derived distances and wormhole handling.
Are there quantitative predictors that can be used to predict when TTGS would help or not, like graph connectivity metrics or nearest-neighbor distances?
Are there environments in which the baselines are already mostly successful where a poor choice of 
 and 
 would result in worse performance?
Limitations:
yes

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.