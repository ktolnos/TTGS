# TTGS Rebuttal Plan

## Review Summary

| Reviewer | Score | Confidence | Originality | Key Stance |
|----------|-------|------------|-------------|------------|
| K7zo | 4 (Weak Accept) | 3 | 3 | Solid but concerned about metric properties, data coverage, hyperparam sensitivity |
| Zg7J | 3 (Weak Reject) | **5** | 2 | Conceptually limited; narrow setting; presentation gaps |
| 1VSa | 4 (Weak Accept) | 4 | 2 | Strong empirics but incremental over SoRB; narrow eval; scaling unclear |
| VfQS | 3 (Weak Reject) | 3 | 2 | Missing geometric-distance baseline; scalability; hyperparam sensitivity |

**Current standing: 4/3/4/3 — borderline. Need to flip at least one reject.**

---

## Review-by-Review Analysis

---

### Reviewer K7zo (Score 4 — Weak Accept)

**Strengths they liked:** Soundness, presentation, significance, generality across 5 learners, thorough ablations and appendices.

**Concerns:**

1. **Metric properties (asymmetry, triangle inequality violation)** — Value-derived distances don't satisfy formal metric axioms. Do edge weights violate these in practice? How does this affect Dijkstra's correctness?
2. **Data coverage limitation** — TTGS fails when intermediate states are absent (manipulation tasks). Wants discussion of potential solutions, even theoretical.
3. **Soft penalty vs. hard threshold tradeoff** — The need for soft penalty reveals that graph usability is sensitive to distance estimate errors.
4. **Hyperparameter sensitivity** — Wants heuristics or guidelines for setting params in new environments.

**Assessment:** Friendly reviewer. Concerns are addressable with discussion + minor experiments. Priority: keep this accept solid.

---

### Reviewer Zg7J (Score 3 — Weak Reject, Confidence 5)

**This is the hardest reviewer to flip.** Highest confidence, rates originality 2/4.

**Concerns:**

1. **Narrow problem setting / limited novelty** — Claims TTGS operates where the hardest GCRL challenges (exploration, unknown rewards, unknown metrics) are already solved. Contribution "reduces to showing Dijkstra suffices in this regime." Wants engagement with broader planning literature (classical planning, motion planning, TAMP).
2. **Reward/norm ambiguity in visual domains** — Section 2 introduces `||s-g||` without specifying what the norm operates on. For pixels, Euclidean distance is meaningless. Does TTGS require ground-truth low-dim state for reward? This should be stated explicitly.
3. **Related work gaps** — Should position relative to classical planning, motion planning, TAMP.
4. **Manipulation results belong in main paper** — The method's scope limitation is a key finding, not an appendix detail.
5. **Baselines introduced only by acronym** — No description of what they do or why they were chosen.
6. **Vague terminology** — "latent capabilities," "locally consistent geometric structure," "global errors" need precise definitions.
7. **Weak value functions** — How does TTGS perform when even local estimates are unreliable?

**Assessment:** Sees this as a well-executed engineering contribution but not a conceptual advance. Must reframe the contribution honestly and address scope.

---

### Reviewer 1VSa (Score 4 — Weak Accept)

**Strengths they liked:** Dramatic gains (0% -> 80%), consistency across 5 learners, soft penalty well-motivated, zero retraining, clear writing, re-implementation seems feasible.

**Concerns:**

1. **Narrow task range** — Strongest results limited to maze navigation. Wants manipulation in main paper.
2. **Dataset required at test time** — Practical deployment constraint, rarely feasible outside research.
3. **Value function calibration sensitivity** — Wants failure mode analysis when V is poorly calibrated.
4. **Computational scaling** — O(N^2) not empirically validated for large datasets. Wants scaling plots.
5. **Asymmetric comparison** — GAS/CompDiffuser require additional training; comparison is not apples-to-apples.
6. **Overstated framing** — Title suggests broad applicability but method is narrower in practice.
7. **Incremental over SoRB** — Main novelty is engineering (soft penalties, adaptive subgoal, broad eval). Valuable but incremental.
8. **Insufficient ablation range** — How sparse can graph be? Simpler distance heuristics? Where does TTGS fail and why?

**Assessment:** Supportive but wants honest scoping and more thorough analysis. Addressable.

---

### Reviewer VfQS (Score 3 — Weak Reject, Confidence 3)

**Concerns:**

1. **Missing geometric-distance baseline** — kNN landmark graph with Euclidean distance + Dijkstra + same subgoal execution rule. This would isolate whether gains come from graph-search-with-waypoints vs. value-derived-distances + wormhole handling. **This is the most actionable concern.**
2. **Scalability** — Compute and space grow with state count for sufficient coverage.
3. **Unclear baseline training details** — How are subgoals sampled during baseline training? Does TTGS gain persist if baselines are retrained with longer-horizon goals?
4. **Hyperparameter sensitivity** — tau and T can have large impact; unclear how to choose beyond broad sweeps.
5. **Can poor tau/T hurt on easy tasks?** — Does TTGS degrade performance where baselines already succeed?
6. **Minor notation issues** — "i.e," typo, mismatched sampling distribution notation.

**Assessment:** Lower confidence (3) — most flippable reject. The geometric baseline experiment is the key ask.

---

## Cross-Cutting Themes

The reviews converge on **five recurring issues**:

| Theme | Raised by | Severity |
|-------|-----------|----------|
| Narrow scope (navigation only) + manipulation in main paper | Zg7J, 1VSa, VfQS | High |
| Incremental over SoRB / need to isolate contribution | Zg7J, 1VSa, VfQS | High |
| Hyperparameter sensitivity / selection guidelines | K7zo, 1VSa, VfQS | Medium-High |
| Scalability / computational cost at scale | 1VSa, VfQS | Medium |
| Value function quality / metric properties | K7zo, Zg7J | Medium |

---

## Rebuttal Experiments

### Experiment 1: Geometric-Distance Baseline (kNN + Euclidean + Dijkstra)

**Addresses:** VfQS Q2 (primary), Zg7J (contribution isolation), 1VSa (simpler heuristics ablation)

**Priority: 1 (HIGHEST) — could flip VfQS**

**Design:** Build a kNN landmark graph over M=4000 sampled states using Euclidean distance in state space (body position). Run Dijkstra with the same adaptive subgoal selection rule. No soft penalty (Euclidean distances don't have the wormhole problem). Compare on giant-stitch tasks against:
- Base learner alone
- TTGS with value-derived distances
- This geometric baseline

**Expected outcome:** Euclidean graph will fail on maze tasks because straight-line distance ignores walls — value-derived distances encode environment structure that Euclidean cannot. This cleanly demonstrates the value function contribution.

**Note:** Table 1 already has TTGS-L2, but that still uses TTGS's soft penalty. The reviewer wants a *fully naive* geometric planner. Also highlight Table 1 more prominently in rebuttal.

---

### Experiment 2: Failure Mode Categorization

**Addresses:** 1VSa Q4, K7zo Q2, Zg7J Q2

**Priority: 2 (HIGH)**

**Design:** On failed episodes across navigation and manipulation tasks, classify each failure into:
1. **Distance estimation error** — path exists in graph but edge weights are wrong, leading to bad routes
2. **Coverage gap** — no connected path through graph from start to goal
3. **Policy execution failure** — good subgoals selected but base policy can't reach them

Log the category for each failed episode. Report distribution as a table.

**Expected outcome:** Navigation failures are primarily policy execution failures. Manipulation failures are primarily coverage gaps. This cleanly explains the performance discrepancy and provides actionable understanding for practitioners.

---

### Experiment 3: Empirical Scaling Plot

**Addresses:** 1VSa Q2, VfQS scalability concern

**Priority: 3 (MEDIUM)**

**Design:** Vary M in {100, 500, 1000, 2000, 4000, 8000, 16000} on 2-3 representative environments. Plot:
- Graph construction time vs. M
- Shortest path computation time vs. M
- Success rate vs. M

**Expected outcome:** Construction time scales quadratically but remains practical (Table 8 shows ~35s for M=4000). Performance plateaus around M=2000-4000. Provides clear guidance on when subsampling is sufficient.

---

### Experiment 4: Metric Property Analysis

**Addresses:** K7zo Q1

**Priority: 4 (MEDIUM)**

**Design:** On 2-3 environments, compute:
- **Asymmetry:** distribution of |d_hat(s,g) - d_hat(g,s)| over sampled pairs
- **Triangle inequality violations:** fraction of triplets where d(s,g) > d(s,m) + d(m,g)
- Stratify by distance (nearby pairs within tau vs. far pairs)

**Expected outcome:** Violations exist but concentrate on far-away pairs (beyond tau). For nearby pairs (the ones TTGS actually relies on for edges), violations are modest. Dijkstra finds good paths because soft penalty downweights the unreliable long edges where violations are worst.

---

### Experiment 5: Degradation on Easy Tasks

**Addresses:** VfQS Q4

**Priority: 5 (MEDIUM-LOW)**

**Design:** On medium-navigate tasks where baselines already achieve >80%, run TTGS with:
- Good hyperparameters
- Deliberately poor hyperparameters (tau=6, tau=96)

Report whether TTGS ever *significantly hurts* performance.

**Expected outcome:** TTGS is generally safe. Worst case: performance matches base policy (when graph is too fragmented to find useful paths, agent defaults to base behavior). Shows the "do no harm" property.

---

### Experiment 6: Weak Value Function Robustness

**Addresses:** Zg7J Q2

**Priority: 6 (MEDIUM)**

**Design:** Take undertrained checkpoints (e.g., 25%, 50% of full training) of HIQL/GCIQL. Apply TTGS. Compare:
- Undertrained base policy alone
- Undertrained base + TTGS
- Fully-trained base policy alone

**Expected outcome:** TTGS still helps even with weaker value functions, as long as local distance estimates are somewhat correlated with true reachability. Below a threshold, TTGS degrades gracefully (defaults to base policy rather than catastrophic failure).

---

## Experiment Priority Queue

| Priority | Experiment | Key Reviewer(s) | Effort | Impact |
|----------|-----------|-----------------|--------|--------|
| 1 | Geometric-distance baseline | VfQS, Zg7J | Medium | Very High (flip VfQS) |
| 2 | Failure mode categorization | 1VSa, K7zo, Zg7J | Medium | High |
| 3 | Scaling plot | 1VSa, VfQS | Low | Medium |
| 4 | Metric property analysis | K7zo | Low-Medium | Medium |
| 5 | Degradation on easy tasks | VfQS | Low | Medium |
| 6 | Weak value function | Zg7J | Medium | Medium |

---

## Rebuttal Writing Strategy

### Response to Zg7J (Priority: HIGHEST — need to neutralize)

**Strategy:** Respectful partial disagreement on novelty; concede presentation gaps; add experiments.

1. **"Reduces to showing Dijkstra suffices"** — Push back respectfully. The contribution is the *insight* that standard TD-learned value functions are locally reliable enough for graph search + the *specific mechanisms* (soft penalty, adaptive subgoal) that make this robust. SoRB (Eysenbach et al., 2019) required online interaction for edge validation and used distributional value ensembles. TTGS is the first purely test-time, training-free graph-search wrapper for frozen GCRL agents. Reference Experiment 1: naive Dijkstra with Euclidean distances fails — the value-derived distance with soft penalty is essential, not a trivial addition.

2. **Narrow problem setting** — Partially concede. Acknowledge navigation is the primary domain. But argue: (a) navigation/locomotion is a large, important problem class (robotics, autonomous systems), (b) the method *correctly identifies* when it cannot help (manipulation) and defaults safely rather than hallucinating, (c) the test-time-only constraint is practically significant — no retraining is a real deployment advantage.

3. **Reward/norm in visual domains** — Clarify: `||s-g||` in Section 2 defines the *GCRL reward function*, not TTGS's distance metric. TTGS uses the learned value function V(s,g) as its distance signal. In visual domains, the value function was trained with whatever reward was used (which may use privileged state info during training). At test time, TTGS only needs V(s,g) — it does **not** require ground-truth state. Will clarify this in revision.

4. **Related work on classical/motion planning** — Agree. Add paragraph positioning TTGS relative to classical planning (A*, PRM, RRT). Key distinction: TTGS operates in learned representation space with noisy distance estimates, whereas classical methods assume known dynamics/geometry. The soft penalty is specifically designed for the learned-distance regime.

5. **Manipulation in main paper** — Agree completely. Move Table 5 + Figure 5 analysis to main text. This demonstrates intellectual honesty about scope.

6. **Baseline descriptions** — Agree. Add 1-2 sentence descriptions of each method.

7. **Vague terminology** — Agree to tighten. Define "locally consistent" as "distance estimates d_hat(s,g) are accurate within tau steps of the true step distance."

---

### Response to VfQS (Priority: HIGH — most flippable)

1. **Geometric baseline** — Run Experiment 1. Additionally, highlight existing Table 1 (TTGS-L2 vs TTGS-value). The combined evidence shows value-derived distances are crucial on structured environments where Euclidean distance ignores obstacles.

2. **Scalability** — Point to Table 8 (runtime analysis already in paper) + Experiment 3 for scaling curve.

3. **Baseline training details** — Clarify: all baselines use default OGBench training with standard goal-sampling distributions. TTGS is applied purely post-hoc — no training changes. The question of whether retraining baselines with different goal distributions would close the gap is interesting but orthogonal: TTGS's value proposition is precisely that it helps *existing* trained agents without any retraining.

4. **Hyperparameter guidelines** — Provide concrete heuristic: *"Set tau near the horizon where the base policy's local success rate drops below 50%. Set T = 2*tau. In practice, the doubling schedule (tau, T) in {(12,24), (24,48), (48,96)} covers most settings with 1-2 trials."* Reference Table 4 showing fixed hyperparameters work across domains.

5. **Degradation on easy tasks** — Experiment 5 results. Also note Table 3: on medium-navigate tasks where baselines are strong, TTGS matches or marginally improves, never significantly hurts.

6. **Notation fixes** — Thank reviewer, will fix.

---

### Response to K7zo (Priority: MEDIUM — maintain accept)

1. **Metric properties** — Experiment 4 results. Acknowledge violations exist. Explain: Dijkstra's optimality guarantee requires non-negative edge weights (which we have) but is robust to asymmetry — it still finds a low-cost path. The soft penalty mitigates the practical impact of triangle inequality violations by discouraging the long edges where violations concentrate.

2. **Data coverage solutions** — Discuss potential solutions: (a) augmenting the graph with states from a learned dynamics model or diffusion model, (b) online data collection to fill coverage gaps, (c) using generative models to interpolate between sparse regions. Frame honestly as future work that could extend TTGS to manipulation domains.

3. **Hyperparameter guidelines** — Same heuristic as VfQS response + reference Table 4 (fixed hyperparameters across domains showing robustness).

---

### Response to 1VSa (Priority: MEDIUM — maintain accept)

1. **Narrow task range** — Move manipulation results to main paper. Honestly scope the contribution to navigation/locomotion with safe default behavior elsewhere.

2. **Dataset at test time** — Clarify: only M=4000 sampled states are needed at test time, not the full dataset. The graph is precomputed once and stored. Memory footprint is negligible (4000 state vectors + distance matrix).

3. **Scaling** — Experiment 3 + Table 8 reference.

4. **Asymmetric comparison with GAS/CompDiffuser** — Add explicit note in revision: "GAS trains a Temporal Distance Representation and CompDiffuser trains a diffusion model — these are not test-time-only methods. We include them as points of reference for the performance achievable with additional training." Reference Table 4: TTGS with fixed hyperparameters outperforms GAS on humanoidmaze and pointmaze.

5. **Incremental over SoRB** — Key differences: (a) SoRB requires *online interaction* to validate edges — fundamentally different setting, (b) SoRB uses hard distance thresholds that fragment the graph (our Appendix E shows this fails), (c) TTGS's soft penalty is a novel contribution that solves the graph connectivity vs. reliability tradeoff. These change the applicable problem setting, not just engineering details.

6. **More ablations** — Reference Experiments 2-5.

---

## Paper Revision Plan

### Main Paper Changes

1. **Move manipulation results to main text** — Table 5 + condensed Figure 5 into Section 4, with honest discussion of when/why TTGS's gains are limited
2. **Add baseline method descriptions** — 1-2 sentences per method in Section 4 (HIQL: hierarchical with latent subgoals; QRL: quasimetric distances; GCIQL: implicit Q-learning for goals; SAW: flattened hierarchy with policy bootstrapping; OTA: option-aware temporal abstraction)
3. **Tighten vague terminology** — Define "locally consistent" precisely; replace "latent capabilities" with concrete language
4. **Clarify Section 2 norm** — State explicitly that ||s-g|| defines the GCRL reward, not TTGS's distance metric. In visual domains, the reward may use privileged state during training, but TTGS only needs V(s,g) at test time
5. **Expand related work** — Add paragraph on classical/motion planning (PRM, RRT, A*) and position TTGS relative to these
6. **Sharpen SoRB distinction** — Be precise: SoRB requires online interaction + distributional ensembles; TTGS works purely at test time with any frozen GCRL agent + standard value functions
7. **Add hyperparameter selection heuristic** in Section 4.2 or appendix
8. **Note asymmetric comparison** with GAS/CompDiffuser explicitly

### New Appendix Content (from experiments)

- Geometric baseline comparison (Experiment 1)
- Failure mode categorization table (Experiment 2)
- Scaling plots: time and performance vs. M (Experiment 3)
- Metric property analysis (Experiment 4)
- Degradation analysis on easy tasks (Experiment 5)
- Weak value function robustness (Experiment 6, if time permits)

### Notation Fixes

- "i.e," -> "i.e.," in Section 2
- Fix mismatched sampling distribution notation (tau vs pi_beta)
