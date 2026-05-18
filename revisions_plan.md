# Camera-Ready Revisions Plan

This document enumerates every revision required for the camera-ready of *Test-Time Graph Search for Goal-Conditioned Reinforcement Learning* (Submission 26883), based on the AC's Paper Decision and the four reviewer threads (K7zo, Zg7J, 1VSa, VfQS). Each item lists: (1) the section / location to edit in [main.tex](main.tex), (2) what the issue is and which reviewer raised it, and (3) the concrete change. Tables and numbers come from [rebuttals.md](rebuttals.md), [replies.md](replies.md), and [replies2.md](replies2.md).

The AC's explicit camera-ready checklist is the spine of this plan:
1. Move manipulation results to the main text.
2. Narrow the framing throughout (abstract, intro, conclusion).
3. Incorporate the hop-ratio predictor and the L2-kNN baseline.
4. Make the evaluation-only nature of $\|s-g\|<\epsilon$ explicit in Section 2.
5. Expand the related work section to engage with the relevant planning literature.

---

## Narrow framing

**Change:**
- clarify in the abstract that the primary gains are in long-horizon locomotion
- In the introduction, state that TTGS is most impactful when (1) the task requires long-horizon stitching, (2) the base policy is locally reliable, and (3) the dataset covers intermediate states. note that these conditions hold for navigation but not for the tested manipulation tasks. Replace "unlocks latent capabilities in diverse base learners".
- In the conclusion, replace broad language with a precise summary: TTGS provides substantial navigation improvements and defaults to base policy behavior when its conditions are not met.

---

## Section 2 (Preliminaries)

### Make $\|s-g\|<\epsilon$ evaluation-only and explain hindsight relabeling (AC, Zg7J)

**Issue:** Zg7J's central concern: Section 2 introduces $\|s-g\|<\epsilon$ as if it were a training input, suggesting privileged access to a state-space metric. The AC explicitly flagged this for the camera-ready: "the manuscript must carry this clarification in the final version, not the discussion."

**Change (target: 2–3 sentences in Section 2; full derivation goes to Appendix B):**
- After the optimal-policy equation, add a short paragraph stating: $r=\mathbf{1}\{\|s-g\|<\epsilon\}$ is the **benchmark's evaluation criterion only** — it is never accessed during training or by TTGS at inference. Base learners train via **hindsight relabeling** using dataset trajectory indices and a per-step penalty (see Appendix B), so $V(s,g)$ encodes a step-distance signal as an *output* of training, not an input. The pipeline requires only (i) an offline dataset of state-action sequences and (ii) a value function trained via hindsight relabeling.
- In Appendix B (where the per-reward-convention $V \to \hat d$ mappings already live), add the TD-learning derivation showing $V(s_t,g)\to-\sum_{i=0}^{k-1}\gamma^i$ emerges from trajectory structure alone, as a concrete demonstration that no norm is computed during training.

### Tighten "locally consistent geometric structure" (Zg7J)

**Note:** Zg7J flagged this as "Minor" in the review. Scoped down to the one definition that actually carries weight; the "latent capabilities" and "global errors" rewordings are cosmetic and can be skipped if the rest of the framing edits already make the contribution precise.

**Change:**
- Define "locally consistent geometric structure" precisely on first use (Section 3.1): value-derived distances are approximately correct for nearby state pairs, even when they are unreliable at longer horizons. (If introducing the term before $\tau$ is formally defined, use "for nearby state pairs"; otherwise tie it to "within the trust region $\tau$".)
- *Optional:* replace "latent capabilities" once on first use with a concrete description if the new framing in Abstract/Intro hasn't already eliminated the phrase. Replace "global errors" → "long-horizon estimation errors" if it sounds better than the current phrasing.

### Section 2 typos (VfQS)

**Location:** [main.tex:151-168](main.tex#L151-L168).

**Change:**
- Fix `i.e,` → `i.e.,` (per VfQS misc note).
- Fix the mismatched expectation/sampling notation: the equation writes $\tau\sim p(s\mid g)$ but the surrounding text uses $p(\tau\mid g)$. Use $\tau\sim p(\tau\mid g)$ consistently. ([main.tex:162](main.tex#L162))

### Add brief baseline descriptions and full names (Zg7J)

**Location:** currently baselines are introduced only by acronym at [main.tex:305](main.tex#L305). Add descriptions with full names to the appendix.

**Change:**
- Add a short paragraph (one sentence per learner) introducing HIQL, QRL, GCIQL, SAW, OTA with full names and a one-line description of how each constructs its value function. Note the selection rationale: HIQL/QRL/GCIQL because they appear in the OGBench evaluation; SAW/OTA for their strong reported performance.

---

## Section 3 (Method) — triangle inequality discussion

**Location:** [main.tex:201-207](main.tex#L201-L207).

**Issue:** K7zo asked whether value-derived distances violate metric properties in practice. Currently the paper just notes the issue qualitatively at line 206.

**Change:**
- Replace the qualitative note with a forward pointer to a new appendix subsection (see §10.1 below) and summarize one sentence of the result inline: violation rates within the trust region are 0.2–13.4% with mean violation magnitude of 0.002–3.1 steps, vs 575–588 unconstrained — i.e., the non-metricity does not materially distort shortest-path planning.

---

## Section 4 (Experiments) — promotions and new content

### Move manipulation results to the main paper (AC, Zg7J, 1VSa)

**Location:** Currently in Appendix C, [main.tex:768-815](main.tex#L768-L815). Should land at the end of Section 4.1 (Main Results) or as a new subsection 4.3.

**Change:**
- Move the manipulation **table** and the surrounding analysis from Appendix C into Section 4 with a heading like "Manipulation tasks: where TTGS does not help, and why."
- **Keep [Figure 5 (manip_viz)](main.tex#L771-L793) in Appendix C.** The figure is large and main-text space is tight; the structural argument (evaluation goals lie outside the data manifold, no intermediate states bridge the gap) can be summarized in 1–2 sentences in the main text with a forward pointer to the figure in the appendix.
- Frame the result honestly: TTGS does not degrade performance, but evaluation goals lie outside the data manifold; this is a structural limitation, not a tuning issue.

### Add the L2-kNN baseline (AC, VfQS)

**Location:** Brief paragraph in the main-text experiments section; full per-$k$ sweep table in the appendix (see "L2-kNN baseline — full per-$k$ sweep" below). The AC required incorporation but did not specify placement; we mirror the manipulation/hop-ratio pattern (concept and headline numbers in main text, full table in appendix).

**Issue:** VfQS asked for a kNN landmark baseline using purely geometric distance + Dijkstra to isolate the contribution of value-derived distances and the soft penalty. The AC called this experiment out specifically: "this experiment cleanly isolates that TTGS's contribution is how to do graph search over value-derived distances, not that graph search helps."

**Change (target: 2–3 sentences in main text + appendix table):**
- Main text: a training-free L2-kNN landmark baseline reaches a competitive peak (59.4% at $k{=}10$ on humanoidmaze-giant-stitch, 66.6% at $k{=}20$ on antmaze-giant-stitch) but the optimal $k$ differs across environments and the method is fragile — disconnections at low $k$ and wormhole-like shortcuts at high $k$. TTGS's soft penalty plus full-graph construction reaches 76.7% / 76.5% on both without per-environment $k$ selection. L2 also presupposes knowing which observation dimensions encode body position, which is unavailable in pixel-based domains. Forward-reference Appendix §10.3 for the full per-$k$ sweep.

### Add the M-sweep / scaling experiment (1VSa)

**Location:** Full table goes in [Appendix F (Runtime)](main.tex#L895-L924) (see "Scaling sweep — full table" below). Main text only carries a one-sentence reference.

**Issue:** 1VSa: "Computational scaling is not empirically validated... practical performance on large offline datasets is not shown."

**Change (main text, one sentence):** Building on the existing runtime discussion, note that we ran an $M$-sweep ($M\in\{125,\ldots,8000\}$, HIQL, 8 seeds) showing that build time grows quadratically but stays under 2 minutes at $M{=}8000$, and path computation stays under 1.5 s; full table in Appendix §10.5.

### Add the hop-ratio predictor (AC, 1VSa)

**Location:** New subsection in Section 4, either alongside the manipulation discussion or as a standalone "Predicting when TTGS helps" subsection. The AC specifically called this out as a camera-ready requirement.

**Issue:** 1VSa asked for a quantitative metric to predict in advance whether TTGS will help on a new task. We proposed the **largest hop ratio** (rollout-free):

$$\frac{\max\bigl(d(s_0, p_0), d(p_0, p_1), \ldots, d(p_{L-1}, p_L), d(p_L, g)\bigr)}{d(s_0, g)}$$

**Change:**
- **In the main text**, define the hop-ratio predictor (the formula above), explain the intuition (low ratio → every hop is small relative to the total task → TTGS regime; high ratio → some hop spans a large fraction of the distance → insufficient coverage), and summarize the empirical pattern in 1–2 sentences: manipulation hop ratios are 4–6× higher than navigation, confirming insufficient intermediate coverage; among locomotion tasks, giant mazes have the lowest ratios and the largest TTGS gains. Forward-reference the appendix for the full table.
- **Move the per-task table to the appendix** (the GCIQL, 8 seeds, 9-row table from [replies2.md:21-32](replies2.md)) — it is wide and main-text space is tight. See "Hop-ratio predictor — full table and detailed discussion" below.
- Keep the recommended procedure for new tasks in the main text (one sentence): compute the hop ratio as a rollout-free preliminary check (≈0.4 threshold); if it is low, run TTGS directly, since the method defaults safely to base policy behavior when its conditions are not met.
- Keep the second-factor caveat in the main text (one sentence): humanoidmaze-giant-stitch (low hop ratio, zero improvement) illustrates that hop ratio is necessary but not sufficient — base policy local competence is the second factor, evidenced by Figure 2c.

### Add subgoal sampling distribution disclosure (VfQS Q1)

**Location:** Appendix H (Hyperparameters), near the baseline descriptions.

**Change:** Add a short paragraph from [rebuttals.md:158](rebuttals.md): HIQL and GCIQL mix future-trajectory goals (50%, geometric sampling, mean ~100 steps), current-state goals (20%), and random dataset goals (30%); QRL uses 100% random goals; SAW and OTA were tuned by their authors for long-range goals on the benchmark. The point is that all baselines are already exposed to long-range goals during training, so TTGS's gains are complementary, not a consequence of poor goal-distance distributions.

---

## Section 5 (Related Work) — substantial expansion (AC, Zg7J)

**Location:** [main.tex:424-429](main.tex#L424-L429).

**Issue:** Zg7J asked for engagement with broader planning literature and pointed to specific prior work whose high-level structure resembles TTGS. The AC explicitly asked for engagement with **GSP**, **IM-DSG**, and **Bagatella et al. NeurIPS 2023** (https://openreview.net/forum?id=QlbZabgMdK), in addition to SoRB.

**Change:**
- Add a new paragraph that places TTGS within the prior tradition of graph-based planning over states with value-derived edge costs. Cite and discuss:
  - **SoRB** (Eysenbach et al., 2019) — already cited; emphasize that it operates online with a growing replay buffer, trains a specialized distributional distance function, and uses hard edge pruning.
  - **GSP / Goal-Space Planning with Subgoal Models** (Lo et al., 2024) — **NEW reference**; training-time method that learns four subgoal-conditioned models online and uses them for potential-based reward shaping.
  - **IM-DSG / Intrinsically Motivated Discovery of Temporally Abstract Graph-Based Models of the World** (Bagaria et al., 2025) — **NEW reference**; incrementally builds a skill graph through online exploration with novelty-driven expansion and optimistic edge probabilities.
  - **Bagatella et al. NeurIPS 2023** — **NEW reference**, https://openreview.net/forum?id=QlbZabgMdK; the AC noted this is even closer in spirit to our paper. (Look up the title and add to [main.bib](main.bib).)
- Finish the paragraph explaining what is novel about TTGS relative to this lineage (from [replies.md:18-19](replies.md)):
  1. Operates purely at test time with zero gradient-based training beyond the base agent's existing value function.
  2. Plug-and-play applicability on top of existing offline GCRL methods (five base learners with consistent gains).
  3. The soft-penalty mechanism, which is necessary precisely because TTGS cannot verify edges via models or online execution (hard thresholds cause up to 99% disconnections, [Appendix E](main.tex#L861)).
  4. The empirical demonstration that standard offline GCRL value functions already encode sufficient local geometric structure for effective planning, making (1)–(3) viable without specialized distance learning.
- Add a sentence positioning TTGS relative to **classical planning, motion planning, and TAMP** (from [rebuttals.md:60](rebuttals.md)): TTGS shares the graph-search structure but differs in that edge costs come from learned, noisy value functions rather than known dynamics or collision checkers — which is what motivates the soft penalty.
- **GAS / CompDiffuser asymmetry** (from [rebuttals.md:88](rebuttals.md), promised to 1VSa): no main-text changes — clarify the existing note in Appendix A. Make explicit that TTGS operates under strictly more restrictive constraints (no additional training, no generative models, no online interaction) yet achieves comparable performance, and that to the best of our knowledge TTGS is the first method that operates purely offline and at test time in the GCRL setting, so no direct baseline in this category exists.

Bib additions for this section are listed in the "Bibliography additions" block at the end of this plan.

---

## Section 6 (Limitations) — expand and tighten (1VSa, K7zo, VfQS)

**Location:** [main.tex:431-438](main.tex#L431-L438).

**Change:**
1. **Dataset access.** Currently silent on this. Add a short paragraph (from [rebuttals.md:117](rebuttals.md)): we require access to a small subset (~0.4% of the dataset, $M{=}4000$) of the original training data at test time, not the full dataset. When the policy is deployed by the same user/organization that trained it, this is immediate; otherwise a small sample or limited interaction is feasible.
2. **Conditions under which TTGS helps.** Restate the three preconditions explicitly.
3. **Coverage gaps and future directions.** From [rebuttals.md:22](rebuttals.md): when intermediate states are missing, two complementary directions could help: (i) improving value learning so existing intermediate states become recognized as reachable, and (ii) using generative models trained on offline data to synthesize additional states.
4. **Failure-mode characterization (qualitative).** Promised to K7zo, 1VSa, and VfQS in [rebuttals.md:22, 121, 166](rebuttals.md). The original 100-failed-episode tally was not rigorous enough for a quantitative claim (no formal protocol, single annotator, no inter-annotator agreement), so phrase the discussion **qualitatively**, not as a percentage. Suggested wording: "Inspecting failed navigation rollouts, we observe that the dominant failure mode is the base policy failing to execute a short hop near a subgoal (the agent gets stuck or overshoots), rather than the planner producing infeasible paths through walls or unreachable regions of the graph. On manipulation, evaluation goals lie outside the data manifold (Appendix C), and TTGS defaults to the base policy." Do not commit to "100 episodes" or "4 agents × 2 environments" counts in the manuscript.

---

## Section 7 (Conclusion) — narrow language (AC, 1VSa)

**Location:** [main.tex:440-447](main.tex#L440-L447).

**Change:**
- Replace broad claims like "substantially improves long-horizon performance across diverse tasks" and "diverse tasks" with task-specific language.
- Use the rebuttal summary from [replies2.md:7](replies2.md): "TTGS provides substantial navigation improvements and defaults to base policy behavior when its conditions are not met."
- Drop or tone down "broad applicability" in line 445.


---

## Hyperparameter heuristic (K7zo, VfQS)

**Location:** [Appendix H (Hyperparameters)](main.tex#L959) already has the heuristic; promote a brief version into the main text at the end of Section 4.2 or in a closing paragraph of Section 4.

**Change:**
- Add 2–3 sentences (from [rebuttals.md:26](rebuttals.md)):
  - Heuristic: set $\tau$ by visually inspecting the shortest paths on the graph. If paths are excessively fragmented into many short hops, $\tau$ is too small; if paths contain many unreliable long edges, $\tau$ is too large. Then set $T = 2\tau$.
  - Note that very large $\tau,T$ degenerate to (start, goal) and recover base learner performance, while very small values force excessive waypoints; we did not observe complete degradation in any experiment ([rebuttals.md:162](rebuttals.md)).

---

## Appendices

### New appendix section: triangle-inequality violations (K7zo)

**Change:** Add the table below from [rebuttals.md:11-16](rebuttals.md) (1M triplets globally and 1M triplets where all pairwise distances fall under $\tau$, 8 seeds):

| Base Learner | Dataset                    | Violation Rate (all) | Violation Rate (under $\tau$) | Mean Positive Violation (all) | Mean Positive Violation (under $\tau$) |
|--------------|----------------------------|----------------------|-------------------------------|-------------------------------|----------------------------------------|
| HIQL         | humanoidmaze-giant-stitch  | 0.2%                 | 13.4%                         | 585.7                         | 3.1                                    |
| HIQL         | antmaze-giant-stitch       | 1.3%                 | 9.1%                          | 587.8                         | 1.2                                    |
| QRL          | humanoidmaze-giant-stitch  | 0.7%                 | 0.3%                          | 0.001                         | 0.002                                  |
| QRL          | antmaze-giant-stitch       | 0.7%                 | 0.2%                          | 0.001                         | 0.002                                  |
| GCIQL        | humanoidmaze-giant-stitch  | 0.4%                 | 8.3%                          | 575.6                         | 1.2                                    |
| GCIQL        | antmaze-giant-stitch       | 5.6%                 | 8.2%                          | 574.9                         | 1.2                                    |

Discussion: violations occur within the trust region but are negligible in magnitude (0.002–3.1 steps vs 575–588 unconstrained). This is consistent with TTGS's strong empirical performance across all base learners despite their different violation profiles.

### Manipulation analysis (Appendix C)

**Location:** [main.tex:768-815](main.tex#L768-L815).

**Change:** The manipulation table moves to the main text (per the §4 manipulation subsection), but **Figure 5 (`manip_viz`) stays in this appendix** — the value-function-geometry visualization is large and there is no main-text space for it. Trim the appendix prose to avoid duplicating what is now in Section 4: keep the figure, its caption, the per-task breakdown if not in the main text, and the extended geometric discussion. The main text should forward-reference this appendix for the visualization.

### L2-kNN baseline — full per-$k$ sweep

**Location:** New appendix subsection.

**Change:** Carry the full table from [rebuttals.md:138-152](rebuttals.md) (HIQL, $M{=}4000$, 8 seeds), since the main text only carries a one-paragraph summary:

| Method        | k  | humanoidmaze-giant-stitch | Disconnect | antmaze-giant-stitch | Disconnect |
|---------------|----|---------------------------|------------|----------------------|------------|
| HIQL          | -- | 3.2 ± 0.5                 | --         | 1.6 ± 0.9            | --         |
| HIQL + TTGS   | -- | 76.7 ± 18.1               | 0.00       | 76.5 ± 5.2           | 0.00       |
| HIQL + L2-kNN | 2  | 3.2 ± 1.4                 | 1.00       | 1.6 ± 1.1            | 1.00       |
| HIQL + L2-kNN | 5  | 28.1 ± 20.6               | 0.64       | 1.6 ± 1.1            | 1.00       |
| HIQL + L2-kNN | 10 | 59.4 ± 15.3               | 0.00       | 29.2 ± 23.9          | 0.60       |
| HIQL + L2-kNN | 20 | 18.8 ± 13.3               | 0.00       | 66.6 ± 3.6           | 0.00       |
| HIQL + L2-kNN | 50 | 3.2 ± 3.9                 | 0.00       | 9.6 ± 7.0            | 0.00       |

Discussion to include in the appendix:
- Define **disconnect ratio**: fraction of evaluation episodes for which no path exists between the start and goal in the kNN graph.
- Describe the L2 distance: Euclidean distance between agent body positions, normalized by the average dataset step length (matches Section 4 / [Table 1](main.tex#L312-L340) handling).
- Note privileged-state caveat: this requires knowing which observation dimensions encode body position, which is unavailable in pixel-based domains; value-derived distances do not.
- Note the subgoal-selection procedure (using $T$) was reused, so this is a strict apples-to-apples comparison of edge-cost derivation, not a different planner.
- Reiterate the headline finding: optimal $k$ differs across environments ($k{=}10$ for humanoidmaze, $k{=}20$ for antmaze); TTGS's soft penalty + full-graph construction reaches 76–77% on both without per-environment tuning.

### Hop-ratio predictor — full table and detailed discussion

**Location:** Appendix.

**Change:** Carry the full per-task table from [replies2.md:21-32](replies2.md) (GCIQL, 8 seeds, sorted by hop ratio):

| Dataset                   | Hop Ratio | GCIQL | +TTGS | $\Delta$ (pp) |
|---------------------------|-----------|-------|-------|---------------|
| pointmaze-giant-stitch    | 0.041     | 0.0   | 98.0  | +98.0         |
| antmaze-giant-stitch      | 0.080     | 0.0   | 32.7  | +32.7         |
| humanoidmaze-giant-stitch | 0.112     | 0.2   | 0.2   | 0.0           |
| humanoidmaze-medium-stitch| 0.140     | 14.0  | 20.2  | +6.2          |
| antmaze-medium-stitch     | 0.207     | 29.5  | 53.0  | +23.5         |
| pointmaze-medium-stitch   | 0.217     | 18.2  | 44.0  | +25.8         |
| scene-play                | 0.458     | 50    | 52    | +2            |
| cube-triple-play          | 0.470     | 4     | 4     | 0             |
| puzzle-4x6-play           | 0.620     | 10    | 10    | 0             |

Discussion to include in the appendix (since the main text only has the formula + a 1–2 sentence summary):
- Restate the formal definition and the computation procedure (Dijkstra on the TTGS graph, then take the max edge cost along the path).
- Note that the metric is rollout-free: it only needs the offline dataset and value function.
- Manipulation hop ratios (0.458–0.620) are 4–6× higher than navigation (0.041–0.217), confirming insufficient intermediate coverage in those datasets.
- Among locomotion tasks, giant mazes have the lowest ratios and the largest TTGS gains.
- humanoidmaze-giant-stitch (low hop ratio, zero improvement) shows that hop ratio is necessary but not sufficient — base policy local competence is a separate, rollout-dependent factor (Figure 2c).
- State the recommended procedure for new tasks: (1) compute the hop ratio as a rollout-free preliminary check, threshold ≈0.4; (2) if low, run TTGS directly, since it defaults safely to base policy behavior when its conditions are not met.

### Scaling sweep — full table

**Location:** [Appendix F (Runtime)](main.tex#L895-L924).

**Change:** Carry the full table from [rebuttals.md:96-111](rebuttals.md) (HIQL, 8 seeds):

| M    | humanoidmaze-giant-stitch | Build (s)   | Path (s) | antmaze-giant-stitch | Build (s)  | Path (s) |
|------|---------------------------|-------------|----------|----------------------|------------|----------|
| 125  | 10.4 ± 10.0               | 7.4 ± 2.0   | 0.06     | 29.2 ± 8.3           | 9.3 ± 4.2  | 0.05     |
| 250  | 13.3 ± 11.6               | 5.9 ± 1.1   | 0.07     | 29.9 ± 12.4          | 8.2 ± 4.4  | 0.06     |
| 500  | 42.4 ± 22.9               | 5.0 ± 1.5   | 0.14     | 51.6 ± 12.1          | 3.7 ± 0.4  | 0.10     |
| 1000 | 60.0 ± 19.6               | 6.9 ± 1.7   | 0.24     | 53.6 ± 16.8          | 5.8 ± 1.4  | 0.20     |
| 2000 | 72.8 ± 13.0               | 10.7 ± 1.7  | 0.44     | 70.9 ± 10.6          | 10.2 ± 1.3 | 0.39     |
| 4000 | 76.7 ± 18.1               | 26.2 ± 1.5  | 0.84     | 76.5 ± 5.2           | 25.5 ± 0.9 | 0.69     |
| 8000 | 78.7 ± 12.8               | 99.8 ± 8.7  | 1.31     | 81.1 ± 2.5           | 88.6 ± 2.5 | 1.13     |

Discussion: success rate increases with $M$; build time grows quadratically but stays under 2 minutes at $M{=}8000$; path computation stays under 1.5 s.

---

## Bibliography additions (in [main.bib](main.bib))

- **Lo et al. 2024**, "Goal-Space Planning with Subgoal Models" — for §6.
- **Bagaria et al. 2025**, "Intrinsically Motivated Discovery of Temporally Abstract Graph-Based Models of the World" — for §6.
- **Bagatella et al. 2023 (NeurIPS)**, OpenReview id `QlbZabgMdK` — for §6 (AC's explicit ask). Look up exact title and authors before adding.
- (Optional, depending on planning-literature paragraph) classical planning, motion planning, or TAMP citations if a single representative reference is needed.

---

## Cross-checking pass before camera-ready

After all edits, verify:

**Framing consistency**
- [ ] Abstract, introduction, and conclusion all qualify the gains as long-horizon **locomotion** (not "diverse tasks") and acknowledge manipulation as a structural limitation.
- [ ] The three preconditions (long-horizon stitching needed; base policy locally reliable; dataset covers intermediate states) are stated in the introduction and re-stated in the limitations.

**Section 2 (Preliminaries)**
- [ ] $\|s-g\|<\epsilon$ is labeled as the benchmark's evaluation criterion only, not a training input.
- [ ] A short hindsight-relabeling paragraph is present in the main text, with the TD-learning derivation moved to Appendix B (no norm computed during training).
- [ ] Typos fixed: `i.e,` → `i.e.,`; expectation/sampling notation uses $\tau\sim p(\tau\mid g)$ consistently.

**Main-text new content**
- [ ] Manipulation table is in Section 4; Figure 5 (`manip_viz`) stays in Appendix C; main text forward-references the figure.
- [ ] L2-kNN headline numbers are in Section 4 (one short paragraph); full per-$k$ table in the appendix.
- [ ] Hop-ratio predictor: formula, intuition, recommended procedure, and the necessary-but-not-sufficient caveat are in Section 4; full per-task table in the appendix.
- [ ] M-sweep: one-sentence reference in Section 4; full table in Appendix F.
- [ ] Triangle-inequality summary (one sentence) is in Section 3 with a forward pointer to the appendix table.
- [ ] Hyperparameter heuristic ($\tau$ via path inspection, $T=2\tau$) is in the main text near the ablations.

**Related work and bibliography**
- [ ] Related-work paragraph engages SoRB, GSP, IM-DSG, and Bagatella et al. and characterizes the four novelty points (test-time-only; plug-and-play; soft-penalty necessity; local-geometry empirical evidence).
- [ ] Classical planning / motion planning / TAMP positioned in one sentence.
- [ ] All three new bib entries (Lo 2024; Bagaria 2025; Bagatella 2023) are added to [main.bib](main.bib) and `\cite` calls compile.

**Appendices**
- [ ] All four new tables (triangle violations, L2-kNN per-$k$, M-sweep, hop-ratio per-task) carry the exact numbers from [rebuttals.md](rebuttals.md) and [replies2.md](replies2.md).
- [ ] Appendix A (GAS / CompDiffuser) makes the asymmetry (no training, no generative models, no online interaction) explicit.
- [ ] Appendix C is trimmed to avoid duplicating the manipulation discussion now in the main text.
- [ ] Appendix H carries the subgoal-sampling-distribution disclosure.

**Limitations**
- [ ] Dataset access (~0.4% subset, $M{=}4000$) is stated.
- [ ] Qualitative failure-mode wording avoids "100 episodes" and per-agent/per-env counts.
