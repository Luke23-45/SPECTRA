# B-PGS: Claims, Proof Status, and Paper Obligations

## 1. Main claim table

| Claim | Type | Status | What is required | Safe manuscript wording |
|---|---|---|---|---|
| B-PGS introduces a bounded uncertainty chart for task-wise log-variance | mathematical | Proved | include Proposition 1 | "B-PGS uses a smooth bounded reparameterization of task log-variance." |
| B-PGS induces bounded positive precision weights for finite uncertainty coordinates | mathematical | Proved | include Theorem 1 from the canonical definition | "For finite uncertainty coordinates, the induced precision weights remain in a fixed positive interval." |
| B-PGS structurally excludes exact precision collapse to zero within its chart | mathematical | Proved | state the consequence of Theorem 1 carefully | "Within the bounded chart, the learned precision weight cannot become exactly zero for any finite uncertainty coordinate." |
| B-PGS uses split optimization for network and uncertainty variables | algorithmic | Proved | specify stop-gradient plus two-stage update rule | "B-PGS updates network and uncertainty variables through separate optimization passes." |
| The scalar uncertainty subproblem has an explicit stationary-point equation for positive fixed surrogate value | mathematical | Proved | include fixed-surrogate derivative calculation and the zero-surrogate edge case | "For fixed positive surrogate value, the scalar uncertainty objective admits an explicit stationary-point condition." |
| B-PGS is novel relative to unconstrained Kendall weighting | positioning | Literature positioning required | careful comparison section and precise wording | "B-PGS differs from classical homoscedastic uncertainty weighting by combining bounded reparameterization with split optimization." |
| B-PGS is more stable than unconstrained Kendall weighting | empirical-plus-theoretical | Partially Supported | synthetic and standard-benchmark evidence under fair settings | "B-PGS is designed to improve optimization control relative to unconstrained uncertainty weighting, and this should be evaluated empirically." |
| B-PGS outperforms alternative uncertainty-weighting methods | empirical | Not Yet Supported | fair baseline study, multi-seed results, standard datasets | "We evaluate whether B-PGS improves over established uncertainty-weighting baselines." |
| B-PGS is superior to analytical alternatives such as Kirchdorfer-style approaches | empirical and conceptual | Not Yet Supported | direct comparison plus clear novelty articulation | "B-PGS should be compared directly against recent analytical alternatives before stronger superiority claims are made." |

## 2. Proof status

### Claims currently safe as proved

The following are safe to present as formal results in the manuscript:

1. bounded range of $s_i(\theta_i)$ for finite $\theta_i$,
2. bounded positive range of $\omega_i(\theta_i)$ for finite $\theta_i$,
3. stationary-point equation for the scalar uncertainty subproblem at fixed positive surrogate value, together with the zero-surrogate edge case,
4. finite initialization via clipped inverse chart,
5. gradient-path separation under the stated split optimization rule.

### Claims that must remain partial

The following should be described conservatively:

1. optimization stability improvements,
2. practical robustness under noisy training,
3. better behavior on highly imbalanced tasks,
4. stronger performance than unconstrained uncertainty weighting.

These can be motivated mathematically, but the paper should still rely on experiments for confirmation.

### Claims that are not theorem statuses

The following are literature-positioning claims rather than mathematical theorems:

1. whether B-PGS is meaningfully distinct from bounded variants of Kendall-style weighting,
2. whether the split optimization view constitutes enough novelty for publication on its own,
3. whether the method is sufficiently differentiated from recent analytical alternatives.

These require careful related-work analysis and fair empirical comparison, not formal proof alone.

### Claims that are not currently paper-safe

Do not state any of the following as established facts:

1. global convergence of the neural training system,
2. impossibility of optimization failure in the broader network dynamics,
3. universal superiority over competing weighting methods,
4. state-of-the-art status,
5. exact equivalence between the pure method and engineering-enhanced variants.

## 3. What must be shown empirically

The paper still needs an empirical package that matches the pure definition.

### Minimum empirical obligations

1. Head-to-head comparison of pure B-PGS against pure unconstrained Kendall weighting under matched training conditions.
2. At least one comparison against a recent analytical or aggregation-based uncertainty-weighting alternative.
3. Multi-seed reporting on at least one standard benchmark, not only synthetic stress tests.
4. A fairness statement confirming that B-PGS did not receive engineering-only advantages in the main comparison.

### Strongly recommended ablations

1. bounded chart versus unbounded uncertainty optimization,
2. split optimization versus single-loss coupled optimization,
3. optional engineering enhancements versus the pure implementation,
4. sensitivity to the interval endpoints $(a_i,b_i)$.

### Evidence needed for stronger claims

If the manuscript wants to claim improved stability, it should show at least one of:

- lower variance in task weights,
- fewer divergent runs,
- more stable loss trajectories,
- or more reliable optimization across seeds.

If the manuscript wants to claim improved performance, it needs standard benchmark results with seed aggregation and a fair baseline protocol.

## 4. What must not be claimed

The manuscript must not claim:

- "B-PGS solves uncertainty weighting,"
- "B-PGS guarantees no starvation,"
- "B-PGS is strictly superior to all alternatives,"
- "B-PGS has a complete end-to-end proof for deep learning optimization,"
- "B-PGS is more than a bounded Kendall variant" without explaining precisely what extra ingredient is being claimed.

Safe alternatives:

- "B-PGS is a bounded and split-optimization variant of uncertainty weighting."
- "B-PGS structurally excludes exact precision collapse to zero within its bounded uncertainty chart."
- "B-PGS is proposed as a new uncertainty-weighting method aimed at improving optimization control."
- "The method has directly provable boundedness properties and empirically testable optimization consequences."

## 5. Reviewer attack surface

### Objection 1: "This is just bounded Kendall."

**Safe response strategy:**  
Admit the inheritance from Kendall directly. Then state the exact novelty boundary: B-PGS combines bounded reparameterization with split optimization, and the paper studies whether that combination changes the optimization behavior in meaningful ways.

**Evidence requirement:**  
Need a direct conceptual comparison section plus an ablation that separates boundedness from split optimization if possible.

### Objection 2: "Decoupling is algorithmic, not probabilistic."

**Safe response strategy:**  
Agree with the characterization. Do not oversell the decoupling as a new probabilistic model. Present it as an algorithmic design choice applied to uncertainty-weighted optimization, stated through the surrogate state $m$ and the split update rule.

**Evidence requirement:**  
Need explicit algorithm statement and computational-graph explanation in the paper.

### Objection 3: "Why should boundedness improve optimization quality?"

**Safe response strategy:**  
Say boundedness gives direct control over the admissible precision interval and prevents the uncertainty coordinates from representing arbitrarily extreme weights. Then let experiments test whether that control improves training behavior.

**Evidence requirement:**  
Need bounded-versus-unbounded comparisons and trajectory visualizations.

### Objection 4: "Why compare to baselines without the same engineering extras?"

**Safe response strategy:**  
This objection should be neutralized by design. The main comparison must use the pure method for B-PGS and comparably plain implementations for the baselines.

**Evidence requirement:**  
Need an explicit fairness statement in the methods section and a clean baseline implementation table.

## 6. Relationship to recent literature

### Kendall et al. 2018

This is the primary reference for the classical homoscedastic uncertainty-weighted objective. B-PGS should be presented as building on this framework rather than replacing it historically.

### Kirchdorfer et al.

This line of work is relevant because it questions standard uncertainty weighting and proposes analytical alternatives. It should be used to sharpen the novelty boundary and motivate direct comparison, not to imply automatic superiority.

### Achituve et al.

This work is relevant if the manuscript discusses uncertainty-aware gradient aggregation or related optimization views. If it is cited, the citation should match that narrower role and should not be used to overstate what B-PGS proves.

## 7. Current manuscript-safe position

At this stage, the safest paper position is:

"B-PGS is a new bounded and split-optimization uncertainty-weighting method. Its core boundedness properties, exclusion of exact precision collapse within the chart, and stationary-point structure can be stated and proved directly. Its practical value relative to unconstrained and recent alternative methods must then be established empirically under fair conditions."
