# B-PGS: Canonical Formal Definition

## 1. Motivation and relation to classical uncertainty weighting

Bayesian Projected Gradient Scaling (B-PGS) is proposed as a bounded and decoupled variant of homoscedastic uncertainty weighting for multi-task learning. The method is motivated by two design goals:

1. keep the learned uncertainty variables inside a controlled range through a smooth reparameterization, and
2. separate the update of network parameters from the update of uncertainty parameters.

The intended contribution is not a claim that all uncertainty-weighting pathologies are solved in full generality. The intended contribution is a precise mathematical method whose bounded chart and split update structure can be analyzed directly and compared fairly against existing alternatives.

## 2. Problem setup and notation

Let $T \ge 1$ be the number of tasks. Let $w \in \mathcal{W}$ denote the model parameters. For each task $i \in \{1,\dots,T\}$, let
$$
\ell_i : \mathcal{W} \to \mathbb{R}_{\ge 0}
$$
be a non-negative task loss.

For each task $i$, B-PGS introduces an unconstrained uncertainty coordinate
$$
\theta_i \in \mathbb{R}.
$$
Let $\theta = (\theta_1,\dots,\theta_T) \in \mathbb{R}^T$.

Fix task-wise bounds
$$
a_i < b_i.
$$
Let $\sigma : \mathbb{R} \to (0,1)$ be a smooth strictly increasing map. In the canonical instance,
$$
\sigma(x) = \frac{1}{1+e^{-x}}.
$$

Define the bounded log-variance map
$$
s_i(\theta_i) = a_i + (b_i-a_i)\sigma(\theta_i),
$$
and the induced precision map
$$
\omega_i(\theta_i) = e^{-s_i(\theta_i)}.
$$

Throughout the canonical definition, the standing assumption is
$$
\ell_i(w) \ge 0 \quad \text{for all } i \text{ and } w.
$$
Under this assumption, the default positive transform is simply
$$
\rho(x) = x.
$$
If a task family admits signed objectives or otherwise violates non-negativity, then a strictly positive surrogate may be introduced later, but that is not part of the canonical definition.

## 3. Classical Kendall objective

The classical homoscedastic uncertainty-weighted objective of Kendall, Gal, and Cipolla (2018) can be written in log-variance form as
$$
\mathcal{L}_{\mathrm{K}}(w,s)
=
\sum_{i=1}^{T}
\left[
\frac{1}{2} e^{-s_i}\ell_i(w)
+
\frac{1}{2}s_i
\right].
$$

In that formulation, the $s_i$ are unconstrained variables. B-PGS keeps the same basic uncertainty-weighting template but modifies both the parameterization and the optimization procedure.

## 4. B-PGS core idea

B-PGS consists of two ingredients.

First, the uncertainty variable is not optimized directly in $s$-space. Instead, an unconstrained coordinate $\theta_i$ is mapped into a bounded open interval $(a_i,b_i)$ through a smooth monotone chart.

Second, B-PGS uses split optimization:

- the network update uses detached precision weights, and
- the uncertainty update uses the current task-loss magnitudes as fixed inputs to a separate uncertainty objective.

This makes B-PGS a bounded and algorithmically decoupled uncertainty-weighting method.

## 5. Formal definition of the bounded uncertainty chart

### Definition 1. Bounded chart

For each task $i$, define
$$
s_i(\theta_i) = a_i + (b_i-a_i)\sigma(\theta_i).
$$

The image of this map is contained in the open interval $(a_i,b_i)$ for every finite $\theta_i$.

### Definition 2. Precision map

For each task $i$, define
$$
\omega_i(\theta_i) = e^{-s_i(\theta_i)}.
$$

Because $s_i(\theta_i)$ is bounded for finite $\theta_i$, the corresponding precision remains strictly positive and bounded.

### Definition 3. Safe inverse initialization

Given a target initial log-variance value $s_i^{(0)} \in [a_i,b_i]$, define
$$
p_i = \frac{s_i^{(0)} - a_i}{b_i-a_i}.
$$
Choose a clipping constant $\varepsilon_{\mathrm{clip}}$ with
$$
0 < \varepsilon_{\mathrm{clip}} < \frac{1}{2},
$$
and define
$$
\tilde p_i = \min(1-\varepsilon_{\mathrm{clip}}, \max(\varepsilon_{\mathrm{clip}}, p_i)).
$$
The initialized uncertainty coordinate is
$$
\theta_i^{(0)} = \log\left(\frac{\tilde p_i}{1-\tilde p_i}\right).
$$

This gives a finite initialization in $\theta$-space even when $s_i^{(0)}$ lies at a boundary value.

## 6. Formal definition of the decoupled objectives

### Definition 4. Network objective

For fixed $\theta$, define
$$
J_{\mathrm{net}}(w \mid \theta)
=
\sum_{i=1}^{T}
\frac{1}{2}\,\operatorname{sg}\!\bigl[\omega_i(\theta_i)\bigr]\ell_i(w),
$$
where $\operatorname{sg}[\cdot]$ denotes stop-gradient.

This objective is used only for updating the model parameters $w$.

### Definition 5. Uncertainty objective

Let
$$
m = (m_1,\dots,m_T) \in \mathbb{R}_{\ge 0}^T
$$
be a surrogate statistic of task difficulty. In the simplest canonical case,
$$
m_i = \ell_i(w).
$$
In later implementation variants, $m_i$ may be replaced by another non-negative statistic derived from task-loss magnitudes, but that is not required by the canonical method.

For fixed surrogate state $m$, define
$$
J_{\mathrm{unc}}(\theta \mid m)
=
\sum_{i=1}^{T}
\left[
\frac{1}{2}\,\omega_i(\theta_i)\rho(m_i)
+
\frac{1}{2}\,s_i(\theta_i)
\right].
$$

Under the standing non-negativity assumption, the canonical choice is $\rho(m_i) = m_i$. If we further take $m_i = \ell_i(w)$, then
$$
J_{\mathrm{unc}}(\theta \mid w)
=
\sum_{i=1}^{T}
\left[
\frac{1}{2}\,\omega_i(\theta_i)\ell_i(w)
+
\frac{1}{2}\,s_i(\theta_i)
\right].
$$

This objective is used only for updating the uncertainty coordinates $\theta$.

## 7. Ideal update rule

The canonical B-PGS update is discrete-time and proceeds in two stages at iteration $t$.

### Step A. Network update

Given $(w_t,\theta_t)$, update the model parameters by applying one optimizer step to
$$
J_{\mathrm{net}}(w \mid \theta_t).
$$
This yields
$$
w_{t+1} = \mathcal{U}_w\!\left(w_t; \nabla_w J_{\mathrm{net}}(w_t \mid \theta_t)\right),
$$
for some chosen optimizer map $\mathcal{U}_w$.

### Step B. Uncertainty update

Choose a surrogate state $m_t$ for the uncertainty update. In the canonical method,
$$
m_{i,t} = \ell_i(w_t).
$$
Then update the uncertainty variables by applying one optimizer step to
$$
J_{\mathrm{unc}}(\theta \mid m_t).
$$
This yields
$$
\theta_{t+1} = \mathcal{U}_\theta\!\left(\theta_t; \nabla_\theta J_{\mathrm{unc}}(\theta_t \mid m_t)\right),
$$
where $m_t$ is treated as fixed during that update.

The essential point is not the choice of optimizer family but the split optimization structure together with a clearly specified non-negative surrogate state.

### Continuous-time interpretation (intuition only)

The discrete B-PGS algorithm may be interpreted informally as a pair of separated flows:

- a network flow for $w$ under frozen precision weights, and
- an uncertainty flow for $\theta$ under frozen task-loss magnitudes.

This interpretation is for intuition only. The canonical object remains the discrete-time update rule above.

## 8. Propositions and proofs

### Proposition 1. Bounded range of the log-variance chart

For every finite $\theta_i \in \mathbb{R}$,
$$
s_i(\theta_i) \in (a_i,b_i).
$$

**Proof.**
Since $\sigma(\theta_i) \in (0,1)$ for every finite $\theta_i$, we have
$$
0 < \sigma(\theta_i) < 1.
$$
Multiplying by $(b_i-a_i) > 0$ and adding $a_i$ gives
$$
a_i < a_i + (b_i-a_i)\sigma(\theta_i) < b_i.
$$
That expression is exactly $s_i(\theta_i)$. $\square$

### Theorem 1. Structural boundedness of the precision map

For every finite $\theta_i \in \mathbb{R}$,
$$
e^{-b_i} < \omega_i(\theta_i) < e^{-a_i}.
$$

**Proof.**
By Proposition 1, $a_i < s_i(\theta_i) < b_i$. Since the map $x \mapsto e^{-x}$ is strictly decreasing and strictly positive,
$$
e^{-b_i} < e^{-s_i(\theta_i)} < e^{-a_i}.
$$
By definition, $e^{-s_i(\theta_i)} = \omega_i(\theta_i)$. $\square$

**Consequence.**
Under the B-PGS chart, exact precision collapse to zero cannot occur for any finite uncertainty coordinate. This is a statement about the uncertainty weight itself, not a theorem about all possible causes of task-level optimization failure.

### Proposition 3. Stationary-point equation for the scalar uncertainty subproblem

Fix a task $i$ and fix a surrogate value $m_i \ge 0$. Under the canonical choice $\rho(m_i)=m_i$, define
$$
f_i(s) = \frac{1}{2}e^{-s}m_i + \frac{1}{2}s.
$$
Then:

- if $m_i > 0$, any stationary point of $f_i$ in $s$ satisfies
$$
e^{-s_i^\star}m_i = 1.
$$
- if $m_i > 0$, this is equivalent to
$$
s_i^\star = \log m_i.
$$
- if $m_i = 0$, then $f_i(s) = \frac{1}{2}s$ and there is no finite stationary point.

**Proof.**
Differentiate:
$$
f_i'(s) = -\frac{1}{2}e^{-s}m_i + \frac{1}{2}
=
\frac{1}{2}\left(1 - e^{-s}m_i\right).
$$
At a stationary point, $f_i'(s)=0$, hence
$$
1 - e^{-s_i^\star}m_i = 0,
$$
which yields
$$
e^{-s_i^\star}m_i = 1.
$$
If $m_i>0$, taking logs gives $s_i^\star = \log m_i$.

If $m_i=0$, then
$$
f_i'(s) = \frac{1}{2} > 0,
$$
so no finite stationary point exists. $\square$

**Boundary note.**
If $\log m_i \notin (a_i,b_i)$, then the stationary point of the unrestricted scalar objective lies outside the range of the chart. In that case, the induced optimization in $\theta_i$ should be described in terms of an infimum approached at the chart boundary, not as a finite minimizer in $\theta_i$.

### Proposition 4. Finite inverse-chart initialization

For every $s_i^{(0)} \in [a_i,b_i]$, the clipped inverse-chart rule above produces a finite value $\theta_i^{(0)} \in \mathbb{R}$.

**Proof.**
By construction,
$$
\varepsilon_{\mathrm{clip}} \le \tilde p_i \le 1-\varepsilon_{\mathrm{clip}},
$$
with $0 < \varepsilon_{\mathrm{clip}} < \frac{1}{2}$. Therefore
$$
0 < \tilde p_i < 1,
$$
so both $\tilde p_i$ and $1-\tilde p_i$ are strictly positive finite real numbers. Hence
$$
\log\left(\frac{\tilde p_i}{1-\tilde p_i}\right)
$$
is finite. $\square$

### Theorem 2. Gradient-path separation under split optimization

Under the intended split optimization procedure:

- $J_{\mathrm{net}}$ contributes gradients to $w$ but not to $\theta$, because $\omega_i(\theta_i)$ is stop-gradient in that objective;
- $J_{\mathrm{unc}}$ contributes gradients to $\theta$ through $s_i(\theta_i)$ and $\omega_i(\theta_i)$, while the surrogate state $m_i$ is treated as a fixed input during that update.

**Proof sketch.**
In $J_{\mathrm{net}}$, the factor $\operatorname{sg}[\omega_i(\theta_i)]$ is treated as constant during backpropagation, so its derivative with respect to $\theta_i$ is zero on that pass. The remaining dependence is through $\ell_i(w)$, hence the gradient flows to $w$.

In $J_{\mathrm{unc}}$, the dependence on $\theta_i$ is explicit through $s_i(\theta_i)$ and $\omega_i(\theta_i)$. Under the split procedure, the surrogate state $m_i$ is treated as fixed during the uncertainty update, so the intended gradient path is to $\theta$. $\square$

## 9. Scope and non-claims

This document defines the canonical mathematical object of B-PGS.

We rigorously claim:

- bounded log-variance and bounded positive precision for finite uncertainty coordinates,
- exclusion of exact precision collapse to zero within the bounded chart,
- gradient-path separation under the stated split optimization rule.

We do not claim:

- full convergence of the nonconvex neural-network training problem,
- universal superiority over all uncertainty-weighting alternatives,
- exact global optimality of the full split training system,
- impossibility of optimization slowdown or failure in downstream applications,
- "guaranteed no starvation" as a theorem about all optimization dynamics.

The clean mathematical definition should be separated from engineering stabilizers, implementation heuristics, and numerical guards. Those may be useful in practice, but they are not part of the core paper claim.

## 10. References

1. Alex Kendall, Yarin Gal, and Roberto Cipolla. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
2. Lukas Kirchdorfer, Tobias Sesterhenn, Christian Bartelt, Heiner Stuckenschmidt, Lukas Schott, and Jan M. Kohler. *Investigating Uncertainty Weighting for Multi-Task Learning: Insights and Analytical Alternative.* International Journal of Computer Vision, 134(8), 2026. Published online December 23, 2025.
3. Idan Achituve, Idit Diamant, Arnon Netzer, Gal Chechik, and Ethan Fetaya. *Bayesian Uncertainty for Gradient Aggregation in Multi-Task Learning.* Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR 235:117-134, 2024.
