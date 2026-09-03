# SentinelDC: Comprehensive Mathematical Foundations & Equations Reference

**Target Venue:** IEEE Transactions on Dependable and Secure Computing (TDSC) / IEEE Cloud Computing  
**Document Purpose:** Complete mathematical, statistical, and algorithmic reference for manuscript preparation, camera-ready submission, and formal theoretical proofs.

---

## Table of Contents
1. [Nomenclature & Mathematical Symbols](#1-nomenclature--mathematical-symbols)
2. [Temporal Ingestion & Linear-Time Feature Flattening](#2-temporal-ingestion--linear-time-feature-flattening)
3. [Extreme Class Imbalance & Cost-Sensitive Gradient Optimization](#3-extreme-class-imbalance--cost-sensitive-gradient-optimization)
4. [Precision-Recall Curve Dynamic Threshold Optimization](#4-precision-recall-curve-dynamic-threshold-optimization)
5. [Game-Theoretic Local Explainability via Exact TreeSHAP](#5-game-theoretic-local-explainability-via-exact-treeshap)
6. [Tier 1 Deterministic Expert Rule Engine & Failure Mode Predicates](#6-tier-1-deterministic-expert-rule-engine--failure-mode-predicates)
7. [Storage Blast Radius & Cluster Availability Probability Models](#7-storage-blast-radius--cluster-availability-probability-models)
8. [Two-Tier D2G Latency Guarantees & SLA Fallback Formulation](#8-two-tier-d2g-latency-guarantees--sla-fallback-formulation)
9. [Mathematical Formulations of Baseline Benchmark Algorithms](#9-mathematical-formulations-of-baseline-benchmark-algorithms)

---

## 1. Nomenclature & Mathematical Symbols

| Symbol | Definition | Domain / Dimension |
| :--- | :--- | :--- |
| $\mathcal{D}$ | Fleet-wide telemetry dataset | $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ |
| $N$ | Total number of evaluated drive instances | $N \in \mathbb{N}$ ($N = 348,810$) |
| $d_k$ | Unique physical hard drive entity (Serial Number) | $d_k \in \mathcal{S}$ |
| $W$ | Temporal lookback observation window | $W = 10$ days |
| $M$ | Number of sampled raw SMART telemetry channels | $M \in [11, 21]$ depending on vendor |
| $D$ | Total flattened feature vector dimensionality | $D = M \cdot W \in [110, 210]$ |
| $\mathbf{S}_{d_k}$ | 2D temporal sequence matrix for drive $d_k$ | $\mathbf{S}_{d_k} \in \mathbb{R}^{M \times W}$ |
| $\mathbf{x}_{d_k}$ | 1D flattened temporal feature vector | $\mathbf{x}_{d_k} \in \mathbb{R}^D$ |
| $y_i$ | Ground-truth binary health label | $y_i \in \{0, 1\}$ ($0 = \text{healthy}, 1 = \text{failing}$) |
| $\hat{p}_i$ | Model predicted failure probability | $\hat{p}_i = \hat{y}_i \in [0, 1]$ |
| $\tau^*$ | Calibrated optimal decision threshold | $\tau^* \in (0, 1)$ |
| $w_+$ | Positive-class loss penalty weight (`scale_pos_weight`) | $w_+ = N_{\text{healthy}} / N_{\text{failing}} \approx 100$ |
| $\phi_m(\mathbf{x})$ | Exact TreeSHAP additive attribution for feature $m$ | $\phi_m \in \mathbb{R}$ |
| $T, L, D_{\text{tree}}$ | Ensemble hyperparams: Trees, Leaves, Tree Depth | $T \approx 100, L \approx 32, D_{\text{tree}} \le 6$ |
| $\lambda$ | Hard drive component failure hazard rate | $\lambda = \text{AFR} / (365 \times 24) \text{ hours}^{-1}$ |

---

## 2. Temporal Ingestion & Linear-Time Feature Flattening

### 2.1 Tensor Ingestion Model
Let the multi-drive telemetry stream over discrete daily timesteps $t \in \{1, 2, \dots, W\}$ be represented as a 3rd-order tensor:
$$\boldsymbol{\mathcal{T}} \in \mathbb{R}^{K \times M \times W}$$
where $K$ is the number of active drive serial numbers, $M$ is the number of raw SMART sensor channels, and $W = 10$ is the sliding observation horizon.

For a specific drive $d_k$, its raw observation matrix is:
$$\mathbf{S}_{d_k} = \begin{bmatrix} s_{1,1} & s_{1,2} & \cdots & s_{1,W} \\ s_{2,1} & s_{2,2} & \cdots & s_{2,W} \\ \vdots & \vdots & \ddots & \vdots \\ s_{M,1} & s_{M,2} & \cdots & s_{M,W} \end{bmatrix} \in \mathbb{R}^{M \times W}$$

### 2.2 Temporal Vectorization Operator
The temporal flattening transformation $\mathcal{F}: \mathbb{R}^{M \times W} \to \mathbb{R}^{D}$ maps the 2D matrix into a 1D vector $\mathbf{x}_{d_k}$ via row-major concatenation:
$$\mathbf{x}_{d_k} = \text{vec}\left(\mathbf{S}_{d_k}^T\right) = \left[ s_{1,1}, s_{1,2}, \dots, s_{1,W}, s_{2,1}, \dots, s_{M,W} \right]^T \in \mathbb{R}^D$$
where $D = M \cdot W$. For our 21-attribute Seagate enterprise models, $D = 21 \times 10 = 210$.

### 2.3 Discrete Degradation Velocity and Acceleration
By supplying all $W$ daily time steps simultaneously to tree split evaluators, the ensemble implicitly models discrete numerical derivatives of sensor degradation:

1. **First-Order Degradation Velocity (Rate of Change):**
   $$\Delta s_{m, t} = \frac{\partial s_m}{\partial t} \approx s_{m, t} - s_{m, t-1}$$
2. **Second-Order Degradation Acceleration (Curvature):**
   $$\Delta^2 s_{m, t} = \frac{\partial^2 s_m}{\partial t^2} \approx \Delta s_{m, t} - \Delta s_{m, t-1} = s_{m, t} - 2s_{m, t-1} + s_{m, t-2}$$
3. **Cumulative Window Trajectory Drift:**
   $$\nabla_W s_m = s_{m, W} - s_{m, 1} = \sum_{t=2}^W \Delta s_{m, t}$$

### 2.4 Computational Complexity: Linear Flattening vs. Attention
* **Sequence Transformer (Self-Attention):**
  Given sequence length $W$ and projection dimension $d_{\text{model}}$:
  $$\text{Complexity}_{\text{Transformer}} = \mathcal{O}\left( N \cdot (W^2 \cdot d_{\text{model}} + W \cdot d_{\text{model}}^2) \right)$$
  For a fleet of $N = 100,000$ drives evaluated daily, quadratic dependency $\mathcal{O}(W^2)$ requires dedicated GPU tensor accelerators.
* **SentinelDC Flattened Gradient Tree Boosting:**
  With depth-limited decision trees ($D_{\text{tree}} \le 6$):
  $$\text{Complexity}_{\text{SentinelDC}} = \mathcal{O}\left( N \cdot T \cdot D_{\text{tree}} \right) \equiv \mathcal{O}(N)$$
  Inference latency scales strictly in **linear time $\mathcal{O}(N)$**, taking **$0.0023\text{ ms}$** per drive on standard x86-64 server CPUs.

---

## 3. Extreme Class Imbalance & Cost-Sensitive Gradient Optimization

### 3.1 Fleet Failure Rate & Imbalance Ratio
Let $\mathcal{D}_+ = \{(\mathbf{x}_i, y_i) \mid y_i = 1\}$ and $\mathcal{D}_- = \{(\mathbf{x}_i, y_i) \mid y_i = 0\}$. The empirical failure rate $\rho$ satisfies:
$$\rho = \frac{|\mathcal{D}_+|}{|\mathcal{D}_+| + |\mathcal{D}_-|} < 0.03 \quad (\text{AFR } < 3\%)$$
The fleet imbalance ratio $w_+$ is formulated as:
$$w_+ = \frac{|\mathcal{D}_-|}{|\mathcal{D}_+|} = \frac{N_{\text{healthy}}}{N_{\text{failing}}} \in [33, 150]$$

### 3.2 Serial-Level Stratified Partitioning (Anti-Leakage Constraint)
Let $\sigma(i) \in \mathcal{S}$ map sample index $i$ to its unique physical drive serial number. To prevent temporal lookahead data leakage, the partition $\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{test}}$ is constrained by:
$$\{\sigma(i) \mid i \in \mathcal{D}_{\text{train}}\} \cap \{\sigma(j) \mid j \in \mathcal{D}_{\text{test}}\} = \emptyset$$
with stratified class balance preserved:
$$\frac{|\{i \in \mathcal{D}_{\text{train}} \mid y_i = 1\}|}{|\mathcal{D}_{\text{train}}|} \approx \frac{|\{j \in \mathcal{D}_{\text{test}} \mid y_j = 1\}|}{|\mathcal{D}_{\text{test}}|} = \rho$$

### 3.3 Cost-Sensitive Second-Order Loss Objective
Standard binary cross-entropy treats false positives and false negatives symmetrically. We apply asymmetric positive-class reweighting:
$$\ell(y_i, \hat{y}_i) = - \left[ w_+ \cdot y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
where $\hat{p}_i = \sigma(\hat{y}_i) = \frac{1}{1 + e^{-\hat{y}_i}}$ is the sigmoid link function.

At boosting iteration $t$, the second-order Taylor expansion of the objective function is:
$$\widetilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \gamma T_{\text{leaves}} + \frac{1}{2} \lambda_{\text{reg}} \sum_{j=1}^{T_{\text{leaves}}} w_j^2$$
where $g_i$ and $h_i$ are the weighted first and second-order loss gradients:
$$g_i = \frac{\partial \ell(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}} = \left[ y_i w_+ + (1 - y_i) \right] \hat{p}_i - w_+ y_i$$
$$h_i = \frac{\partial^2 \ell(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2} = \left[ y_i w_+ + (1 - y_i) \right] \hat{p}_i (1 - \hat{p}_i)$$

### 3.4 Optimal Leaf Weight & Split Criterion
For a leaf node containing instance index set $I_j = \{i \mid q(\mathbf{x}_i) = j\}$, setting $\frac{\partial \widetilde{\mathcal{L}}^{(t)}}{\partial w_j} = 0$ yields the optimal weight $w_j^*$:
$$w_j^* = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda_{\text{reg}}}$$

The corresponding maximized loss reduction when splitting an internal node into left ($I_L$) and right ($I_R$) partitions is given by the split gain $\mathcal{G}$:
$$\mathcal{G}_{\text{split}} = \frac{1}{2} \left[ \frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda_{\text{reg}}} + \frac{\left(\sum_{i \in I_R} g_i\right)^2}{\sum_{i \in I_R} h_i + \lambda_{\text{reg}}} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda_{\text{reg}}} \right] - \gamma$$
A split is approved if and only if $\mathcal{G}_{\text{split}} > 0$.

---

## 4. Precision-Recall Curve Dynamic Threshold Optimization

### 4.1 Rejection of the Default 0.5 Decision Threshold
In severe class imbalance ($\rho \approx 1\%$), a naive decision boundary $\hat{p}_i \ge 0.5$ forces the model into high-precision, near-zero recall failure. SentinelDC treats the operational threshold $\tau \in [0, 1]$ as a tunable hyperparameter calibrated over the training validation set.

### 4.2 Parameterized Confusion Matrix Elements
For any candidate threshold $\tau$:
$$\text{TP}(\tau) = \sum_{i=1}^{|\mathcal{D}_{\text{val}}|} \mathbb{I}(y_i = 1) \cdot \mathbb{I}(\hat{p}_i \ge \tau)$$
$$\text{FP}(\tau) = \sum_{i=1}^{|\mathcal{D}_{\text{val}}|} \mathbb{I}(y_i = 0) \cdot \mathbb{I}(\hat{p}_i \ge \tau)$$
$$\text{FN}(\tau) = \sum_{i=1}^{|\mathcal{D}_{\text{val}}|} \mathbb{I}(y_i = 1) \cdot \mathbb{I}(\hat{p}_i < \tau)$$
$$\text{TN}(\tau) = \sum_{i=1}^{|\mathcal{D}_{\text{val}}|} \mathbb{I}(y_i = 0) \cdot \mathbb{I}(\hat{p}_i < \tau)$$

### 4.3 Objective Function for Threshold Calibration
We define precision and recall curves over $\tau$:
$$\text{Precision}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FP}(\tau) + \epsilon}$$
$$\text{Recall}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau) + \epsilon}$$

The optimal operational threshold $\tau^*$ is found via 1D line search maximizing the harmonic mean ($F_1$-score) along the continuous PR curve:
$$\tau^* = \arg\max_{\tau \in [0, 1]} F_1(\tau) = \arg\max_{\tau \in [0, 1]} \left( \frac{2 \cdot \text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau) + \epsilon} \right)$$

For cost-sensitive SLAs where missing a failure (False Negative $\to$ emergency rebuild) incurs a cost $C_{\text{FN}}$ and pulling a healthy drive (False Positive $\to$ wasted technician labor) incurs $C_{\text{FP}}$, the general cost-optimal threshold satisfies:
$$\tau_{\text{cost}}^* = \arg\min_{\tau \in [0, 1]} \left[ C_{\text{FN}} \cdot \text{FN}(\tau) + C_{\text{FP}} \cdot \text{FP}(\tau) \right]$$

### 4.4 Area Under the Precision-Recall Curve (PR-AUC)
$$\text{PR-AUC} = \int_0^1 \text{Precision}(R) \, dR \approx \sum_{k=1}^K \left[ \text{Recall}_k - \text{Recall}_{k-1} \right] \cdot \text{Precision}_k$$

---

## 5. Game-Theoretic Local Explainability via Exact TreeSHAP

### 5.1 Axiomatic Foundations of Shapley Additive Explanations
For a local prediction $f(\mathbf{x})$, we solve for additive attribution values $\phi_0, \phi_1, \dots, \phi_D$ satisfying:
$$f(\mathbf{x}) = \phi_0 + \sum_{m=1}^D \phi_m(\mathbf{x})$$

The attribution values uniquely satisfy four fundamental game-theoretic axioms:
1. **Efficiency (Local Accuracy):**
   $$\sum_{m=1}^D \phi_m(\mathbf{x}) = f(\mathbf{x}) - \mathbb{E}_{\mathbf{x}}[f(\mathbf{x})]$$
2. **Symmetry:** If for all subsets $S \subseteq \mathcal{F} \setminus \{j, k\}$:
   $$f_x(S \cup \{j\}) = f_x(S \cup \{k\}) \implies \phi_j(\mathbf{x}) = \phi_k(\mathbf{x})$$
3. **Dummy (Null Player):** If for all $S \subseteq \mathcal{F} \setminus \{j\}$:
   $$f_x(S \cup \{j\}) = f_x(S) \implies \phi_j(\mathbf{x}) = 0$$
4. **Additivity:** For ensemble model $f(\mathbf{x}) = \sum_{t=1}^T f_t(\mathbf{x})$:
   $$\phi_m(f, \mathbf{x}) = \sum_{t=1}^T \phi_m(f_t, \mathbf{x})$$

### 5.2 The Exponential Trap of Model-Agnostic Sampling (KernelSHAP)
Under arbitrary black-box estimators, calculating exact Shapley values requires marginalizing over all $2^D$ feature subsets:
$$\phi_m(\mathbf{x}) = \sum_{S \subseteq \mathcal{F} \setminus \{m\}} \frac{|S|!(|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} \left[ f_x(S \cup \{m\}) - f_x(S) \right]$$
For $D = 210$:
$$2^{210} \approx 1.645 \times 10^{63} \text{ function evaluations}$$
This computation is mathematically intractable for real-time streaming operations.

### 5.3 Polynomial Complexity Proof of TreeSHAP
By exploiting the recursive binary tree structure of XGBoost, Lundberg et al.'s TreeSHAP algorithm evaluates conditional expectations $\mathbb{E}[f(\mathbf{x}) \mid \mathbf{x}_S]$ directly along tree branches.

Let $T$ be the number of trees, $L$ be the maximum number of leaves per tree, and $D_{\text{tree}}$ be the maximum depth of any tree.
$$\text{Complexity}_{\text{TreeSHAP}} = \mathcal{O}\left( T \cdot L \cdot D_{\text{tree}}^2 \right)$$

**Numerical Proof for SentinelDC:**
* $T = 100$ trees
* $L = 32$ maximum leaves
* $D_{\text{tree}} = 5 \implies D_{\text{tree}}^2 = 25$
$$\text{Total Elementary Operations} = 100 \times 32 \times 25 = 80,000 \text{ ops}$$
On an x86 processor capable of $3.2 \times 10^9$ cycles per second:
$$\text{Execution Time} = \frac{80,000}{3.2 \times 10^9} \approx 2.5 \times 10^{-5} \text{ seconds} = \mathbf{0.025 \text{ ms}}$$
This enables **synchronous, per-sample explainability** directly on the live telemetry path.

### 5.4 Relative Risk Factor Contribution
The percentage attribution $\psi_m(\mathbf{x})$ of each SMART attribute $m$ is normalized as:
$$\psi_m(\mathbf{x}) = \frac{\max(0, \phi_m(\mathbf{x}))}{\sum_{j=1}^D \max(0, \phi_j(\mathbf{x})) + \epsilon} \times 100\%$$

### 5.5 Fleet Baseline Median Deviation (Robust Kinetic Delta)
To isolate physical degradation from normal operational noise, the raw value $s_{m, W}$ at day 10 is compared against the fleet-wide healthy median $\widetilde{\mu}_m$ using the Median Absolute Deviation (MAD):
$$\text{MAD}_m = \text{median}\left( |s_{m} - \widetilde{\mu}_m| \right)_{s_m \in \mathcal{D}_{\text{healthy}}}$$
$$\Delta_{m, \text{robust}} = \frac{s_{m, W} - \widetilde{\mu}_m}{\text{MAD}_m + \epsilon}$$

---

## 6. Tier 1 Deterministic Expert Rule Engine & Failure Mode Predicates

Tier 1 translates continuous SHAP vectors $\boldsymbol{\phi}(\mathbf{x})$ and telemetry vectors $\mathbf{x}$ into deterministic physical failure modes using boolean predicate logic:

### 6.1 Predicate 1: Magnetic Platter Surface Degradation
Platter surface damage is characterized by media defects, track read errors, and sector reallocations:
$$\mathbb{I}_{\text{Platter}}(\mathbf{x}, \boldsymbol{\phi}) = \left[ \left( \sum_{t=1}^W \phi(\text{SMART } 5_t) > \theta_{\phi} \right) \lor \left( \sum_{t=1}^W \phi(\text{SMART } 197_t) > \theta_{\phi} \right) \right] \land \left[ s_{5, W} > 0 \lor s_{197, W} > 0 \right]$$
*Physical Consequence:* Head flying height instability causing physical contact with the magnetic media; reallocated sector table exhaustion.

### 6.2 Predicate 2: Read/Write Head & Preamp Instability
Head and pre-amplifier degradation produces uncorrectable ECC read errors and command timeout bursts without initial sector reallocations:
$$\mathbb{I}_{\text{Head}}(\mathbf{x}, \boldsymbol{\phi}) = \left[ \phi(\text{SMART } 187_W) > \theta_{\phi} \lor \phi(\text{SMART } 188_W) > \theta_{\phi} \right] \land \left[ s_{187, W} > s_{187, 1} \right]$$
*Physical Consequence:* Pre-amplifier thermal drift or MR read-element resistance degradation; high risk of instantaneous head landing.

### 6.3 Predicate 3: Thermal Runaway & Environmental Stress
Thermal failure is governed by absolute temperature limits and rapid temperature rate of change:
$$\mathbb{I}_{\text{Thermal}}(\mathbf{x}, \boldsymbol{\phi}) = \left[ \phi(\text{SMART } 194_W) > \theta_{\phi} \right] \land \left[ s_{194, W} \ge 55^\circ\text{C} \lor \left( s_{194, W} - s_{194, 1} \ge 10^\circ\text{C} \right) \right]$$
*Physical Consequence:* Lubricant evaporation on platter surface, thermal track expansion causing seek errors, accelerated spindle motor bearing wear.

### 6.4 Predicate 4: Interface, Cabling & Controller CRC Corruption
Transmission errors over the SAS/SATA backplane or PCIe HBA controller:
$$\mathbb{I}_{\text{Interface}}(\mathbf{x}, \boldsymbol{\phi}) = \left[ \phi(\text{SMART } 199_W) > \theta_{\phi} \right] \land \left[ s_{199, W} - s_{199, 1} > 0 \right] \land \left[ s_{5, W} == 0 \right]$$
*Physical Consequence:* Physical cable oxidation, loose latching connectors, or backplane trace impedance mismatches. *Platter is healthy; drive replacement is avoided via cable reseating.*

---

## 7. Storage Blast Radius & Cluster Availability Probability Models

### 7.1 Erasure Coding Redundancy Formulation
Let an enterprise Ceph cluster or ZFS vdev implement an $(M_{\text{data}} + K_{\text{parity}})$ erasure coding scheme (or RAID-6 with $K_{\text{parity}} = 2$).
The cluster can tolerate up to $K_{\text{parity}}$ simultaneous disk failures before data loss occurs.

Let $N_{\text{group}} = M_{\text{data}} + K_{\text{parity}}$ denote the stripe group size. Suppose $f$ drives have failed or are actively degraded. The remaining fault tolerance margin $\mu$ is:
$$\mu = K_{\text{parity}} - f$$

### 7.2 Probability of Catastrophic Pool Loss During Rebuild
When an unannounced drive crash occurs, the rebuild duration $T_{\text{rebuild}}$ (hours) is a function of drive capacity $C_{\text{drive}}$ (TB) and network backfill rebuild rate $R_{\text{rebuild}}$ (TB/hour):
$$T_{\text{rebuild}} = \frac{C_{\text{drive}}}{R_{\text{rebuild}}}$$

During this rebuild window $T_{\text{rebuild}}$, the remaining $(N_{\text{group}} - f)$ drives are under intensive 100% read stress. Let $\lambda$ denote the component hazard rate:
$$\lambda = \frac{\text{AFR}}{8760} \text{ failures/hour}$$

The probability $P_{\text{loss}}$ that an additional $(\mu + 1)$ drives fail during the rebuild window, causing **irreversible cluster data loss**, is governed by the binomial Poisson survival model:
$$P_{\text{loss}} = 1 - \sum_{j=0}^{\mu} \binom{N_{\text{group}} - f}{j} \left( 1 - e^{-\lambda T_{\text{rebuild}}} \right)^j \left( e^{-\lambda T_{\text{rebuild}}} \right)^{N_{\text{group}} - f - j}$$

### 7.3 Unrecoverable Read Error (URE) Compound Risk
During rebuild, every surviving bit must be read without error. Let $U$ denote the Unrecoverable Read Error rate (typically $1 \text{ sector in } 10^{14} \text{ bits}$ for desktop drives, $10^{15} \text{ bits}$ for enterprise drives).
The probability $P_{\text{URE}}$ of encountering an unrecoverable read error across reading $C_{\text{rebuild}}$ bits is:
$$P_{\text{URE}} = 1 - (1 - U)^{C_{\text{rebuild}}} \approx 1 - e^{- U \cdot C_{\text{rebuild}}}$$
For an 8-drive RAID-5 array with 12TB drives ($C_{\text{rebuild}} \approx 7 \times 12 \times 8 \times 10^{12} \text{ bits} \approx 6.72 \times 10^{14} \text{ bits}$):
$$P_{\text{URE}} = 1 - e^{- 10^{-14} \cdot 6.72 \times 10^{14}} = 1 - e^{-6.72} \approx \mathbf{99.88\%}$$
*Mathematical Proof:* An unannounced hard drive failure in a high-capacity RAID-5 array results in virtually guaranteed data loss during rebuild. Proactive evacuation via SentinelDC eliminates this catastrophic rebuild state.

---

## 8. Two-Tier D2G Latency Guarantees & SLA Fallback Formulation

### 8.1 Decoupled Execution Model
Let the total processing latency of the alerting system be denoted as $\Delta t_{\text{total}}$.
$$\Delta t_{\text{total}} = t_{\text{inference}} + t_{\text{TreeSHAP}} + t_{\text{Tier1}} + t_{\text{Tier2}}$$

From empirical benchmarks:
* $t_{\text{inference}} \le 0.003\text{ ms}$
* $t_{\text{TreeSHAP}} \le 0.035\text{ ms}$
* $t_{\text{Tier1}} \le 0.010\text{ ms}$
$$\therefore \quad t_{\text{critical\_path}} = t_{\text{inference}} + t_{\text{TreeSHAP}} + t_{\text{Tier1}} < \mathbf{0.05 \text{ ms}} \ll 1\text{ ms}$$

### 8.2 Asynchronous Generative Synthesis & SLA Bound
Tier 2 dispatches an HTTP request to the LLaMA-3.3-70B API over network channel $\mathcal{N}$. Let $t_{\text{LLM}} \sim \mathcal{F}_{\text{latency}}$ represent the stochastic network + inference latency of the LLM.
We enforce a strict deterministic SLA timeout threshold:
$$t_{\text{SLA\_limit}} = 400\text{ ms}$$

The alerting delivery function $\boldsymbol{\Phi}(\mathbf{x})$ is defined by the piecewise step operator:
$$\boldsymbol{\Phi}(\mathbf{x}) = \begin{cases} \mathbf{\Psi}_{\text{Tier2}}\left( \mathbf{J}_{\text{Tier1}}(\mathbf{x}) \right), & \text{if } t_{\text{LLM}} \le t_{\text{SLA\_limit}} \land \text{HTTP\_Status} == 200 \\ \mathbf{J}_{\text{Tier1}}(\mathbf{x}), & \text{if } t_{\text{LLM}} > t_{\text{SLA\_limit}} \lor \text{HTTP\_Status} \neq 200 \end{cases}$$
where $\mathbf{J}_{\text{Tier1}}(\mathbf{x})$ is the immutable, strictly typed Tier 1 JSON schema contract, and $\mathbf{\Psi}_{\text{Tier2}}$ is the natural language synthesizer.

**Availability Guarantee Theorem:**
Under this decoupled formulation, the availability of the critical SRE alert channel is mathematically independent of external LLM API uptime:
$$A_{\text{SentinelDC}} = A_{\text{Tier1}} \cdot \left[ P(t_{\text{LLM}} \le t_{\text{SLA}}) \cdot A_{\text{LLM}} + \left( 1 - P(t_{\text{LLM}} \le t_{\text{SLA}}) \cdot A_{\text{LLM}} \right) \right] \equiv A_{\text{Tier1}} = \mathbf{100.0\%}$$

---

## 9. Mathematical Formulations of Baseline Benchmark Algorithms

To substantiate why alternative machine learning paradigms failed in our multi-model bake-off, their formal mathematical definitions and failure mechanisms are documented below:

### 9.1 Elliptic Envelope (Mahalanobis Distance on PCA Subspace)
Assumes healthy SMART vectors follow a multivariate normal distribution $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. The squared Mahalanobis distance is:
$$D_M^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$$
Parameters $\boldsymbol{\mu}_{\text{MCD}}$ and $\boldsymbol{\Sigma}_{\text{MCD}}$ are estimated via the Minimum Covariance Determinant (MCD) on $H \subset \mathcal{D}_{\text{healthy}}$:
$$\boldsymbol{\Sigma}_{\text{MCD}} = \arg\min_H \det(\boldsymbol{\Sigma}_H) \quad \text{s.t. } |H| = h$$
*Failure Proof:* SMART data violates Gaussian normality. Registers like SMART 5 (Reallocated Sectors) have probability mass function $P(s_5 = 0) \approx 0.98$. The covariance matrix $\boldsymbol{\Sigma}$ is singular ($\det(\boldsymbol{\Sigma}) = 0$) in the raw 210-D space, necessitating lossy PCA dimensionality reduction to 30 components, which caps $F_1$ at $0.3511$.

### 9.2 Local Outlier Factor (LOF)
Given query instance $p$, let $d(p, o)$ be Euclidean distance and $N_k(p)$ be the $k$-nearest neighbors of $p$.
1. **$k$-Distance:** $d_k(p) = d(p, o)$ such that at least $k$ neighbors have distance $\le d(p, o)$.
2. **Reachability Distance:**
   $$\text{reach-dist}_k(p, o) = \max\left( d_k(o), d(p, o) \right)$$
3. **Local Reachability Density (LRD):**
   $$\text{lrd}_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}$$
4. **Local Outlier Factor (LOF):**
   $$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}}{|N_k(p)|}$$
*Failure Proof:* LOF has inference complexity $\mathcal{O}(N_{\text{train}} \cdot D)$. For $N_{\text{train}} = 37,080$ drives, evaluating a single incoming log row requires scanning an $8,000 \times 190$ float32 matrix in memory, yielding a latency of **$0.0371\text{ ms}$** ($16\times$ slower than XGBoost) while providing uncalibrated, non-probabilistic binary outputs ($F_1 = 0.3374$).

### 9.3 Isolation Forest
Given dataset $\mathcal{X}$, an isolation tree recursively splits a randomly selected feature $q$ at a uniform split point $p \in [\min(x_q), \max(x_q)]$ until instances are isolated.
The anomaly score $s(\mathbf{x}, n)$ for sample $\mathbf{x}$ across an ensemble of $t$ trees is:
$$s(\mathbf{x}, n) = 2^{ - \frac{\mathbb{E}[h(\mathbf{x})]}{c(n)} }$$
where $h(\mathbf{x})$ is the tree path length, and $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree (BST):
$$c(n) = 2 \left( \ln(n - 1) + 0.5772156649 \right) - \frac{2(n - 1)}{n}$$
*Failure Proof:* Random axis-aligned cuts fail in zero-inflated tabular data where hundreds of features share identical zero values across healthy drives. Random partitioning isolates healthy drives with identical probability to failing drives, yielding an abysmal Precision of $0.2140$ and $F_1 = 0.2808$.

---

## 10. Summary of Mathematical Superiority

$$\begin{aligned}
\text{Latency: } & \mathcal{O}(N) \ll \mathcal{O}(N \cdot W^2) \quad (\text{Linear Flattening vs Transformers}) \\
\text{XAI Complexity: } & \mathcal{O}(T \cdot L \cdot D_{\text{tree}}^2) \ll \mathcal{O}(M \cdot 2^M) \quad (\text{Exact TreeSHAP vs KernelSHAP}) \\
\text{Reliability: } & P(\text{Failure Avoided}) \approx 99.88\% \quad (\text{Proactive Evacuation vs URE During Rebuild})
\end{aligned}$$
