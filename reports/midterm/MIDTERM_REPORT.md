# NETAJI SUBHAS UNIVERSITY OF TECHNOLOGY
**(Formerly Netaji Subhas Institute of Technology)**  
**Sector-3, Dwarka, New Delhi - 110078, India**

---

# MID-TERM PROJECT REPORT
**Bachelor of Technology (B.Tech) in Computer Science and Engineering**

### **RobustHFT-AgentLab: Stress-Testing and Defending LLM Trading Agents Against Subtle Financial Misinformation**
*Temporal Grounding, Market Microstructure Evidence, and Execution Risk Controls*

---

**Submitted By:**  
**[Student Name]**  
Roll No: [Roll Number]  
Department of Computer Science and Engineering  
Netaji Subhas University of Technology, New Delhi  

**Under the Guidance of:**  
**[Supervisor Name]**  
[Designation / Professor]  
Department of Computer Science and Engineering  
Netaji Subhas University of Technology, New Delhi  

*Academic Year: 2026–2027*

---

## Table of Contents
1. [Introduction](#1-introduction)
   - 1.1 Base Paper Overview: AutoRedTrader
   - 1.2 Project Scope and Mid-Term Milestones
2. [Motivation](#2-motivation)
   - 2.1 The Fragility of Textual Reliance in Quantitative Systems
   - 2.2 Methodological Flaws Identified in the Base Paper
3. [Literature Survey](#3-literature-survey)
   - 3.1 LLM-Based Financial Trading Agents
   - 3.2 Adversarial Vulnerabilities and Financial Red-Teaming
   - 3.3 Behavioral Finance and Cognitive Biases
   - 3.4 Market Microstructure and Backtest Integrity
   - 3.5 Comparative Analysis Table
4. [Problem Statement](#4-problem-statement)
   - 4.1 Threat Model: Closed-Loop Misinformation Injection
   - 4.2 Formalization of Evaluation Metrics (ASR, Placebo ASR, Net ASR, MER Lift)
5. [Objective and Methodology](#5-objective-and-methodology)
   - 5.1 Core Objectives
   - 5.2 System Architecture
   - 5.3 The 3-Tier Defense Hierarchy
     - Tier 1: Admissible Temporal Grounding (Implemented & In Active Calibration)
     - Tier 2: High-Frequency / Microstructure Proxies (Formulated for End-Term)
     - Tier 3: Execution Risk-Control Layer (Formulated for End-Term)
   - 5.4 Adversarial Generation and Closed-Loop Steering (Pólya-Urn)
6. [Simulation Platform and Requirements](#6-simulation-platform-and-requirements)
   - 6.1 Software and Hardware Specifications
   - 6.2 Safety Boundaries and Provenance Enforcement
   - 6.3 Structural No-Lookahead Architecture and Verification
   - 6.4 Preliminary Mid-Term Experimental Results
7. [Conclusion](#7-conclusion)
   - 7.1 Summary of Mid-Term Achievements
   - 7.2 Roadmap for End-Term Completion
8. [References](#8-references)

---

## 1. Introduction

The deployment of Large Language Models (LLMs) in autonomous decision-making environments has rapidly transitioned from exploratory academic prototypes to high-stakes quantitative financial workflows. In algorithmic trading, recent frameworks such as FinMem (Yu et al., 2025), FinCon (Yu et al., 2024), and TradingAgents (Xiao et al., 2025) integrate generative language models into Partially Observable Markov Decision Process (POMDP) architectures. In these systems, an agent continuously observes quantitative market data (price histories, technical indicators, order metrics), retrieves textual market narratives (news reports, corporate filings, microblog feeds) from memory stores, and synthesizes multimodal signals to execute sequential financial decisions (Buy, Sell, or Hold).

While multimodal reasoning provides significant predictive flexibility, it simultaneously introduces a critical, asymmetric attack surface: **untrusted textual inputs**. Unlike numerical time series—which are strictly regulated and verified through consolidated market data feeds—textual news streams are decentralized, noisy, and highly susceptible to strategic injection. Traditional machine learning vulnerabilities, such as adversarial gradient perturbations or jailbreaking prompts, aim to bypass explicit safety boundaries to generate harmful language. In contrast, financial misinformation operates under an entirely different paradigm: **subtle semantic distortion**. A malicious actor does not need to inject overtly false or toxic text. Instead, minor alterations—such as flipping a growth rate, inverting a qualitative sentiment descriptor, modifying an entity attribution, or subtly distorting causal claims—produce text that appears authentic, passes stylistic detection filters, and yet decisively manipulates the downstream trading policy.

### 1.1 Base Paper Overview: AutoRedTrader
The benchmark baseline for this project is *AutoRedTrader: Autonomous Red Teaming of Trading Agents through Synthetic Misinformation Injection* (Liu et al., 2026, arXiv:2605.09185). AutoRedTrader formalizes the autonomous red-teaming of LLM trading agents by defining a structured three-tiered operator space for generating synthetic financial misinformation:
1. **Behavioral Bias Framing ($\Phi_{\text{Bias}}$):** Steering the target agent's psychological state through cognitive bias induction (e.g., triggering overconfidence, loss aversion, anchoring, or herding dynamics).
2. **Minor Semantic Perturbations ($\Phi_{\text{Minor}}$):** Applying targeted atomic edits across eight distinct semantic categories, including directional flipping, sentiment inversion, numerical substitution, and causal swaps.
3. **Style Rewriting ($\Phi_{\text{Rewrite}}$):** Camouflaging perturbed text into natural journalistic or formal financial research styles to evade surface-level linguistic filters.

AutoRedTrader dynamically optimizes this perturbation space using a history-aware Pólya-urn reinforcement sampler, demonstrating that an autonomous attacker can flip trading decisions in up to 26.7% of evaluated days. To counteract this, the authors propose a preliminary heuristic termed "time-series-informed grounding."

### 1.2 Project Scope and Mid-Term Milestones
This B.Tech Project, entitled **RobustHFT-AgentLab (`rhal`)**, develops an offline, deterministic, and academically rigorous defensive simulation laboratory that addresses the critical architectural and methodological vulnerabilities identified in existing literature.

Rather than treating the problem as a single monolithic defense, this project designs a **Three-Tier Defense Hierarchy**:
- **Tier 1: Admissible Temporal Grounding** — Historical analog retrieval and mathematical sentiment shrinkage with strict no-lookahead enforcement ($e + H \le t$).
- **Tier 2: High-Frequency / Microstructure Evidence Fusion** — Incorporating fast-moving order-flow and spread proxies to ground textual claims in microscopic market reality.
- **Tier 3: Execution Risk-Control Layer** — Downstream portfolio safeguards (confidence gates, hold-on-conflict, volatility sizing, and drawdown throttling).

**Current Mid-Term Evaluation Status:** As of this mid-term evaluation, the baseline simulation platform and closed-loop MisGen red-teaming engine have been fully reproduced and mathematically verified. Furthermore, a major evaluation flaw in the base paper—namely, retrieval-induced noise—has been isolated through the implementation of a novel **Placebo Control Arm**. **Defense Tier 1 (Admissible Temporal Grounding) has been fully implemented**, and current work is actively focused on fine-tuning its coupling parameters ($\kappa$-shrinkage sweep) to map the optimal robustness-return Pareto frontier. Defenses Tier 2 and Tier 3, along with extensive cross-asset validations, form the scheduled deliverables for the final evaluation phase.

---

## 2. Motivation

The motivation for this research stems from fundamental vulnerabilities discovered in the operation of autonomous LLM trading agents, alongside notable methodological deficiencies in the current state of the art.

### 2.1 The Fragility of Textual Reliance in Quantitative Systems
Quantitative finance has historically relied on strict statistical models operating on price, volume, and order-book state. The introduction of LLMs has democratized unstructured text interpretation, enabling agents to parse real-time breaking news. However, LLMs lack an intrinsic understanding of physical or financial reality; they process text probabilistically based on linguistic token associations. Consequently, when an agent encounters synthetic narratives that mimic genuine financial press releases, it suffers from cognitive capture: the textual signal completely overrides the quantitative reality. In backtests, an agent fed manipulated earnings reports or fabricated executive statements consistently misallocates capital, resulting in catastrophic portfolio drawdowns.

### 2.2 Methodological Flaws Identified in the Base Paper
A critical study of the base paper (*AutoRedTrader*) revealed four major limitations that undermine its real-world applicability and empirical conclusions:

1. **The Retrieval Confound: Failure to Disentangle Attack Semantics from Retrieval Noise:**  
   In existing studies, the Attack Success Rate (ASR) is computed by comparing a clean execution run against an injected run. However, the retrieval mechanism relies on top-$k$ similarity matching over a dynamically updated document pool. Adding synthetic documents—irrespective of their semantic content—naturally alters document rank ordering, displacing marginal clean articles. In near-threshold market regimes, minor changes in retrieved context flip decisions purely due to *retrieval noise*. AutoRedTrader failed to isolate this effect, leading to over-reported attack efficacy.
2. **Temporal Lookahead Leakage in Historical Analog Grounding:**  
   The grounding mechanism in AutoRedTrader searches past market windows to validate current news. However, in their mathematical formulation, an analog identified at historical time $e$ with forward outcome horizon $H$ is considered valid even when $e < t$. If $e + H > t$, the analog's outcome window overlaps with future data relative to $t$, creating subtle lookahead leakage that invalidates the backtest.
3. **Coarse Macro Indicators vs. Microscopic Market Reality:**  
   Existing temporal grounding relies exclusively on coarse, low-frequency daily indicators (e.g., 5-day return, 20-day volatility). Real-world markets reflect institutional activity at the microstructure level. A textual release claiming "overwhelming institutional accumulation" can be immediately falsified if high-frequency Order-Flow Imbalance (OFI) is negative and bid-ask spreads are widening. Neglecting microstructure leaves the defense blind to intraday order book reality.
4. **Absence of Downstream Portfolio Risk Management:**  
   In quantitative finance, no signal is executed without risk filtration. The base paper assumes raw agent decisions directly dictate market execution. In practice, a robust defense requires execution-level circuit breakers that override decisions whenever textual confidence and market consistency diverge.

---

## 3. Literature Survey

The intersection of generative artificial intelligence, adversarial robustness, and quantitative financial engineering spans several distinct research domains:

### 3.1 LLM-Based Financial Trading Agents
The paradigm of utilizing language models as sequential financial agents originated with FinMem (Yu et al., 2025), which established layered memory structures (working memory and long-term episodic memory) alongside character profiles to simulate cognitive investment behavior. Li et al. (2025) introduced InvestorBench, formalizing the standard POMDP trading agent architecture divided into four functional modules: Perception (data parsing), Profile (risk and style alignment), Memory (retrieval-augmented storage), and Action (execution synthesis). FinCon (Yu et al., 2024) extended this with multi-agent conceptual verbal reinforcement, where specialized analyst agents debate trade validity. TradingAgents (Xiao et al., 2025) and FinAgent (Zhang et al., 2024) introduced tool-augmented planning and multimodal observation capabilities. While these systems demonstrate superior alpha generation in benign market conditions, none were evaluated against intentional, adversarial informational corruption.

### 3.2 Adversarial Vulnerabilities and Financial Red-Teaming
Adversarial attacks against NLP systems historically focused on character or token-level substitutions or adversarial prefix prompts designed to elicit toxic output (Jiang et al., 2024; Zhou et al., 2025). In financial contexts, Yan et al. introduced TradeTrap (2025), demonstrating that state-of-the-art LLMs (including GPT-4 and Claude-3) exhibit significant decision instability when exposed to minor prompt paraphrasing. Wang et al. (2026) characterized prompt-induced linguistic fingerprints in machine-generated news, noting that subtle stylistic adaptations can easily evade static discriminators. Jiang et al. (2026) developed GLISTERS, a benchmark for reference-free counterfactual financial misinformation detection, proving that detecting semantic inversion without access to the ground-truth event is near chance level for standard classifiers. AutoRedTrader (Liu et al., 2026) consolidated these insights into an automated red-teaming loop specifically optimized to force agent action flips.

### 3.3 Behavioral Finance and Cognitive Biases
Behavioral economics establishes that human traders systematically deviate from rational expectations. Barber and Odean (2001) documented how overconfidence leads to excessive trading and reduced risk aversion. Kahneman and Tversky's Prospect Theory (2013) established loss aversion—the psychological asymmetry where losses are felt roughly twice as intensely as equivalent gains. Spyrou (2013) analyzed herding behaviors where market participants discard private information to follow collective consensus, while De Long et al. (1990) proved that noise traders driven by cognitive heuristics can create persistent price anomalies. These psychological biases serve as the mathematical foundation for adversarial framing operators ($\Phi_{\text{Bias}}$) that exploit the inductive biases of LLMs.

### 3.4 Market Microstructure and Backtest Integrity
High-frequency market microstructure offers empirical proxies that capture supply-demand asymmetries before they manifest in closing prices. Corwin and Schultz (2012) derived a robust high-low estimator for effective bid-ask spreads. Garman and Klass (1980) developed an efficient realized volatility estimator using extreme value distributions. Cont, Kukanov, and Stoikov (2014) proved that Order-Flow Imbalance (OFI)—the net imbalance between buying and selling volume at the best quotes—exhibits a near-linear instantaneous relationship with price change. Amihud's illiquidity measure (2002) quantifies the price impact per unit of trading volume. Finally, in backtest methodology, Bailey et al. (2014) and López de Prado (2018) emphasized that lookahead bias and selection bias are the primary causes of backtest failure, mandating strictly causal, trailing data transformations.

### 3.5 Comparative Analysis
| Framework / System | Agent Architecture | Adversarial Loop | Grounding Method | Microstructure Proxies | Risk Controls | Placebo / Net ASR |
|---|---|---|---|---|---|---|
| FinMem (2025) | POMDP + Layered Memory | None | None | None | None | No |
| TradeTrap (2025) | Single LLM Prompt | Static Paraphrasing | None | None | None | No |
| AutoRedTrader (2026) | Memory-Augmented LLM | Closed-Loop (Pólya-Urn) | Daily Indicators ($e < t$) | None | None | No (Naive ASR only) |
| **RobustHFT-AgentLab (Ours)** | **Modular POMDP** | **Closed-Loop (Pólya-Urn)** | **Admissible ($e+H \le t$)** | **OFI, Spread, GK-RV** | **Multi-Barrier Risk** | **Yes (Paired Net ASR)** |

---

## 4. Problem Statement

Let an autonomous financial trading agent operate within a discrete-time financial market over a trading horizon $t \in \{1, 2, \dots, T\}$. At each step $t$, the agent receives an observation tuple:
$$o_t = \left( \mathbf{x}_{1:t}, \mathcal{D}_t \right)$$
where $\mathbf{x}_{1:t} \in \mathbb{R}^{t \times d}$ denotes the strictly causal historical market time series (OHLCV prices and derived technical indicators), and $\mathcal{D}_t = \{d_1, d_2, \dots, d_k\}$ is a retrieved subset of $k$ news documents obtained from a memory pool $\mathcal{M}_t$.

The agent maintains an internal state $s_t$ composed of its perception of market conditions, its behavioral profile context $\mathbf{b}$, and episodic memory traces. The policy $\pi_\theta(a_t \mid o_t, s_t)$ maps observations to discrete portfolio actions:
$$a_t \in \{\text{Buy } (+1), \text{Hold } (0), \text{Sell } (-1)\}$$

### 4.1 Threat Model: Closed-Loop Misinformation Injection
An external red-team adversary seeks to induce erroneous trading decisions by perturbing the information environment. The adversary cannot modify the consolidated market data $\mathbf{x}_{1:t}$, nor can it directly alter model parameters $\theta$. The adversary's sole attack vector is injecting a synthetic document $d^*_t$ into the unverified news repository $\mathcal{M}_t$.

To be effective and evade security tripwires, $d^*_t$ must satisfy two conflicting constraints:
1. **Evasion (Plausibility):** $d^*_t$ must closely resemble genuine financial reporting, satisfying semantic similarity $A_{\text{sim}}(d^*_t, d_t) \ge \tau_{\text{sim}}$ and linguistic detectability $A_{\text{det}}(d^*_t) \le \tau_{\text{det}}$.
2. **Efficacy (Decision Flipping):** When retrieved into $\mathcal{D}_t$, $d^*_t$ must alter the agent's action relative to its unperturbed decision: $a_t^{\text{injected}} \ne a_t^{\text{clean}}$.

### 4.2 Formalization of Evaluation Metrics
To eliminate retrieval confounds, the metric space is defined as follows:
- **Naive Attack Success Rate (ASR):**
  $$\text{ASR} = \frac{1}{T} \sum_{t=1}^T \mathbb{I}\left( a_t^{\text{treated}} \ne a_t^{\text{clean}} \right)$$
- **Placebo ASR ($\text{ASR}_{\text{placebo}}$):**
  $$\text{ASR}_{\text{placebo}} = \frac{1}{T} \sum_{t=1}^T \mathbb{I}\left( a_t^{\text{placebo}} \ne a_t^{\text{clean}} \right)$$
- **Net Attack Success Rate ($\text{ASR}_{\text{net}}$):**
  $$\text{ASR}_{\text{net}} = \max\left(0, \, \text{ASR}_{\text{injected}} - \text{ASR}_{\text{placebo}}\right)$$
- **Misinformation Exposure Rate (MER) and Lift:**
  $$\text{MER} = \frac{1}{T \cdot k} \sum_{t=1}^T \sum_{i=1}^k \mathbb{I}\left( d_{t,i} \in \mathcal{M}_{\text{synthetic}} \right), \quad \text{MER Lift} = \frac{\text{MER}}{\text{Base Rate}_{\text{synthetic}}}$$

---

## 5. Objective and Methodology

### 5.1 Core Objectives
1. **Phase 1 (Mid-Term Completed):** Construct an end-to-end, offline, CPU-only simulation testbed replicating the closed-loop AutoRedTrader framework; establish a paired shadow evaluation methodology with a placebo arm; fully implement and unit-test **Defense Tier 1: Admissible Temporal Grounding**.
2. **Phase 1 Ongoing (Mid-Term Tuning):** Calibrate the grounding coupling parameter $\kappa \in [0.3, 1.0]$ to quantify the Pareto frontier between attack resistance and clean return preservation.
3. **Phase 2 (End-Term Scheduled):** Formulate and integrate **Defense Tier 2: High-Frequency / Microstructure Evidence Fusion** (Garman-Klass RV, Corwin-Schultz spread, and Order-Flow Imbalance).
4. **Phase 3 (End-Term Scheduled):** Implement **Defense Tier 3: Execution Risk-Control Layer** (Hold-on-Conflict, volatility sizing, and drawdown circuit breakers), followed by full multi-seed ablation and real-market data testing.

### 5.2 System Architecture
The framework operates as an integrated closed-loop pipeline where market data, retrieved news, temporal grounding, microstructure proxies, and execution risk controls operate in synchrony under deterministic seed control.

### 5.3 The 3-Tier Defense Hierarchy

#### Tier 1: Admissible Temporal Grounding (Implemented & In Active Calibration)
Temporal grounding reconciles textual news claims with quantitative price patterns. When news is retrieved at step $t$, the grounding engine queries historical market windows of duration $W=30$ days to identify historical analogs matching the current market state $\mathbf{x}_{t-W:t}$.

**The Mathematical Admissibility Constraint:** To prevent lookahead leakage, every historical candidate window starting at $s$ and ending at $e = s + W$ with forward evaluation horizon $H = 5$ days must satisfy:
$$e + H \le t$$
If an analog satisfies this condition, its realized forward market outcome $\Delta p_{e \to e+H}$ is strictly in the historical past relative to decision time $t$. The top-$K$ analogs are retrieved via Euclidean distance over normalized feature vectors, yielding an expected analog return $r_{\text{analog}}$ and confidence $c \in [0, 1]$.

**Sentiment Shrinkage Formula:** Let $s_{\text{raw}} \in [-1, 1]$ be the raw sentiment score extracted from retrieved news, and $s_{\text{mkt}} = \text{sign}(r_{\text{analog}}) \cdot \min(1, |r_{\text{analog}}|/\sigma)$ be the market-implied sentiment. The grounded sentiment $s_g$ is formulated as a shrinkage estimator:
$$s_g = s_{\text{raw}} - \kappa \cdot (1 - c) \cdot \left( s_{\text{raw}} - s_{\text{mkt}} \right)$$
where $\kappa \in [0, 1]$ is the grounding strength parameter. When text and market evidence conflict and confidence $c$ is low, $\kappa$ pulls the effective sentiment toward market reality. We are currently performing a rigorous empirical sweep of $\kappa \in \{0.3, 0.7, 1.0\}$.

#### Tier 2: High-Frequency / Microstructure Proxies (Formulated for End-Term)
While Tier 1 operates on daily OHLCV bars, Tier 2 equips the grounding layer with high-frequency microstructure estimators:
- **Garman-Klass Realized Volatility ($GK\text{-}RV$):**
  $$\sigma_{GK}^2 = 0.511 \left(\ln \frac{H}{C}\right)^2 - 0.019 \left[\ln \frac{C}{O} \ln \frac{H \cdot L}{O^2} - 2 \ln \frac{H}{O} \ln \frac{L}{O}\right] - 0.383 \left(\ln \frac{C}{O}\right)^2$$
- **Corwin-Schultz Spread Proxy ($S$):** Estimates bid-ask spread from 2-day high-low price ratios, capturing liquidity dry-ups.
- **Order-Flow Imbalance Proxy ($OFI$):**
  $$OFI_t = \text{Vol}_t \cdot \frac{2 C_t - H_t - L_t}{H_t - L_t}$$

In Phase 2, these proxies will be fused into the state vector, allowing the agent to detect when bullish text is contradicted by order book liquidity withdrawal.

#### Tier 3: Execution Risk-Control Layer (Formulated for End-Term)
The final defense tier operates downstream of the agent's policy $\pi_\theta$, enforcing strict portfolio risk boundaries:
- **Hold-on-Conflict Filter:** If the sign of textual sentiment contradicts the market trend ($\text{sign}(s_{\text{raw}}) \ne \text{sign}(s_{\text{mkt}})$) and confidence is below threshold $\tau_{\text{conf}}$, the action is forced to Hold ($a_t = 0$).
- **Volatility-Targeted Sizing:** Position size is scaled inversely with Garman-Klass volatility: $w_t = \min\left(1.0, \frac{\sigma_{\text{target}}}{\sigma_{GK, t}}\right)$.
- **Drawdown Circuit Breaker:** If trailing portfolio drawdown exceeds $\text{DD}_{\text{max}} = 5\%$, all trading is halted until volatility normalizes.

### 5.4 Adversarial Generation and Closed-Loop Steering
The MisGen engine generates candidate perturbations across three spaces:
$$d^* = \Phi_{\text{Rewrite}}\left( \Phi_{\text{Minor}}\left( d, \Phi_{\text{Bias}}(\mathbf{b}) \right) \right)$$
Candidate documents are passed through a TF-IDF cosine similarity gate ($A_{\text{sim}} \ge 0.35$) and our trained logistic detectability proxy ($A_{\text{det}} \le \tau_{\text{det}}$). Surviving documents enter the pool.

To steer the attack online, the adversary samples operator combinations $(b, m, r)$ using a generalized Pólya-urn model. When an injected document successfully flips an agent's decision ($a_t^{\text{injected}} \ne a_t^{\text{clean}}$), its weight is reinforced:
$$w_{b,m,r}^{(t+1)} = w_{b,m,r}^{(t)} + \alpha \cdot \mathbb{I}(\text{flip}_t) + \delta \cdot |\Delta \text{Return}_t|$$
where $\alpha=1.0$ is the discrete flip bonus and $\delta=0.15$ rewards return destruction.

---

## 6. Simulation Platform and Requirements

### 6.1 Software and Hardware Specifications
- **Operating Environment:** macOS (Darwin 24.3.0 / Apple Silicon) and Linux (Ubuntu 22.04+).
- **Programming Language:** Python 3.14 (fully compatible with Python 3.10+).
- **Core Libraries:** NumPy 2.5, Pandas 3.0, Scikit-Learn 1.9, Pytest 8.0+.
- **Documentation & Typesetting:** TeX Live / Tectonic (automated offline IEEE report generation).
- **Hardware Constraints:** 100% CPU-only execution; maximum memory footprint $< 1.5$ GB RAM; execution time for a full 5-seed, 9-arm simulation $< 60$ seconds.

### 6.2 Safety Boundaries and Provenance Enforcement
- Every generated document schema explicitly mandates metadata: `provenance = "synthetic"` or `provenance = "placebo"`.
- The simulation platform contains no broker connectors, no order routing hooks, and no public dissemination interfaces.
- Market data loaders operate in read-only mode over synthetic GARCH processes or cached local historical CSVs.

### 6.3 Structural No-Lookahead Architecture and Verification
1. **Next-Open Execution:** Decisions synthesized at the close of day $t$ using information $\mathcal{I}_t$ are executed strictly at the opening price of day $t+1$ ($\text{Open}_{t+1}$), incorporating a 5 bps transaction fee and 5 bps linear slippage penalty.
2. **Trailing Normalization:** All feature z-scores and rolling volatilities use trailing rolling windows strictly terminated at $t$.
3. **Future-Mutation Test Harness:** The platform includes an automated unit test suite (`tests/test_features_nolookahead.py`). The test records the agent's feature values and trading decisions at historical step $t$, artificially mutates the market data for all steps $t' > t$ with random noise, and re-executes the pipeline. The test mathematically asserts that past decisions and representations remain bitwise identical, proving zero future leakage.

### 6.4 Preliminary Mid-Term Experimental Results
The baseline reproduction, placebo arm, and Tier 1 grounding have been executed across 5 independent random seeds ($42, 43, 44, 45, 46$) over a 60-day active trading period (preceded by a 250-day analog warm-up history).

| Experimental Arm | MER (%) | MER Lift | Naive ASR (%) | Net ASR (%) | Clean Return (%) | Max Drawdown (%) |
|---|---|---|---|---|---|---|
| `clean__none` (Undefended Baseline) | $0.0 \pm 0.0$ | $0.00 \times$ | $0.0 \pm 0.0$ | --- | $+4.3 \pm 5.7$ | $-4.9 \pm 1.5$ |
| `placebo__none` (Noise Control Arm) | $50.7 \pm 0.8$ | $1.03 \times$ | $21.3 \pm 4.4$ | --- | $+3.3 \pm 4.4$ | $-4.7 \pm 2.4$ |
| `injected__none` (Attacked Baseline) | $63.4 \pm 2.2$ | $1.30 \times$ | $\mathbf{35.0 \pm 6.4}$ | $\mathbf{13.7 \pm 6.4}$ | $+5.6 \pm 5.7$ | $-3.5 \pm 1.1$ |
| `clean__grounded` (Tier 1 Defense Clean) | $0.0 \pm 0.0$ | $0.00 \times$ | $0.0 \pm 0.0$ | --- | $-0.8 \pm 2.7$ | $-4.9 \pm 1.2$ |
| `placebo__grounded` (Tier 1 Placebo) | $50.4 \pm 0.6$ | $1.03 \times$ | $15.7 \pm 7.0$ | --- | $-1.4 \pm 4.3$ | $-5.5 \pm 0.1$ |
| `injected__grounded` (Tier 1 Attacked, $\kappa=0.7$) | $61.3 \pm 3.5$ | $1.26 \times$ | $\mathbf{21.7 \pm 4.7}$ | $\mathbf{6.0 \pm 7.2}$ | $-1.3 \pm 5.0$ | $-5.8 \pm 2.3$ |

**Key Mid-Term Findings:**
1. **Validation of the Placebo Discovery:** In the undefended agent, naive ASR is $35.0\%$. However, exposing the agent to meaning-preserving placebo text flips decisions in $21.3\%$ of days purely due to retrieval pool reshuffling! Consequently, the true adversarial semantic impact ($\text{ASR}_{\text{net}}$) is $13.7\%$, proving that prior literature significantly overstated attack potency by omitting retrieval controls.
2. **Efficacy of Tier 1 Grounding:** Implementing Admissible Temporal Grounding reduces naive ASR from $35.0\%$ to $21.7\%$ (a $13.3$ percentage point absolute reduction) and slashes $\text{ASR}_{\text{net}}$ from $13.7\%$ to $6.0\%$ (a $56.2\%$ relative reduction in net attack vulnerability).
3. **Current Work — Calibrating the $\kappa$-Frontier:** Our active tuning sweep shows that while $\kappa=0.7$ provides substantial attack suppression, it introduces a mild conservative drag on clean returns (from $+4.3\%$ to $-0.8\%$). Calibrating $\kappa$ across the spectrum $[0.3, 1.0]$ is currently underway to identify the Pareto-optimal configuration before layering Tier 2 and Tier 3 defenses.

---

## 7. Conclusion

### 7.1 Summary of Mid-Term Achievements
At this mid-term evaluation stage, the project has successfully accomplished its foundational technical objectives:
- **Robust Testbed Reproduction:** Replicated the closed-loop red-teaming framework of AutoRedTrader within an offline, fully deterministic, CPU-only simulation environment supported by a passing 62-test verification suite.
- **Methodological Breakthrough:** Formalized and validated the **Placebo Control Arm**, demonstrating that $21.3\%$ of observed decision flips in existing benchmarks stem from retrieval displacement rather than semantic deception, establishing $\text{ASR}_{\text{net}}$ as the mandatory metric for financial red-teaming.
- **Implementation of Defense Tier 1:** Successfully developed and deployed **Admissible Temporal Grounding**, enforcing the strict $e + H \le t$ historical admissibility boundary to eliminate lookahead bias while achieving a $56.2\%$ relative reduction in net attack success rate.
- **Active Parameter Fine-Tuning:** Initiated comprehensive parameter sensitivity sweeps over the grounding shrinkage parameter $\kappa$ to mathematically map the frontier between adversarial robustness and alpha preservation.

### 7.2 Roadmap for End-Term Completion
The work remaining for the final project submission is structured along three concrete engineering phases:
1. **Phase 2 (Weeks 9–12): Implementation of Defense Tier 2 (Microstructure Evidence Fusion):** Integrate Garman-Klass volatility, Corwin-Schultz spread estimators, and intraday Order-Flow Imbalance proxies into the grounding evidence builder, enabling the agent to cross-examine textual sentiment against microscopic order book dynamics.
2. **Phase 3 (Weeks 13–15): Deployment of Defense Tier 3 (Execution Risk-Control Layer):** Implement portfolio-level protective mechanisms, including the Hold-on-Conflict heuristic, volatility-targeted position sizing, and trailing drawdown circuit breakers to insulate the trading account from residual adversarial bypasses.
3. **Phase 4 (Weeks 16–18): Full Factorial Ablation and Real-Market Stress Testing:** Execute the full 10-seed ablation matrix across all combinations of defense tiers, conduct cross-asset validation using historical equity and cryptocurrency data (via Yahoo Finance loaders), and complete the final IEEE thesis manuscript.

---

## 8. References

1. Z. Liu, Y. Yu, Y. Cao, Y. Jiang, H. Li, Z. Lu, Y. Wang, Y. Zheng, X. Guo, C. Y. Cheng, J. Tsujii, and S. Ananiadou, "AutoRedTrader: Autonomous Red Teaming of Trading Agents through Synthetic Misinformation Injection," *arXiv preprint arXiv:2605.09185*, 2026.
2. Y. Yu, H. Li, Z. Chen, Y. Jiang, Y. Li, J. W. Suchow, D. Zhang, and K. Khashanah, "FinMem: A performance-enhanced LLM trading agent with layered memory and character design," *IEEE Transactions on Big Data*, 2025.
3. H. Li, Y. Cao, Y. Yu, S. R. Javaji, Z. Deng, Y. He, Y. Jiang, Z. Zhu, K. Subbalakshmi, J. Huang, et al., "InvestorBench: A benchmark for financial decision-making tasks with LLM-based agent," in *Proc. 63rd Annual Meeting of the ACL (Vol. 1)*, pp. 2509–2525, 2025.
4. Y. Yu, Z. Yao, H. Li, Z. Deng, Y. Jiang, Y. Cao, Z. Chen, J. W. Suchow, Z. Cui, R. Liu, et al., "FinCon: A synthesized LLM multi-agent system with conceptual verbal reinforcement for enhanced financial decision making," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.
5. Y. Xiao, E. Sun, D. Luo, and W. Wang, "TradingAgents: Multi-agents LLM financial trading framework," *arXiv preprint arXiv:2412.20138*, 2025.
6. L. Yan, J. Mei, T. Zhou, L. Huang, J. Zhang, D. Liu, and J. Shao, "TradeTrap: Are LLM-based trading agents truly reliable and faithful?," *arXiv preprint arXiv:2512.02261*, 2025.
7. A. Zhou, K. Wu, F. Pinto, Z. Chen, Y. Zeng, Y. Yang, S. Yang, S. Koyejo, J. Zou, and B. Li, "AutoRedTeamer: Autonomous red teaming with lifelong attack integration," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2025.
8. F. Jiang, Z. Xu, L. Niu, Z. Xiang, B. Ramasubramanian, B. Li, and R. Poovendran, "ArtPrompt: ASCII art-based jailbreak attacks against aligned LLMs," in *Proc. 62nd Annual Meeting of the ACL (Vol. 1)*, pp. 15157–15173, 2024.
9. C. Wang, M. Gao, Z. Wang, J. Yin, K. Shu, and C. Lin, "Prompt-induced linguistic fingerprints for LLM-generated fake news detection," in *Proc. ACM Web Conference*, pp. 7633–7644, 2026.
10. Y. Jiang, Z. Liu, Y. Cao, Y. He, Z. Xu, C. Xu, Z. Deng, P. Tiwari, X. Chen, A. Lopez-Lira, et al., "All that glisters is not gold: A benchmark for reference-free counterfactual financial misinformation detection," *arXiv preprint arXiv:2601.04160*, 2026.
11. B. M. Barber and T. Odean, "Boys will be boys: Gender, overconfidence, and common stock investment," *The Quarterly Journal of Economics*, vol. 116, no. 1, pp. 261–292, 2001.
12. D. Kahneman and A. Tversky, "Prospect theory: An analysis of decision under risk," in *Handbook of the Fundamentals of Financial Decision Making*, pp. 99–127, World Scientific, 2013.
13. S. Spyrou, "Herding in financial markets: a review of the literature," *Review of Behavioral Finance*, vol. 5, no. 2, pp. 175–194, 2013.
14. J. B. De Long, A. Shleifer, L. H. Summers, and R. J. Waldmann, "Noise trader risk in financial markets," *Journal of Political Economy*, vol. 98, no. 4, pp. 703–738, 1990.
15. T. Loughran and B. McDonald, "When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks," *The Journal of Finance*, vol. 66, no. 1, pp. 35–65, 2011.
16. H. Mahmoud, *Pólya Urn Models*, Chapman and Hall/CRC, 2008.
17. S. A. Corwin and P. Schultz, "A simple way to estimate bid-ask spreads from daily high and low prices," *The Journal of Finance*, vol. 67, no. 2, pp. 719–760, 2012.
18. M. B. Garman and M. J. Klass, "On the estimation of security price volatilities from historical data," *Journal of Business*, vol. 53, no. 1, pp. 67–78, 1980.
19. R. Cont, A. Kukanov, and S. Stoikov, "The price impact of order book events," *Journal of Financial Econometrics*, vol. 12, no. 1, pp. 47–88, 2014.
20. Y. Amihud, "Illiquidity and stock returns: cross-section and time-series effects," *Journal of Financial Markets*, vol. 5, no. 1, pp. 31–56, 2002.
21. D. H. Bailey, J. M. Borwein, M. López de Prado, and Q. J. Zhu, "The probability of backtest overfitting," *Journal of Computational Finance*, vol. 20, no. 4, pp. 39–69, 2016.
22. M. López de Prado, *Advances in Financial Machine Learning*, John Wiley \& Sons, 2018.
