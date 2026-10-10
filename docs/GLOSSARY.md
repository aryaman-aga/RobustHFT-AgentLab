# GLOSSARY.md — Technical Terminology Reference for RobustHFT-AgentLab

This document defines key concepts, metrics, financial vocabulary, red-teaming operators, and architectural components used across **RobustHFT-AgentLab (`rhal`)** and the base paper (*AutoRedTrader*).

---

## 1. Core Evaluation & Attack Metrics

| Term | Full Name | Definition / Formula | Purpose in `rhal` |
|---|---|---|---|
| **ASR** | Attack Success Rate | $\text{ASR} = \frac{\text{Count}(\text{action}_{\text{treated}} \neq \text{action}_{\text{shadow}})}{N}$ | Measures the raw proportion of trading decisions flipped by injected misinformation compared to a shadow clean baseline. |
| **Placebo ASR** | Baseline Structural Noise | ASR measured when visible news pool is augmented with meaning-preserving paraphrases. | Establishes the noise floor caused purely by retrieval pool reshuffling (top-$k$ shifts). |
| **Net ASR** | Net Attack Success Rate | $\text{Net ASR} = \text{ASR}_{\text{injected}} - \text{ASR}_{\text{placebo}}$ | **Our Extension.** Isolates true semantic adversarial impact from structural retrieval noise. |
| **MER** | Misinformation Exposure Rate | $\text{MER} = \frac{\text{Synthetic Docs Retrieved}}{k \times \text{Trading Days}}$ | Percentage of retrieved documents visible to the agent's perception module that carry synthetic misinformation. |
| **MER Lift** | Exposure Amplification | $\text{MER Lift} = \frac{\text{MER}}{\text{Pool Synthetic Fraction}}$ | **Our Extension.** Ratio of misinformation in retrieved top-$k$ vs. total pool share. Values $>1$ mean the attacker successfully game retrieval ranking. |
| **CR** | Cumulative Return | Total fractional portfolio gain/loss over the evaluation horizon. | Measures financial performance under clean and attacked conditions. |
| **Max Drawdown** | Maximum Drawdown | $\max_{t} \frac{\text{Peak}_t - \text{Portfolio}_t}{\text{Peak}_t}$ | Quantifies peak-to-trough downside risk exposure under attack. |

---

## 2. Red-Teaming & MisGen Architecture

| Term | Category | Definition |
|---|---|---|
| **MisGen** | Engine | The closed-loop red-teaming engine that perturbs financial news items and steers attack selection using feedback loops. |
| **Pólya-Urn Sampler** | Sampling Algorithm | A history-aware reinforcement algorithm where successful decision-flipping strategies increase their sampling probability ($w_s = 1 + \alpha h_s$). |
| **Behavioral Bias Prompt** | Attack Dimension ($\Phi_{\text{Bias}}$) | Injected agent profile context mimicking human cognitive biases (Loss Aversion, Anchoring, Herding, Overconfidence). |
| **Minor Perturbation** | Attack Dimension ($\Phi_{\text{Minor}}$) | 8 fine-grained semantic edits: `flipping` (direction reversal), `sentiment` (intensity scale), `numerical` (financial numbers), `temporal` (timestamps), `entity` (company names), `concept` (business terms), `causal` (reasoning), `other`. |
| **Style Rewrite** | Attack Dimension ($\Phi_{\text{Rewrite}}$) | Polishing perturbed text into standard `news` style or `academic` formal style to evade automated detectors. |
| **Similarity Filter ($A_{\text{sim}}$)** | Quality Gate | TF-IDF cosine filter ensuring generated misinformation retains textual similarity to the original clean source. |
| **Detectability Model ($A_{\text{det}}$)** | Quality Gate / Classifier | A calibrated classifier (`outputs/models/detector.pkl`) predicting whether news text is synthetic or clean at a target clean False Positive Rate (FPR = 10%). |
| **Placebo Arm** | Baseline Control | An experimental arm using meaning-preserving LLM paraphrases (`provenance="placebo"`) to benchmark natural retrieval drift. |

---

## 3. Defense, Grounding & Microstructure Components

| Term | Category | Definition |
|---|---|---|
| **Temporal Grounding** | Defense Layer | Cross-referencing retrieved textual news claims against historical price/market time-series windows to verify consistency. |
| **Admissibility Rule** | Grounding Constraint | Strict non-lookahead rule ($e + H \le t$) ensuring historical analog windows ending at $e$ with horizon $H$ only use data $\le t$. |
| **Text–Market Consistency ($c$)** | Grounding Metric | $c = 1 - \frac{|s_{\text{text}} - s_{\text{mkt}}|}{2} \in [0, 1]$. Quantifies agreement between news sentiment ($s_{\text{text}}$) and market momentum ($s_{\text{mkt}}$). |
| **Consistency Shrinkage ($\kappa$)** | Grounding Formula | $s_g = s - \kappa(1 - c)(s - s_{\text{mkt}})$. Pulls text-driven sentiment toward market reality when consistency $c$ is low. |
| **Microstructure Features** | High-Frequency Proxies | Features derived from price/volume dynamics: **Garman–Klass Volatility** (OHLC vol), **Order Flow Imbalance (OFI)**, **Amihud Illiquidity**, **Volume Shock**, **Spread Proxy**, **Range Ratio**. |
| **Risk Layer** | Execution Control | Post-policy guardrail enforcing confidence thresholds, **hold-on-conflict** rules, volatility-targeted position sizing, stop-loss/take-profit triggers, and drawdown throttling. |
| **Hold-on-Conflict** | Risk Override | Automatically overrides a trade to **Hold** if text sentiment and market evidence point in opposite directions. |

---

## 4. Agent Framework & Simulation Concepts

| Term | Category | Definition |
|---|---|---|
| **POMDP** | Agent Formalism | **Partially Observable Markov Decision Process**. The agent observes noisy prices and retrieved top-$k$ news without full market state knowledge. |
| **Shadow Clean Agent** | Experiment Design | An unattacked agent running on identical clean market data in lockstep with the treated (attacked) agent to provide paired step-by-step baselines. |
| **No-Lookahead Invariant** | Integrity Constraint | Strict design rule: decision at index $t$ depends *only* on rows $\le t$. Verified by automated future-mutation tests. |
| **Surrogate Policy** | Policy Implementation | Transparent, deterministic rule-based agent policy used for fast, reproducible red-teaming experiments without API latency. |
