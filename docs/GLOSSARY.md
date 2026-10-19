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
| **Placebo Arm** | Baseline Control | An experimental arm that injects meaning-preserving, rule-based paraphrases (synonym and connector substitutions; `provenance="placebo"`) to measure flips caused by retrieval drift alone. |

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
| **Hold-on-Conflict** | Risk Override | Overrides a new trade to **Hold** when text and market evidence conflict (consistency $c < 0.6$, i.e. $|s_{\text{text}} - s_{\text{mkt}}| > 0.8$). Backfires against an adaptive attacker, who creates the conflict. |

---

## 4. Agent Framework & Simulation Concepts

| Term | Category | Definition |
|---|---|---|
| **POMDP** | Agent Formalism | **Partially Observable Markov Decision Process**. The agent observes noisy prices and retrieved top-$k$ news without full market state knowledge. |
| **Shadow Clean Agent** | Experiment Design | An unattacked agent running on identical clean market data in lockstep with the treated (attacked) agent to provide paired step-by-step baselines. |
| **No-Lookahead Invariant** | Integrity Constraint | Strict design rule: decision at index $t$ depends *only* on rows $\le t$. Verified by automated future-mutation tests. |
| **Surrogate Policy** | Policy Implementation | Transparent, deterministic rule-based agent policy used for fast, reproducible red-teaming experiments without API latency. |

---

## 5. Measurement Controls and Extensions

| Term | Category | Definition |
|---|---|---|
| **Frozen Attack / Attack Tape** | Experiment Design | The documents and bias prompts of one adaptive attack (against the undefended agent), recorded and replayed verbatim against every defense, so defenses face identical attacks. |
| **Placebo Ladder** | Measurement Control | Placebo → count-matched placebo → bias-matched placebo → frozen attack. Consecutive differences split ASR into retrieval noise, article count, bias prompts and misinformation content. |
| **Common Random Numbers (CRN)** | Experiment Design | The attacker is seeded by injection type and draws from streams keyed by (day, article), so every arm of a seed shares the same random numbers. |
| **cf-ASR** | Metric | One-step counterfactual ASR: each day a copy of the clean twin's state decides with the attacked agent's documents and bias; removes flips caused by drift between the twins. |
| **Position Gap** | Metric | Mean absolute difference between the attacked agent's and the clean twin's target positions; captures changes in size that ASR (a label flip) misses. |
| **Reader Mode** | Agent Mode | `agent.text_scorer=llm`: an LLM rates each retrieved article on [−1, 1]; everything else in the defense layer is unchanged. |
| **Decider Mode** | Agent Mode | `agent.mode=llm`: the LLM chooses Buy/Sell/Hold itself, with the market evidence in its prompt (prompt-level grounding). |
| **Text-Free Agent** | Baseline Defense | Trades on market evidence only; cannot be attacked through text but gives up the information in genuine news. |
| **Detection Filter** | Baseline Defense | Drops retrieved articles the detectability model flags, at a chosen false-alarm rate calibrated on pre-window news. |
| **Robust Aggregation** | Baseline Defense | Weighted median or trimmed mean of article ratings instead of the weighted mean. |
| **Defense-Aware Attacker** | Attack Variant | `misgen.evidence_aware`: among acceptable edits, picks the one that agrees (`aligned`, evades grounding) or conflicts (`conflict`, triggers hold-on-conflict) with the day's public market evidence. |
| **Shrink-on-Conflict** | Risk Rule | `risk.conflict_mode=shrink`: on conflict, keeps the action but moves the position only a fraction $c$ of the way to its target. |
| **Price-Impact Stress Test** | Stress Test | The recorded attack also moves the shared price path by $\gamma$ daily standard deviations per unit of summed sentiment shift; both twins trade the moved prices. |
| **Real-Price Mode** | Data Setting | `data.source=real`: each seed is a different 400-day window of a real price history, with template news about the fictional company tied to the real returns. |
| **Analog Scale** | Grounding Parameter | Typical H-day move that maps analog forward returns into the market signal; `auto` estimates it from pre-window prices. |

