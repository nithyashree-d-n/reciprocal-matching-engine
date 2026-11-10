# Sequential Reciprocal Matchmaker
### Online Multi-Stage Decision System for Reciprocal Recommendation under Delayed Feedback

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Standard%20Library%20Only-brightgreen.svg)](#dependencies)
[![Docker Ready](https://img.shields.io/badge/Docker-Offline%20Container%20Ready-blue.svg)](Dockerfile)
[![Tests Passing](https://img.shields.io/badge/Tests-22%2F22%20Passing-success.svg)](#testing--verification)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An intelligent, offline-capable decision engine for sequential reciprocal matching under uncertainty. Designed for two-sided platforms (matching, mentoring, peer networks, and executive recruitment), this system manages the trade-offs between **active information acquisition**, **reciprocal dealbreaker validation**, **global graph allocation**, and **censored delayed feedback**.

---

## 📌 Executive Summary

Traditional recommender systems optimize one-sided click-through or engagement rates. However, reciprocal matching introduces fundamentally harder algorithmic challenges:
1. **Bilateral Mutual Consent:** An introduction succeeds only if *both* parties accept. A high compatibility score cannot override a hard dealbreaker in either direction.
2. **Opportunity Cost & Global Allocation:** Introducing Person A to Person B removes both from the candidate pool for up to 30 days, impacting the entire network. Local greedy matching causes severe multi-pair allocation losses.
3. **Budgeted Information Discovery:** Profiles arrive with incomplete attributes. The system has a strict daily budget to either unlock hard constraints or clarify high-impact soft preferences.
4. **Delayed, Right-Censored Feedback:** Real outcomes (responses, first meetings, mutual second-meeting intentions) take days or weeks to mature. Unobserved feedback cannot be naively treated as failure.

This repository implements an **Online Markov Decision Process (MDP)** solver that operates over 60-day decision horizons followed by 40-day outcome maturation windows across a changing population of 200 members.

---

## 🏗️ System Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │            Daily Observable State            │
                          │   (Available Members, Disclosures, Feedback) │
                          └──────────────────────┬───────────────────────┘
                                                 │
                        STAGE 1: Active Clarification (Ask Phase)
                        Two-Tier Value-of-Information (Budget: 12 units)
                                                 │
                                                 ▼
                           Updated State with Fresh Disclosures
                                                 │
                                                 ▼
                        STAGE 2: Multi-Stage Funnel Scoring
                         P(Accept) × P(Date | Accept) × P(MSMI | Date)
                                                 │
                                                 ▼
                        STAGE 3: Global Graph Matching & Timing
                        Max-Weight Allocation with Dynamic Reservation
                                                 │
                                                 ▼
                        STAGE 4: Delayed Feedback & Online Beliefs
                         Empirical Bayes updating in persistent memory
```

### 1. Two-Tier Value-of-Information Clarification (`adaptive_asks`)
- **Tier 1 (Hard Constraint Unlock):** Evaluates unconstrained candidate reach across demographic density (geographic zones, age distributions). Spends 6–9 budget units unlocking members who expand the feasible match graph.
- **Tier 2 (Precision Feature De-risking):** Identifies pairs with hard feasibility and spends remaining units querying high-impact soft attributes (`relationship_goal`, `relationship_pace`) to de-risk introductions.

### 2. Multi-Stage Funnel Scoring (`score_pair`)
Conversion is factorized through a 3-stage funnel:
$$\mathbb{E}[\text{MSMI}(a, b)] = P(\text{Both Accept } a, b) \times P(\text{Date Occurs} \mid \text{Mutual Accept}) \times P(\text{Both MSMI} \mid \text{Date})$$
- Evaluates soft attribute congruence across `relationship_goal` ($\beta = 1.0$), `relationship_pace` ($\beta = 0.6$), `lifestyle` ($\beta = 0.35$), and `conversations` ($\beta = 0.25$).
- Adjusts for individual member responsiveness and historical response rates tracked via online memory.

### 3. Global Graph Allocation with Dynamic Reservation (`adaptive_match`)
- Formulates pairing as a **Maximum Weight Matching** problem on a general graph $G = (V, E)$, avoiding the multi-pair allocation loss of greedy rankers.
- Implements a **Dynamic Reservation Price Schedule** $\theta(t)$:
  - **Days 0–15 (Arrival Phase):** $\theta = 0.4$ (holds candidates for higher-potential upcoming arrivals).
  - **Days 16–45 (Exploitation Phase):** $\theta = 0.0$ (matches all positive-utility feasible pairs).
  - **Days 46–55 (Coverage Phase):** $\theta = -0.2$ (expands coverage before the decision horizon closes).
  - **Days 56–59 (End-Game):** $\theta = 0.1$ (prioritizes high-certainty pairs ensuring dates mature within the 30-day window).

### 4. Online Belief Updating under Delayed Feedback
- Handles right-censored feedback via event tracking in explicit JSON `memory`.
- Updates empirical response probabilities for members who consistently accept or fail to respond by deadlines.

---

## 📊 Benchmark & Evaluation Results

Tested across 6 standardized scenario families (20 independent seeds each, 120 episodes total):
1. **Development:** Standard synthetic population.
2. **Sparse:** Geographic fragmentation across 12 zones (severe graph sparsity).
3. **Cold Start:** 40% higher missingness in initial member profiles.
4. **Delayed:** Extended scheduling latency before meetings occur.
5. **Feature Shift:** Inverted soft attribute importance weights.
6. **Temporal Drift:** Sentiment degradation and decreased acceptance rates post day 35.

### Performance Comparison

| Policy Strategy | Mean MSMI / 100 Members | Member Coverage | Mutual Acceptance Rate | Ask Budget Efficiency | Allocation Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Adaptive Policy (Ours)** | **0.83** | **44.0%** | **8.5%** | **High (VoI 2-Tier)** | **Max-Weight + Reservation** |
| Greedy Baseline | 0.50 | 40.8% | 7.6% | Moderate (Hard-only) | Greedy Pair Ranker |
| Random Feasible | 0.25 | 41.2% | 6.7% | Moderate (Hard-only) | Random Feasible Edges |
| No-Asks Baseline | 0.33 | 17.4% | 2.8% | Zero (0 Asks) | Greedy Pair Ranker |

> **Key Takeaway:** The adaptive policy delivers a **+66% higher primary MSMI score** over the greedy baseline by eliminating allocation bottlenecks, de-risking high-value pairings, and avoiding premature commitments during the early arrival phase.

---

## 📁 Repository Structure

```
.
├── policy.py                 # Core executable decision policy (Adaptive, Greedy, Random, No-Asks)
├── kit.py                    # Complete synthetic simulator, reciprocal constraints & data generator
├── evaluate.py               # 60-day decision + 40-day follow-up evaluation harness
├── verify_data.py            # SHA-256 integrity validator for synthetic data pools
├── Dockerfile                # Production-grade offline CPU container definition
├── requirements.txt          # Dependency specification (standard library core + optional dev)
├── data/                     # 10 independent synthetic pools (2,000 members, 71 data files)
│   ├── public_01/ ... public_06/  # Training pools
│   ├── public_07/ ... public_08/  # Validation pools
│   └── public_09/ ... public_10/  # Development-test pools
├── docs/
│   ├── DATA_CONTRACT.md      # Strict field vocabulary, types, and constraint definitions
│   ├── POLICY_INTERFACE.md   # Two-phase JSON stdin/stdout executable protocol
│   ├── SUBMISSION.md         # Reproducibility and container execution checklist
│   └── FAQ.md                # Technical clarifications and design assumptions
├── examples/
│   └── baseline_results/     # Verified reference benchmark outputs
└── test_*.py                 # Unit tests (contract, release invariants, container sandbox)
```

---

## 🚀 Quickstart & Reproduction

### Prerequisites
- Python 3.10 or later
- Standard library only (no external packages required for execution)

### 1. Verify Dataset & Run Unit Tests
```bash
# Verify integrity of all 71 synthetic data files
python verify_data.py

# Execute the complete unit test suite (22 tests)
python -m unittest -v
```

### 2. Run Single Episode Evaluation
```bash
# Evaluate the Adaptive Policy on Seed 101
python evaluate.py --seeds 101 --output results/adaptive_run.json
```

### 3. Compare Across All Baseline Strategies
```bash
# Adaptive Policy (Ours)
python evaluate.py --baseline adaptive --seeds 101,102,103 --output results/adaptive.json

# Greedy Baseline
python evaluate.py --baseline greedy --seeds 101,102,103 --output results/greedy.json

# No-Clarification Baseline
python evaluate.py --baseline no_asks --seeds 101,102,103 --output results/no_asks.json

# Random Feasible Baseline
python evaluate.py --baseline random --seeds 101,102,103 --output results/random.json
```

---

## 🐳 Isolated Offline Docker Execution

The policy can be packaged into an offline, sandboxed Docker container (no network access, 2 CPU cores, 1 GiB RAM, 10s timeout per invocation):

```bash
# Build the container image
docker build -t sequential-policy:latest .

# Run evaluation inside the isolated container
python evaluate.py --image sequential-policy:latest --seeds 101 --output results/docker_eval.json
```

---

## 📐 Mathematical Formulation & Constraints

### 1. Reciprocal Hard Constraints (Strict Feasibility)
A pair $(u, v)$ is feasible if and only if all symmetric conditions hold:
- **Age Window:** Both members fall within each other's stated age bounds (`age_min` $\le$ `age` $\le$ `age_max` in both directions):
  $$\text{AgeMin}(u) \le \text{Age}(v) \le \text{AgeMax}(u) \quad \land \quad \text{AgeMin}(v) \le \text{Age}(u) \le \text{AgeMax}(v)$$
- **Gender Preference:** Each member's gender matches the other's acceptable gender list:
  $$\text{gender}(v) \in \text{WhoToMeet}(u) \quad \land \quad \text{gender}(u) \in \text{WhoToMeet}(v)$$
- **Geographic Compatibility:** Each member accepts the other's geographic zone:
  $$\text{zone}(v) \in \text{AcceptableZones}(u) \quad \land \quad \text{zone}(u) \in \text{AcceptableZones}(v)$$
- **Schedule Overlap:** Both members share at least one overlapping availability window:
  $$\text{Schedule}(u) \cap \text{Schedule}(v) \neq \emptyset$$
- **Smoking & Children Dealbreakers:** Strict non-smoker and children criteria are mutually satisfied.

### 2. Primary Optimization Metric
$$\text{MSMI per 100 Arrived Members} = 100 \times \frac{M}{N}$$

Where:
- **$M$**: Count of pairs where first date occurs within 30 days of assignment, and *both* members submit positive second-meeting intentions within 3 days post-date.
- **$N$**: Total members who arrived by decision day 59.

---

## 🛠️ Tech Stack & Highlights

- **Core Language:** Python 3.10+ (Standard Library: `itertools`, `hashlib`, `json`, `math`, `random`)
- **Execution Mode:** Offline CPU-only inference, sub-millisecond per-step latency
- **Protocol:** Stateless process invocation with explicit JSON state memory persistence
- **Containerization:** Docker (Debian slim, non-root user, read-only rootfs)

---

## 📄 License & Attribution

- **Source Code:** Released under the [MIT License](LICENSE).
- **Synthetic Data:** Distributed under the [Synthetic Data License](DATA_LICENSE.md). All profiles, conversations, and outcomes are entirely synthetic.
