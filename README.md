
Awareness-Based Cognitive Filtering Agent (ABC-Agent)
Experiments validating an Awareness-Guided Cognitive Filter Theory through RL Benchmarks (CartPole + Stress Tests)

---

Overview
This repository contains two core experiments that implement and validate a new cognitive theory:
Awareness-Based Cognitive Filtering (ABC) Theory
The theory proposes that:
An intelligent agent performs better and stays more stable when it filters internal noise and distractions through an awareness-guided modulation signal.
---
Repository Structure
├── PExperiment01.py       # Experiment 1: Alpha sensitivity, multi-seed stability, hyperparameter robustness
├── PExperiment02.py       # Experiment 2: Baseline vs Awareness (clean vs stress/noisy environments)
├── exp1_results.csv
├── exp2_results.csv
├── exp1_graph.png
├── exp2_plot_comparison.png
├── exp2_plot_stress.png
└── README.md
---
Experiment 1 – Awareness Parameter Study
Validates
Stability across random seeds
Sensitivity of awareness parameter (alpha)
Hyperparameter robustness

Outputs
exp1_results.csv
exp1_graph.png

---
Experiment 2 – Baseline vs Awareness Under Stress
Validates
Awareness agent outperforms baseline
Awareness agent stays stable under noisy/stress conditions
Baseline collapses under noise → proves necessity of cognitive filter

Outputs
exp2_results.csv
exp2_plot_comparison.png
exp2_plot_stress.png

---
Key Findings
1. Awareness improves average performance
2. Awareness remains stable across seeds
3. Awareness resists noise/distraction
4. Baseline model fails under stress
5. Strong evidence for the Awareness-Based Cognitive Filter Theory

---
How to Run
Install dependencies:
pip install numpy pandas matplotlib gymnasium
Run Experiment 1:
python PExperiment01.py
Run Experiment 2:
python PExperiment02.py

---
License
MIT License — open for research use.

---
Citation
Prathmesh Wathore (2025). Awareness-Based Cognitive Filtering in Reinforcement Learning.
GitHub Repository.















Yeh FINAL README.md hai — seedha paste karo and commit.
