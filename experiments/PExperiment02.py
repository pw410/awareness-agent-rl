# experiment2.py
# Compare Baseline vs Awareness agent on CartPole-v1 (clean vs noisy observations).
# Requirements: python3, numpy, pandas, matplotlib, gymnasium
#
# If you see "No module named 'gymnasium'":
#   pip install gymnasium
#
# Usage: python experiment2.py
# Output: exp2_results.csv, exp2_plot_comparison.png, exp2_plot_stress.png

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from statistics import mean

# ---------------------------
# Tiny deterministic helpers
# ---------------------------
EPISODE_MAX_STEPS = 200  # standard for CartPole
RUNS_PER_SEED = 25       # number of episodes averaged per seed
SEEDS = [101, 102, 103, 104, 105]

# ---------------------------
# Agents (Baseline vs Awareness)
# ---------------------------
class AwarenessAgent:
    def __init__(self, alpha=0.0, noise_gain=0.0, seed=None):
        """
        alpha: awareness strength (0 for baseline)
        noise_gain: internal noise multiplier (not the external stress noise)
        seed: rng seed for weight init
        """
        self.alpha = float(alpha)
        self.noise_gain = float(noise_gain)
        rng = np.random.RandomState(seed)
        # simple linear policy weights for 4-dim CartPole => 2 actions
        # shape (4,2)
        self.w = rng.randn(4, 2) * 0.1

    def ramanujan_cf(self, m):
        # simple 3-element ramanujan-style scalar function (keeps it deterministic)
        # ensures >0
        return 1.0 + 1.0/(1.0 + abs(m[0])) + 1.0/(1.0 + abs(m[1])) + 1.0/(1.0 + abs(m[2]))

    def policy(self, obs):
        """
        obs: 1D numpy array of observation (CartPole: 4)
        returns action (0 or 1)
        """
        # small internal noise vector m (simulate cognitive filter input)
        m = np.tanh(np.random.randn(3) * self.noise_gain)
        cf = self.ramanujan_cf(m)               # cognitive filter scalar
        mod = self.w * (1.0 + self.alpha * cf)  # modulated weights
        logits = obs @ mod                      # linear logits (2,)
        # softmax then argmax
        exps = np.exp(logits - np.max(logits))
        probs = exps / (np.sum(exps) + 1e-9)
        return int(np.argmax(probs))


# ---------------------------
# Episode runner
# ---------------------------
def run_episode(env, agent, obs_noise=0.0):
    """
    env: gymnasium env
    agent: AwarenessAgent
    obs_noise: standard deviation of gaussian noise added to observations (stress)
    returns total reward for one episode
    """
    obs, _ = env.reset()   # gymnasium reset -> (obs, info)
    total = 0.0
    for step in range(EPISODE_MAX_STEPS):
        # external noisy observation (stress test)
        if obs_noise > 0.0:
            obs_noisy = obs + np.random.normal(scale=obs_noise, size=obs.shape)
        else:
            obs_noisy = obs
        action = agent.policy(obs_noisy)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            break
    return total

# ---------------------------
# Evaluate function
# ---------------------------
def evaluate_variant(alpha, noise_gain, seeds, obs_noise):
    """
    alpha: awareness strength for agent
    noise_gain: agent's internal noise parameter
    seeds: list of random seeds to run
    obs_noise: external observation noise (stress test)
    returns: list of mean_total_rewards (one per seed)
    """
    seed_scores = []
    for s in seeds:
        np.random.seed(s)
        env = gym.make("CartPole-v1", render_mode=None)
        # make deterministic-ish for each seed
        env.reset(seed=s)
        agent = AwarenessAgent(alpha=alpha, noise_gain=noise_gain, seed=s + 7)
        runs = [run_episode(env, agent, obs_noise=obs_noise) for _ in range(RUNS_PER_SEED)]
        seed_scores.append(float(np.mean(runs)))
        env.close()
    return seed_scores

# ---------------------------
# Experiment: Baseline vs Awareness
# ---------------------------
def main():
    # two variants:
    # Baseline: alpha=0.0 (no awareness)
    # Awareness: alpha=0.10 (use your CF modulation)
    baseline_params = {"alpha": 0.0, "noise_gain": 0.0}
    awareness_params = {"alpha": 0.10, "noise_gain": 0.2}

    # two conditions:
    # Clean (obs_noise = 0.0)
    # Stress (obs_noise = 0.5)  -> fairly disruptive for CartPole
    conditions = {"clean": 0.0, "stress": 0.5}

    results = []
    for cond_name, obs_noise in conditions.items():
        print(f"\nRunning condition: {cond_name} (obs_noise={obs_noise})")
        # baseline
        b_scores = evaluate_variant(baseline_params["alpha"], baseline_params["noise_gain"], SEEDS, obs_noise)
        print(" Baseline per-seed means:", [round(x, 2) for x in b_scores])
        # awareness
        a_scores = evaluate_variant(awareness_params["alpha"], awareness_params["noise_gain"], SEEDS, obs_noise)
        print(" Awareness per-seed means:", [round(x, 2) for x in a_scores])

        # aggregate
        results.append({
            "condition": cond_name,
            "variant": "baseline",
            "seed_scores": b_scores,
            "mean": float(np.mean(b_scores)),
            "std": float(np.std(b_scores, ddof=1))
        })
        results.append({
            "condition": cond_name,
            "variant": "awareness",
            "seed_scores": a_scores,
            "mean": float(np.mean(a_scores)),
            "std": float(np.std(a_scores, ddof=1))
        })

    # Save CSV (tidy)
    tidy_rows = []
    for r in results:
        for i, s in enumerate(SEEDS):
            tidy_rows.append({
                "condition": r["condition"],
                "variant": r["variant"],
                "seed": s,
                "mean_reward_seed": r["seed_scores"][i]
            })
    df_tidy = pd.DataFrame(tidy_rows)
    csv_name = "exp2_results.csv"
    df_tidy.to_csv(csv_name, index=False)
    print(f"\nSaved results CSV: {csv_name}")

    # Summary table (means and std)
    rows = []
    for r in results:
        rows.append({
            "condition": r["condition"],
            "variant": r["variant"],
            "mean": r["mean"],
            "std": r["std"]
        })
    df_summary = pd.DataFrame(rows)
    print("\nSummary:\n", df_summary)

    # ---------------------------
    # Plot: Bar chart comparison (clean vs stress)
    # ---------------------------
    # pivot
    pivot = df_summary.pivot(index="variant", columns="condition", values="mean")
    pivot_std = df_summary.pivot(index="variant", columns="condition", values="std")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(pivot.index))
    width = 0.35
    ax.bar(x - width/2, pivot["clean"], width, yerr=pivot_std["clean"], label="Clean", capsize=6)
    ax.bar(x + width/2, pivot["stress"], width, yerr=pivot_std["stress"], label="Stress (noisy obs)", capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.str.capitalize())
    ax.set_ylabel("Mean total reward (avg across seeds)")
    ax.set_title("Experiment 2 — Baseline vs Awareness (clean & stress)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plot_name = "exp2_plot_comparison.png"
    plt.savefig(plot_name, dpi=300)
    plt.close()
    print(f"Saved figure: {plot_name}")

    # ---------------------------
    # Plot: Per-seed scattering (shows stability across seeds)
    # ---------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    # collect per-seed
    for r in results:
        xs = [r["variant"] + "_" + r["condition"]] * len(SEEDS)
        ys = r["seed_scores"]
        # jitter x for visibility
        jitter = (np.random.rand(len(ys)) - 0.5) * 0.08
        xvals = np.arange(len(results))  # we will map positions
    # We'll plot grouped: baseline_clean, awareness_clean, baseline_stress, awareness_stress
    labels = []
    yvals = []
    for grp in ["baseline_clean", "awareness_clean", "baseline_stress", "awareness_stress"]:
        var, cond = grp.split("_")
        labels.append(var.capitalize() + "\n" + cond.capitalize())
        # find matching result
        match = next((r for r in results if r["variant"] == var and r["condition"] == cond), None)
        if match:
            yvals.append(match["seed_scores"])
        else:
            yvals.append([0]*len(SEEDS))
    # plot box + scatter
    positions = np.arange(len(labels))
    # boxplot
    ax2.boxplot(yvals, positions=positions, widths=0.6, patch_artist=True,
                boxprops=dict(facecolor="#D1E8FF", color="#2B7BB9"))
    # scatter seeds
    for i, arr in enumerate(yvals):
        jitter = (np.random.rand(len(arr)) - 0.5) * 0.12
        ax2.scatter(np.full(len(arr), positions[i]) + jitter, arr, color="#2B7BB9", alpha=0.9, s=30)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Mean reward per seed (avg of runs)")
    ax2.set_title("Experiment 2 — Per-seed stability (box + seeds)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plot_name2 = "exp2_plot_stress.png"
    plt.savefig(plot_name2, dpi=300)
    plt.close()
    print(f"Saved figure: {plot_name2}")

    print("\nExperiment 2 complete. Files generated:")
    print(" -", csv_name)
    print(" -", plot_name)
    print(" -", plot_name2)
    print("\nYou can attach these in your paper (CSV + PNG).")

if __name__ == "__main__":
    main()
