import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 160

# -------------------------------
# Awareness Agent (Your Theory)
# -------------------------------
class AwarenessAgent:
    def __init__(self, alpha, noise_gain):
        self.alpha = alpha
        self.noise_gain = noise_gain
        self.w = np.random.randn(4, 2) * 0.1

    def ramanujan_cf(self, m):
        return 1 + 1/(1 + m[0] + 1/(1 + m[1] + 1/(1 + m[2])))

    def policy(self, obs):
        m = np.tanh(np.random.randn(3) * self.noise_gain)
        cf = self.ramanujan_cf(m)
        mod = self.w * (1 + self.alpha * cf)
        logits = obs @ mod
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.argmax(probs)


# -------------------------------
# Run one episode
# -------------------------------
def run_episode(env, agent):
    obs = env.reset()[0]
    total = 0
    for _ in range(200):
        act = agent.policy(obs)
        obs, r, done, _, _ = env.step(act)
        total += r
        if done:
            break
    return total


# -------------------------------
# Experiment Loop
# -------------------------------
def evaluate(alpha, noise, seeds):
    env = gym.make("CartPole-v1")
    scores = []
    for s in seeds:
        np.random.seed(s)
        agent = AwarenessAgent(alpha, noise)
        runs = [run_episode(env, agent) for _ in range(25)]
        scores.append(np.mean(runs))
    env.close()
    return scores


ALPHAS = [0.00, 0.05, 0.10, 0.20, 0.30]
NOISE = 0.2
SEEDS = [101,102,103,104,105]

results = {}

for a in ALPHAS:
    scores = evaluate(a, NOISE, SEEDS)
    results[a] = scores
    print(f"Alpha={a}: {scores}  Mean={np.mean(scores):.2f}")

# Save CSV
df = pd.DataFrame({
    "alpha": list(results.keys()),
    "mean_reward": [np.mean(v) for v in results.values()],
    "std": [np.std(v) for v in results.values()]
})
df.to_csv("exp1_results.csv", index=False)


# -------------------------------
# Graph (NCA Publication Style)
# -------------------------------
plt.figure(figsize=(7,5))
plt.plot(df["alpha"], df["mean_reward"], marker="o", linewidth=2)
plt.fill_between(df["alpha"],
                 df["mean_reward"] - df["std"],
                 df["mean_reward"] + df["std"],
                 alpha=0.3)

plt.title("Experiment 1 – Alpha Sensitivity & Multi-Seed Stability")
plt.xlabel("Alpha (Awareness Strength)")
plt.ylabel("Mean Total Reward")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("exp1_graph.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nExperiment 1 Complete.\nCSV saved: exp1_results.csv\nGraph saved: exp1_graph.png")
