import numpy as np
import matplotlib.pyplot as plt
from env.k_arm_bandit import KArmBandit
from policy.gradient_policy import GradientPolicy

k = 10
steps = 1000
epochs = 2000
mu = 4
sigma = 1

policies = {
    "α = 0.1 (baseline)": GradientPolicy(alpha=0.1, k=k),
    "α = 0.4 (baseline)":      GradientPolicy(alpha=0.4, k=k),
    "α = 0.1 (no baseline)":     GradientPolicy(alpha=0.1, k=k, baseline=False),
    "α = 0.4 (no baseline)":     GradientPolicy(alpha=0.4, k=k, baseline=False),
}

results = {}

for name, pol in policies.items():
    bandit = KArmBandit(policy=pol, k=k, mu=mu, sigma=sigma, stationary=True)
    avg_rewards, optimal_pct, best_mean = bandit.run(steps=steps, epochs=epochs)
    results[name] = (avg_rewards, optimal_pct, best_mean)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name, (avg_rewards, _, best_mean) in results.items():
    plt.plot(avg_rewards, label=f"{name}")

plt.axhline(
    np.mean([v[2] for v in results.values()]),
    color="red",
    linestyle="--",
    label="Best Possible Mean"
)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward over Time")
plt.yticks(np.arange(3.1, 4.6, 0.1))
plt.legend()

plt.subplot(1, 2, 2)
for name, (_, optimal_pct, _) in results.items():
    plt.plot(optimal_pct, label=f"{name}")
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("Optimal Action Percentage over Time")
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()
plt.show()