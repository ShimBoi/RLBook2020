import numpy as np
import matplotlib.pyplot as plt
from policy.epsilon_greedy_policy import EpsilonGreedyPolicy
from policy.incremental_policy import IncrementalPolicy
from env.k_arm_bandit import KArmBandit

k = 10
steps = 10000
epochs = 2000
mu = 0
sigma = 1

policies = {
    "α = 0.1":     IncrementalPolicy(epsilon=0.1, alpha=0.1, k=k, initial=0),
    "ε = 0.1":      IncrementalPolicy(epsilon=0.1, k=k, initial=0),
}

results = {}

for name, pol in policies.items():
    bandit = KArmBandit(policy=pol, k=k, mu=mu, sigma=sigma, stationary=False)
    avg_rewards, optimal_pct, best_mean = bandit.run(steps=steps, epochs=epochs)
    results[name] = (avg_rewards, optimal_pct, best_mean)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name, (avg_rewards, _, best_mean) in results.items():
    plt.plot(avg_rewards, label=f"{name}")

plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward over Time")
plt.yticks(np.arange(0.1, 2.0, 0.1))
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