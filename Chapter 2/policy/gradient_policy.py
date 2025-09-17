import numpy as np
from policy.base_policy import BasePolicy

def softmax(x):
    num = np.exp(x - np.max(x))
    return num / np.sum(num)

class GradientPolicy(BasePolicy):
    def __init__(self, alpha=0.1, k=10, baseline=True):
        self.k = k
        self.preferences = [0] * k
        self.probs = [1/k] * k
        self.alpha = alpha
        self.baseline = baseline
        self.total_reward = 0.0
        self.total_steps = 0

    def action(self):
        self.probs = softmax(self.preferences)
        return np.random.choice(np.arange(self.k), p=self.probs)
    
    def update(self, action, reward):
        self.total_steps += 1
        self.total_reward += reward
        baseline = self.total_reward / self.total_steps if self.baseline else 0

        self.preferences[action] += self.alpha * ((reward - baseline) * (1 - self.probs[action]))

        for a in range(self.k):
            if a != action:
                self.preferences[a] -= self.alpha * ((reward - baseline) * self.probs[a])

    def reset(self):
        self.preferences = [0] * self.k
        self.probs = [1/self.k] * self.k
        self.total_reward = 0.0
        self.total_steps = 0