import numpy as np
from policy.base_policy import BasePolicy

class IncrementalPolicy(BasePolicy):
    def __init__(self, epsilon=0, k=10, initial=0, alpha=0):
        self.epsilon = epsilon
        self.k = k
        self.initial = initial
        self.q_values = [initial] * k
        self.action_counts = [0] * k
        self.alpha = alpha

    def action(self):
        prob = np.random.random()
        if prob < self.epsilon:
            return np.random.randint(0, self.k)
        else:
            return np.argmax(self.q_values)
    
    def update(self, action, reward):
        step_size = self.alpha
        if self.alpha == 0:
            self.action_counts[action] += 1
            step_size = 1 / self.action_counts[action]

        self.q_values[action] += step_size * (reward - self.q_values[action])

    def reset(self):
        self.q_values = [self.initial] * self.k
        self.action_counts = [0] * self.k