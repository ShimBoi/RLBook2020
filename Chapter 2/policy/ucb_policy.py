import numpy as np
from policy.base_policy import BasePolicy

class UCBPolicy(BasePolicy):
    def __init__(self, c=0, k=10, initial=0):
        self.k = k
        self.initial = initial
        self.q_values = [initial] * k
        self.action_counts = [0] * k
        self.t = 0
        self.c = c

    def action(self):
        self.t += 1
        ucb_values = []
        for a in range(self.k):
            if self.action_counts[a] == 0:
                return a
            else:
                score = self.q_values[a] + self.c * np.sqrt(np.log(self.t)) / self.action_counts[a]
                ucb_values.append(score)

        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        num_action_taken = self.action_counts[action]

        self.action_counts[action] += 1

        # if denominator is 0, then assume some initial value
        if num_action_taken == 0:
            return

        # Q_t(a) = sum of rewards when a taken prior to t / number of times a taken prior to t
        self.q_values[action] += 1 / self.action_counts[action] * (reward - self.q_values[action])

    def reset(self):
        self.q_values = [self.initial] * self.k
        self.action_counts = [0] * self.k
        self.t = 0