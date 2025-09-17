import numpy as np
from policy.base_policy import BasePolicy

class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, epsilon=0, k=10, initial=0):
        self.epsilon = epsilon
        self.k = k
        self.initial = initial
        self.q_values = [initial] * k
        self.action_counts = [0] * k
        self.reward_counts = [0] * k

    def action(self):
        prob = np.random.random()
        if prob < self.epsilon:
            return np.random.randint(0, self.k)
        else:
            return np.argmax(self.q_values)
    
    def update(self, action, reward):
        num_action_taken = self.action_counts[action]
        sum_action_rewards = self.reward_counts[action]

        self.action_counts[action] += 1
        self.reward_counts[action] += reward
        # if denominator is 0, then assume some initial value
        if num_action_taken == 0:
            return

        # Q_t(a) = sum of rewards when a taken prior to t / number of times a taken prior to t
        self.q_values[action] = sum_action_rewards / num_action_taken

    def reset(self):
        self.q_values = [self.initial] * self.k
        self.action_counts = [0] * self.k
        self.reward_counts = [0] * self.k