import numpy as np

class KArmBandit:
    
    def __init__(self, policy=None, k=10, mu=0, sigma=1, stationary=True):
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.q_stars = np.random.normal(mu, sigma, k)
        self.stationary = stationary
        self.policy = policy

    def pull(self, action):
        reward = np.random.normal(self.q_stars[action], 1)

        if not self.stationary:
            self.q_stars += np.random.normal(0, 0.01, self.k)

        return reward

    def reset(self):
        self.policy.reset()
        self.q_stars = np.random.normal(self.mu, self.sigma, self.k)
        if not self.stationary:
            pass

    def run(self, steps=1000, epochs=2000):
        rewards = np.zeros((epochs, steps))
        optimal_actions = np.zeros((epochs, steps), dtype=int)
        best_means = np.zeros(epochs)

        for epoch in range(epochs):
            self.reset()

            for step in range(steps):
                optimal_action = np.argmax(self.q_stars)
                best_means[epoch] = self.q_stars[optimal_action]
                action = self.policy.action()
                reward = self.pull(action)
                self.policy.update(action, reward)

                rewards[epoch, step] = reward
                optimal_actions[epoch, step] = (action == optimal_action)

        avg_rewards = rewards.mean(axis=0)
        optimal_action_percents = optimal_actions.mean(axis=0) * 100
        highest_possible_avg_reward = best_means.mean()

        return avg_rewards, optimal_action_percents, highest_possible_avg_reward
