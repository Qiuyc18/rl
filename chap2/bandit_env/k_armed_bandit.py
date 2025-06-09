import numpy as np


class KArmedBandit:
    def __init__(self, k: int, mu: float = 0.0, sigma: float = 1.0):
        """
        k: numbers of the arm
        mu: mean of the reward distribution
        sigma: standard deviation of the reward distribution
        """
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.q_true = []  # True action values
        self.reset()

    def reset(self):
        self.q_true = np.random.normal(self.mu, self.sigma, self.k)
        return None

    def step(self, action: int) -> float:
        reward = np.random.rand() + self.q_true[action]
        return reward
    
    def optimal_action(self) -> int:
        """
        Return optimal action in current environment.
        """
        return int(np.argmax(self.q_true))
    