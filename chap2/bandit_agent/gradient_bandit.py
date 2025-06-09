from .base_agent import BaseAgent
import numpy as np


class GradientBanditAgent(BaseAgent):
    def __init__(self,
                 k: int,
                 alpha: float = 0.1,
                 use_baseline: bool = True):
        """
        :param alpha: (0, 1] for Nonstationary Problem
        :param use_baseline: whether to use baseline
        """
        super().__init__(k)
        assert (alpha > 0.0) and (alpha <= 1.0)
        self.alpha = alpha
        self.use_baseline = use_baseline

        self.H = np.zeros(k)
        self.pi = np.ones(k) / k
        self.avg_reward = 0.0

    def select_action(self) -> int:
        # action = max(range(self.k), lambda a: self.pi[a])
        action = np.random.choice(self.k, p=self.pi)
        return int(action)

    def update(self, action, reward):
        # If not use_baseline, self.avg_reward remains 0.0
        if self.use_baseline:
            self.avg_reward += (reward - self.avg_reward) / self.time            

        for a in range(self.k):
            if a == action:
                self.H[a] += self.alpha * (reward - self.avg_reward) * (1 - self.pi[a])
            else:
                self.H[a] -= self.alpha * (reward - self.avg_reward) * self.pi[a]

        expH = np.exp(self.H - np.max(self.H))
        self.pi = expH / np.sum(expH)