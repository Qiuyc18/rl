import numpy as np
from .base_agent import BaseAgent


class OptimisticInitAgent(BaseAgent):
    def __init__(self,
                 k: int,
                 q_init: float = 5.0,
                 alpha: float = None):
        super().__init__(k)
        self.q_init = q_init
        self.Q = [q_init] * k
        self.alpha = alpha
    
    def select_action(self) -> int:
        return int(np.argmax(self.Q))
    
    def update(self, action: int, reward: float):
        self.N[action] += 1
        if self.alpha is None:
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
        else:
            self.Q[action] += self.alpha * (reward - self.Q[action])