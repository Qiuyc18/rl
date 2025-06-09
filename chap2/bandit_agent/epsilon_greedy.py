from .base_agent import BaseAgent
import random


class EpsilonGreedyAgent(BaseAgent):
    def __init__(self, 
                 k: int, 
                 epsilon: float = 0.1,
                 alpha: float = None):
        """
        :param epsilon: Hyperparam of the probability to explore
        :param alpha: None for Stationary Problem and (0, 1] for Nonstationary Problem
        """
        super().__init__(k)
        self.epsilon = epsilon
        self.alpha = alpha
    
    def select_action(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.k)
        
        return max(range(self.k), key=lambda a: self.Q[a])
    
    def update(self, action: int, reward: float):
        self.N[action] += 1
        if self.alpha is None:
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
        else:
            self.Q[action] += self.alpha * (reward - self.Q[action])
        