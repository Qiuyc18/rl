from .base_agent import BaseAgent
import numpy as np


class UCBAgent(BaseAgent):
    def __init__(self,
                 k: int,
                 c: float = 0.1):
        """
        :param c: Hyperparam to control the degree of exploration 
        """
        super().__init__(k)
        self.c = c;

    def select_action(self) -> int:
        # At beginning, every N_t(a) == 0 and should be explored fist 
        for a in range(self.k):
            if self.N[a] == 0:
                return a
            
        ucb_values = [
            self.Q[a] + self.c * np.sqrt(np.log(self.time) / self.N[a])
            for a in range(self.k)
        ]
        return int(np.argmax(ucb_values))

    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
