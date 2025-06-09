from abc import ABC, abstractmethod
import random


class BaseAgent(ABC):
    def __init__(self, k:int):
        self.k = k
        self.time = 1
        self.Q = [0.0] * k
        self.N = [0] * k
        self.history = []
        self.optimal_actions = []

    @abstractmethod
    def select_action(self) -> int:
        """
        Select an action based on the current policy.
        """
        return random.randint(0, self.k - 1)

    @abstractmethod
    def update(self, action: int, reward: float):
        """
        Update the agent's knowledge based on the action taken and the received reward.
        :param action: The action taken by the agent.
        :param reward: The reward received from the environment.
        """
        pass

    def step(self, env):
        """
        Take a step in the environment.
        :param env: The environment in which the agent operates.
        :return: The action selected and the reward received from the environment.
        """
        action = self.select_action()
        reward = env.step(action)
        self.update(action, reward)
        self.time += 1
        self.history.append(reward)

        # TODO: calculate optimal action
        optimal = env.optimal_action()
        is_optimal = int(action == optimal)
        self.optimal_actions.append(is_optimal)

        return action, reward
    
    def get_history(self):
        """
        Get the history of actions and rewards.
        :return: A list of reward.
        """
        return self.history

    def get_optimal_actions(self):
        """
        Get the list of whether optimal action is selected.
        """
        return self.optimal_actions