import matplotlib.pyplot as plt
import numpy as np


class BanditVisualizer:
    def __init__(self):
        # label -> runs_data
        self.data_dict = {}

    def add_data(self, label: str, runs_data):
        """
        Add data for a specific label.
        :param label: The label for the data (e.g., 'epsilon-greedy', 'UCB', etc.)
        :param runs_data: A list or array of rewards from multiple runs.
        """
        self.data_dict[label] = np.array(runs_data)

    def plot(self, 
             title: str = None, 
             xlabel: str = "Step", 
             ylabel: str = "Average Reward",
             alpha=0):
        """
        Plot the average rewards for each label.
        :param title: The title of the plot.
        :param xlabel: The label for the x-axis.
        :param ylabel: The label for the y-axis.
        """
        plt.figure()
        for label, data in self.data_dict.items():
            avg_rewards = np.mean(data, axis=0)
            steps = np.arange(data.shape[1]) 
            plt.plot(avg_rewards, label=label, alpha=alpha)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    