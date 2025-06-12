import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table


ACTIONS = [
    np.array([ 1,  0]),     # South
    np.array([-1,  0]),     # North
    np.array([ 0,  1]),     # East
    np.array([ 0, -1])      # West
]
ACTIONS_FIGS=['↓', '↑', '→', '←']
ACTION_PROB = 0.25
POS_A = [0, 1]
POS_A_PRIME = [4, 1]
POS_B = [0, 3]
POS_B_PRIME = [2, 3]

class GridworldEnv:
    """
    Gridworld environment for MDP and RL algorithms.
    States are grid cells, actions move the agent in four directions.
    """
    def __init__(
            self,
            nrows = 5,
            ncols = 5,
            start = np.array([0, 0]),
        ):
        """
        Initialize the Gridworld.

        Args:
            nrows: number of rows
            ncols: number of cols
            start: [row, col] start position.
        """
        self.nrows = nrows
        self.ncols = ncols
        self.start = np.array(start, dtype=int)
        self.pos = self.start.copy()

    def set_pos(self, pos):
        self.pos = np.array(pos, dtype=int)
        return self.pos.copy()
    
    def reset(self) -> np.ndarray:
        """
        Reset agent to start position.

        Returns:
            current position as (row, col) array.
        """
        self.pos = self.start.copy()
        return self.pos.copy()

    def step(self, action: np.ndarray):
        """
        Take an action in the environment.

        Args:
            action: one of the ACTIONS vectors.

        Returns:
            pos_next: new position array
            reward: float reward for the transition
        """
        if list(self.pos) == POS_A:
            self.pos = np.array(POS_A_PRIME)
            return self.pos.copy(), 10
        if list(self.pos) == POS_B:
            self.pos = np.array(POS_B_PRIME)
            return self.pos.copy(), 5
        
        next_pos = self.pos + action
        r, c = next_pos
        if r < 0 or r >= self.nrows or c < 0 or c >= self.ncols:
            reward = -1
            next_pos = self.pos.copy()
        else:
            reward = 0
        self.pos = next_pos
        return self.pos.copy(), reward
    

class RandomAgent:
    """
    Agent with equal probability to go each direction
    """
    def __init__(self, gamma = 0.9, nrows = 5, ncols = 5, env = GridworldEnv) -> None:
        self.policy = np.array([0.25] * 4)
        self.gamma = gamma
        self.nrows = nrows
        self.ncols = ncols
        self.env = env()


    def select_action(self) -> np.array:
        idx = np.random.randint(0, 4)
        action = ACTIONS[idx]

        return action
    
    
    def cal_v_from_state(self):
        """
        Using DP to caluculate Expectation
        """
        self.v = np.zeros((self.nrows, self.ncols))
        while True:
            value = np.zeros_like(self.v)
            for row in range(self.nrows):
                for col in range(self.ncols):
                    for idx, action in enumerate(ACTIONS):
                        self.env.set_pos(np.array([row, col]))
                        next_pos, reward = self.env.step(action)
                        next_row, next_col = next_pos[0], next_pos[1]
                        # Bellman Equation
                        value[row, col] += self.policy[idx] * (reward + self.gamma * self.v[next_row, next_col])
            if np.sum(np.abs(self.v - value)) < 1e-4:
                break
            self.v = value

        return self.v
    
    
    def cal_v_star(self):
        self.v_star = np.zeros((self.nrows, self.ncols))
        while True:
            v_star_new = np.zeros_like(self.v_star)
            for row in range(self.nrows):
                for col in range(self.ncols):
                    action_returns = []
                    for idx, action in enumerate(ACTIONS):
                        self.env.set_pos(np.array([row, col]))
                        next_pos, reward = self.env.step(action)
                        next_row, next_col = next_pos  
                        action_returns.append(reward + self.gamma * self.v_star[next_row, next_col])
                    v_star_new[row, col] = max(action_returns)
            if np.max(np.abs(v_star_new - self.v_star)) < 1e-4:
                break
            self.v_star = v_star_new

        return self.v_star
    

    def cal_policy_star(self):
        self.policy_star = np.zeros((self.nrows, self.ncols, len(ACTIONS)))
        for row in range(self.nrows):
            for col in range(self.ncols):
                action_returns = []
                for idx, action in enumerate(ACTIONS):
                    self.env.set_pos(np.array([row, col]))
                    next_pos, reward = self.env.step(action)
                    next_row, next_col = next_pos
                    action_returns.append(reward + self.gamma * self.v_star[next_row, next_col])
                max_val = max(action_returns)
                best_idxs = [1 if v == max_val else 0 for i, v in enumerate(action_returns) ]
                self.policy_star[row, col] = np.array(best_idxs)

        return self.policy_star
    

    def plot_v(self):
        assert hasattr(self, 'v')
        self.v = np.round(self.v, decimals=1)
        _, ax = plt.subplots()
        ax.set_axis_off()
        table = Table(ax, bbox=[0, 0, 1, 1])

        width, height = 1.0 / self.ncols, 1.0 / self.nrows
        for (row, col), value in np.ndenumerate(self.v):
            text = str(value)

            if [row, col] == POS_A:
                text += " (A)"
            elif [row, col] == POS_A_PRIME:
                text += " (A')"
            elif [row, col] == POS_B:
                text += " (B)"
            elif [row, col] == POS_B_PRIME:
                text += " (B')"

            table.add_cell(row, col, width, height, loc='center', 
                           facecolor='white', text=text)
        for i in range(len(self.v)):
            table.add_cell(i, -1, width, height, text=i + 1, loc='right',
                           edgecolor='none', facecolor='none')
            table.add_cell(-1, i, width, height/2, text=i + 1, loc='center',
                           edgecolor='none', facecolor='none')

        ax.add_table(table)
        save_path = './images/figure_3_2.png'
        plt.savefig(save_path)
        print(f">>> Result saved to {save_path}")
        plt.close()


    def plot_v_star(self):
        assert hasattr(self, 'v_star')
        self.v_star = np.round(self.v_star, decimals=1)
        _, ax = plt.subplots()
        ax.set_axis_off()
        table = Table(ax, bbox=[0, 0, 1, 1])

        width, height = 1.0 / self.ncols, 1.0 / self.nrows
        for (row, col), value in np.ndenumerate(self.v_star):
            text = str(value)

            if [row, col] == POS_A:
                text += " (A)"
            elif [row, col] == POS_A_PRIME:
                text += " (A')"
            elif [row, col] == POS_B:
                text += " (B)"
            elif [row, col] == POS_B_PRIME:
                text += " (B')"

            table.add_cell(row, col, width, height, loc='center', 
                           facecolor='white', text=text)
        for i in range(len(self.v_star)):
            table.add_cell(i, -1, width, height, text=i + 1, loc='right',
                           edgecolor='none', facecolor='none')
            table.add_cell(-1, i, width, height/2, text=i + 1, loc='center',
                           edgecolor='none', facecolor='none')

        ax.add_table(table)
        save_path = './images/figure_3_5.png'
        plt.savefig(save_path)
        print(f">>> Result saved to {save_path}")
        plt.close()


    def plot_policy_star(self):
        assert hasattr(self, 'policy_star')
        _, ax = plt.subplots()
        ax.set_axis_off()
        table = Table(ax, bbox=[0, 0, 1, 1])

        width, height = 1.0 / self.ncols, 1.0 / self.nrows
        for row in range(self.nrows):
            for col in range(self.ncols):
                a_star = self.policy_star[row, col, :]
                text = ''
                for idx, flag in enumerate(a_star):
                    if flag:
                        text += ACTIONS_FIGS[idx]

                if [row, col] == POS_A:
                    text += " (A)"
                elif [row, col] == POS_A_PRIME:
                    text += " (A')"
                elif [row, col] == POS_B:
                    text += " (B)"
                elif [row, col] == POS_B_PRIME:
                    text += " (B')"

                table.add_cell(row, col, width, height, loc='center',
                            facecolor='white', text=text)
        for i in range(len(self.v_star)):
            table.add_cell(i, -1, width, height, text=i + 1, loc='right',
                           edgecolor='none', facecolor='none')
            table.add_cell(-1, i, width, height/2, text=i + 1, loc='center',
                           edgecolor='none', facecolor='none')
            
        ax.add_table(table)
        save_path = './images/figure_3_5_policy.png'
        plt.savefig(save_path)
        print(f">>> Result saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    matplotlib.use('Agg')

    agent = RandomAgent()
    # agent.cal_v_from_state()
    # agent.plot_v()
    agent.cal_v_star()
    agent.plot_v_star()
    agent.cal_policy_star()
    agent.plot_policy_star()