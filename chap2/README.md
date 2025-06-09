# Code collection for *Chapter 2 multi‑armed bandit*

## Project layout

```
chap2/
├── bandit_agent/     # Implementations of different bandit algorithms
├── bandit_env/       # Simple stochastic bandit environment
├── experiments/      # Experiment runner and YAML config
├── output/           # Sample plots generated from experiments
├── utils/            # Helper utilities (plotting etc.)
├── main.py           # Entry point to run an experiment
└── main.ipynb        # Jupyter version of the example
```

### Agents
The following agents are provided inside `chap2/bandit_agent`:

- **EpsilonGreedyAgent** – standard ε‑greedy action selection with optional constant step size.
- **OptimisticInitAgent** – chooses greedily from optimistically initialised values.
- **UCBAgent** – implements the Upper Confidence Bound algorithm for exploration.
- **GradientBanditAgent** – policy gradient method with optional baseline.

### Environment
`chap2/bandit_env/k_armed_bandit.py` defines a simple stochastic environment where each arm returns a reward sampled from a normal distribution.  It also exposes `optimal_action()` for evaluation purposes.

### Experiments
Experiments are configured using `chap2/experiments/experiment.yaml`.  The `ExperimentRunner` class loads this configuration, instantiates the agents and runs multiple trials in parallel if desired.

## Running
1. Install the Python dependencies (e.g. using `pip`):
   ```bash
   pip install numpy matplotlib tqdm pyyaml
   ```
2. Execute an experiment:
   ```bash
   cd chap2
   python -m main
   ```
   The parameters (number of bandit arms, steps, strategies, etc.) are read from `./chap2/experiments/experiment.yaml`.
3. After running, plots summarising the reward curves are displayed and can also be found under `./chap2/output`.

You can alternatively open `./chap2/main.ipynb` for an interactive notebook version of the same experiment.

## Visualisation
`utils/plotting.py` contains `BanditVisualizer` for plotting the average reward of each strategy across runs.