import os
import yaml
import importlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from bandit_env.k_armed_bandit import KArmedBandit
from tqdm import tqdm


def _run_one(args):
    k, steps, AgentClass, params = args
    env = KArmedBandit(k)
    agent = AgentClass(k, **params)
    for _ in range(steps):
        agent.step(env)
    return agent.get_history()


class ExperimentRunner:
    """
    Load from experiment.yaml
    """
    def __init__(self, config_path: str = None):
        base_dir = os.path.dirname(__file__)
        if config_path is None:
            config_path = os.path.join(base_dir, 'experiment.yaml')
        elif os.path.isdir(config_path):
            config_path = os.path.join(config_path, 'experiment.yaml')
        self.config_path = os.path.abspath(config_path)

        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Can't find config file: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.k = self.config.get('k', 10)
        self.runs = self.config['runs']                 # 实验轮数
        self.steps = self.config['steps']               # 每轮步数
        self.strategies = self.config['strategies']     # 策略列表
    
    def run(self, parallel: bool = False) -> dict:
        """
        Run the experiment based on the loaded configuration.
        :param parallel: Whether to run strategies in parallel.
        :return: Dictionary containing the results of the experiment.
        """
        results = {}
        for strat in self.strategies:
            name = strat['name']
            class_path = strat['class_path']
            AgentClass = self._load_class(class_path)

            # hyperparameters: dict or list
            raw_params = strat.get('hyperparameters', [{}])
            if isinstance(raw_params, dict):
                hyper_list = [raw_params]
            elif isinstance(raw_params, list) and all(isinstance(p, dict) and len(p) == 1 for p in raw_params):
                keys = [list(p.keys())[0] for p in raw_params]
                if len(set(keys)) == len(raw_params):
                    combined = {k: next(p[k] for p in raw_params if k in p) for k in keys}
                    hyper_list = [combined]
                else:
                    hyper_list = raw_params
            else:
                hyper_list = raw_params

            for params in hyper_list:
                label = self._make_label(name, params)
                print(f">>> Running {label}\n>>> params: {params}\n>>> parallel: {parallel}")
                data = self._run_strategy(AgentClass, params, parallel)
                results[label] = data
                print(">>> Done!")
        return results
        
    def _run_strategy(self, AgentClass, params: dict, parallel: bool) -> np.ndarray:
        """
        Run multiple rounds for different strageties and hyperparams
        """        
        args_list = [(self.k, self.steps, AgentClass, params)] * self.runs
        if parallel:
            with ProcessPoolExecutor() as pool:
                histories = list(tqdm(
                    pool.map(_run_one, args_list),
                    total=self.runs,
                    desc=f"Parallel: {AgentClass.__name__}"
                ))
        else:
            histories = [ _run_one(a) for a in tqdm(args_list, desc=...) ]

        return np.array(histories)
    
    def _load_class(self, class_path:str):
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _make_label(self, name: str, params: dict) -> str:
        if params:
            ps = ','.join(f"{k}={v}" for k, v in params.items())
            return f"{name}({ps})"
        else:
            return name
        

