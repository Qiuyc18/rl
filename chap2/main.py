from experiments.experiment_runner import ExperimentRunner
from utils.plotting import BanditVisualizer


def main():
    runner = ExperimentRunner()
    results = runner.run(parallel=True)

    vis = BanditVisualizer()
    for label, data in results.items():
        vis.add_data(label, data)
    vis.plot(title='Results', alpha=0.8)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()      
    main()