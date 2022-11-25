import optuna
import gc

from hgnnmodule import get_flow


def func_search(trial):
    return {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
        'n_layers': trial.suggest_int('n_layers', 2, 3),
        'lr': trial.suggest_categorical('lr', [1e-3, 5e-3, 1e-2]),
        'dropout': trial.suggest_uniform('dropout', 0.0, 0.5)
    }


def hpo_experiment(args, **kwargs):

    tool = AutoML(args, n_trials=100, func_search=func_search)
    result = tool.run()
    print("\nFinal results:\n")
    print(result)
    return result


class AutoML:
    def __init__(self, args, n_trials=3, **kwargs):
        self.args = args
        self.flow = None
        self.seed = kwargs.pop("seed") if "seed" in kwargs else [1]
        assert "func_search" in kwargs
        self.func_search = kwargs["func_search"]
        self.metric = kwargs["metric"] if "metric" in kwargs else None
        self.n_trials = n_trials
        self.best_result = None
        self.best_params = None
        self.default_params = kwargs

    def _objective(self, trials):
        gc.collect()
        args = self.args
        cur_params = self.func_search(trials)
        args.__setattr__('hyperparams', cur_params)
        for key, value in cur_params.items():
            args.__setattr__(key, value)

        # Build new flow
        flow = get_flow(args.flow, args)
        result = flow.train()[0]['test']['Micro_f1'].item()
        if isinstance(result, tuple):
            result = (result[0] + result[1]) / 2
        if self.best_result is None or result > self.best_result:
            self.best_result = result
            self.best_params = cur_params
        return result

    def run(self):
        study = optuna.create_study(direction="maximize")
        # Optimizer uses Tree-structured Parzen Estimator as a default
        study.optimize(self._objective, n_trials=self.n_trials, n_jobs=1)
        print(study.best_params)
        return self.best_result
