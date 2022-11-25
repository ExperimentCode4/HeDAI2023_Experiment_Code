from utils.logger import Logger
from . import get_flow
from .utils import set_random_seed
from .utils.FeatInitExperiment import FeatInitExperiment
from .utils.SequentialModelExperiment import SequentialModelExperiment
from .utils.hpo import hpo_experiment


def hgnnmodule(args):
    # Add or set random seed

    args.seed = 0
    args.logger = Logger(args)

    # Add best configuration to args
    set_random_seed(args.seed)

    # If we want hyperparameter tuning
    if getattr(args, "use_hpo", False):
        hpo_experiment(args)

    else:
        flow = get_flow(args.flow, args)
        result = flow.train()
        return result
