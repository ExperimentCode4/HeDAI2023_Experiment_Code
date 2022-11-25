from utils.logger import Logger
from .extractionflow.MIMICIVFlow import MimicIVExtractor
from .utils.utils import set_random_seed


def datasetmodule(args):
    # Add or set random seed
    if not getattr(args, 'seed', False):
        args.seed = 0

    set_random_seed(args.seed)

    args.logger = Logger(args)

    # If specific trainer flow is needed, we load it,
    # otherwise just the general task is loaded
    flow = None
    if args.dataset == 'mimiciv':
        flow = MimicIVExtractor(args)
    else:
        exit(f"Extraction flow for dataset {args.dataset} not implemented")

    # Extract dataset
    result = flow.run_flow()
    return result
