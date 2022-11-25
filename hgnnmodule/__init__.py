from .models import *
from .utils import *
from .utils.FeatInitExperiment import FeatInitExperiment
from .utils.SequentialModelExperiment import SequentialModelExperiment


def get_flow(flow_name, args):
    if flow_name == 'feat_init':
        return FeatInitExperiment(args)
    elif flow_name == 'seq_model':
        return SequentialModelExperiment(args)
