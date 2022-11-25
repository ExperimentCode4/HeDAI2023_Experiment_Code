import torch
from abc import ABC, abstractmethod

from hgnnmodule.utils.NodeClassificationTask import NodeClassificationTask


class BaseFlow(ABC):
    candidate_optimizer = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD
    }

    def __init__(self, args):
        """

        Parameters
        ----------
        args

        Attributes
        -------------
        evaluate_interval: int
            the interval of evaluation in validation
        """
        super(BaseFlow, self).__init__()
        self.evaluator = None
        self.evaluate_interval = 1
        self._checkpoint = None

        self.args = args
        self.path = args.path
        self.logger = self.args.logger
        self.model_name = args.model
        self.device = args.device
        self.task = NodeClassificationTask(args)
        self.hg = self.task.get_graph().to(self.device)
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.optimizer = args.optimizer
        self.loss_fn = self.task.get_loss_fn()
        self.loss_func = self.task.loss_func
        self.input_feature = None


    @abstractmethod
    def train(self):
        pass
    
    def _full_train_step(self):
        r"""
        Train with a full_batch graph
        """
        raise NotImplementedError
    
    def _mini_train_step(self):
        r"""
        Train with a mini_batch seed nodes graph
        """
        raise NotImplementedError
    
    def _full_test_step(self, modes):
        r"""
        Test with a full_batch graph
        """
        raise NotImplementedError
    
    def _mini_test_step(self, modes):
        r"""
        Test with a mini_batch seed nodes graph
        """
        raise NotImplementedError
