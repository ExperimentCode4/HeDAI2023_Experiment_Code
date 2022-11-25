import torch.nn.functional as F
import torch.nn as nn
from hgnnmodule.utils.NodeClassificationDataset import NodeClassificationDataset
from hgnnmodule.utils import Evaluator
from hgnnmodule.utils.BCELoss import BCELoss
from hgnnmodule.utils.HierarchicalLoss import HierarchicalLoss


class NodeClassificationTask:

    def __init__(self, args):
        super(NodeClassificationTask, self).__init__()
        self.dataset = NodeClassificationDataset(args.dataset)
        self.logger = args.logger
        self.loss_func = args.loss_func
        self.loss_fn = None
        self.evaluator = None
        if hasattr(args, 'validation'):
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_idx(args.validation)
        else:
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_idx()
        self.evaluator = Evaluator(args.seed)
        self.labels = self.dataset.get_labels().to(args.device)
        self.multi_label = self.dataset.multi_label
        self.extras = self.dataset.extras
        self.device = args.device

        if hasattr(args, 'evaluation_metric'):
            self.evaluation_metric = args.evaluation_metric
        else:
            self.evaluation_metric = 'f1'

    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
        if self.multi_label:
            if self.loss_func == 'bce':
                return BCELoss(
                    nn.BCELoss(),
                    self.dataset.category
                )
            # HMCN variant from http://proceedings.mlr.press/v80/wehrmann18a/wehrmann18a.pdf
            # This variant uses a linear layer from the final convolution to each hierarchical level
            elif self.loss_func == 'hmcn_out':
                return HierarchicalLoss(
                    self.loss_func,
                    nn.BCELoss(),
                    self.dataset.category,
                    self.extras['diag_level'],
                    self.extras['rollup_maps'],
                    self.device
                )
        return F.cross_entropy

    def get_evaluator(self, name):
        if name == 'acc':
            return self.evaluator.cal_acc
        elif name == 'f1_lr':
            return self.evaluator.nc_with_LR
        elif name == 'f1':
            return self.evaluator.f1_node_classification

    def optimize_threshold(self, logits, mode='valid'):
        if mode == 'test':
            mask = self.test_idx
        elif mode == 'valid':
            mask = self.val_idx
        elif mode == 'train':
            mask = self.train_idx

        self.evaluator.fine_tune_threshold(logits[mask], self.labels[mask])

    def evaluate(self, logits, mode='test', info=True):
        if mode == 'test':
            mask = self.test_idx
        elif mode == 'valid':
            mask = self.val_idx
        elif mode == 'train':
            mask = self.train_idx
            
        if self.evaluation_metric == 'acc':
            acc = self.evaluator.cal_acc(self.labels[mask], logits[mask])
            return dict(Accuracy=acc)
        elif self.evaluation_metric == 'f1':
            f1_dict = self.evaluator.f1_node_classification(self.labels[mask], logits[mask])
            return f1_dict
        else:
            raise ValueError('The evaluation metric is not supported!')

    def get_idx(self):
        return self.train_idx, self.val_idx, self.test_idx

    def get_labels(self):
        return self.labels
