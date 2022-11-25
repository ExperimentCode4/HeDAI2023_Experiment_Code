import dgl
import torch
from tqdm import tqdm

from hgnnmodule.models.HSAGE import HSAGE
from hgnnmodule.utils.EarlyStopping import EarlyStopping
from hgnnmodule.utils.NodeClassificationTask import NodeClassificationTask


class FeatInitExperiment:
    candidate_optimizer = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD
    }

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.path = self.args.path
        self.task = NodeClassificationTask(args)
        self.args.category = self.task.dataset.category
        self.hyperparams = self.set_hyperparameters()
        self.category = self.args.category
        self.experiment_name = self.args.experiment_name
        self.hg = self.task.get_graph().to(self.device)
        self.feats = self.hg.ndata.pop('h', {})
        self.cur_layer = None
        self.model = None
        self.evaluator = None
        self.evaluate_interval = 1
        self._checkpoint = None
        self.logger = self.args.logger
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.optimizer = args.optimizer
        self.loss_fn = self.task.get_loss_fn()
        self.loss_func = self.task.loss_func
        self.input_feature = None

        self.train_idx, self.valid_idx, self.test_idx = self.task.get_idx()
        self.labels = self.task.get_labels().to(self.device)

        if self.args.mini_batch_flag:
            sampler = dgl.dataloading.MultiLayerNeighborSampler([self.args.fanout] * self.args.n_layers)

            self.train_loader = dgl.dataloading.DataLoader(
                self.hg,
                {self.category: self.train_idx.to(self.device)},
                sampler,
                batch_size=self.args.batch_size,
                device=self.device,
                shuffle=True,
                num_workers=0
            )

    def create_model(self, graph, feats, h_dim, out_dim, n_layers, dropout, lr, weight_decay, loss_func, extras,
                     device):
        if self.args.model == 'HSAGE':
            self.model = HSAGE(graph, feats, h_dim, out_dim, n_layers, dropout, loss_func, extras, device).to(device)
        self.opt = self.candidate_optimizer[self.optimizer](self.model.parameters(),
                                                            lr=lr,
                                                            weight_decay=weight_decay)

    def train(self):

        # Start Logging
        self.logger.start_log()

        # Log Hyperparameters
        self.logger.log_value('hyperparameters', self.hyperparams)
        self.logger.log_value('experiment_name', self.experiment_name)

        # Set early stopping criteria
        stopper = EarlyStopping(self.args.patience, self.path['output_fold'] + 'tmp_graph.pt')

        # Set training task and labels
        self.labels = self.task.extras['rollup_labels'][f'l{self.task.extras["diag_level"]}'].to(self.device)
        self.task.labels = self.labels

        # Create model
        self.create_model(
            self.hg,
            self.feats,
            self.args.h_dim,
            self.labels.shape[1],
            self.args.n_layers,
            self.args.dropout,
            self.args.lr,
            self.args.weight_decay,
            self.args.loss_func,
            self.task.extras,
            self.device
        )

        epoch = 0
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:

            # One iteration of training and evaluation
            self._train_step()
            metric_dict, losses = self._evaluate()

            train_loss = losses['train']
            valid_loss = losses['valid']
            self.logger.info(
                f"[Train Info] Epoch: {epoch}, Train loss: {train_loss:.3f}, Valid loss: {valid_loss:.3f}."
                + self.logger.metric2str(metric_dict))

            # Log metrics and losses
            self.logger.log_series(metric_dict)
            self.logger.log_series({'train': {'loss': train_loss}, 'val': {'loss': valid_loss}})

            # Ealy stopping mechanism
            if stopper.step(losses['valid'], self.model):
                self.logger.info('Early Stop!\tEpoch:' + str(epoch))
                break

        stopper.load_model(self.model)

        metric_dict, losses = self._evaluate()
        self.logger.info('[Test Info]' + self.logger.metric2str(metric_dict))
        self.logger.log_values(metric_dict)

        # Stop Logging
        self.logger.stop_log()

        return metric_dict, epoch

    def set_hyperparameters(self):
        if hasattr(self.args, 'hyperparams'):
            return self.args.hyperparams
        else:
            return {
                'h_dim': self.args.h_dim,
                'n_layers': self.args.n_layers,
                'lr': self.args.lr,
                'dropout': self.args.dropout
            }

    def _evaluate(self):
        if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
            metric_dict, losses = self._mini_test_step(modes=['valid', 'test'])
        else:
            metric_dict, losses = self._full_test_step(modes=['train', 'valid', 'test'])
        return metric_dict, losses

    def _train_step(self):
        if self.args.mini_batch_flag:
            train_loss = self._mini_train_step()
        else:
            train_loss = self._full_train_step()
        return train_loss

    def _full_train_step(self):
        self.model.train()
        inputs = self.model.input_feature()
        outputs = self.model(self.hg, inputs)
        loss = self.loss_fn(outputs, self.labels, self.train_idx)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def _mini_train_step(self):
        loss_all = 0.0
        self.model.train()
        loader_tqdm = tqdm(self.train_loader, ncols=120, position=0)
        for i, (input_nodes, output_nodes, mfgs) in enumerate(loader_tqdm, 1):
            inputs = {ntype: self.model.feats[ntype][feats] for ntype, feats in input_nodes.items()}
            inputs = {ntype: self.model.activation(self.model.input[ntype](feats)) for ntype, feats in inputs.items()}

            train_idx = output_nodes[self.category]
            outputs = self.model(mfgs, inputs)[self.category]
            labels = self.labels[train_idx]

            loss = self.loss_fn(outputs, labels, None)
            loss_all += loss.item()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return loss_all / i

    def _full_test_step(self, modes: list) -> (dict, dict):
        """

        Parameters
        ----------
        mode: list[str]
            `train`, 'test', 'valid' are optional in list.

        Returns
        -------
        metric_dict: dict[str, float]
            score of evaluation metric
        info: dict[str, str]
            evaluation information
        loss: dict[str, float]
            the loss item
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.model.feats
            inputs = {ntype: self.model.activation(self.model.input[ntype](feats)) for ntype, feats in inputs.items()}

            outputs = self.model(self.hg, inputs)
            masks = {}
            for mode in modes:
                if mode == "train":
                    masks[mode] = self.train_idx
                elif mode == "valid":
                    masks[mode] = self.valid_idx
                elif mode == "test":
                    masks[mode] = self.test_idx

            # Optimize threshold based on validation data
            self.task.optimize_threshold(outputs[self.category], mode='valid')

            # Calculate metrics
            metric_dict = {key: self.task.evaluate(outputs[self.category], mode=key) for key in masks}
            loss_dict = {key: self.loss_fn(outputs, self.labels, mask).item() for key, mask in masks.items()}

            return metric_dict, loss_dict

    def _mini_test_step(self, modes):
        # TODO: This can definately be improved upon regarding runtime,
        # see the evaluation from the full_test_step
        self.model.eval()
        with torch.no_grad():
            metric_dict = {}
            loss_dict = {}
            loss_all = 0.0
            for mode in modes:
                if mode == 'train':
                    loader_tqdm = tqdm(self.train_loader, ncols=120)
                elif mode == 'valid':
                    loader_tqdm = tqdm(self.val_loader, ncols=120)
                elif mode == 'test':
                    loader_tqdm = tqdm(self.test_loader, ncols=120)
                y_trues = []
                y_predicts = []
                for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm, 1):
                    blocks = [blk.to(self.device) for blk in blocks]
                    seeds = seeds[self.category]
                    lbl = self.labels[seeds].to(self.device)
                    logits = self.model(blocks)[self.category]
                    loss = self.loss_fn(logits, lbl)

                    loss_all += loss.item()
                    y_trues.append(lbl.detach().cpu())
                    y_predicts.append(logits.detach().cpu())
                loss_all /= i
                y_trues = torch.cat(y_trues, dim=0)
                y_predicts = torch.cat(y_predicts, dim=0)
                evaluator = self.task.get_evaluator(name='f1')
                metric_dict[mode] = evaluator(y_trues, y_predicts.argmax(dim=1).to('cpu'))
                loss_dict[mode] = loss_all
        return metric_dict, loss_dict
