from torch.nn.modules.loss import _WeightedLoss
import torch as th


class HierarchicalLoss(_WeightedLoss):
    def __init__(self, hml_variant, loss_func, category, in_level, rollup_maps, device):
        super(HierarchicalLoss, self).__init__()
        self.in_level = in_level
        self.category = category
        self.rollup_maps = rollup_maps
        self.loss_func = loss_func
        self.device = device
        self.hml_variant = hml_variant

    def forward(self, preds, labs, mask=None):
        if self.hml_variant == 'hmcn_out':
            return self.hmcn_out(preds, labs, mask)

    def hmcn_out(self, preds, labs, mask):
        # Calculate full loss:
        loss = self.loss_func(preds['S'][mask] if mask is not None else preds['S'], labs[mask] if mask is not None else labs)

        # TODO: If this works, the labs can be precomputed
        for level, aggregation in self.rollup_maps.items():

            agg_labs = self.rollup(labs[mask] if mask is not None else labs, aggregation, th.max)
            agg_loss = self.loss_func(preds[level][mask] if mask is not None else preds[level], agg_labs)
            loss += agg_loss

        return loss

    def forward1(self, preds, labs):
        # The forward calculation of a classic HML loss over the tree
        loss = self.loss_func(preds, labs)

        # For each level in the tree (starting from the depth of the prediction)
        # we calculate the loss on different levels and add them together
        for aggregation in self.rollup_maps.values():

            # Clone predictions
            cloned_preds = preds.clone()
            cloned_labs = labs.clone()

            agg_preds = self.rollup(cloned_preds, aggregation, th.max)
            agg_labs = self.rollup(cloned_labs, aggregation, th.max)

            level_loss = self.loss_func(agg_preds, agg_labs)
            loss += level_loss

        return loss

    # noinspection PyUnboundLocalVariable
    def rollup(self, input_tensor: th.Tensor, aggregation: dict, agg_func: callable) -> th.tensor:

        agg_tensor = th.zeros(size=(input_tensor.shape[0], len(aggregation)), dtype=th.float32).to(self.device)
        for i, agg_indexes in enumerate(aggregation.values()):
            agg_tensor[:, i] = agg_func(input_tensor[:, agg_indexes], dim=1)[0]

        return agg_tensor

    def rollup_tensor(self, tensor: th.Tensor, rollup_map: dict, agg_func: callable) -> th.tensor:
        # Construct new tensor with rolled_up dimensions
        new_tensor = th.zeros(size=(tensor.shape[0], len(rollup_map)), dtype=th.float32).to(self.device)
        for key, values in rollup_map.items():
            new_tensor[:, key] = agg_func(tensor[:, values], dim=1)[0]

        return new_tensor