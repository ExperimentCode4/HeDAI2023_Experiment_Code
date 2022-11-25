from torch.nn.modules.loss import _WeightedLoss


class BCELoss(_WeightedLoss):
    def __init__(self, loss_func, category):
        super(BCELoss, self).__init__()
        self.loss_func = loss_func
        self.category = category

    def forward(self, preds, labs, mask):
        if mask is None:
            return self.loss_func(preds, labs)
        else:
            return self.loss_func(preds[self.category][mask], labs[mask])
