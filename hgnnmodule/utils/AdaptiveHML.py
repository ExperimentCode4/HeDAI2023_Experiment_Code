from torch.nn.modules.loss import _WeightedLoss
import torch as th


class AdaptiveHMLLoss(_WeightedLoss):
    """
    Parameters
    ----------
    in_level: Initial or final level of input tensors and labels
    rollup_maps: Maps with rules for how to rollup label and prediction tensors
    loss_func: Loss function, could be BCELoss or BCEWithLogitsLoss
    device: The device to use (cuda)
    adaptivity: The type of adaptive HML to use, currenly supports (layered, adaptive)
        - layered: trains on the simplest L1 task, when epochs increase it switches
                   to an L2 task, etc, until final saturation on the in_level
        - adaptive: Trains on the in_level task, when epochs increase it switches
                    to the full hierarch HML loss function
    Returns:
    -------
    loss: The calculated loss between preds and labels
    """
    # Initialize HML Loss with
    def __init__(self, in_level, rollup_maps, loss_func, device, adaptivity):
        super(AdaptiveHMLLoss, self).__init__()
        self.in_level = in_level
        self.rollup_maps = rollup_maps
        self.loss_func = loss_func
        self.device = device
        self.adaptivity = self.set_adaptivity(adaptivity)
        self.plateau = 0

    def forward(self, preds, labs):

        # Calculate layered loss
        if self.adaptivity == 'layered':
            return self.calc_layered_loss(preds, labs)

        # Calculate adaptive loss
        if self.adaptivity == 'adaptive':
            return self.calc_adaptive_loss(preds, labs)

        if self.adaptivity == 'hml':
            loss = self.loss_func(preds, labs)
            return self.calc_hml_loss(preds, labs, loss)

    def calc_layered_loss(self, preds, labs):
        # Roll tensors to a desired level, depending on
        # the plateau, then calculate that single loss
        level = min(5 - (4 - self.plateau), 5)

        # We only roll_to_level if it is less than 5
        if level < 5:
            preds, labs = self.roll_to_level(preds, labs, level)

        return self.loss_func(preds, labs)

    def calc_adaptive_loss(self, preds, labs):
        # Calculate flat loss
        loss = self.loss_func(preds, labs)

        if self.plateau != 0:
            # Calculate HML Loss
            loss = self.calc_hml_loss(preds, labs, loss)

        return loss

    def calc_hml_loss(self, preds, labs, loss):
        for depth in range(self.in_level - 1, 0, -1):
            # Get rollup map
            rollup_map = self.rollup_maps[f'rollup_maps_l{depth}']

            # Roll up preds and labs
            new_labs = self.rollup_tensor(labs, rollup_map, th.max)
            new_preds = self.rollup_tensor(preds, rollup_map, th.max)

            # Calculate new Loss
            rollup_loss = self.loss_func(new_preds, new_labs)

            # Add this to original loss
            loss += rollup_loss

        return loss / self.in_level

    def roll_to_level(self, preds, labs, out_level):
        # Get rollup map
        rollup_map = self.rollup_maps[f'rollup_maps_l{out_level}']

        labs = self.rollup_tensor(labs, rollup_map, th.max)
        preds = self.rollup_tensor(preds, rollup_map, th.max)

        return preds, labs

    def rollup_tensor(self, tensor: th.Tensor, rollup_map: dict, agg_func: callable) -> th.tensor:
        # Construct new tensor with rolled_up dimensions
        new_tensor = th.zeros(size=(tensor.shape[0], len(rollup_map)), dtype=th.float32).to(self.device)
        for key, values in rollup_map.items():
            new_tensor[:, key] = agg_func(tensor[:, values], dim=1)[0]

        return new_tensor

    def increase_plateau(self):
        self.plateau += 1

    def set_adaptivity(self, adaptivity):
        if adaptivity in ['layered', 'adaptive', 'hml']:
            return adaptivity
        exit(f'Adaptivity of HML: {adaptivity} not implemented')
