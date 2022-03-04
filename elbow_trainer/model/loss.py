import torch.nn.functional as F
from monai.losses import DiceLoss


def nll_loss(output, target):
    return F.nll_loss(output, target)


def dice_loss(output, target):
    crit = DiceLoss()
    loss = crit(output, target)
    return loss


