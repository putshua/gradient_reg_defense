from typing import Any
import torch
import torch.nn as nn

from torchattacks.attack import Attack

class Uniform(Attack):
    r"""
    Add Uniform Noise.

    Arguments:
        model (nn.Module): model to attack.
        std (nn.Module): standard deviation (Default: 0.1).

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.GN(model)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.1):
        super().__init__("Uniform", model)
        self.std = eps

    def forward(self, images, labels=None):
        images = images.clone().detach().to(self.device)
        adv_images = images + self.std*(torch.rand_like(images)*2-1)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        return adv_images
                