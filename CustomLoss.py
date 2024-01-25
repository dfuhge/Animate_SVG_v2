import torch
from torch import nn, Tensor
from torch.nn import functional


class CustomLoss(nn.CrossEntropyLoss):
    ignore_index: int
    label_smoothing: float

    def __init__(self) -> None:
        # TODO experiment with weights
        weight = torch.tensor([1, 1, 1, 1])
        super().__init__(weight, None, None, 'mean')
        self.ignore_index = -100
        self.label_smoothing = 0.0

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # TODO depending on input, ignore certain features
        #  A) by setting result and target to 0
        #  B) by setting target to result
        #  C) by setting result to target

        return functional.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)