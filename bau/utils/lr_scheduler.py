# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from bisect import bisect_right
import math
import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            total_epochs,
            warmup_epochs=10,
            start_decay_epoch=30,
            eta_min=0,
            warmup_factor=1.0 / 3,
            warmup_method="linear",
            last_epoch=-1,
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.start_decay_epoch = start_decay_epoch
        self.eta_min = eta_min
        self.warmup_factor = warmup_factor
        self.warmup_method = warmup_method

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )

        super(WarmupCosineDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = self.get_warmup_factor_at_epoch(self.last_epoch)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        elif self.last_epoch < self.start_decay_epoch:
            return [base_lr for base_lr in self.base_lrs]
        else:
            T_max = self.total_epochs - self.start_decay_epoch
            T_cur = self.last_epoch - self.start_decay_epoch
            return [
                self.cosine_decay_lr(base_lr, T_cur, T_max, self.eta_min)
                for base_lr in self.base_lrs
            ]

    def get_warmup_factor_at_epoch(self, epoch):
        if self.warmup_method == "constant":
            return self.warmup_factor
        elif self.warmup_method == "linear":
            alpha = epoch / self.warmup_epochs
            return self.warmup_factor * (1 - alpha) + alpha
        else:
            return 1

    def cosine_decay_lr(self, base_lr, T_cur, T_max, eta_min):
        return eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_max)) / 2
