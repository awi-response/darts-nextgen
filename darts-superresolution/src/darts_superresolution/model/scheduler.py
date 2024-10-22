"""@file source.py
@author Ryan Missel

Class definition for the CosineAnnealingWarmRestarts with both Max-LR Decay and global LinearWarmup.

https://github.com/qu-gg/pytorch-cosine-annealing-with-decay-and-initial-warmup/blob/main/source.py
"""

import math
import warnings

from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False, warmup_steps=350, decay=1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestartsWithDecayAndLinearWarmup, self).__init__(optimizer, last_epoch, verbose)

        # Decay attributes
        self.decay = decay
        self.initial_lrs = self.base_lrs

        # Warmup attributes
        self.warmup_steps = warmup_steps
        self.current_steps = 0

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        return [
            (self.current_steps / self.warmup_steps)
            * (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        """Step could be called after every batch update"""
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if self.T_cur + 1 == self.T_i:
            if self.verbose:
                print(f"multiplying base_lrs by {self.decay:.4f}")
            self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1

            if self.current_steps < self.warmup_steps:
                self.current_steps += 1

            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class CosineAnnealingWithDecay(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    """

    def __init__(self, optimizer, T_0, eta_min=0, last_epoch=-1, gamma=0.9, T_mult=2, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        # if T_mult < 1 or not isinstance(T_mult, int):
        #     raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.steps_per_cycle = T_0
        self.step_in_cycle = 0
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.cycle = 0
        self.gamma = gamma
        self.current_steps = 0
        self.T_mult = T_mult
        super(CosineAnnealingWithDecay, self).__init__(optimizer, last_epoch, verbose)

        self.max_lr_0 = self.base_lrs
        # self.max_lr = self.base_lrs

        # Decay attributes
        # self.decay = decay
        # self.initial_lrs = self.base_lrs

        # Warmup attributes

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )
        # print("Get_lr: ", self.step_in_cycle, self.steps_per_cycle)
        return [
            (
                self.eta_min
                + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.step_in_cycle / self.steps_per_cycle)) / 2
            )
            for base_lr in self.base_lrs
        ]
        # else:
        #     return [
        #         (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.step_in_cycle / self.steps_per_cycle)))
        #         for base_lr in self.base_lrs
        #     ]

    def step(self, epoch=None):
        """Step could be called after every batch update"""
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if self.current_steps != 0 and self.current_steps % (self.steps_per_cycle * 2) == 0:
            self.step_in_cycle = 0
            # self.steps_per_cycle = self.steps_per_cycle * self.T_mult

        if self.current_steps != 0 and self.current_steps % self.steps_per_cycle == 0:
            self.cycle += 1
            if self.cycle % 2 != 0:
                self.base_lrs = self.max_lr_0
                # self.max_lr = self.max_lr_0 * (self.gamma ** self.cycle)
                self.base_lrs = [base_lr * (self.gamma**self.cycle) for base_lr in self.base_lrs]
                # self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        # if self.T_cur + 1 == self.T_i:
        #     if self.verbose:
        #         print("multiplying base_lrs by {:.4f}".format(self.decay))
        #     self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]

        if epoch is None:
            epoch = self.last_epoch + 1

        self.step_in_cycle += 1
        self.current_steps += 1
        # self.T_cur = self.T_cur + 1

        # if self.current_steps < self.warmup_steps:
        #     self.current_steps += 1

        # if self.T_cur >= self.T_i:
        #     self.T_cur = self.T_cur - self.T_i
        #     self.T_i = self.T_i * self.T_mult

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
