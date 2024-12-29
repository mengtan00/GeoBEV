# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel

__all__ = ['SequentialControlHook']


@HOOKS.register_module()
class SequentialControlHook(Hook):
    """ """

    def __init__(self, temporal_start_epoch=1):
        super().__init__()
        self.temporal_start_epoch=temporal_start_epoch

    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.with_prev=flag
        else:
            runner.model.module.with_prev = flag

    def before_run(self, runner):
        self.set_temporal_flag(runner, False)

    def before_train_epoch(self, runner):
        if runner.epoch > self.temporal_start_epoch:
            self.set_temporal_flag(runner, True)

@HOOKS.register_module()
class SequentialControlHook1(Hook):
    """ """

    def __init__(self, temporal_start_epoch=1, warmup_ratio=1e-9):
        super().__init__()
        self.temporal_start_epoch=temporal_start_epoch
        self.warmup_ratio = warmup_ratio

    def set_loss_weight(self, runner, times):
        if is_parallel(runner.model.module):
            runner.model.module.module.loss_distill_weight *= times
        else:
            runner.model.module.loss_distill_weight *= times

    def before_run(self, runner):
        self.set_loss_weight(runner, self.warmup_ratio)

    def before_train_epoch(self, runner):
        if runner.epoch >= self.temporal_start_epoch:
            self.set_loss_weight(runner, 1/self.warmup_ratio)
            self.warmup_ratio = 1