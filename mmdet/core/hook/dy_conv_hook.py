from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class DyConvHook(Hook):

    def __init__(self, per_iter=8089):
        self.step = 0
        self.per_iter = per_iter

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass


    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        self.step += 1
        #dy_conv =[runner.model.module.bbox_head.dy_scale_net, runner.model.module.bbox_head.dy_quality_net]
        dy_conv =[runner.model.module.bbox_head.atten1]
        if self.step % self.per_iter == 0:
            for mdl in dy_conv:
                #for _mdl in mdl:
                mdl.updata_temperature() # typo here

    def after_iter(self, runner):
        pass