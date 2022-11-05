# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn.functional as F
from mmcv.runner import auto_fp16
import torch


@DETECTORS.register_module()
class HRGFL(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(HRGFL, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
    
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs): # resize 4x larger
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])
        
        if return_loss:
            img = F.interpolate(img, size=(2*img.size(-2), 2*img.size(-1)), mode='bicubic')
        else:
            for i in range(len(img)):
                img[i] = F.interpolate(img[i], size=(2*img[i].size(-2), 2*img[i].size(-1)), mode='bicubic')

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)
