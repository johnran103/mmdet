# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector

from ..necks import FPN
from ..backbones import MobileNetV2

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.core import bbox2result
from mmcv.runner import BaseModule, auto_fp16

class mobilenetv2_fpn(nn.Module):
    def __init__(self, load_weights=False):
        super(mobilenetv2_fpn,self).__init__()
        self.mobilenet = MobileNetV2()
        self.in_channels = [24, 32, 96, 1280]
        self.scales = [333, 167, 84, 42]
        self.fpn = FPN(self.in_channels, 256, len(self.scales))
        self.fuse1 = nn.Conv2d(256*4,256,1,padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.bn2 = nn.BatchNorm2d(num_features=1)
        self.fuse2 = nn.Conv2d(256,1,1,padding=0)

    def forward(self, input):
        ret = self.mobilenet(input)
        
        ret = list(self.fpn(ret))

        #_scale = (333, 333)
        for i in range(4):
            ret[i] = F.interpolate(ret[i], size=(333,333), mode='bilinear')
        
        ret = torch.cat(ret,dim=1)

        ret = self.fuse1(ret)
        ret = self.bn1(ret)
        ret = self.relu(ret)

        ret = self.fuse2(ret)
        ret = self.bn2(ret)
        ret = self.relu(ret)

        return ret


@DETECTORS.register_module()
class DyMPDetGT(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 scale_path=None,
                 quality_path=None):
        super(DyMPDetGT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
        # self.scale_net = mobilenetv2_fpn()
        # self.quality_net = mobilenetv2_fpn()
        # self.load_checkpoint_freeeze_param(scale_path=scale_path, quality_path=quality_path)
    
    def load_checkpoint_freeeze_param(self, scale_path=None, quality_path=None):
        if scale_path is not None:
            self.scale_net.load_state_dict(torch.load(scale_path,map_location='cpu'))

        if quality_path is not None:
            self.quality_net.load_state_dict(torch.load(quality_path,map_location='cpu'))
        
        for name, param in self.scale_net.named_parameters():
            param.requires_grad = False

        for name, param in self.quality_net.named_parameters():
            param.requires_grad = False

        for m in self.scale_net.modules():
            if isinstance(m, _BatchNorm):
                print(f'BN {m} freeezed')
                m.eval()

        for m in self.quality_net.modules():
            if isinstance(m, _BatchNorm):
                print(f'BN {m} freeezed')
                m.eval()

        # self.scale_net.to('cuda:0')
        # self.quality_net.to('cuda:0')

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      scale_map=None,
                      quality_map=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        
        # with torch.no_grad():
        #     scale_out = self.scale_net(img)
        #     quality_out = self.quality_net(img)
        
        # print(f'scale_out {scale_out}')
        # print(f'quality_out {quality_out}')

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, scale_map=scale_map, quality_map=quality_map, gt_bboxes_ignore=gt_bboxes_ignore)

        #losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, scale_map=scale_out, quality_map=quality_out, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
    
    def simple_test(self, img, img_metas, scale_map=None, quality_map=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)

        # with torch.no_grad():
        #     scale_out = self.scale_net(img)
        #     quality_out = self.quality_net(img)

        # assert 1 == 0, 'NO SIMPLE TEST'

        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale, scale_maps=scale_map, quality_maps=quality_map)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, scale_map=None, quality_map=None, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)

        # scale_outs = []
        # quality_outs = []

        # with torch.no_grad():
        #     for img in imgs:
        #         scale_out = self.scale_net(img)
        #         scale_outs.append(scale_out)
        #         quality_out = self.quality_net(img)
        #         quality_outs.append(quality_out)

        for i in range(len(scale_map)):
            _s = scale_map[i]
            #print(self.scale_net.mobilenet.conv1.conv.weight)
            #print(_s.sum())
            import numpy as np
            import cv2
            print(_s.size())
            cv2.imwrite("/home/ranqinglin/work_dirs/ddet/scale{}.jpg".format(i), (_s).permute(1,2,0).cpu().numpy() * 255)
            #noise=torch.tensor(0.1 * np.random.rand(_s.size()[0], _s.size()[1], _s.size()[2])).to(imgs[0].device)
            #mask = _s > 0.7
            #_s[mask]=0
            #_s = _s.clamp(0,0.7)
        exit(0)
        #print(scale_map)

        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale,scale_maps=scale_map, quality_maps=quality_map)

        # results_list = self.bbox_head.aug_test(
        #     feats, img_metas, rescale=rescale,scale_maps=scale_outs, quality_maps=quality_outs)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
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
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)
