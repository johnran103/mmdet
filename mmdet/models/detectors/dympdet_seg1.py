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
import sys
import os
sys.path.append(os.path.abspath('/home/ranqinglin/work_dirs/ddet/scale_map_net'))
from model.PSPNet import OneModel
import argparse
import cv2


class arg:
    def __init__(self):
        self.num_classes = 9
        self.layers = 50
        self.zoom_factor = 8
        self.vgg = False
        self.pretrained = False

def visualize(img,img1):
    mean=torch.tensor([123.675, 116.28 , 103.53 ])
    std=torch.tensor([58.395,57.12,57.375])
    img = img.permute(1,2,0)
    img = img * std[None,None,:]
    img = img + mean[None,None,:]
    
    img = img.numpy()
    # for box in gt_bboxes:
    #     cv2.rectangle(img,(int(box[0]),int(box[1])), (int(box[2]),int(box[3])),(0,255,0))
    cv2.imwrite('./tmp.jpg', img)
    #label = psudo_label_generator.get(img, x['gt_bboxes'])
    cv2.imwrite('./scale.jpg', torch.sum(torch.tensor(img1),dim=0).numpy() * 255)


@DETECTORS.register_module()
class DyMPDetSeg1(SingleStageDetector):

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
        super(DyMPDetSeg1, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
        model_parser = arg()
        self.scale_net = OneModel(model_parser)
        self.load_checkpoint_freeeze_param(scale_path=scale_path)
    
    def load_checkpoint_freeeze_param(self, scale_path=None, quality_path=None):
        if scale_path is not None:
            self.scale_net.load_state_dict(torch.load(scale_path,map_location='cpu'))

        for name, param in self.scale_net.named_parameters():
            param.requires_grad = False

        for m in self.scale_net.modules():
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
        
        with torch.no_grad():
            _img = F.interpolate(img, size=(1024,1024),mode='bilinear')
            scale_out = self.scale_net(_img)
            scale_out = F.sigmoid(scale_out)
            #visualize(_img[0].cpu(), scale_out[0].cpu())
        
        # print(f'scale_out {scale_out}')
        # print(f'quality_out {quality_out}')

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, scale_map=scale_out, quality_map=None, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
    
    def simple_test(self, img, img_metas, rescale=False):
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

        with torch.no_grad():
            _img = F.interpolate(img, size=(1024,1024),mode='bilinear')
            scale_out = self.scale_net(_img)
            scale_out = F.sigmoid(scale_out)

        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale, scale_map=scale_out, quality_map=None)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
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

        scale_outs = []

        with torch.no_grad():
            for img in imgs:
                _img = F.interpolate(img, size=(1024,1024),mode='bilinear')
                scale_out = self.scale_net(_img)
                scale_out = F.sigmoid(scale_out)
                scale_outs.append(scale_out)

        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale,scale_maps=scale_outs, quality_maps=None)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

