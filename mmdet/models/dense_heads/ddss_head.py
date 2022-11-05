import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32


import nltk
from nltk.cluster.kmeans import KMeansClusterer

from mmdet.core import (anchor_inside_flags, bbox_overlaps, build_assigner,
                        build_sampler, images_to_levels, multi_apply,
                        reduce_mean, unmap)
from mmdet.core.utils import filter_scores_and_topk
from ..builder import HEADS, build_loss
from .gfl_head import GFLHead


@HEADS.register_module()
class DensityHead(GFLHead):
    def __init__(self,
                loss_density_map=dict(type='MSELoss', loss_weight=10),
                **kwargs):
        super(DensityHead, self).__init__(**kwargs)
        self.loss_density_map_weight = loss_density_map['loss_weight']
        

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)

        self.cls_mappper = nn.Conv2d(self.feat_channels * 2, self.feat_channels, 1, padding=0)
        self.reg_mapper = nn.Conv2d(self.feat_channels * 2, self.feat_channels, 1, padding=0)

        self.density_convs = nn.ModuleList()
        for i in range(4):
            chn = self.feat_channels * 5 if i == 0 else self.feat_channels
            self.density_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.density_mapper = nn.Conv2d(
            self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides]) # ?
    

    # init all parameters 
    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.density_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.gfl_cls, std=0.01)
        normal_init(self.gfl_reg, std=0.01)
        normal_init(self.cls_mappper, std=0.01)
        normal_init(self.reg_mapper, std=0.01)
        normal_init(self.density_mapper, std=0.01)
    

    def forward_density_map_single(self, feats_all_scale): # depend on FPN
        final_feats = []
        scale_num = len(feats_all_scale)

        size = feats_all_scale[0].size()[-2:]
        for i in range(scale_num):
            final_feats.append(F.interpolate(feats_all_scale[i][None, :, :, :], size=size).squeeze(0))
        final_feats = torch.cat(final_feats)

        # print(f'get final_feats size {final_feats.size()}')
        # print(final_feats.size())

        final_feats = final_feats[None,:,:,:]

        for density_conv in self.density_convs:
            final_feats = density_conv(final_feats)

        final_feats = self.density_mapper(final_feats)
        final_feats = final_feats.squeeze(0).squeeze(0) # interpolate to ori_size
        
        # assert type(final_feats) == torch.Tensor
        # print(f'get final_feats size {final_feats.size()}')
        # print(final_feats.size())

        return final_feats
        
    def forward_density_map(self, feats):
        feats_list = []
        _N = feats[0].size(0)
        _N_scale = len(feats)
        
        ret = []

        for i in range(int(_N)):
            tmp_list = []
            for j in range(_N_scale):
                tmp_list.append(feats[j][i])
            ret.append(self.forward_density_map_single(tmp_list))

        return tuple(ret)

    
    def forward(self, feats, train=False):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification and quality (IoU)
                    joint scores for all scale levels, each is a 4D-tensor,
                    the channel number is num_classes.
                bbox_preds (list[Tensor]): Box distribution logits for all
                    scale levels, each is a 4D-tensor, the channel number is
                    4*(n+1), n is max value of integral set.
                density_map (list[Tensor]): density map for all imgs.
        """

        # print('fuck'*20)
        # print(f'We get FPN size here!')
        # for idx, it in enumerate(feats):
        #     print(f'FPN idx {idx}')
        #     print(f'FPN size {it.size()}')
        
        density_map = self.forward_density_map(feats)
        ret = multi_apply(self.forward_single, feats, self.scales, density_map=density_map)
        
        if train:
            ret = (ret[0], ret[1], list(density_map))
            return ret
        else:
            return ret 

    # define interaction here
    def forward_single(self, x, scale, density_map=None):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls and quality joint scores for a single
                    scale level the channel number is num_classes.
                bbox_pred (Tensor): Box distribution logits for a single scale
                    level, the channel number is 4*(n+1), n is max value of
                    integral set.
        """
        cls_feat = x
        reg_feat = x

        _N = x.size(0)

        # for cls_conv in self.cls_convs:
        #     cls_feat = cls_conv(cls_feat)
        # for reg_conv in self.reg_convs:
        #     reg_feat = reg_conv(reg_feat)

        cls_feat_cat = x
        reg_feat_cat = x

        cls_feat_list = []
        reg_feat_list = []
        for i in range(int(_N)):
            size = x[i].size()[-2:]
            d_map = F.interpolate(density_map[i][None, None, :, :], size=size).squeeze(0)
            cls_feat_list.append(torch.cat([cls_feat_cat[i] * d_map, cls_feat_cat[i]]).unsqueeze(0))
            reg_feat_list.append(torch.cat([reg_feat_cat[i] * d_map, reg_feat_cat[i]]).unsqueeze(0))

        cls_feat = torch.cat(cls_feat_list)
        reg_feat = torch.cat(reg_feat_list)
        cls_feat = self.cls_mappper(cls_feat)
        reg_feat = self.reg_mapper(reg_feat)

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_score = self.gfl_cls(cls_feat)
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        return cls_score, bbox_pred

    def get_loss_density_map(self, density_map, gt_density_map):
        _N = len(density_map)
        loss_func = torch.nn.MSELoss(size_average=False)
        loss = 0
        for d_map, gt_d_map in zip(density_map, gt_density_map):
            print(gt_d_map.size())
            print(d_map.size())
            loss += loss_func(F.interpolate(gt_d_map[None, :, :],size=d_map.size()[-2:]).squeeze(0), d_map)
        loss /= _N
        return loss * self.loss_density_map_weight
    
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             density_map,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_density_map,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            density_map(list[Tensor]):density map predict with shape(H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_density_map (list[Tensor]): gt density map witg shape (H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_dfl,\
            avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.prior_generator.strides,
                num_total_samples=num_total_samples)
        #print(f'len gt_density_map {len(gt_density_map)}')
        #print(f'len gt_density_map[0] {gt_density_map[0].size()}')
        # density_map = map(lambda x: x.unsqueeze(0), density_map)
        # density_map = torch.cat(list(density_map))
        # #print(f'gt_density_map {gt_density_map.size()}')
        # gt_density_map = gt_density_map.squeeze(1)

        #print(f'gt_density_map.sum() {gt_density_map.sum()}')

        loss_density_map = self.get_loss_density_map(density_map, gt_density_map)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl, loss_density_map=loss_density_map)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      gt_density_map=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x, train=True)
        #print(len(outs))
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas, gt_density_map)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, gt_density_map)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list
    
