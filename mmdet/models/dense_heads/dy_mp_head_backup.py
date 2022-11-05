
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


class attention1d(nn.Module):
    def __init__(self, in_planes=1, ratios=16, K=4, temperature=34, init_weight=True): # quality map
        super(attention1d, self).__init__()
        assert temperature % 3 == 1
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios)
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        self.K = K
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        _N, _C, _H, _W = x.size()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv1d(nn.Module):
    '''
    Args:
        x(Tensor):  shape (batch, in_channel, height, width)
        quality_map(Tensor):  shape (batch, 1, height, width)
    
    Return:
        output(Tensor):  shape (batch, out_channel, height, width)
    

    Note:
        in_channel must eqal to out_channel
    '''


    def __init__(self, in_planes, out_planes, ratio=16.0, stride=1, padding=0, dilation=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv1d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.K = K
        self.attention = attention1d(1, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self): # maybe problematic
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x, quality_map):# a different version of dynamic convlution, is another kind of spatial attention
        residule = x
        batch_size, in_planes, height, width = x.size()
        softmax_attention = self.attention(quality_map).permute(0, 2, 3, 1) # (N, H, W, K)
        #x = x.view(1, -1, width, height)# 变化成一个维度进行组卷积
        #weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同)
        #weight = weight.view(self.K, self.in_planes, self.out_planes)
        # print(f'softmax_attention {softmax_attention.size()}')
        # print(f'self.weight {self.weight.size()}')
        weight = self.weight.view(self.K, -1)
        aggregate_weight = torch.matmul(softmax_attention, weight).view(batch_size, height, width, self.out_planes, self.in_planes)# (N, H, W, C2, C1)
        aggregate_weight = aggregate_weight.permute(3, 0, 4, 1, 2)  # (C2, N, C1, H, W)
        output = aggregate_weight * x[None, :, :, :, :]
        # if self.bias is not None:
        #     aggregate_bias = torch.matmul(softmax_attention, self.bias).permute(0, 3, 1, 2) # (N, C1, H, W)
        #     print(aggregate_bias.size())
        #     print(softmax_attention.size())
        #     output = output + aggregate_bias
        output = output.sum(dim=0) # (N, C1, H, W)
        return residule + output


@HEADS.register_module()
class DYMPHead(GFLHead):
    """
    """

    def __init__(self,
                 num_words=200,
                 beta=0,
                 gamma=10,
                 proxies_list =[2, 3, 2, 5, 4, 8, 8, 4, 3, 3],
                 fuse_stage=3,
                 **kwargs):
        self.num_words=num_words
        self.proxies_list = proxies_list
        self.beta = beta
        self.gamma = gamma
        self.fuse_stage = fuse_stage
        super(DYMPHead, self).__init__(**kwargs)
        
        assert self.num_classes == len(self.proxies_list)
        
    
    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.BN1 = nn.BatchNorm2d(self.feat_channels)
        self.BN2 = nn.BatchNorm2d(5 * self.feat_channels)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.scale_convs = nn.ModuleList()
        self.scale_mappers = nn.ModuleList()
        self.fuse_dy_convs = nn.ModuleList()
        self.dy_level1 = Dynamic_conv1d(self.feat_channels, 4) # here 8 means dynamic network C2
        self.dy_level2 = Dynamic_conv1d(self.feat_channels, 4)
        self.dy_level3 = Dynamic_conv1d(self.feat_channels, 4)
        self.dy_level4 = Dynamic_conv1d(self.feat_channels, 4)
        self.dy_level5 = Dynamic_conv1d(self.feat_channels, 4)
        
        self.compress_feature = ConvModule(self.feat_channels * 5, self.feat_channels, 1, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        self.decompress_feature = ConvModule(self.feat_channels, self.feat_channels * 5, 1, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)

        for i in range(self.fuse_stage):
            self.fuse_dy_convs.append(Dynamic_conv1d(self.feat_channels, 4))

        # for i in range(self.fuse_stage):
        #     self.fuse_dy_convs.append(ConvModule(
        #             self.feat_channels,
        #             self.feat_channels,
        #             3,
        #             stride=1,
        #             padding=1,
        #             conv_cfg=self.conv_cfg,
        #             norm_cfg=self.norm_cfg))

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
        
        for i in range(self.fuse_stage):
            self.scale_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg)
            )
            self.scale_mappers.append(nn.Conv2d(self.feat_channels, 1, 3, padding=1))

        self.quality_conv = nn.Conv2d(self.feat_channels * 5, self.feat_channels, 3, padding=1)
        assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls_conv = nn.Conv2d(
            self.feat_channels, self.feat_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        self.quality_mapper = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])
        
        self._init_BoIW()
        self._init_proxies()

    def _init_BoIW(self):
        embedding = torch.randn(self.num_classes+1, self.num_words, self.feat_channels)
        self.register_buffer("_embedding", embedding)
        self.register_buffer("_pos_embedding_ptr", torch.zeros(self.num_classes+1, dtype=torch.long))
    
    def _init_proxies(self):
        self.accumulate_proxies = [self.proxies_list[0]]
        for nums in self.proxies_list[1:]:
            self.accumulate_proxies.append(self.accumulate_proxies[-1] + nums)
        self.proxies = nn.Parameter(torch.randn(sum(self.proxies_list), self.feat_channels))
        prob = []
        for nums in self.proxies_list:
            prob += [1/nums] * nums
        _proxies_prob = torch.Tensor(prob)
        self.register_buffer("_proxies_prob", _proxies_prob)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.scale_convs:
            normal_init(m.conv, std=0.01)
        for m in self.scale_mappers:
            normal_init(m, std=0.01)
        # for m in self.fuse_dy_convs: # debug purpose
        #     normal_init(m, std=0.01) 
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.gfl_cls_conv, std=0.01)
        normal_init(self.gfl_reg, std=0.01)
        normal_init(self.quality_conv, std=0.01)
        normal_init(self.quality_mapper, std=0.01)
        nn.init.normal_(self.proxies, 0, 0.01)

    def forward_proxy(self, feat, is_training = False):
        centers = F.normalize(self.proxies, p=2, dim=1)
        feat = F.normalize(feat, p=2, dim=1)
        
        simInd = feat.matmul(centers.t())
        simClasses = []
        pre_pos = 0
        for i in range(self.num_classes):
            sub_sim = simInd[:, pre_pos:pre_pos+self.proxies_list[i]]
            prob = F.softmax(sub_sim * self.gamma, dim=1)
            sub_sim = torch.sum(prob*sub_sim, dim=1)
            simClasses.append(sub_sim[:,None])
            pre_pos += self.proxies_list[i]
        simClasses = torch.cat(simClasses, dim=1) * self.gamma
        if is_training:
            return simClasses, simInd
        return simClasses
    
    def forward_single(self, x, scale, quality_map=None, top_size=None):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            quality_map (Tensor): (N, 1, H, W) 

        Returns:
            tuple:
                cls_score (Tensor): Cls and quality joint scores for a single
                    scale level the channel number is num_classes.
                bbox_pred (Tensor): Box distribution logits for a single scale
                    level, the channel number is 4*(n+1), n is max value of
                    integral set.
        """

        # determin which layer is
        
        # print(x.size()[-1])
        # print(top_size)

        if x.size()[-1] == top_size:
            new_quality_map = F.interpolate(quality_map, size=x.size()[-2:])
            x = self.dy_level1(x, new_quality_map)
        elif x.size()[-1] == top_size // 2:
            new_quality_map = F.interpolate(quality_map, size=x.size()[-2:])
            x = self.dy_level2(x, new_quality_map)
        elif x.size()[-1] == top_size // 4:
            new_quality_map = F.interpolate(quality_map, size=x.size()[-2:])
            x = self.dy_level3(x, new_quality_map)
        elif x.size()[-1] == top_size // 8:
            new_quality_map = F.interpolate(quality_map, size=x.size()[-2:])
            x = self.dy_level4(x, new_quality_map)
        else:
            new_quality_map = F.interpolate(quality_map, size=x.size()[-2:])
            x = self.dy_level5(x, new_quality_map)

        cls_feat = x
        reg_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        cls_feat = self.gfl_cls_conv(cls_feat)
        if self.training:
            return cls_feat, bbox_pred
        _shape = cls_feat.shape
        cls_feat = cls_feat.permute(0, 2, 3, 1).reshape(
            -1, self.feat_channels).contiguous()
        cls_score = self.forward_proxy(cls_feat)
        cls_score = cls_score.reshape(_shape[0], _shape[2], _shape[3], -1).permute(0,3,1,2).contiguous()
        return cls_score, bbox_pred

    def sink(self, M, _ut = None, reg = 0.1, numItermax=100, stopThr=1e-8, cuda = False):
        # we assume that no distances are null except those of the diagonal of
        us = M.new_ones((M.size()[0],)) / M.size()[0]
        us = us.view(-1, 1)
        if _ut is None:
            ut = M.new_ones((M.size()[1],)) / M.size()[1]
        else:
            ut = _ut
        ut = ut.view(-1, 1)
        Nini = len(us)
        Nfin = len(ut)
        alpha = M.new_ones(Nini)
        alpha = alpha.view(-1, 1)
        beta = M.new_ones(Nfin)
        beta = beta.view(-1, 1)
        K = torch.exp(-M / reg)
        cpt = 0
        err = 1
        while cpt < numItermax:
            alpha_bak = alpha
            alpha = us/torch.mm(K, beta)
            beta = ut/torch.mm(K.t(), alpha)
            err = (alpha_bak - alpha).abs().sum(-1).mean()
            if err < stopThr:
                break
            cpt += 1
        return alpha.view(-1,1) * K * beta.view(1, -1)
    
    @torch.no_grad()
    def _update_dictionary(self, features, labels, max_step = 10):
        for class_idx in  range(self.num_classes + 1):
            pos_feat = features[labels == class_idx]
            # neg_feat = features[labels != class_idx]
            pos_sz = pos_feat.shape[0]
            if pos_sz > 0 :
                step = min(max_step, pos_sz)
                select_list = random.sample([i for i in range(pos_sz)], step)
                selected_feature = pos_feat[select_list]
                ptr = int(self._pos_embedding_ptr[class_idx])
                if ptr + step > self.num_words:
                    dlt = ptr + step - self.num_words
                    self._embedding[class_idx, ptr:self.num_words,:] = selected_feature[:step-dlt,:]
                    self._embedding[class_idx, 0:dlt,:] = selected_feature[step-dlt:,:]
                else:
                    self._embedding[class_idx, ptr:ptr+step,:] = selected_feature
                self._pos_embedding_ptr[class_idx] = (ptr + step) % self.num_words
    @torch.no_grad()
    def _update_dictionary_quality(self, features, labels, ious, max_step = 10):
        for class_idx in  range(self.num_classes + 1):
            pos_feat = features[labels == class_idx]
            pos_ious = ious[labels == class_idx]
            # neg_feat = features[labels != class_idx]
            if class_idx != self.num_classes:
                pos_feat = pos_feat[pos_ious > 0.9]
            pos_sz = pos_feat.shape[0]
            if pos_sz > 0 :
                step = min(max_step, pos_sz)
                # print(step)
                select_list = random.sample([i for i in range(pos_sz)], step)
                selected_feature = pos_feat[select_list]
                ptr = int(self._pos_embedding_ptr[class_idx])
                if ptr + step > self.num_words:
                    dlt = ptr + step - self.num_words
                    self._embedding[class_idx, ptr:self.num_words,:] = selected_feature[:step-dlt,:]
                    self._embedding[class_idx, 0:dlt,:] = selected_feature[step-dlt:,:]
                else:
                    self._embedding[class_idx, ptr:ptr+step,:] = selected_feature
                self._pos_embedding_ptr[class_idx] = (ptr + step) % self.num_words
    

    def loss_op_all_leval(self, feats, labels, scores, avg_factor):
        new_feats = []
        new_labels = []
        new_scores = []
        for cls_feat, label,score in zip(feats, labels, scores):
            cls_feat = cls_feat.permute(0, 2, 3, 1).reshape(
                -1, self.feat_channels).contiguous()
            label = label.reshape(-1)
            new_feats.append(cls_feat)
            new_labels.append(label)
            new_scores.append(score)

            # print(label.shape)
            # print(cls_feat.shape)
        feats =  torch.cat(new_feats, dim=0)
        labels = torch.cat(new_labels, dim=0)
        scores = torch.cat(new_scores, dim=0)
        # print(feats.shape)
        # print(labels.shape)
        loss_op = self.loss_op(feats, labels)
        # self._update_dictionary_quality(feats, labels, scores, 10)
        self._update_dictionary(feats, labels, 10)
        
        # loss_op = loss_op / avg_factor
        return loss_op



    def loss_op(self, feat, labels):
        pre_pos = 0
        feat = F.normalize(feat, p=2, dim=1)
        centers = F.normalize(self.proxies, p=2, dim=1)
        sim_ind = 1 - (feat.matmul(centers.t()) + 1)/2
        loss_op = [sim_ind.sum() * 0]
        for cls_id in range(self.num_classes):
            cls_pos_ind = labels == cls_id
            dis = sim_ind[cls_pos_ind][:, pre_pos:pre_pos+self.proxies_list[cls_id]]
            ut = self._proxies_prob[pre_pos:pre_pos+self.proxies_list[cls_id]]
            pre_pos += self.proxies_list[cls_id]
            if dis.shape[0] == 0:
                continue
            P = self.sink(dis, _ut=ut, reg=0.1)
            loss_op.append((P * dis).sum())
        # print(loss_op)
        return sum(loss_op) / self.num_classes
    
    def contrastive(self, features, labels, weighted=None, avg_factor = None):
        device = features.device
        # compute logits
        anchor_count = features.shape[0]
        features = F.normalize(features, p=2, dim=1)
        embeddings = self._embedding.reshape(-1, self.feat_channels)
        contrast_feature = F.normalize(embeddings, p=2, dim=1)
        logits = torch.matmul(features, contrast_feature.T)

        logits = logits.reshape(anchor_count, self.num_classes+1, self.num_words)
        exp_logits = torch.exp(logits).sum(2)
        pos = labels != self.num_classes

        embedding_labels = torch.arange(self.num_classes+1).to(device)
        mask = (labels[:, None] == embedding_labels[None,:]).float()

        con_exp_logits = exp_logits
        con_exp_logits = con_exp_logits.sum(1)

        pos_exp_logits = exp_logits * mask
        pos_exp_logits = pos_exp_logits.sum(1)

        log_prob = torch.log(pos_exp_logits/con_exp_logits)

        if not weighted is None:
            log_prob = log_prob * weighted
        if avg_factor:
            loss = -log_prob.sum() / avg_factor / self.num_words
        else:
            loss = -log_prob.mean() / self.num_words

        return loss
    
    def loss_single(self, anchors, cls_feat, bbox_pred, labels, label_weights,
                    bbox_targets, stride, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        _shape = cls_feat.shape
        cls_feat = cls_feat.permute(0, 2, 3, 1).reshape(
            -1, self.feat_channels).contiguous()
        # cls_score = self.(cls_feat)
        cls_score, simInd = self.forward_proxy(cls_feat, is_training=True)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(pos_anchor_centers,
                                                 pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            if torch.any(weight_targets > 0):
                # regression loss
                loss_bbox = self.loss_bbox(
                    pos_decode_bbox_pred,
                    pos_decode_bbox_targets,
                    weight=weight_targets,
                    avg_factor=1.0)

                # dfl loss
                loss_dfl = self.loss_dfl(
                    pred_corners,
                    target_corners,
                    weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                    avg_factor=4.0)
            else:
                loss_bbox = bbox_pred.sum() * 0
                loss_dfl = bbox_pred.sum() * 0
                weight_targets = torch.tensor(0).cuda()
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = torch.tensor(0).cuda()

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)
        loss_cls_emb = self.contrastive(features = cls_feat, labels = labels, weighted = label_weights, avg_factor = num_total_samples)
        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum(), loss_cls_emb * self.beta, score
    
    def loss_quality_all_level(self, quality_map, gt_quality_map):
        '''
            quality_map:(N, 1, H, W)
            gt_quality_map:(N, 1, H1, W1)
        '''

        sz = quality_map.size()[-2:]
        gt_quality_map = F.interpolate(gt_quality_map, size=sz)

        _N = quality_map.size(0)
        loss_func = torch.nn.L1Loss()
        loss = loss_func(quality_map, gt_quality_map)
        
        return 10 * loss / _N

    def loss_scale_all_level_all_stage(self, scale_maps, gt_scale_map):
        sz = scale_maps[0].size()[-2:]
        gt_scale_map = F.interpolate(gt_scale_map, size=sz)
        loss = 0
        loss_func = torch.nn.L1Loss()

        num_stage = len(scale_maps)
        _N = scale_maps[0].size(0)

        # for i in range(num_stage):
        #     print(i)
        #     print(scale_maps[i].size())
        #     print(gt_scale_map.size())

        for i in range(num_stage):
            loss += loss_func(scale_maps[i], gt_scale_map)
        
        loss /= _N

        return loss

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_feats,
             bbox_preds,
             scale_maps,
             quality_map,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_scale_map,
             gt_quality_map,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            quality_map: (Tensor), predicted quality map for all positions.
            scale_maps: list(Tensor), predicted scale map for all positions. 
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_quality_map:(Tensor), gt quality map.
            gt_scale_map:(Tensor), gt scale map.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_feats]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_feats[0].device
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
            avg_factor, losses_cls_emb, scores = multi_apply(
                self.loss_single,
                anchor_list,
                cls_feats,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.anchor_generator.strides,
                num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        losses_op = self.loss_op_all_leval(cls_feats, labels_list, scores, avg_factor=num_total_samples) * self.beta
        losses_quality = self.loss_quality_all_level(quality_map, gt_quality_map)
        #losses_scale = self.loss_scale_all_level_all_stage(scale_maps, gt_scale_map)
        
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl, loss_op=losses_op,loss_emb=losses_cls_emb, loss_quality=losses_quality) #losses_scale=losses_scale)
        
    def update_ot(self):
        with torch.no_grad():
            pre_pos = 0
            for idx in range(self.num_classes):
                nums = self.proxies_list[idx]
                NUM_CLUSTERS = nums
                data = self._embedding[idx].cpu().numpy()
                kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25, avoid_empty_clusters=True)
                assigned_clusters = kclusterer.cluster(data, assign_clusters=True)
                _sum = self.num_words
                counter = Counter(assigned_clusters)
                probs = []
                for idx,key in enumerate(counter):
                    _nums = counter[key]
                    prob = _nums/_sum
                    probs.append(prob)
                probs = sorted(probs)
                for idx,prob in enumerate(probs):
                    self._proxies_prob[pre_pos+idx] = prob
                pre_pos += nums
        
    def enable_emd_training(self):
        self.beta = 1

    def predict_quality(self, feats):
        '''
            feats: list[(Tensor)]

            return:
            quality_map: Tensor
        '''
        sz = feats[2].size()[-2:] # mid-level feature map
        num_scale = len(feats)
        _N = len(feats[0])
        indv_feat = [[] for _ in range(_N)]
        for i in range(num_scale):
            for j in range(_N):
                indv_feat[j].append(F.interpolate(feats[i][j][None, :, :, :], size=sz)) # (1, 256, 168, 168)

        for i in range(_N):
            indv_feat[i] = torch.cat(indv_feat[i], dim=1)
        
        indv_feat = torch.cat(indv_feat)
        indv_feat = self.quality_conv(indv_feat)
        indv_feat = self.relu(indv_feat)
        indv_feat = self.BN1(indv_feat)
        quality_map = self.quality_mapper(indv_feat)

        return quality_map

    def fuse_info_via_dyconv(self, fuse_stage, feats, scale_map):
        '''
        Args:
            feats: feature of all scale for all image

        Return:
            dynamic conv feats.
            
        '''

        return self.fuse_dy_convs[fuse_stage](feats, scale_map)


    def get_new_feat(self, feats, training=None):
        '''
        Args:
            feats: feature of all scale for all image
        '''
        sz = feats[0].size()[-2:]
        num_scale = len(feats)
        _N = len(feats[0])
        indv_feat = [[] for _ in range(_N)]
        for i in range(num_scale):
            for j in range(_N):
                indv_feat[j].append(F.interpolate(feats[i][j][None, :, :, :], size=sz))# (1, 256, 168, 168)
    
        for i in range(_N):
            indv_feat[i] = torch.cat(indv_feat[i], dim=1) # (1, 256 * 5, 168, 168)
        
        indv_feat = torch.cat(indv_feat) #(N, 256 * 5, 168, 168)
        
        scale_maps = []
        indv_feat = self.compress_feature(indv_feat)

        for i in range(self.fuse_stage):
            scale_map = self.scale_convs[i](indv_feat)
            scale_map = self.scale_mappers[i](scale_map) # (2, 256, 170, 170)
            indv_feat = self.fuse_info_via_dyconv(i, indv_feat, scale_map)
            indv_feat = self.BN1(indv_feat)
            indv_feat = self.relu(indv_feat)

            scale_maps.append(scale_map)
        
        indv_feat = self.decompress_feature(indv_feat)

        feats = list(torch.split(indv_feat, self.feat_channels, dim=1)) # split as scale
        div = 2
        for i in range(1, num_scale):
            feats[i] = F.interpolate(feats[i], size=(feats[i].size()[-2] // div, feats[i].size()[-1] // div))
            div *= 2
        
        if training:
            return tuple(feats), scale_maps
        else:
            return tuple(feats)

    def forward(self, feats, training=False):
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
        """
        #feats = self.get_new_feat(feats,training=training)
        
        if training:
            feats, scale_maps = feats, None

        quality_map = self.predict_quality(feats)
        
        ret = multi_apply(self.forward_single, feats, self.scales, quality_map=quality_map, top_size=feats[0].size()[-1])


        if training:
            ret = (ret[0], ret[1], scale_maps, quality_map)
            return ret
        else:
            return ret

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      gt_quality_map=None,
                      gt_scale_map=None,
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
            gt_quality_map (Tensor): Ground truth for quality
            gt_scale_map (Tensor): Ground truth for scale

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x, training=True)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas, gt_scale_map, gt_quality_map)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, gt_scale_map, gt_quality_map)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

