
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from inspect import signature
from mmcv.ops import batched_nms

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

        self.weight = nn.Parameter(torch.randn(K, in_planes), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, in_planes), requires_grad=True)
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self): # maybe problematic
        nn.init.kaiming_uniform_(self.weight)

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x, attn_map):# a different version of dynamic convlution, is another kind of spatial attention
        residule = x
        batch_size, in_planes, height, width = x.size()
        softmax_attention = self.attention(attn_map).permute(0, 2, 3, 1) # (N, H, W, K)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同)
        aggregate_weight = torch.matmul(softmax_attention, self.weight)
        aggregate_weight = aggregate_weight.permute(0, 3, 1, 2)
        output = x * aggregate_weight

        if self.bias is not None:
            aggregate_bias = torch.matmul(softmax_attention, self.bias).permute(0, 3, 1, 2) # (N, C, H, W)
            output = output + aggregate_bias

        return residule + output

class ChannelAttention(nn.Module):
    def __init__(self, inplanes):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.MaxPool2d(1)
        self.avg_pool = nn.AvgPool2d(1)
        # 通道注意力，即两个全连接层连接
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=inplanes // 16, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=inplanes // 16, out_channels=inplanes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc(self.max_pool(x))
        avg_out = self.fc(self.avg_pool(x))
        # 最后输出的注意力应该为非负
        out = self.sigmoid(max_out + avg_out)
        return out
  
 
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 压缩通道提取空间信息
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 经过卷积提取空间注意力权重
        x = torch.cat([max_out, avg_out], dim=1)
        out = self.conv1(x)
        # 输出非负
        out = self.sigmoid(out)
        return out


@HEADS.register_module()
class DYMPHead1(GFLHead):
    """
    """

    def __init__(self,
                 num_words=200,
                 beta=0,
                 gamma=10,
                 proxies_list =[2, 3, 2, 5, 4, 8, 8, 4, 3, 3],
                 fuse_stage=5,
                 **kwargs):
        self.num_words=num_words
        self.proxies_list = proxies_list
        self.beta = beta
        self.gamma = gamma
        self.fuse_stage = fuse_stage
        super(DYMPHead1, self).__init__(**kwargs)

        self.atten1 = attention1d(in_planes=1, ratios=16, K=4)
        in_channel = 256 * 5
        self.c_attens = nn.ModuleList()
        self.s_attens = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(4):
            self.c_attens.append(ChannelAttention(in_channel))
            self.s_attens.append(SpatialAttention())
            self.bns.append(nn.BatchNorm2d(in_channel))

        self.diff_k_conv = nn.ModuleList()
        k_shape = [1,3,5,7]
        for i in range(4):
            k = k_shape[i]
            self.diff_k_conv.append(nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=k, padding= k // 2, bias=False))
        
        self.relu = nn.ReLU()
        
        assert self.num_classes == len(self.proxies_list)
        
    
    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
    
        # self.dy_scale_net = nn.ModuleList()
        # self.dy_quality_net = nn.ModuleList()
        
        # for i in range(self.fuse_stage):
        #     self.dy_scale_net.append(Dynamic_conv1d(self.feat_channels, 4))
        #     self.dy_quality_net.append(Dynamic_conv1d(self.feat_channels, 4))

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
        self.gfl_cls_conv = nn.Conv2d(
            self.feat_channels, self.feat_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
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
        # for m in self.fuse_dy_convs: # debug purpose
        #     normal_init(m, std=0.01) 
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.gfl_cls_conv, std=0.01)
        normal_init(self.gfl_reg, std=0.01)
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
    
    def forward_single(self, x, scale):
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

        feats =  torch.cat(new_feats, dim=0)
        labels = torch.cat(new_labels, dim=0)
        scores = torch.cat(new_scores, dim=0)
       
        loss_op = self.loss_op(feats, labels)
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

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_feats,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
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
        
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl, loss_op=losses_op,loss_emb=losses_cls_emb) #losses_scale=losses_scale)
        
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

    def dynamic_net_scale(self, feat, scale_map, weight):

        ret = 0
        for i in range(4):
            residue = feat
            out = self.diff_k_conv[i](feat)
            out = self.bns[i](out)
            out = self.relu(out)

            out = self.c_attens[i](out) * out
            out = self.s_attens[i](out) * out
 
            out += residue
            ret += out * weight[i]

        return ret
        
    

    def refine_feat_by_scale(self, feats, scale_map=None):
        '''
        Args:
            feats: feature of all scale for all image
        '''

        # num_scale = len(feats)
        # ret = []

        # for i in range(num_scale):
        #     _feat = feats[i]
        #     sz = _feat.size()[-2:]
        #     _scale_map = F.interpolate(scale_map, size=sz)
        #     ret.append(self.dy_scale_net[i](_feat, _scale_map))
        
        ret = []
        all_feats = []
        num_scale = len(feats)
        sz = [x.size() for x in feats]
        mid_sz = sz[1][-2:]
        scale_map = F.interpolate(scale_map, size=mid_sz)

        #print(f'mid_sz {mid_sz} sz {sz}') , [torch.Size([4, 256, 168, 168]), torch.Size([4, 256, 84, 84]), torch.Size([4, 256, 42, 42]), torch.Size([4, 256, 21, 21]), torch.Size([4, 256, 11, 11])]

        _feats = []
        for i in range(len(feats)):
            _feats.append(F.interpolate(feats[i], size=mid_sz))

        shape = [256 for i in range(len(feats))]
        weight = self.atten1(scale_map)
        for i in range(len(_feats[0])):
            _input = []
            for j in range(num_scale):
                _input.append(_feats[j][i])
            _input = torch.cat(_input)[None,...]
            _out = self.dynamic_net_scale(_input, scale_map, weight[i]) # split here
            _out = torch.split(_out, shape, dim=1)
            all_feats.append(_out)

        for i in range(num_scale):
            _sz = sz[i]
            _feats = []
            for j in range(len(all_feats)):
                _feats.append(F.interpolate(all_feats[j][i], size=_sz[-2:]))
            _feats = torch.cat(_feats)
            ret.append(_feats)

        return tuple(ret)

    def refine_feat_by_quality(self, feats, quality_map=None):
        '''
        Args:
            feats: feature of all quality for all image
        '''

        num_scale = len(feats)
        ret = []
        
        for i in range(num_scale):
            _feat = feats[i]
            sz = _feat.size()[-2:]
            _quality_map = F.interpolate(quality_map, size=sz)
            ret.append(self.dy_quality_net[i](_feat, _quality_map))
            
        return tuple(ret)

    def forward(self, feats, scale_map=None, quality_map=None):
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

        feats = self.refine_feat_by_scale(feats, scale_map=scale_map)
        #feats = self.refine_feat_by_quality(feats, quality_map=quality_map)
        
        ret = multi_apply(self.forward_single, feats, self.scales)

        return ret

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      scale_map=None, 
                      quality_map=None,
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
        outs = self(x, scale_map=scale_map, quality_map=quality_map)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list
    
    def simple_test(self, feats, img_metas, rescale=False, scale_maps=None, quality_maps=None):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale, scale_maps=scale_maps, quality_maps=quality_maps)

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """

        outs = self.forward(feats)

        results_list = self.get_bboxes(
            *outs, img_metas=img_metas, rescale=rescale)
        return results_list

    def aug_test(self, feats, img_metas, rescale=False, scale_maps=None, quality_maps=None):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5), where
                5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,), The length of list should always be 1.
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale, scale_maps=scale_maps, quality_maps=quality_maps)

    def aug_test_bboxes(self, feats, img_metas, rescale=False, scale_maps=None, quality_maps=None):
        """Test det bboxes with test time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The length of list should always be 1.
        """
        # check with_nms argument
        gb_sig = signature(self.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        gbs_sig = signature(self._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'

        aug_bboxes = []
        aug_scores = []
        aug_labels = []
        for x, img_meta, scale_map, quality_map in zip(feats, img_metas, scale_maps, quality_maps):
            # only one image in the batch
            outs = self.forward(x, scale_map=scale_map, quality_map=quality_map)
            bbox_outputs = self.get_bboxes(
                *outs,
                img_metas=img_meta,
                cfg=self.test_cfg,
                rescale=False,
                with_nms=False)[0]
            aug_bboxes.append(bbox_outputs[0])
            aug_scores.append(bbox_outputs[1])
            if len(bbox_outputs) >= 3:
                aug_labels.append(bbox_outputs[2])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_labels = torch.cat(aug_labels, dim=0) if aug_labels else None

        if merged_bboxes.numel() == 0:
            det_bboxes = torch.cat([merged_bboxes, merged_scores[:, None]], -1)
            return [
                (det_bboxes, merged_labels),
            ]

        det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores,
                                            merged_labels, self.test_cfg.nms)
        det_bboxes = det_bboxes[:self.test_cfg.max_per_img]
        det_labels = merged_labels[keep_idxs][:self.test_cfg.max_per_img]

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])

        return [
            (_det_bboxes, det_labels),
        ]

    