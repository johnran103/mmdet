import argparse
from collections import defaultdict
from email.policy import default
from mmdet.apis import init_detector, show_result_pyplot, inference_detector
import warnings
import cv2
import mmcv
import torch
import math
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
import numpy as np
from mmdet.core import UnifiedForegroundPacking
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import time
import json

CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')

colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 128, 255),
    (128, 255, 128),
    (255, 128, 128),
    (255, 255, 255),

]

class COCOevalNew(COCOeval):
    
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super(COCOevalNew, self).__init__(cocoGt=None, cocoDt=None, iouType='segm')

        from collections import defaultdict
        self.wrongCls = defaultdict(list)


    def evaluateImg(self, imgId, catId, aRng, maxDet):
        
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''

        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]

        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]

        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]

        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)

        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))

        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                if(len(self.wrongCls[tind]) == 0):
                    for _ in range(11):
                        self.wrongCls[tind].append([])
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
                    
                    if gt[m]['category_id'] != catId:
                        self.wrongCls[tind][gt[m]['category_id']].append(catId)

        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        
        
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }


def compute_iof(pos1, pos2):
    left1, top1, right1, down1 = pos1
    left2, top2, right2, down2 = pos2
    area1 = (right1 - left1) * (down1 - top1)
    area2 = (right2 - left2) * (down2 - top2)
    # 计算中间重叠区域的坐标
    left = max(left1, left2)
    right = min(right1, right2)
    top = max(top1, top2)
    bottom = min(down1, down2)
    if left >= right or top >= bottom:
        return 0
    else:
        inter = (right - left) * (bottom - top)
        return inter / min(area1, area2)


def compute_iou(pos1, pos2):
    left1, top1, right1, down1 = pos1
    left2, top2, right2, down2 = pos2
    area1 = (right1 - left1) * (down1 - top1)
    area2 = (right2 - left2) * (down2 - top2)
    # 计算中间重叠区域的坐标
    left = max(left1, left2)
    right = min(right1, right2)
    top = max(top1, top2)
    bottom = min(down1, down2)
    if left >= right or top >= bottom:
        return 0
    else:
        inter = (right - left) * (bottom - top)
        return inter / (area1 + area2 - inter)



# modify test
class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results, bbox=None, img_data=None):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None

        if img_data is None:
            img = mmcv.imread(results['img'])
        else:
            img = img_data
        if bbox:
            x1, x2, y1, y2, _ = bbox
            img = img[x1:x2, y1:y2, :]
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def my_inference_detector(model, data):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result[0]


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep



def display_merge_result(results, img, img_name, w, h):
    w = math.ceil(w)
    h = math.ceil(h)
    img_data = cv2.imread(img)
    new_img = np.zeros((h, w, 3))
    for result in results:
        x1, y1, w, h, n_x, n_y, scale_factor = [math.floor(_) for _ in result]
        if w == 0 or h == 0:
            continue
        new_img[n_y:n_y + h * scale_factor, n_x:n_x + w * scale_factor, :] = cv2.resize(
            img_data[y1:y1 + h, x1:x1 + w, :], (w * scale_factor, h * scale_factor))
    return new_img


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mp_det_config', help='')
    parser.add_argument('mp_det_config_ckpt', help='')
    parser.add_argument('dataset_anno', help='')
    parser.add_argument('dataset_root', help='')
    args = parser.parse_args()
    return args


def main():
    device = 'cuda'
    args = parse_args()
    
    mp_det_config = args.mp_det_config
    mp_det_config_ckpt = args.mp_det_config_ckpt
 
    mp_det = init_detector(mp_det_config, mp_det_config_ckpt, device=device)
    dataset_anno = args.dataset_anno
    dataset_root = args.dataset_root

    coco = COCO(dataset_anno)  # 导入验证集
    size = len(list(coco.imgs.keys()))
    results = []
    times = []
    # shape_set = set()
    cnt = 1
    rm_cnt = 1e-6
    sum_rm = 0
    for key in range(size):
        print(cnt, size, end='\r')
        cnt += 1
        # if cnt > 10:
        #     continue
        image_id = key
        img_name = coco.imgs[key]['file_name']
        img = os.path.join(dataset_root, img_name)
        data = dict(img=img)
        
        second_results = my_inference_detector(mp_det, LoadImage()(data))
        
        finale_results = second_results

        for idx in range(len(finale_results)):
            finale_results[idx] = np.array(finale_results[idx])

        for idx, result in enumerate(finale_results):
            result = np.array(result)
            if result.shape[0] == 0:
                continue
            keep = py_cpu_nms(result,0.6)

            for bbox in result[keep]:
                rm_cnt += 1
                x1, y1, x2, y2, score = bbox
                sum_rm += score
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                image_result = {
                    'image_id': image_id,
                    'category_id': idx,
                    'score': float(score),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                }
                results.append(image_result)
            
    # write output

    json.dump(results, open('{}_bbox_result_tmp.json'.format('UAV'), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = coco
    coco_pred = coco_true.loadRes('{}_bbox_result_tmp.json'.format('UAV'))

    # run COCO evaluation
    coco_eval = COCOevalNew(coco_true, coco_pred, 'bbox')
    # coco_eval.params.imgIds = image_ids
    coco_eval.params.maxDets = [10, 100 , 500]

    wrongCls = coco_eval.wrongCls
    for t in wrongCls.keys():
        for ii in range(len(wrongCls[t])):
            wrongCls[t][ii] = Counter(wrongCls[t][ii])
    print(f'wrongCls {wrongCls}')

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    main()
