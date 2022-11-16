import torch
import torch.nn as nn
import argparse
import sys
import os
sys.path.append(os.path.abspath('..'))
from utils.loss import calc_loss
from model.PSPNet import OneModel
from torch.utils.data import DataLoader
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.datasets.samplers import GroupSampler
import cv2
from mmcv.parallel import collate
from functools import partial
from utils.mIOU import miou
import torch.nn.functional as F
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmodel')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def visualize(img,img1, i):
    mean=torch.tensor([123.675, 116.28 , 103.53 ])
    std=torch.tensor([58.395,57.12,57.375])
    img = img.permute(1,2,0)
    img = img * std[None,None,:]
    img = img + mean[None,None,:]
    
    img = img.numpy()
    # for box in gt_bboxes:
    #     cv2.rectangle(img,(int(box[0]),int(box[1])), (int(box[2]),int(box[3])),(0,255,0))
    print(img)
    cv2.imwrite(f'./o_img/tmp_{i}.jpg', img)
    #label = psudo_label_generator.get(img, x['gt_bboxes'])
    print(img1)
    cv2.imwrite(f'./s_img/scale{i}.jpg', torch.sum(torch.tensor(img1.permute(1,2,0)),dim=-1).numpy() * 255)
       


class GetPsudoSegLabel:
    def __init__(self):
        self.areaRng = [[0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['small', 'medium', 'large']
        self.ratio = [0.75, 1.5]

    def get(self, img_h, img_w, boxes):
        label = np.zeros((img_h, img_w, 3*3))
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            if h != 0:
                ratio = w / h
            else:
                ratio = 1
            idx=-1
            idy=-1
            area = w * h
            for i in range(3):
                if area >= self.areaRng[i][0] and area < self.areaRng[i][1]:
                    idx = i
                    break
            if ratio < self.ratio[0]:
                idy = 0
            elif ratio < self.ratio[1]:
                idy = 1
            else:
                idy = 2
            label[y1:y2,x1:x2,idx*3+idy] = 1.0
        
        return torch.tensor(label[None,...])

if __name__=="__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)

    device = torch.device('cuda:0')

    checkpoint = torch.load('../ckpt/checkpoints_seg/epoch_3.pth',map_location='cpu')

    psudo_label_generator = GetPsudoSegLabel()

    dataset=build_dataset(cfg.data.test)

    samples_per_gpu = cfg.data.samples_per_gpu

    shuffle = True
    
    sampler = GroupSampler(dataset,samples_per_gpu) if shuffle else None

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=8,
        batch_sampler=None,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False)

    model=OneModel(cfg.model).to(device)
    model.load_state_dict(checkpoint) 
    model.eval()

    print(f'>>>>>>>>>>>>>>>>>>>> testing !!! <<<<<<<<<<<<<<<<')
    with torch.no_grad():
        _loss = []
        mious = []
        for i, x in enumerate(data_loader):
            #print(x['img'][0].size())
            imgs = x['img'][0].to(device)
            
            img_h, img_w = imgs.size()[-2:]
            gt_bboxes = x['gt_bboxes'][0]
            seg_label = seg_label.append(psudo_label_generator.get(img_h, img_w, _bboxes))
            seg_label = seg_label.to(device)

            _output = model(imgs)
            _output = F.sigmoid(_output)
            visualize(imgs[0].cpu(), _output[0].cpu(), i)
            loss = calc_loss(_output.permute(0,2,3,1), seg_label)
            mious.append(miou(seg_label,_output[0].cpu()))

        print(f'>>>>>>>>>>>>>>>>>>>> loss {sum(_loss)/len(_loss)} <<<<<<<<<<<<<<<<')
        print(f'>>>>>>>>>>>>>>>>>>>> miou {sum(mious)/len(mious)} <<<<<<<<<<<<<<<<')



