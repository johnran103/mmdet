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

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmodel')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


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
    cv2.imwrite('./scale.jpg', torch.sum(torch.tensor(img1),dim=-1).numpy() * 255)
       


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

    device = torch.device('cuda:1')

    psudo_label_generator = GetPsudoSegLabel()

    dataset=build_dataset(cfg.data.train)

    samples_per_gpu = cfg.data.samples_per_gpu

    shuffle = True
    
    sampler = GroupSampler(dataset,samples_per_gpu) if shuffle else None

    data_loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=8,
        batch_sampler=None,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False)

    model=OneModel(cfg.model).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.model.lr_rate, momentum=0.95) #SGDM
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [8,10], gamma=0.1)
    epoch=12 # 1x

    model.train()
    for _epoch in range(epoch):
        print(f'start >>>>>>>>>>>>>>>>>>>> {_epoch} epoch <<<<<<<<<<<<<<<<')
        epoch_loss = []
        for i, x in enumerate(data_loader):
            imgs = x['img'].data[0].to(device)
            img_h, img_w = imgs.size()[-2:]
            gt_bboxes = x['gt_bboxes'].data[0]
            seg_label = []
            for _bboxes in gt_bboxes:
                seg_label.append(psudo_label_generator.get(img_h, img_w, _bboxes))
            seg_label=torch.cat(seg_label, dim=0)
            seg_label = seg_label.to(device)
            # visualize(imgs[0].cpu(), seg_label[0].cpu())

            _output = model(imgs)
            loss = calc_loss(_output.permute(0,2,3,1), seg_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            if i % 200 == 0:
                print(f'>>>>iter {i} loss {loss.item()}')

        if (_epoch + 1) % 4 == 0:
            torch.save(model.state_dict(),'../ckpt/checkpoints_seg/epoch_'+str(_epoch)+".pth")
        
        print(f'>>>>>>>>>>>>>>>>>>>> epoch {_epoch} loss {sum(epoch_loss)/len(epoch_loss)} <<<<<<<<<<<<<<<<')
        scheduler.step()



