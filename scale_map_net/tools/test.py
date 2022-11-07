import os
import torch
import torch.nn as nn
import random
from PIL import Image
import matplotlib.pyplot as plt

from s_net import MCNN, res50_fpn, mobilenetv2_fpn, VDSR
from dataloader import ScaleDataset, ScaleDatasetEDSR

import cv2

import numpy as np

device = torch.device("cuda:1")
mcnn = VDSR()

#img_root = '/home/ranqinglin/work_dirs/ddet/data/VisDrone_Dataset_COCO_Format/images/instance_UFP_UAVtrain'
#gt_dmap_root = '/home/ranqinglin/work_dirs/ddet/data/VisDrone_Dataset_COCO_Format/images/train_scale_map'


gt_dmap_root = './dummy_gt'
img_root = './dummy_input'

checkpoint = torch.load('./checkpoints/epoch_14.pth',map_location='cpu')        # 从本地读取

# print(checkpoint)

mcnn.load_state_dict(checkpoint)   # 加载
mcnn.to(device)

mcnn.eval()

mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

gt_dmap_root_test = './dummy_gt'
img_root_test = './dummy_input'


test_dataset = ScaleDatasetEDSR(img_root_test,gt_dmap_root_test,4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

criterion=nn.MSELoss(size_average=False).to(device)

def save_img(et_dmap, name):
    et_dmap = et_dmap.squeeze(0)
    et_dmap = et_dmap.squeeze(0)
    et_dmap = et_dmap.detach().cpu().numpy()
    print(et_dmap)
    et_dmap = et_dmap * 500

    print(et_dmap)
    et_dmap = Image.fromarray(np.uint8(et_dmap))
    et_dmap.save(name)

#boxes = [[48, 784, 200, 236], [280, 792, 152, 228], [578, 108, 168, 182], [758, 860, 168, 182], [890, 698, 166, 182], [516, 882, 192, 148], [600, 790, 200, 134], [666, 708, 200, 132], [152, 1139, 192, 126], [56, 1185, 198, 114], [516, 1237, 188, 118], [714, 652, 184, 120], [48, 1239, 184, 120], [1131, 616, 184, 120], [624, 1115, 182, 116], [556, 1179, 182, 112], [764, 1215, 132, 152], [226, 1103, 196, 98], [388, 1475, 90, 116], [462, 1517, 88, 118], [1615, 1123, 68, 144], [663, 1481, 76, 128], [554, 1513, 82, 112], [889, 1481, 68, 124], [1640, 278, 52, 132], [1640, 36, 52, 112], [1311, 1477, 54, 106], [1406, 554, 38, 66], [516, 632, 38, 64], [999, 1247, 50, 42], [1062, 292, 40, 40], [985, 1163, 36, 36]]

with torch.no_grad():
    img_scale = (1332,1332)
    img_name = '0000331_01801_d_0000850.jpg'
    input_img = plt.imread(os.path.join(img_root, img_name))
    gt_img = np.load(os.path.join(gt_dmap_root, img_name).replace('jpg','npy'))
    # L = list(np.max(gt_img,axis=0))
    # L = sorted(L, reverse=True)
    # print(L[:30])
    gt_img = gt_img[:,:] * 255
    gt_dmap = Image.fromarray(np.uint8(gt_img))
    gt_dmap.save('./gt.jpg')

    img = cv2.resize(input_img, img_scale)

    save_input_img = Image.fromarray(np.uint8(img))
    save_input_img.save('./save_input_img.jpg')

    img = img.transpose((2,0,1))

    mean = np.float64(mean.reshape(3, 1, 1))
    std = 1 / np.float64(std.reshape(3, 1, 1))

    img_numpy = img - mean
    img_numpy = img_numpy * std

    img_tensor = torch.tensor(img_numpy, dtype=torch.float).to(device).unsqueeze(0)


    with torch.no_grad():
        test_mse = 0
        output_sum = 0
        for _data, _gt in test_dataloader:
            _data = _data.to(device)
            _gt = _gt.to(device)
            _out = mcnn(_data)
            output_sum += torch.sum(_out).item()
            test_mse += criterion(_out, _gt).item()
                    
        print(f'output_sum {output_sum}')
        print(f'test_mse {test_mse}')

    img_output = mcnn(img_tensor)

    save_img(img_output, "./01.jpg")

