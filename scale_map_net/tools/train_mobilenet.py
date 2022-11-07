import os
import torch
import torch.nn as nn
import random
from PIL import Image

from s_net import MCNN, res50_fpn, mobilenetv2_fpn
from dataloader import ScaleDataset,ScaleDatasetAug

import numpy as np
import math

pi=3.1415926

if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    device=torch.device("cuda:0")
    mcnn=mobilenetv2_fpn().to(device)

    criterion=nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.SGD(mcnn.parameters(), lr=1e-6,
                                momentum=0.95)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 50, 80], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25], gamma=0.1)
    
    img_root = '/home/ranqinglin/work_dirs/ddet/data/VisDrone_Dataset_COCO_Format/images/instance_UFP_UAVtrain'
    gt_dmap_root = '/home/ranqinglin/work_dirs/ddet/data/VisDrone_Dataset_COCO_Format/images/train_scale_map'

    #gt_dmap_root = './dummy_gt'
    #img_root = './dummy_input'

    gt_dmap_root_test = './dummy_gt'
    img_root_test = './dummy_input'

    test_dataset = ScaleDatasetAug(img_root_test,gt_dmap_root_test,4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


    dataset = ScaleDatasetAug(img_root,gt_dmap_root,4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    #training phase
    if not os.path.exists('./checkpoints_aug'):
        os.mkdir('./checkpoints_aug')
    
    train_loss_list = []
    epoch_list = []

    max_epoch = 30

    for epoch in range(0,max_epoch):
        mcnn.train()

        epoch_loss=0

        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            #print(img.size())
            #print(gt_dmap.size())
            # forward propagation
            idx = gt_dmap < 1e-10
            idx1 = gt_dmap > 1e-10
            et_dmap = mcnn(img)
            #print(torch.sum(et_dmap))

            w1 = min(math.cos((max_epoch-epoch)/max_epoch*(pi/2)), 0.5)
            w2 = max(math.cos(epoch/max_epoch*(pi/2)), 0.5)

            s = w1 + w2
            w1 = 1/(s+1e-10) * w1
            w2 = 1/(s+1e-10) * w2

            #print(w1, w2)

            loss1 = w1 * criterion(et_dmap[idx], gt_dmap[idx])
            loss2 = w2 * criterion(et_dmap[idx1], gt_dmap[idx1])
            loss = loss1 + loss2

            epoch_loss+=loss.item()
            if i % 100 == 0:
                print(f'iter {i}, loss {loss.item()}')
                mcnn.eval()
                part_ouput = []
                with torch.no_grad():
                    test_mse = 0
                    output_sum = 0
                    for _data, _gt in test_dataloader:
                        #print('fuck')
                        #print(len(test_dataloader))
                        _data = _data.to(device)
                        _gt = _gt.to(device)
                        _out = mcnn(_data)
                        output_sum += torch.sum(_out).item()
                        test_mse += criterion(_out, _gt).item()
                        part_ouput.append(_out[0,0,0,0].item())
                        part_ouput.append(_out[0,0,0,50].item())
                        part_ouput.append(_out[0,0,0,100].item())
                        part_ouput.append(_out[0,0,0,200].item())
                    
                    print(f'output_sum {output_sum}')
                    print(f'test_mse {test_mse}')
                    print(f'part_output {part_ouput}')

                mcnn.train()
            #print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(mcnn.state_dict()['fuse.0.bias'])
        

        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss/len(dataloader))
        print(f'epoch {epoch} loss {epoch_loss/len(dataloader)}')

        if (epoch + 1) % 5 == 0:
            torch.save(mcnn.state_dict(),'./checkpoints_aug/epoch_'+str(epoch)+".pth")

        scheduler.step()

    import time
    print(time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))

        