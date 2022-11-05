import os
import torch
import torch.nn as nn
import random
from PIL import Image

from q_net import MCNN
from dataloader import QualityDataset

import numpy as np


if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    device=torch.device("cuda:0")
    mcnn=MCNN().to(device)
    criterion=nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.SGD(mcnn.parameters(), lr=1e-6,
                                momentum=0.95)
    
    img_root = '/home/ranqinglin/work_dirs/ddet/data/VisDrone_Dataset_COCO_Format/images/instance_UFP_UAVtrain'
    gt_dmap_root = '/home/ranqinglin/work_dirs/ddet/data/VisDrone_Dataset_COCO_Format/images/train_quality_map'

    dataset = QualityDataset(img_root,gt_dmap_root,4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 50, 80], gamma=0.1)


    #training phas[e
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    
    train_loss_list = []
    epoch_list = []

    for epoch in range(0,100):
        mcnn.train()

        epoch_loss=0

        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            #print(img.size())
            #print(gt_dmap.size())
            # forward propagation
            et_dmap=mcnn(img)
            # calculate loss
            loss=criterion(et_dmap,gt_dmap)
            epoch_loss+=loss.item()
            if i % 100 == 0:
                print(f'iter {i}, loss {loss.item()}')
            #print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss/len(dataloader))
        print(f'epoch {epoch} loss {epoch_loss/len(dataloader)}')

        if (epoch + 1) % 20 == 0:
            torch.save(mcnn.state_dict(),'./checkpoints/epoch_'+str(epoch)+".param")

        scheduler.step()

        mcnn.eval()
        index = 100
        img, gt_dmap = dataset[index]
        
        img = img.unsqueeze(0).to(device)
        gt_dmap = gt_dmap.squeeze(0)
        et_dmap = mcnn(img)
        et_dmap = et_dmap.squeeze(0)
        et_dmap = et_dmap.squeeze(0)
        et_dmap = et_dmap.detach().cpu().numpy()

        gt_dmap = gt_dmap.numpy() * 255
        et_dmap = et_dmap * 255

        #print(gt_dmap)
        #print(et_dmap)

        gt_dmap = Image.fromarray(np.uint8(gt_dmap))
        et_dmap = Image.fromarray(np.uint8(et_dmap))

        gt_dmap.save('./gt_dmap.jpg')
        et_dmap.save('./et_dmap.jpg')

    import time
    print(time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))

        