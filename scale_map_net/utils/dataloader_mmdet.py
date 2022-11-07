from torch.utils.data import Dataset
import os
import numpy as np
import torch
import cv2
import json
from collections import defaultdict


class SegDataset(Dataset):
    '''
    transform bbox to segmentation map
    '''
    def __init__(self,img_root,bbox_json):
        '''
        img_root: the root path of img.
        
        '''
        self.img_root = img_root

        self.img_names = [filename for filename in os.listdir(img_root) \
                           if os.path.isfile(os.path.join(img_root, filename))]

        self.bbox = defaultdict(list)
        with open(bbox_json, "r") as f:
            _json_file = json.load(f)
        
        self.name_to_id = {}
        self.id_to_name = {}

        for _item in _json_file["images"]:
            self.name_to_id[_item["file_name"]] = _item["id"]
            self.id_to_name[_item["id"]] = _item["file_name"]

        for _item in _json_file['annotations']:
            self.bbox[_item['image_id']].append(_item['bbox'])

        for key in self.bbox.keys():
            self.bbox[key] = sorted(self.bbox[key], key=lambda x:x[2]*x[3], reverse=True)

        self.areaRng = [[0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['small', 'medium', 'large']
        self.ratio = [0.75, 1.5]

        self.n_samples = len(self.img_names)
        self.box_size = []


    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
         
        assert index <= len(self), 'index range error'
        img_name=self.img_names[index]

        img=cv2.imread(os.path.join(self.img_root, img_name))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, img_c = img.shape

        label = np.zeros((img_h, img_w, 3*3))
        boxes = self.bbox[self.name_to_id[img_name]]
        #print(boxes)
        for box in boxes:
            
            x, y, w, h = box
            cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0))
            ratio = w / h
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
            
            label[y:y+h,x:x+w,idx*3+idy] = 1.0
        
        return img, label


# test code
if __name__=="__main__":
    #gt_dmap_root = '/home/ranqinglin/work_dirs/ddet/data/VisDrone_Dataset_COCO_Format/images/train_scale_map'
    #img_root = '/home/ranqinglin/work_dirs/ddet/data/VisDrone_Dataset_COCO_Format/images/instance_UFP_UAVtrain'

    #gt_dmap_root = './dummy_gt'
    #img_root = './dummy_input'
    img_root = '/home/ranqinglin/work_dirs/ddet/data/VisDrone_Dataset_COCO_Format/images/instance_UFP_UAVtrain'
    bbox_json = '/home/ranqinglin/work_dirs/ddet/data/VisDrone_Dataset_COCO_Format/annotations/instances_UFP_UAVtrain.json'
    dataset = SegDataset(img_root, bbox_json)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i,(img,label) in enumerate(dataloader):
        print(img.shape, label.shape)
        print(label.sum())

        cv2.imwrite('./input.jpg', img[0].numpy())
        #cv2.imwrite('./scale.jpg', torch.sum(label[0],dim=-1).numpy() * 255)
        cv2.imwrite('./scale.jpg', label[0][:,:,3].numpy() * 255)
        break