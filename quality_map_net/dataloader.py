from curses import reset_shell_mode
from tkinter import image_names
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import mmcv

from PIL import Image

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

import copy

from mmdet.core import PolygonMasks

import inspect

class QualityDataset(Dataset):
    '''
    crowdDataset
    '''
    def __init__(self,img_root,gt_dmap_root,gt_downsample=4):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        '''
        self.img_root = img_root
        self.gt_dmap_root = gt_dmap_root
        self.gt_downsample = gt_downsample

        self.img_names = [filename for filename in os.listdir(img_root) \
                           if os.path.isfile(os.path.join(img_root, filename))]
        self.n_samples = len(self.img_names)

        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
         
        assert index <= len(self), 'index range error'
        img_name=self.img_names[index]

        img_scale = (1332,1332)

        img=plt.imread(os.path.join(self.img_root, img_name))
        if len(img.shape)==2: # expand grayscale image to three channel.
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img), 2)
   
        gt_dmap=np.load(os.path.join(self.gt_dmap_root, img_name.replace('.jpg','.npy')))
        img = cv2.resize(img, img_scale)

        if self.gt_downsample > 1: # to downsample image and density-map to match deep-model.
            ds_rows=int(img.shape[0]//self.gt_downsample)
            ds_cols=int(img.shape[1]//self.gt_downsample)
            img = cv2.resize(img,(ds_cols * self.gt_downsample, ds_rows * self.gt_downsample))
            img = img.transpose((2,0,1)) # convert to order (channel,rows,cols)

            gt_dmap = cv2.resize(gt_dmap,(ds_cols, ds_rows))
            gt_dmap = gt_dmap[np.newaxis,:,:]

            img_numpy = torch.tensor(img,dtype=torch.float).numpy()
            img_numpy = np.float32(img_numpy) if img_numpy.dtype != np.float32 else img_numpy.copy()

            mean = np.float64(self.mean.reshape(3, 1, 1))
            stdinv = 1 / np.float64(self.std.reshape(3, 1, 1))

            img_numpy = img_numpy - mean
            img_numpy = img_numpy * stdinv
            #cv2.subtract(img_numpy, mean, img_numpy)  # inplace
            #cv2.multiply(img_numpy, stdinv, img_numpy)  # inplace
            
            img_tensor = torch.tensor(img_numpy, dtype=torch.float)
            gt_dmap_tensor = torch.tensor(gt_dmap, dtype=torch.float)

        return img_tensor, gt_dmap_tensor


class QualityDatasetEDSR(Dataset):
    '''
    crowdDataset
    '''
    def __init__(self,img_root,gt_dmap_root,gt_downsample=4):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        '''
        self.img_root = img_root
        self.gt_dmap_root = gt_dmap_root
        self.gt_downsample = gt_downsample

        self.img_names = [filename for filename in os.listdir(img_root) \
                           if os.path.isfile(os.path.join(img_root, filename))]
        self.n_samples = len(self.img_names)

        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
         
        assert index <= len(self), 'index range error'
        img_name=self.img_names[index]

        img_scale = (1332,1332)

        img=plt.imread(os.path.join(self.img_root, img_name))
        if len(img.shape)==2: # expand grayscale image to three channel.
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img), 2)
   
        gt_dmap=np.load(os.path.join(self.gt_dmap_root, img_name.replace('.jpg','.npy')))
        img = cv2.resize(img, img_scale)

        if self.gt_downsample > 1: # to downsample image and density-map to match deep-model.
            ds_rows=int(img.shape[0]//self.gt_downsample)
            ds_cols=int(img.shape[1]//self.gt_downsample)
            #img = cv2.resize(img,(ds_cols * self.gt_downsample, ds_rows * self.gt_downsample))
            img = cv2.resize(img,(ds_cols, ds_rows))
            img = img.transpose((2,0,1)) # convert to order (channel,rows,cols)

            gt_dmap = cv2.resize(gt_dmap,(ds_cols, ds_rows))
            gt_dmap = gt_dmap[np.newaxis,:,:]

            img_numpy = torch.tensor(img,dtype=torch.float).numpy()
            img_numpy = np.float32(img_numpy) if img_numpy.dtype != np.float32 else img_numpy.copy()

            mean = np.float64(self.mean.reshape(3, 1, 1))
            stdinv = 1 / np.float64(self.std.reshape(3, 1, 1))

            img_numpy = img_numpy - mean
            img_numpy = img_numpy * stdinv
            #cv2.subtract(img_numpy, mean, img_numpy)  # inplace
            #cv2.multiply(img_numpy, stdinv, img_numpy)  # inplace
            
            img_tensor = torch.tensor(img_numpy, dtype=torch.float)
            gt_dmap_tensor = torch.tensor(gt_dmap, dtype=torch.float)

        return img_tensor, gt_dmap_tensor


albu_train_transforms = [
    dict(
        type='HorizontalFlip',
        p=0.5),
    dict(
        type='Rotate',
        limit=180,
        p=0.5),
]

albu_train_transforms1 = [dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='ChannelShuffle', p=0.1),
]

dict1 = {'mask': 'image'}

class Albu:

    def __init__(self, transforms):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)

        self.transforms = transforms
       
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           additional_targets=dict1)

    def albu_builder(self, cfg):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    def __call__(self, results):
        # dict to albumentations format
        
        results = self.aug(**results)

        return results


class Albu1(Albu):
    def __init__(self, transforms):
        super(Albu1, self).__init__(transforms)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms])


class QualityDatasetAug(Dataset):
    '''
    crowdDataset
    '''
    def __init__(self,img_root,gt_dmap_root,gt_downsample=4):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        '''
        self.img_root = img_root
        self.gt_dmap_root = gt_dmap_root
        self.gt_downsample = gt_downsample

        self.img_names = [filename for filename in os.listdir(img_root) \
                           if os.path.isfile(os.path.join(img_root, filename))]
        self.n_samples = len(self.img_names)

        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
         
        assert index <= len(self), 'index range error'
        img_name=self.img_names[index]

        img_scale = (1332,1332)

        #img=plt.imread(os.path.join(self.img_root, img_name))

        img = cv2.imread(os.path.join(self.img_root, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #print(img.shape)

        if len(img.shape)==2: # expand grayscale image to three channel.
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img), 2)
   
        gt_dmap=np.load(os.path.join(self.gt_dmap_root, img_name.replace('.jpg','.npy')))
        img = cv2.resize(img, img_scale)

        if self.gt_downsample > 1: # to downsample image and density-map to match deep-model.
            ds_rows=int(img.shape[0]//self.gt_downsample)
            ds_cols=int(img.shape[1]//self.gt_downsample)
            img = cv2.resize(img,(ds_cols * self.gt_downsample, ds_rows * self.gt_downsample))

            gt_dmap = cv2.resize(gt_dmap,(ds_cols, ds_rows))
            #gt_dmap = gt_dmap[:,:,np.newaxis]

            img_numpy = torch.tensor(img,dtype=torch.float).numpy()
            img_numpy = np.float32(img_numpy) if img_numpy.dtype != np.float32 else img_numpy.copy()

            # p = np.random.rand() < 0.5
            # if p :
            #     img_numpy, gt_dmap = flip(img_numpy ,gt_dmap, 'horizontal')

            # img_numpy = np.array([i for i in range(3*64)]).reshape(3,8,8)
            # gt_dmap = np.array([i for i in range(3*64)]).reshape(3,8,8)

            # print(img_numpy[0][0][0])
            # print(img_numpy[0][0][-1])

            # save_input_img = Image.fromarray(np.uint8(img_numpy))
            # save_input_img_gt = Image.fromarray(np.uint8(gt_dmap * 255))
            # save_input_img.save('./save_input_img_0.jpg')
            # save_input_img_gt.save('./save_input_img_gt_0.jpg')

            results = {}
            results['image'] = img_numpy
            results['mask'] = gt_dmap

            aug1 = Albu(transforms=albu_train_transforms)
            results = aug1(results)

            img_numpy = results['image']
            gt_dmap = results['mask']

            # print(img_numpy[0][0][0])
            # print(img_numpy[0][0][-1])

            # save_input_img = Image.fromarray(np.uint8(img_numpy))
            # save_input_img_gt = Image.fromarray(np.uint8(gt_dmap * 255))
            # save_input_img.save('./save_input_img_1.jpg')
            # save_input_img_gt.save('./save_input_img_gt_1.jpg')

            reuslts = {}
            reuslts['image'] = img_numpy

            aug2 = Albu1(transforms=albu_train_transforms1)
            results = aug2(results)

            mean = np.float64(self.mean.reshape(3, 1, 1))
            stdinv = 1 / np.float64(self.std.reshape(3, 1, 1))

            img_numpy = img_numpy.transpose((2,0,1)) # convert to order (channel,rows,cols)
            gt_dmap = gt_dmap[:,:,np.newaxis]
            gt_dmap = gt_dmap.transpose((2,0,1))

            img_numpy = img_numpy - mean
            img_numpy = img_numpy * stdinv
            
            img_tensor = torch.tensor(img_numpy, dtype=torch.float)
            gt_dmap_tensor = torch.tensor(gt_dmap, dtype=torch.float)

        return img_tensor, gt_dmap_tensor


# test code
if __name__=="__main__":
    #gt_dmap_root = '/home/ranqinglin/work_dirs/ddet/data/VisDrone_Dataset_COCO_Format/images/train_scale_map'
    #img_root = '/home/ranqinglin/work_dirs/ddet/data/VisDrone_Dataset_COCO_Format/images/instance_UFP_UAVtrain'

    gt_dmap_root = './dummy_gt'
    img_root = './dummy_input'
    dataset = ScaleDatasetAug(img_root, gt_dmap_root, gt_downsample=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i,(img,gt_dmap) in enumerate(dataloader):
        # plt.imshow(img)
        # plt.figure()
        # plt.imshow(gt_dmap)
        # plt.figure()
        
        #print(img)
        #print(gt_dmap)
        print(len(dataloader))
        print(img.shape, gt_dmap.shape)
        break