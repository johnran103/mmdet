import torch
import torch.nn as nn

from mmdet.models import ResNet, FPN, MobileNetV2

import torch.nn.functional as F

class MCNN(nn.Module):
    '''
    Implementation of Multi-column CNN for crowd counting
    '''
    def __init__(self, load_weights=False):
        super(MCNN,self).__init__()

        self.branch1=nn.Sequential(
            nn.Conv2d(3,16,9,padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,7,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,16,7,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,8,7,padding=3),
            nn.ReLU(inplace=True)
        )

        self.branch2=nn.Sequential(
            nn.Conv2d(3,20,7,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20,40,5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40,20,5,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20,10,5,padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch3=nn.Sequential(
            nn.Conv2d(3,24,5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48,24,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24,12,3,padding=1),
            nn.ReLU(inplace=True)
        )

        self.fuse=nn.Sequential(nn.Conv2d(30,1,1,padding=0))

        self.relu=nn.ReLU(inplace=True)

        if not load_weights:
            self._initialize_weights()

    def forward(self,img_tensor):
        x1=self.branch1(img_tensor)
        x2=self.branch2(img_tensor)
        x3=self.branch3(img_tensor)
        x=torch.cat((x1,x2,x3),1)
        x=self.fuse(x)
        x=self.relu(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



'''
Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
'''


'''
Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])

'''


class res50_fpn(nn.Module):
    def __init__(self, load_weights=False):
        super(res50_fpn,self).__init__()

        self.resnet = ResNet(50)
        self.in_channels = [256, 512, 1024, 2048]
        self.scales = [333, 167, 84, 42]
        self.fpn = FPN(self.in_channels, 256, len(self.scales))
        self.fuse1 = nn.Conv2d(256*4,256,1,padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.bn2 = nn.BatchNorm2d(num_features=1)
        self.fuse2 = nn.Conv2d(256,1,1,padding=0)

    def forward(self, input):
        ret = self.resnet(input)

        ret = list(self.fpn(ret))

        #_scale = (333, 333)
        for i in range(4):
            ret[i] = F.interpolate(ret[i], size=(333,333), mode='bilinear')
        
        ret = torch.cat(ret,dim=1)

        ret = self.fuse1(ret)
        ret = self.bn1(ret)
        ret = self.relu(ret)

        ret = self.fuse2(ret)
        ret = self.bn2(ret)
        ret = self.relu(ret)

        return ret


class mobilenetv2_fpn(nn.Module):
    def __init__(self, load_weights=False):
        super(mobilenetv2_fpn,self).__init__()
        self.mobilenet = MobileNetV2()
        self.in_channels = [24, 32, 96, 1280]
        self.scales = [333, 167, 84, 42]
        self.fpn = FPN(self.in_channels, 256, len(self.scales))
        self.fuse1 = nn.Conv2d(256*4,256,1,padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.bn2 = nn.BatchNorm2d(num_features=1)
        self.fuse2 = nn.Conv2d(256,1,1,padding=0)

    def forward(self, input):
        ret = self.mobilenet(input)
        
        ret = list(self.fpn(ret))

        #_scale = (333, 333)
        for i in range(4):
            ret[i] = F.interpolate(ret[i], size=(333,333), mode='bilinear')
        
        ret = torch.cat(ret,dim=1)

        ret = self.fuse1(ret)
        ret = self.bn1(ret)
        ret = self.relu(ret)

        ret = self.fuse2(ret)
        ret = self.bn2(ret)
        ret = self.relu(ret)

        return ret


# test code
if __name__=="__main__":
    img=torch.rand((1,3,800,1200),dtype=torch.float)
    mcnn=MCNN()
    out_dmap=mcnn(img)
    print(out_dmap.shape)