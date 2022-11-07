import torch
import torch.nn as nn

from mmdet.models import ResNet, FPN, MobileNetV2

import torch.nn.functional as F

from common import default_conv, ResBlock, BasicBlock

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

# parser.add_argument('--act', type=str, default='relu',
#                     help='activation function')
# parser.add_argument('--pre_train', type=str, default='',
#                     help='pre-trained model directory')
# parser.add_argument('--extend', type=str, default='.',
#                     help='pre-trained model directory')
# parser.add_argument('--n_resblocks', type=int, default=16,
#                     help='number of residual blocks')
# parser.add_argument('--n_feats', type=int, default=64,
#                     help='number of feature maps')
# parser.add_argument('--res_scale', type=float, default=1,
#                     help='residual scaling')
# parser.add_argument('--shift_mean', default=True,
#                     help='subtract pixel mean from the input')
# parser.add_argument('--dilation', action='store_true',
#                     help='use dilated convolution')
# parser.add_argument('--precision', type=str, default='single',
#                     choices=('single', 'half'),
#                     help='FP precision for test (single | half)')

# https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/model/edsr.py

class EDSR(nn.Module): # not converge
    def __init__(self, conv=default_conv):
        super(EDSR, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3 
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            conv(n_feats, 1, kernel_size)
        ]

       
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        

    def forward(self, x):

        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x 

    # def load_state_dict(self, state_dict, strict=True):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             if isinstance(param, nn.Parameter):
    #                 param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 if name.find('tail') == -1:
    #                     raise RuntimeError('While copying the parameter named {}, '
    #                                        'whose dimensions in the model are {} and '
    #                                        'whose dimensions in the checkpoint are {}.'
    #                                        .format(name, own_state[name].size(), param.size()))
    #         elif strict:
    #             if name.find('tail') == -1:
    #                 raise KeyError('unexpected key "{}" in state_dict'
    #                                .format(name))


class VDSR(nn.Module):
    def __init__(self, conv=default_conv):
        super(VDSR, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3 
       
        def basic_block(in_channels, out_channels, act):
            return BasicBlock(
                conv, in_channels, out_channels, kernel_size,
                bias=True, bn=True, act=act
            )

        # define body module
        m_body = []
        m_body.append(basic_block(3, n_feats, nn.ReLU(True)))
        for _ in range(n_resblocks - 2):
            m_body.append(basic_block(n_feats, n_feats, nn.ReLU(True)))
        m_body.append(basic_block(n_feats, 1, nn.ReLU(True)))

        self.body = nn.Sequential(*m_body)

    def forward(self, x):

        res = self.body(x)


        return res


# test code
if __name__=="__main__":
    img=torch.rand((1,3,1332,1332),dtype=torch.float)
    mcnn=mobilenetv2_fpn()
    for m in mcnn.modules():
        print(m)
    #out_dmap=mcnn(img)
    #print(out_dmap.shape)