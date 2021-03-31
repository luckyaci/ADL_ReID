from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from getpass import getuser
import torch
from torch import nn
import torch.nn.functional as F
import random

from frameworks.models.backbone import ResNet_Backbone
from frameworks.models.weight_utils import weights_init_kaiming
from frameworks.training.smooth import SmoothingForImage 

import torchvision

fea_dims_small = {'layer2': 128, 'layer3': 256, 'layer4': 512}
fea_dims = {'layer2': 512, 'layer3': 1024, 'layer4': 2048}


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ResNetBuilder(nn.Module):
    """ __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }
    in_planes = 2048

    def __init__(self, num_pids=None, last_stride=1):
        super(ResNetBuilder, self).__init__()
        depth = 152

        final_layer = 'layer4'
        self.final_layer = final_layer

        pretrained = True

        # Construct base (pretrained) resnet
        if depth not in ResNetBuilder.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNetBuilder.__factory[depth](pretrained=pretrained)
        if depth < 50:
            out_planes = fea_dims_small[final_layer]
        else:
            out_planes = fea_dims[final_layer]

        i = 0
        for module in self.base.modules():
            #print(module)
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.momentum = None
                #print(module.momentum)
                i+=1
        print(i) """

    def __init__(self, num_pids=None, last_stride=1):
        super().__init__() 
        self.num_pids = num_pids
        print(num_pids)
        self.base = ResNet_Backbone(last_stride)
        model_path = '/home/' + getuser() + '/.torch/models/resnet50-19c8e357.pth'
        self.base.load_param(model_path)
        bn_neck = nn.BatchNorm1d(2048, momentum=None)
        bn_neck.bias.requires_grad_(False)
        self.bottleneck = nn.Sequential(bn_neck)
        self.bottleneck.apply(weights_init_kaiming)

        
        embedding_size = 2048
        self.RFM = RFM(embedding_size)
        if self.num_pids is not None:
            self.classifier = nn.Linear(2048, self.num_pids, bias=False)
            self.classifier.apply(weights_init_classifier)

            #camera embedding and classfier
            
            self.DAL = DAL_regularizer(embedding_size)


            if self.num_pids == 4101:
                cam_num = 15
            elif self.num_pids == 751:
                cam_num = 6
            else:
                cam_num = 8
            self.cam_classifier = nn.Sequential(nn.Linear(embedding_size, embedding_size, bias=False) \
                                                , nn.BatchNorm1d(embedding_size)
                                                , nn.ReLU(inplace=True)
                                                ,nn.Linear(embedding_size, cam_num, bias=False))
            self.cam_classifier.apply(weights_init_classifier)


            #momentum updating
            self.agent = SmoothingForImage(momentum=0.9, num=1)            
            
    def forward(self, x, path=None):

        x = self.base(x)
        feat_before_bn = x

        feat_before_bn = F.avg_pool2d(feat_before_bn, feat_before_bn.shape[2:])
        feat_before_bn = feat_before_bn.view(feat_before_bn.shape[0], -1)
        
        detach_fea = feat_before_bn.detach()

        if path:
            detach_fea = self.agent.get_soft_label(path, detach_fea)

        detach_fea.requires_grad_()
        detach_fea = feat_before_bn

        latent_code = self.RFM(detach_fea)
        pzx = torch.cat((feat_before_bn, latent_code), 1 )
        random_index = random.sample(range(0,latent_code.size(0)),latent_code.size(0))

        random_code = latent_code[random_index]
        pzpx = torch.cat((feat_before_bn, random_code), 1 )

        
        feat_after_bn = self.bottleneck(feat_before_bn)



        if self.num_pids is not None:

            classification_results = self.classifier(feat_after_bn)
            #classification_results = self.classifier(latent_code)
            pzx_scores, pzpx_scores = self.DAL(pzx, pzpx)

            cam_score = self.cam_classifier(latent_code)
            #cam_score = self.cam_classifier(feat_after_bn)




            return feat_after_bn, classification_results, cam_score, pzx_scores, pzpx_scores
        else:
            return latent_code

    def get_optim_policy(self):
        base_param_group = filter(lambda p: p.requires_grad, self.base.parameters())
        add_param_group = filter(lambda p: p.requires_grad, self.bottleneck.parameters())
        cls_param_group = filter(lambda p: p.requires_grad, self.classifier.parameters())
        RFM_param_group = filter(lambda p: p.requires_grad, self.RFM.parameters())
        DAL_param_group = filter(lambda p: p.requires_grad, self.DAL.parameters())
        cam_cls_param_group = filter(lambda p: p.requires_grad, self.cam_classifier.parameters())

        all_param_groups = []
        all_param_groups.append({'params': base_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': add_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': cls_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': RFM_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': DAL_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': cam_cls_param_group, "weight_decay": 0.0005})        
        return all_param_groups

#camera clssfier
class RFM(nn.Module):
    '''
    Camera feature extrctor "E" in paper.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.seq = nn.Sequential(nn.BatchNorm1d(n_in)
                                , nn.Linear(n_in, n_in)
                                , nn.BatchNorm1d(n_in)
                                , nn.ReLU(inplace=True)
                                , nn.Linear(n_in, n_in)
                                , nn.BatchNorm1d(n_in)
                                , nn.ReLU(inplace=True))
    
    def forward(self, xs):
        return self.seq(xs)

class DAL_regularizer(nn.Module):
    '''
    Disentangled Feature Learning module in paper.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.discrimintor = nn.Sequential(nn.Linear(n_in*2, n_in)
                                         , nn.ReLU(inplace=True)
                                         , nn.Linear(n_in, n_in)
                                         , nn.ReLU(inplace=True)
                                         , nn.Linear(n_in, 1)
                                         , nn.Sigmoid())
        
    
    def forward(self, pzx, pzpx):
        pzx_scores = self.discrimintor(pzx)
        pzpx_scores = self.discrimintor(pzpx)

        return pzx_scores, pzpx_scores



def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}