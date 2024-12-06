from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from .pooling import GeneralizedMeanPoolingP
from .vit import vit_base_patch16_224_TransReID
from .mobilenetv2 import mobilenetv2_x1_4


class resnet50(nn.Module):
    def __init__(self, num_classes=0, pretrained=True):
        super(resnet50, self).__init__()
        self.num_classes = num_classes
        
        # resnet50 backbone
        resnet = torchvision.models.resnet50(pretrained=pretrained)

        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, nn.InstanceNorm2d(256),
            resnet.layer2, nn.InstanceNorm2d(512),
            resnet.layer3, nn.InstanceNorm2d(1024),
            resnet.layer4, GeneralizedMeanPoolingP(output_size=(1, 1))
        )

        # pooling
        self.pool = GeneralizedMeanPoolingP(output_size=(1, 1))

        # bnneck and classifier
        self.bn_neck = nn.BatchNorm1d(2048)
        init.constant_(self.bn_neck.weight, 1)
        init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)
        if self.num_classes > 0:
                self.classifier = nn.Linear(2048, self.num_classes, bias=False)

    def forward(self, x):
        x = self.base(x)
        emb = self.pool(x)
        emb = emb.view(x.size(0), -1)
        f = self.bn_neck(emb)
        if self.training:
            logits = self.classifier(f)
            return emb, F.normalize(f), logits
        else:
            return F.normalize(f)


class mobilenetv2(nn.Module):
    def __init__(self, num_classes=0, pretrained=True):
        super(mobilenetv2, self).__init__()
        self.num_classes = num_classes
        
        # mobilenetv2 backbone
        mobilenet = mobilenetv2_x1_4(num_classes=num_classes)

        if pretrained:
            model_path = './checkpoints/mobilenetv2_1.4-bc1cc36b.pth' # your pre-trained weight path here
            pretrain_dict = torch.load(model_path)
            model_dict = mobilenet.state_dict()
            pretrain_dict = {
                k: v
                for k, v in pretrain_dict.items()
                if k in model_dict and model_dict[k].size() == v.size()
            }
            model_dict.update(pretrain_dict)
            mobilenet.load_state_dict(model_dict)
        
        self.base = mobilenet.base

        # pooling
        self.pool = GeneralizedMeanPoolingP(output_size=(1, 1))

        # cnn backbone
        self.bn_neck = nn.BatchNorm1d(1792)
        init.constant_(self.bn_neck.weight, 1)
        init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)
        if self.num_classes > 0:
                self.classifier = nn.Linear(1792, self.num_classes, bias=False)

    def forward(self, x):
        x = self.base(x)
        emb = self.pool(x)
        emb = emb.view(x.size(0), -1)
        f = self.bn_neck(emb)
        if self.training:
            logits = self.classifier(f)
            return emb, F.normalize(f), logits
        else:
            return F.normalize(f)


class vit_base_patch16(nn.Module):
    def __init__(self, num_classes=0, pretrained=True):
        super(vit_base_patch16, self).__init__()
        last_stride = 1
        self.neck = 'bnneck'
        self.neck_feat = 'after'
        self.in_planes = 768
        self.num_classes = num_classes        

        # vit backbone
        self.base = vit_base_patch16_224_TransReID(camera=0, view=0, local_feature=False)
        
        if pretrained:
            model_path = './checkpoints/jx_vit_base_p16_224-80ecf9dd.pth' # your pre-trained weight path here
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # bnneck and classifier
        self.bn_neck = nn.BatchNorm1d(768)
        init.constant_(self.bn_neck.weight, 1)
        init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)
        if self.num_classes > 0:
                self.classifier = nn.Linear(768, self.num_classes, bias=False)

    def forward(self, x):
        cls_token = self.base(x)  # class token
        f = self.bn_neck(cls_token)  # bnneck

        if self.training:
            logits = self.classifier(f)
            return cls_token, torch.nn.functional.normalize(f), logits
        else:
            return torch.nn.functional.normalize(f)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
