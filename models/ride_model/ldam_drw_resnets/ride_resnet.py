"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

#from utils import autocast


__all__ = ['Classifier', 'ResClassifier', 'ResNet', 'resnet50']
model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_experts=3, dropout=None, num_classes=1000, use_norm=False, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, share_layer3=False):
        self.inplanes = 64
        self.num_experts = num_experts
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes = self.next_inplanes

        self.share_layer3 = share_layer3

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 192
            else:
                layer3_output_dim = 256

        if layer4_output_dim is None:
            if reduce_dimension:
                layer4_output_dim = 384
            else:
                layer4_output_dim = 512

        if self.share_layer3:
            self.layer3 = self._make_layer(block, layer3_output_dim, layers[2], stride=2)
        else:
            self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, layers[2], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.layer4s = nn.ModuleList([self._make_layer(block, layer4_output_dim, layers[3], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.avgpool = nn.AvgPool2d(7, stride=1)        
        self.use_dropout = True if dropout else False
        
        
        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)


        ##### ResLT implementation 
        def ResLTBlock(): 
            return nn.Sequential(
                nn.Conv2d(self.next_inplanes, self.next_inplanes * 3, 1, bias=False),
                nn.BatchNorm2d(self.next_inplanes * 3),
                nn.ReLU(inplace=True)  ) 
        self.ResLTBlocks = nn.ModuleList([ResLTBlock() for _ in range(num_experts)])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.next_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.next_inplanes, planes))

        return nn.Sequential(*layers)


    def _separate_part(self, x, ind):
        if not self.share_layer3:
            x = (self.layer3s[ind])(x)
        x = (self.layer4s[ind])(x)

        if self.training:
           x = (self.ResLTBlocks[ind])(x)
           x = self.avgpool(x)
        else:
           x = self.avgpool(x)
           x = (self.ResLTBlocks[ind])(x)
        
        #if self.use_dropout:
        #    out = self.dropout(out)
        return x

    def forward(self, x):
        #with autocast():
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if self.share_layer3:
            x = self.layer3(x)

        outs = []
        for ind in range(self.num_experts):
            outs.append(self._separate_part(x, ind))
        return outs

class Classifier(nn.Module):
    def __init__(self, config, feat_in, num_classes, use_norm=False, s=30, num_experts=3):
        super(Classifier, self).__init__()
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if use_norm:
            self.linears = nn.ModuleList([NormedLinear(feat_in, num_classes) for _ in range(num_experts)])
        else:
            self.linears = nn.ModuleList([nn.Linear(feat_in, num_classes) for _ in range(num_experts)])
            s = 1
        self.s = s


    def forward(self, x):
        self.feat = x
        outs = []
        self.logits = outs
        for ind in range(len(x)):
            out = self.avgpool(x[ind])
            out = torch.flatten(out, 1) 
            if self.use_dropout:
                out = self.dropout(out)
            out = (self.linears[ind])(out)
            outs.append(out * self.s) 
        final_out = torch.stack(outs, dim=1).mean(dim=1)
        return final_out

class ResClassifier(nn.Module):
      def __init__(self, config, feat_in, num_classes=10, use_norm=False, dropout=None, scale=1, num_experts=3, s=30):
          super(ResClassifier, self).__init__()
          #nc=[16, 32, 64]
          #nc=[c * scale for c in nc]
          self.num_experts = num_experts
          self.logits=[]
          self.gamma = config.ResLT.gamma
          
          if use_norm: 
              self.linears = nn.ModuleList([NormedLinear(feat_in, num_classes) for _ in range(num_experts)])
          else:
              self.linears = nn.ModuleList([nn.Linear(feat_in, num_classes) for _ in range(num_experts)])
              s = 1
          self.s = s            

          self.linear = nn.ModuleList([nn.Linear(feat_in, num_classes) for _ in range(num_experts)]) #nc[2]
          #self.model = ResNet_Cifar(BasicBlock, [5, 5, 5], num_classes=num_classes, nc=nc)
          self.use_dropout = dropout
          self.use_dropout = True if dropout else False

          if self.use_dropout: 
              print('Using dropout.') 
              self.dropout = nn.Dropout(p=dropout)
          
          self.avgpool = nn.AdaptiveAvgPool2d((1, 1))    
          #self.feat = []
      def forward(self, x):
          #out = self.model(x)
          outs = []       
          self.logits = outs
          for ind in range(len(x)):            
              out = torch.flatten(x[ind], 1)
              c = out.size(1) // 3 
              #bt = out.size(0)
              x1, x2, x3 = out[:,:c], out[:,c:c*2], out[:,c*2:c*3]  
              out = torch.cat((x1, x2, x3),dim=0) 
              #out = self.avgpool(out).view(out.size(0),-1)    
              
              if self.use_dropout:
                  out = self.dropout(out)
              if self.training:
                  out = self.linears[ind](out)
              else:
                  weight = self.linears[ind].weight
                  norm = torch.norm(weight, 2, 1, keepdim=True)
                  weight = weight / torch.pow(norm, self.gamma)
                  out = torch.mm(out, torch.t(weight)) 
              out = out * self.s
              outs.append(out)
          self.logits = outs  
          logits= torch.stack(outs, dim=1).mean(dim=1)
          c = logits.size(0) // 3
          #self.feat = torch.stack(self.feat, dim=1)         
          final_logits = [logits[:c,:], logits[c:c*2,:], logits[c*2:,:]]
          return final_logits
  

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)