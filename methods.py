import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, extra_info=None):
    return lam * criterion(pred, y_a, extra_info) + (1 - lam) * criterion(pred, y_b, extra_info)


def H2T(x1, x2, rho = 0.3):          
    if len(x1.shape) == 4:
        fea_num = x1.shape[1]
        index = torch.randperm(fea_num).cuda()
        slt_num = int(rho*fea_num)
        index = index[:slt_num]
        x1[:,index,:,:] = x2[:,index,:,:] 
                          #torch.rand(x2[:,index,:,:].shape).to(x2.device) #torch.zeros(x2[:,index,:,:].shape).to(x2.device) 
                          #x2[:,index,:,:]
    else:
        for i in range(len(x1)):
            fea_num = x1[i].shape[1]
            index = torch.randperm(fea_num).cuda()
            slt_num = int(rho*fea_num)
            index = index[:slt_num]
            x1[i][:,index,:,:] = x2[i][:,index,:,:]    
    return x1


class H2Tv2(nn.Module):
    def __init__(self, beta = 0.2, cls_num_list=None, use_cuda=True):
        super().__init__()
        self.beta = None
        self.cls_num_list = cls_num_list
        self.use_cuda = use_cuda
        
    def forward(self, x1, x2, dis, y1, y2): 

        dis = 1/(1+dis) 
        p = (dis-0.5)/0.5*0.4+0.6
        
        one_hot1 = F.one_hot(y1,p.size()[1]).float().to(y1.device)                 
        p1 = (p*one_hot1).sum(dim=1)
        p2 = 1-p1

        x = torch.mul(x1, p1.unsqueeze(1)) + torch.mul(x2, p2.unsqueeze(1)) 
        return x


class LearnableWeightScaling(nn.Module):
    def __init__(self, num_classes):
        super(LearnableWeightScaling, self).__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        return self.learned_norm * x
