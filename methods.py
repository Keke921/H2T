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
    if type(x1).__name__ == 'Tensor':
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

class LearnableWeightScaling(nn.Module):
    def __init__(self, num_classes):
        super(LearnableWeightScaling, self).__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        return self.learned_norm * x
