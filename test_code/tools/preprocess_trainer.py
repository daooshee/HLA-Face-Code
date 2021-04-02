import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from data.zerodce_image import ImageLoader

import math
import numpy as np
from IPython import embed

class PreDataloader():
    def __init__(self, batch_size, num_workers, trg_file):
        train_dataset = ImageLoader(trg_file)
        self.loader = data.DataLoader(train_dataset, batch_size,
                                   num_workers=num_workers,
                                   shuffle=True,
                                   pin_memory=True)
        self.loader_iterator = iter(self.loader)

    def __call__(self, use_cuda):
        try:
            images = next(self.loader_iterator)
        except:
            self.loader_iterator = iter(self.loader)
            images = next(self.loader_iterator)
        if use_cuda:
            images = images.cuda()
        return images


class ZeroDCE_L_color(nn.Module):
    def __init__(self):
        super(ZeroDCE_L_color, self).__init__()

    def forward(self, x, eps=1e-8):
        b,c,h,w = x.shape
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg, 2)
        Drb = torch.pow(mr-mb, 2)
        Dgb = torch.pow(mb-mg, 2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2) + eps, 0.5)
        return k


class ZeroDCE_L_spa(nn.Module):
    def __init__(self):
        super(ZeroDCE_L_spa, self).__init__()
        kernel_left  = torch.FloatTensor([[0,0,0],[-1,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0,0,0],[0,1,-1],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_up    = torch.FloatTensor([[0,-1,0],[0,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_down  = torch.FloatTensor([[0,0,0],[0,1,0],[0,-1,0]]).unsqueeze(0).unsqueeze(0)
        
        self.weight_left = kernel_left
        self.weight_right = kernel_right
        self.weight_up = kernel_up
        self.weight_down = kernel_down
        self.pool = nn.AvgPool2d(4)
    
    def forward(self, org, enhance):
        b,c,h,w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool =  self.pool(org_mean)     
        enhance_pool = self.pool(enhance_mean)  

        if org_pool.is_cuda:
            weight_diff = torch.max(torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
            E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool-org_pool)
        else:
            weight_diff = torch.max(torch.FloatTensor([1]) + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]), torch.FloatTensor([0])), torch.FloatTensor([0.5]))
            E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5])), enhance_pool-org_pool)

        if org_pool.is_cuda:
            D_org_letf = F.conv2d(org_pool, self.weight_left.cuda(), padding=1)
            D_org_right = F.conv2d(org_pool, self.weight_right.cuda(), padding=1)
            D_org_up = F.conv2d(org_pool, self.weight_up.cuda(), padding=1)
            D_org_down = F.conv2d(org_pool, self.weight_down.cuda(), padding=1)
            D_enhance_letf = F.conv2d(enhance_pool, self.weight_left.cuda(), padding=1)
            D_enhance_right = F.conv2d(enhance_pool, self.weight_right.cuda(), padding=1)
            D_enhance_up = F.conv2d(enhance_pool, self.weight_up.cuda(), padding=1)
            D_enhance_down = F.conv2d(enhance_pool, self.weight_down.cuda(), padding=1)
        else:
            D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
            D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
            D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
            D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)
            D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
            D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
            D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
            D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down).mean()

        if torch.isinf(E):
            print('ZeroDCE_L_spa is inf')
            embed()

        return E


class ZeroDCE_L_exp(nn.Module):

    def __init__(self, patch_size, mean_val):
        super(ZeroDCE_L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        if x.is_cuda:
            d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        else:
            d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]), 2))
        return d


class ZeroDCE_L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(ZeroDCE_L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        assert batch_size > 0
        assert count_h > 0
        assert count_w > 0
        h_tv = torch.pow((x[:,:,1:,:] - x[:,:,:h_x-1,:]), 2).sum()
        w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:,:w_x-1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

