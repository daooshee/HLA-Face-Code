import math
import numpy as np
from IPython import embed

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhanceNet(nn.Module):
    def __init__(self):
        super(EnhanceNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        number_f = 64
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f,48,3,1,1,bias=True) 

    def forward(self, x_ori):
        x1 = self.relu(self.e_conv1(x_ori))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(x3+x4))
        x6 = self.relu(self.e_conv6(x2+x5))
        x_r = F.tanh(self.e_conv7(x1+x6))
        r1,r2,r3,r4,r5,r6,r7,r8, \
            r9,r10,r11,r12,r13,r14,r15,r16 = torch.split(x_r, 3, dim=1)

        x = x_ori + r1*(torch.pow(x_ori,2)-x_ori)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        x = x + r4*(torch.pow(x,2)-x)       
        x = x + r5*(torch.pow(x,2)-x)       
        x = x + r6*(torch.pow(x,2)-x)   
        x = x + r7*(torch.pow(x,2)-x)
        x = x + r8*(torch.pow(x,2)-x)
        x = x + r9*(torch.pow(x,2)-x)
        x = x + r10*(torch.pow(x,2)-x)
        x = x + r11*(torch.pow(x,2)-x)
        x = x + r12*(torch.pow(x,2)-x)      
        x = x + r13*(torch.pow(x,2)-x)      
        x = x + r14*(torch.pow(x,2)-x)  
        x = x + r15*(torch.pow(x,2)-x)
        enhance_image = x + r16*(torch.pow(x,2)-x)
        return enhance_image