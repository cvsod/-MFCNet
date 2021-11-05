#!/usr/bin/python3
#coding=utf-8

import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
#         self.initialize()

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

            
#     def initialize(self):
#          weight_init(self)
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
#             print('ggg: ', self.model(input).size())
            return self.model(input)   
            
#     def initialize(self):
#          weight_init(self)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            create_label = True
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            create_label = True
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
#                 print('111: ', pred.size(), target_tensor.cuda().size())
                loss += self.loss(pred, target_tensor.cuda()) ##
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
#             print('222: ', input[-1].size(), target_tensor.cuda().size())
            return self.loss(input[-1], target_tensor.cuda())


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Sigmoid):
            pass
        elif isinstance(m, nn.Tanh):
            pass
        elif isinstance(m, nn.MaxPool2d):
            pass
        elif isinstance(m, nn.Upsample):
            pass
        elif isinstance(m, nn.LeakyReLU):
            pass
        elif isinstance(m, nn.AvgPool2d):
            pass
        else:
            m.initialize()

class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()
        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x, edge_feat):
        
        if x.size()[2:] != edge_feat.size()[2:]: #down是正常的大小
            edge_feat = F.interpolate(edge_feat, size=x.size()[2:], mode='bilinear')
        hx = torch.cat([x, edge_feat], 1)
        hx = self.conv0(hx)
        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)
        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)
        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)
        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)
        hx5 = self.relu5(self.bn5(self.conv5(hx)))
        hx = self.upscore2(hx5)
        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)
        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)
        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)
        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)
        return x + residual
            
    def initialize(self):
         weight_init(self)
        
        
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate,bias=True),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True),	
        )
        self.branch3 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate,bias=True),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True),	
        )
        self.branch4 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate,bias=True),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True),	
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
                nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True),)
        
    def forward(self, x):
        [b,c,row,col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x,2,True)
        global_feature = torch.mean(global_feature,3,True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row,col), None, 'bilinear', True)
        
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result
    
    def initialize(self):
         weight_init(self)
               
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('../res/resnet50-19c8e357.pth'), strict=False)


class ConvGRUCell(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim, kernel_size=3):
        super(ConvGRUCell, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_dim1 + in_dim2, out_dim, 4, 2, 1, bias=False)
        self.reset_gate = nn.Sequential(
            nn.Conv2d(in_dim2+ out_dim , out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Conv2d(in_dim2 + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.hidden = nn.Sequential(
            nn.Conv2d(in_dim2 + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Tanh()
        )

    def forward(self, en_f, old_state, depth_d): #enf sod feat
        if depth_d.size()[2:] != old_state.size()[2:]:
            depth_d = F.interpolate(depth_d, size=old_state.size()[2:], mode='bilinear')
        state_hat = self.upsample(torch.cat([old_state, depth_d], 1)) ## upsample: old_state+depth_d ->out_dim
        r = self.reset_gate(torch.cat([en_f, state_hat], dim=1)) ## reset: en_f+ old_state+depth_d ->out_dim
        z = self.update_gate(torch.cat([en_f, state_hat], dim=1))
        new_state = r * state_hat
        hidden_info = self.hidden(torch.cat([en_f, new_state], dim=1))
        output = (1-z) * state_hat + z * hidden_info
        return output, new_state        
        
    def initialize(self):
        weight_init(self)
        
        
# class ConvGRUCell(nn.Module):
#     def __init__(self,  in_dim1, out_dim, kernel_size=3):
#         super(ConvGRUCell, self).__init__()
# #         self.upsample = nn.ConvTranspose2d(in_dim1, out_dim, 4, 2, 1, bias=False)
#         self.reset_gate = nn.Sequential(
#             nn.Conv2d(in_dim1+ out_dim , out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
#             nn.BatchNorm2d(out_dim),
#             nn.Sigmoid()
#         )
#         self.update_gate = nn.Sequential(
#             nn.Conv2d(in_dim1 + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
#             nn.BatchNorm2d(out_dim),
#             nn.Sigmoid()
#         )
#         self.hidden = nn.Sequential(
#             nn.Conv2d(in_dim1 + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
#             nn.BatchNorm2d(out_dim),
#             nn.Tanh()
#         )

#     def forward(self, old_state, depth_d): #old_state sod feat
#         if depth_d.size()[2:] != old_state.size()[2:]:
#             depth_d = F.interpolate(depth_d, size=old_state.size()[2:], mode='bilinear')
# #         state_hat = self.upsample(depth_d) ## upsample: old_state+depth_d ->out_dim
#         state_hat = depth_d
#         r = self.reset_gate(torch.cat([old_state, state_hat], dim=1)) ## reset: en_f+ old_state+depth_d ->out_dim
#         z = self.update_gate(torch.cat([old_state, state_hat], dim=1))
#         new_state = r * state_hat
#         hidden_info = self.hidden(torch.cat([old_state, new_state], dim=1))
#         output = (1-z) * state_hat + z * hidden_info
#         return output       
        
#     def initialize(self):
#         weight_init(self)        
        
        
class CAF(nn.Module):
    def __init__(self, in_dim1, in_dim2):
        super(CAF, self).__init__()
        self.conv1h = nn.Conv2d(in_dim1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(64)
        
        self.conv11h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn11h   = nn.BatchNorm2d(64)        
        
        self.conv2h = nn.Conv2d(in_dim2, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(64)
        
        self.conv22h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn22h   = nn.BatchNorm2d(64)        
        
        self.conv_se1 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
#         self.bn_se1   = nn.BatchNorm2d(32) 
        self.conv_se2 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
#         self.bn_se2   = nn.BatchNorm2d(64)    
        
        self.m_conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.m_bn1h   = nn.BatchNorm2d(64)
        
#         self.m_conv2h = nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1)
#         self.m_bn2h   = nn.BatchNorm2d(64)
        
#         self.f_conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.f_bn1h   = nn.BatchNorm2d(64)
        
#         self.f_conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.f_bn2h   = nn.BatchNorm2d(64)
        
        self.out_conv = nn.Conv2d(64*4, 64, kernel_size=3, stride=1, padding=1)
        self.out_bn   = nn.BatchNorm2d(64)
        
        self.fin_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fin_bn   = nn.BatchNorm2d(64)        
        
    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]: #down是正常的大小
            left = F.interpolate(left, size=down.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left)), inplace=True)
        out11h = F.relu(self.bn11h(self.conv11h(out1h)), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(down)), inplace=True)
        out22h = F.relu(self.bn22h(self.conv22h(out2h)), inplace=True)
        fuse = out1h * out2h
#         tmp_f = torch.cat([fuse, out1h, out2h], 1)
        tmp = F.adaptive_avg_pool2d(fuse, (1,1))
        tmp = self.conv_se1(tmp)
        tmp = self.conv_se2(tmp)
        tmp = F.sigmoid(tmp)
        tmp = fuse*tmp
        out1 = F.relu(self.m_bn1h(self.m_conv1h(tmp)), inplace=True) #+ out1h
#         out2 = F.relu(self.m_bn2h(self.m_conv2h(tmp)), inplace=True) + out2h
#         out1 = F.relu(self.f_bn1h(self.f_conv1h(out1)), inplace=True) 
#         out2 = F.relu(self.f_bn2h(self.f_conv2h(out2)), inplace=True) 
        fuse = torch.cat([out1+out11h+out22h, out1, out11h, out22h], 1)
        res =  F.relu(self.out_bn(self.out_conv(fuse)), inplace=True)
        res =  F.relu(self.fin_bn(self.fin_conv(res)), inplace=True)
        return res
    
    def initialize(self):
         weight_init(self)

class CAF_3(nn.Module):
    def __init__(self, in_dim1, in_dim2, in_dim3):
        super(CAF_3, self).__init__()
        self.conv1h = nn.Conv2d(in_dim1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(64)

        self.conv11h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn11h   = nn.BatchNorm2d(64)        
        
        self.conv2h = nn.Conv2d(in_dim2, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(64)

        self.conv22h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn22h   = nn.BatchNorm2d(64)        
        
#         self.conv3h = nn.Conv2d(in_dim3, 64, kernel_size=3, stride=1, padding=1)
#         self.bn3h   = nn.BatchNorm2d(64)

#         self.conv33h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn33h   = nn.BatchNorm2d(64)        
        
        self.conv_se1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv_se2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
#         self.m_1_conv = nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1)
#         self.m_1_bn   = nn.BatchNorm2d(64)
        
        self.m_conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.m_bn1h   = nn.BatchNorm2d(64)
        
#         self.m_conv2h = nn.Conv2d(64*4, 64, kernel_size=3, stride=1, padding=1)
#         self.m_bn2h   = nn.BatchNorm2d(64)
        
#         self.m_conv3h = nn.Conv2d(64*4, 64, kernel_size=3, stride=1, padding=1)
#         self.m_bn3h   = nn.BatchNorm2d(64)
        
#         self.f_conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.f_bn1h   = nn.BatchNorm2d(64)
        
#         self.f_conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.f_bn2h   = nn.BatchNorm2d(64)
        
#         self.f_conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.f_bn3h   = nn.BatchNorm2d(64)
        
        self.out_conv = nn.Conv2d(64*4, 64, kernel_size=3, stride=1, padding=1)
        self.out_bn   = nn.BatchNorm2d(64)
        
        self.fin_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fin_bn   = nn.BatchNorm2d(64)
        
    def forward(self, left, down, edge):
        if down.size()[2:] != left.size()[2:]: #down是正常的大小
            left = F.interpolate(left, size=down.size()[2:], mode='bilinear')
        if down.size()[2:] != edge.size()[2:]:
            edge = F.interpolate(edge, size=down.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left)), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(down)), inplace=True)
#         out3h = F.relu(self.bn3h(self.conv3h(edge)), inplace=True)

        out11h = F.relu(self.bn11h(self.conv11h(out1h)), inplace=True)
        out22h = F.relu(self.bn22h(self.conv22h(out2h)), inplace=True)
#         out33h = F.relu(self.bn33h(self.conv33h(out3h)), inplace=True)      
        fuse = out1h*out2h#*out33h
#         tmp_f = torch.cat([fuse,out1h, out2h, out3h], 1)
#         tmp_f = F.relu(self.m_1_bn(self.m_1_conv(tmp_f)), inplace=True)
        tmp = F.adaptive_avg_pool2d(fuse, (1,1))
        tmp = self.conv_se1(tmp)
        tmp = self.conv_se2(tmp)
        tmp = F.sigmoid(tmp)
        tmp = fuse*tmp
        out1 = F.relu(self.m_bn1h(self.m_conv1h(tmp)), inplace=True) #+ out1h
#         out2 = F.relu(self.m_bn2h(self.m_conv2h(tmp)), inplace=True) + out2h
#         out3 = F.relu(self.m_bn3h(self.m_conv3h(tmp)), inplace=True) + out3h
#         out1 = F.relu(self.f_bn1h(self.f_conv1h(out1)), inplace=True) 
#         out2 = F.relu(self.f_bn2h(self.f_conv2h(out2)), inplace=True) 
#         out3 = F.relu(self.f_bn3h(self.f_conv3h(out3)), inplace=True) 
        fuse = torch.cat([out11h+out22h+out1,out11h, out22h, out1], 1)
        res =  F.relu(self.out_bn(self.out_conv(fuse)), inplace=True) + edge
        res = F.relu(self.fin_bn(self.fin_conv(res)), inplace=True)
        return res
    
    def initialize(self):
         weight_init(self)            
        
class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v   = nn.BatchNorm2d(64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left )), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse  = out2h*out2v
        out3h = F.relu(self.bn3h(self.conv3h(fuse )), inplace=True)+out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse )), inplace=True)+out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        return out4h, out4v

    def initialize(self):
        weight_init(self)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cfm45  = CFM()
        self.cfm34  = CFM()
        self.cfm23  = CFM()

    def forward(self, out2h, out3h, out4h, out5v, fback=None):
        if fback is not None:
            refine5      = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4      = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3      = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2      = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            out5v        = out5v+refine5
            out4h, out4v = self.cfm45(out4h+refine4, out5v)
            out3h, out3v = self.cfm34(out3h+refine3, out4v)
            out2h, pred  = self.cfm23(out2h+refine2, out3v)
        else:
            out4h, out4v = self.cfm45(out4h, out5v)
            out3h, out3v = self.cfm34(out3h, out4v)
            out2h, pred  = self.cfm23(out2h, out3v)
        return out2h, out3h, out4h, out5v, pred

    def initialize(self):
        weight_init(self)

        
        
class F3Net_depth(nn.Module):
    def __init__(self, cfg):
        super(F3Net_depth, self).__init__()
        self.cfg      = cfg
        self.bkbone   = ResNet()
        self.aspp = ASPP(dim_in=2048, dim_out=256, rate=1,bn_mom = 0.0003)
        ##
        self.depth_caf1 = CAF(256,1024)
        self.depth_caf2 = CAF(64,512)
        self.depth_caf3 = CAF(64, 256)
        self.depth_out = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        self.m_gru1 = ConvGRUCell(256, 64, 64)
        self.m_gru2 = ConvGRUCell(64, 64, 64)
        self.m_gru3 = ConvGRUCell(64, 64, 64)
        
        self.edge_caf = CAF(256, 256) ##edge feat
        self.linearr6 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        self.sod_caf1 = CAF_3(256,1024, 64)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.sod_caf2 = CAF_3(64,512, 64)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.sod_caf3 = CAF_3(64, 256, 64)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.sod_caf4 = CAF_3(64, 64, 64)
        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
#         self.refunet = RefUnet(2,64)
        
        self.initialize()

    def forward(self, x, shape=None):
        shape = x.size()[2:] if shape is None else shape
        out1h, out2h, out3h, out4h, out5v = self.bkbone(x)
        out_aspp = self.aspp(out5v)
#         print('xxx: ', out_aspp.size(), out2h.size())
        edge_feat = self.edge_caf(out_aspp, out2h)

        depth_f1 = self.depth_caf1(out_aspp ,out4h)
        depth_f2 = self.depth_caf2(depth_f1, out3h)
        depth_f3 = self.depth_caf3(depth_f2, out2h)
        depth_pred = F.interpolate(self.depth_out(depth_f3), size=shape, mode='bilinear')

        sod_f1 = self.sod_caf1(out_aspp ,out4h, edge_feat)
        sod_pred_4 = F.interpolate(self.linearr5(sod_f1), size=shape, mode='bilinear')

        m_f1, new_state = self.m_gru1(sod_f1, out_aspp, depth_f1)

        sod_f2 = self.sod_caf2(m_f1,out3h, edge_feat)
        sod_pred_3 = F.interpolate(self.linearr4(sod_f2), size=shape, mode='bilinear')
        m_f2, new_state = self.m_gru2(sod_f2,new_state, depth_f2)

        sod_f3 = self.sod_caf3(m_f2, out2h, edge_feat)
        
        edge = F.interpolate(self.linearr6(edge_feat), size=shape, mode='bilinear')       #3  
        sod_pred_2 = F.interpolate(self.linearr3(sod_f3), size=shape, mode='bilinear') #2
        m_f3, _ = self.m_gru3(sod_f3, new_state, depth_f3)

        sod_f4 = self.sod_caf4(m_f3, out1h, edge_feat)
        sod_pred_1 = self.linearr2(sod_f4)
        sod_pred_1 = F.interpolate(sod_pred_1, size = shape, mode='bilinear') #test

#         sod_pred = self.refunet(sod_pred_1, edge)

#         sod_pred = F.interpolate(sod_pred, size=shape, mode='bilinear')
       
        return depth_f3, depth_pred, edge, sod_pred_1, sod_pred_2, sod_pred_3, sod_pred_4
#         return sod_f1, m_f1, sod_f2, m_f2, sod_f3, m_f3

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)        

class F3Net(nn.Module):
    def __init__(self, cfg):
        super(F3Net, self).__init__()
        self.cfg      = cfg
        self.bkbone   = ResNet()
        self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d( 512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d( 256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearp2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def forward(self, x, shape=None):
        out2h, out3h, out4h, out5v        = self.bkbone(x)
        out2h, out3h, out4h, out5v        = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(out5v)
        out2h, out3h, out4h, out5v, pred1 = self.decoder1(out2h, out3h, out4h, out5v)
        out2h, out3h, out4h, out5v, pred2 = self.decoder2(out2h, out3h, out4h, out5v, pred1)

        shape = x.size()[2:] if shape is None else shape
        pred1 = F.interpolate(self.linearp1(pred1), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.linearp2(pred2), size=shape, mode='bilinear')

        out2h = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear')
        out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear')
        out4h = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear')
        out5h = F.interpolate(self.linearr5(out5v), size=shape, mode='bilinear')
        return pred1, pred2, out2h, out3h, out4h, out5h


    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
