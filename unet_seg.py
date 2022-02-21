import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
class Double_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Double_Conv2d, self).__init__()
        self.double_conv2d = torch.nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv2d(x)

#for normal unet attention
class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer (in upsampling)
        :param skip_connection: activation from corresponding encoder layer (on downsampling side)
        :return: output activations
        """
        g1 = self.W_gate(gate) 
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1) #two vectors,g1 and x1 are summed element-wise. --> aligned weights becoming larger while unaligned weights become relatively smaller #resultant vector goes through a ReLU
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

#for coordinate-attention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
        
#main block for coordinate-attention
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
	
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        #a_h = a_h.expand(-1, -1, h, w)
        #a_w = a_w.expand(-1, -1, h, w)

        out = identity * a_w * a_h

        return out

#for Triple Attention
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out
        
#for CBAM
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
        
#for StripPooling
class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(SPBlock, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1),  padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu =  nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = F.interpolate(x1,(h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = F.interpolate(x2, (h, w))

        x3 = self.relu(x1 + x2)
        x3 = x * torch.sigmoid(self.conv3(x3))
        return x3
class seg_UNet(nn.Module):
    def __init__(self):
        super(seg_UNet, self).__init__()
        self.conv1 = Double_Conv2d(3, 32)
        self.conv2 = Double_Conv2d(32, 64)
        self.conv3 = Double_Conv2d(64, 128)
        self.conv4 = Double_Conv2d(128, 256)
        self.conv5 = Double_Conv2d(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = Double_Conv2d(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = Double_Conv2d(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = Double_Conv2d(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = Double_Conv2d(64, 32)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)#enhancement head
        self.seg_out= nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)#segmentation head

    def forward(self, x):
        conv1 = self.conv1(x)
        # print(conv1.shape)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        # print(conv2.shape)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        # print(conv3.shape)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        # print(conv4.shape)
        pool4 = F.max_pool2d(conv4, kernel_size=2)

        conv5 = self.conv5(pool4)
        # print(conv5.shape)

        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)
        # print(conv6.shape)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)
        # print(conv7.shape)
        
        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)
        # print(conv8.shape)

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)
        # print(conv9.shape)

        conv10 = self.conv10(conv9)
        # print(conv10.shape)
        # out = F.pixel_shuffle(conv10, 2)
        enh_out = F.pixel_shuffle(conv10, 1)
        seg_out = self.seg_out(conv9)

        return enh_out,seg_out

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             m.weight.data.normal_(0.0, 0.02)
    #             if m.bias is not None:
    #                 m.bias.data.normal_(0.0, 0.02)
    #         if isinstance(m, nn.ConvTranspose2d):
    #             m.weight.data.normal_(0.0, 0.02)

class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()
        self.conv1 = Double_Conv2d(3, 32)
        self.conv2 = Double_Conv2d(32, 64)
        self.conv3 = Double_Conv2d(64, 128)
        self.conv4 = Double_Conv2d(128, 256)
        self.conv5 = Double_Conv2d(256, 512)
        
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.Att5 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.conv6 = Double_Conv2d(512, 256)
        
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.Att4 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.conv7 = Double_Conv2d(256, 128)
        
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.Att3 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.conv8 = Double_Conv2d(128, 64)
        
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.Att2 = AttentionBlock(F_g=32, F_l=32, n_coefficients=16)
        self.conv9 = Double_Conv2d(64, 32)
        
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)#e1
        pool1 = F.max_pool2d(conv1, kernel_size=2)#e2

        conv2 = self.conv2(pool1)#e2
        pool2 = F.max_pool2d(conv2, kernel_size=2)#e3

        conv3 = self.conv3(pool2)#e3
        pool3 = F.max_pool2d(conv3, kernel_size=2)#e4

        conv4 = self.conv4(pool3)#e4
        pool4 = F.max_pool2d(conv4, kernel_size=2)#e5

        conv5 = self.conv5(pool4)#e5
        
        up6 = self.up6(conv5)# d5 = self.Up5(e5)
        
        s4 = self.Att5(gate=up6, skip_connection=conv4)
        up6 = torch.cat([s4, up6], 1)# d5 # concatenate attention-weighted skip connection with previous layer output
        conv6 = self.conv6(up6)# d5

        up7 = self.up7(conv6)# d4
        
        s3 = self.Att4(gate=up7 , skip_connection=conv3)
        up7 = torch.cat([s3, up7], 1)# d4
        conv7 = self.conv7(up7)# d4
        
        up8 = self.up8(conv7)# d3
        
        s2 = self.Att3(gate=up8, skip_connection=conv2)        
        up8 = torch.cat([s2, up8], 1)# d3
        conv8 = self.conv8(up8)# d3

        up9 = self.up9(conv8)# d2
        
        s1 = self.Att2(gate=up9, skip_connection=conv1)
        up9 = torch.cat([s1, up9], 1)# d2
        conv9 = self.conv9(up9)# d2

        conv10 = self.conv10(conv9)
        out = F.pixel_shuffle(conv10, 1)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

        
class CA_seg_UNet(nn.Module):
    def __init__(self):
        super(CA_seg_UNet, self).__init__()
        self.conv1 = Double_Conv2d(3, 32)
        self.conv2 = Double_Conv2d(32, 64)
        self.conv3 = Double_Conv2d(64, 128)
        self.conv4 = Double_Conv2d(128, 256)
        self.conv5 = Double_Conv2d(256, 512)
        
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.CA5 =CoordAtt(512,512)
        self.conv6 = Double_Conv2d(512, 256)
        
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.CA4 =CoordAtt(256,256)
        self.conv7 = Double_Conv2d(256, 128)
        
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.CA3 =CoordAtt(128,128)
        self.conv8 = Double_Conv2d(128, 64)
        
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.CA2 =CoordAtt(64,64)
        self.conv9 = Double_Conv2d(64, 32)

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)#enhancement head
        self.seg_out= nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)#segmentation head

    def forward(self, x):
        conv1 = self.conv1(x)
        # print(conv1.shape)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        # print(conv2.shape)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        # print(conv3.shape)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        # print(conv4.shape)
        pool4 = F.max_pool2d(conv4, kernel_size=2)


        conv5 = self.conv5(pool4)
        #print('conv5.shape : ',conv5.shape)# torch.Size([8, 512, 32, 32])

        up6 = self.up6(conv5)
        #print('up6.shape : ',up6.shape)#torch.Size([8, 256, 64, 64])
        up6 = torch.cat([up6, conv4], 1)
        #print('up62.shape : ',up6.shape) #torch.Size([8, 512, 64, 64])
        up6 = self.CA5(up6)
        conv6 = self.conv6(up6)
        # print(conv6.shape)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        up7 = self.CA4(up7)
        conv7 = self.conv7(up7)
        # print(conv7.shape)
        
        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        up8 = self.CA3(up8)
        conv8 = self.conv8(up8)
        # print(conv8.shape)

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        up9 = self.CA2(up9)
        conv9 = self.conv9(up9)
        # print(conv9.shape)
       
        conv10 = self.conv10(conv9)
        # print(conv10.shape)
        # out = F.pixel_shuffle(conv10, 2)
        enh_out = F.pixel_shuffle(conv10, 1)
        seg_out = self.seg_out(conv9)

        return enh_out,seg_out

class gray_TA_CA_seg_UNet(nn.Module):
    def __init__(self):
        super(gray_TA_CA_seg_UNet, self).__init__()
        self.conv1 = Double_Conv2d(3, 32)
        self.conv2 = Double_Conv2d(32, 64)
        self.conv3 = Double_Conv2d(64, 128)
        self.conv4 = Double_Conv2d(128, 256)
        self.conv5 = Double_Conv2d(256, 512)                
        
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.TA5 =TripletAttention()
        self.CA5 =CoordAtt(512,512)
        self.conv6 = Double_Conv2d(512, 256)
        
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.TA4 =TripletAttention()
        self.CA4 =CoordAtt(256,256)
        self.conv7 = Double_Conv2d(256, 128)
        
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.TA3 =TripletAttention()
        self.CA3 =CoordAtt(128,128)
        self.conv8 = Double_Conv2d(128, 64)
        
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.TA2 =TripletAttention()
        self.CA2 =CoordAtt(64,64)
        self.conv9 = Double_Conv2d(64, 32)

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)#enhancement head
        self.seg_out= nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)#segmentation head

    def forward(self, x, gray):
    
        gray2 = F.max_pool2d(gray, kernel_size=2)
        gray3 = F.max_pool2d(gray2, kernel_size=2)
        gray4 = F.max_pool2d(gray3, kernel_size=2)
        gray5 = F.max_pool2d(gray4, kernel_size=2)    
    
        conv1 = self.conv1(x)
        # print(conv1.shape)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        # print(conv2.shape)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        # print(conv3.shape)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        # print(conv4.shape)
        pool4 = F.max_pool2d(conv4, kernel_size=2)


        conv5 = self.conv5(pool4)
        conv5 = conv5 * gray5
        #print('conv5.shape : ',conv5.shape)# torch.Size([8, 512, 32, 32])

        up6 = self.up6(conv5)
        conv4 = conv4 * gray4
        #print('up6.shape : ',up6.shape)#torch.Size([8, 256, 64, 64])
        up6 = torch.cat([up6, conv4], 1)
        #print('up62.shape : ',up6.shape) #torch.Size([8, 512, 64, 64])
        up6 = self.TA5(up6)
        up6 = self.CA5(up6)
        conv6 = self.conv6(up6)
        # print(conv6.shape)

        up7 = self.up7(conv6)
        conv3 = conv3 * gray3
        up7 = torch.cat([up7, conv3], 1)
        up7 = self.TA4(up7)
        up7 = self.CA4(up7)
        conv7 = self.conv7(up7)
        # print(conv7.shape)
        
        up8 = self.up8(conv7)
        conv2 = conv2 * gray2
        up8 = torch.cat([up8, conv2], 1)
        up8 = self.TA3(up8)
        up8 = self.CA3(up8)
        conv8 = self.conv8(up8)
        # print(conv8.shape)

        up9 = self.up9(conv8)
        conv1 = conv1 * gray
        up9 = torch.cat([up9, conv1], 1)
        up9 = self.TA2(up9)
        up9 = self.CA2(up9)
        conv9 = self.conv9(up9)
        # print(conv9.shape)
       
        conv10 = self.conv10(conv9)
        # print(conv10.shape)
        # out = F.pixel_shuffle(conv10, 2)
        enh_out = F.pixel_shuffle(conv10, 1)
        seg_out = self.seg_out(conv9)

        return enh_out,seg_out

class gray_TA_CA_edge_seg_UNet(nn.Module):
    def __init__(self):
        super(gray_TA_CA_edge_seg_UNet, self).__init__()
        self.conv1 = Double_Conv2d(4, 32)
        self.conv2 = Double_Conv2d(32, 64)
        self.conv3 = Double_Conv2d(64, 128)
        self.conv4 = Double_Conv2d(128, 256)
        self.conv5 = Double_Conv2d(256, 512)                
        
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.TA5 =TripletAttention()
        self.CA5 =CoordAtt(512,512)
        self.conv6 = Double_Conv2d(512, 256)
        
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.TA4 =TripletAttention()
        self.CA4 =CoordAtt(256,256)
        self.conv7 = Double_Conv2d(256, 128)
        
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.TA3 =TripletAttention()
        self.CA3 =CoordAtt(128,128)
        self.conv8 = Double_Conv2d(128, 64)
        
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.TA2 =TripletAttention()
        self.CA2 =CoordAtt(64,64)
        self.conv9 = Double_Conv2d(64, 32)

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)#enhancement head
        self.seg_out= nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)#segmentation head

    def forward(self, x, gray, edge):
    
        gray2 = F.max_pool2d(gray, kernel_size=2)
        gray3 = F.max_pool2d(gray2, kernel_size=2)
        gray4 = F.max_pool2d(gray3, kernel_size=2)
        gray5 = F.max_pool2d(gray4, kernel_size=2)  
        
        x = torch.cat([x, edge], 1)  
    
        conv1 = self.conv1(x)
        # print(conv1.shape)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        # print(conv2.shape)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        # print(conv3.shape)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        # print(conv4.shape)
        pool4 = F.max_pool2d(conv4, kernel_size=2)


        conv5 = self.conv5(pool4)
        conv5 = conv5 * gray5
        #print('conv5.shape : ',conv5.shape)# torch.Size([8, 512, 32, 32])

        up6 = self.up6(conv5)
        conv4 = conv4 * gray4
        #print('up6.shape : ',up6.shape)#torch.Size([8, 256, 64, 64])
        up6 = torch.cat([up6, conv4], 1)
        #print('up62.shape : ',up6.shape) #torch.Size([8, 512, 64, 64])
        up6 = self.TA5(up6)
        up6 = self.CA5(up6)
        conv6 = self.conv6(up6)
        # print(conv6.shape)

        up7 = self.up7(conv6)
        conv3 = conv3 * gray3
        up7 = torch.cat([up7, conv3], 1)
        up7 = self.TA4(up7)
        up7 = self.CA4(up7)
        conv7 = self.conv7(up7)
        # print(conv7.shape)
        
        up8 = self.up8(conv7)
        conv2 = conv2 * gray2
        up8 = torch.cat([up8, conv2], 1)
        up8 = self.TA3(up8)
        up8 = self.CA3(up8)
        conv8 = self.conv8(up8)
        # print(conv8.shape)

        up9 = self.up9(conv8)
        conv1 = conv1 * gray
        up9 = torch.cat([up9, conv1], 1)
        up9 = self.TA2(up9)
        up9 = self.CA2(up9)
        conv9 = self.conv9(up9)
        # print(conv9.shape)
       
        conv10 = self.conv10(conv9)
        # print(conv10.shape)
        # out = F.pixel_shuffle(conv10, 2)
        enh_out = F.pixel_shuffle(conv10, 1)
        seg_out = self.seg_out(conv9)

        return enh_out,seg_out
        
class gray_TA_SP_CA_seg_UNet(nn.Module):
    def __init__(self):
        super(gray_TA_SP_CA_seg_UNet, self).__init__()
        self.conv1 = Double_Conv2d(3, 32)
        self.conv2 = Double_Conv2d(32, 64)
        self.conv3 = Double_Conv2d(64, 128)
        self.conv4 = Double_Conv2d(128, 256)
        self.conv5 = Double_Conv2d(256, 512)                
        
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.TA5 =TripletAttention()
        self.SP5 =SPBlock(512,512)
        self.CA5 =CoordAtt(512,512)
        self.conv6 = Double_Conv2d(512, 256)
        
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.TA4 =TripletAttention()
        self.SP4 =SPBlock(256,256)
        self.CA4 =CoordAtt(256,256)
        self.conv7 = Double_Conv2d(256, 128)
        
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.TA3 =TripletAttention()
        self.SP3 =SPBlock(128,128)
        self.CA3 =CoordAtt(128,128)
        self.conv8 = Double_Conv2d(128, 64)
        
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.TA2 =TripletAttention()
        self.SP2 =SPBlock(64,64)
        self.CA2 =CoordAtt(64,64)
        self.conv9 = Double_Conv2d(64, 32)

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)#enhancement head
        self.seg_out= nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)#segmentation head

    def forward(self, x, gray):
    
        gray2 = F.max_pool2d(gray, kernel_size=2)
        gray3 = F.max_pool2d(gray2, kernel_size=2)
        gray4 = F.max_pool2d(gray3, kernel_size=2)
        gray5 = F.max_pool2d(gray4, kernel_size=2)    
    
        conv1 = self.conv1(x)
        # print(conv1.shape)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        # print(conv2.shape)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        # print(conv3.shape)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        # print(conv4.shape)
        pool4 = F.max_pool2d(conv4, kernel_size=2)


        conv5 = self.conv5(pool4)
        conv5 = conv5 * gray5
        #print('conv5.shape : ',conv5.shape)# torch.Size([8, 512, 32, 32])

        up6 = self.up6(conv5)
        conv4 = conv4 * gray4
        #print('up6.shape : ',up6.shape)#torch.Size([8, 256, 64, 64])
        up6 = torch.cat([up6, conv4], 1)
        #print('up62.shape : ',up6.shape) #torch.Size([8, 512, 64, 64])
        up6 = self.TA5(up6)
        up6 = self.SP5(up6)
        up6 = self.CA5(up6)
        conv6 = self.conv6(up6)
        # print(conv6.shape)

        up7 = self.up7(conv6)
        conv3 = conv3 * gray3
        up7 = torch.cat([up7, conv3], 1)
        up7 = self.TA4(up7)
        up7 = self.SP4(up7)
        up7 = self.CA4(up7)
        conv7 = self.conv7(up7)
        # print(conv7.shape)
        
        up8 = self.up8(conv7)
        conv2 = conv2 * gray2
        up8 = torch.cat([up8, conv2], 1)
        up8 = self.TA3(up8)
        up8 = self.SP3(up8)
        up8 = self.CA3(up8)
        conv8 = self.conv8(up8)
        # print(conv8.shape)

        up9 = self.up9(conv8)
        conv1 = conv1 * gray
        up9 = torch.cat([up9, conv1], 1)
        up9 = self.TA2(up9)
        up9 = self.SP2(up9)
        up9 = self.CA2(up9)
        conv9 = self.conv9(up9)
        # print(conv9.shape)
       
        conv10 = self.conv10(conv9)
        # print(conv10.shape)
        # out = F.pixel_shuffle(conv10, 2)
        enh_out = F.pixel_shuffle(conv10, 1)
        seg_out = self.seg_out(conv9)

        return enh_out,seg_out

        
class gray_CBAM_TA_CA_seg_UNet(nn.Module):
    def __init__(self):
        super(gray_CBAM_TA_CA_seg_UNet, self).__init__()
        self.conv1 = Double_Conv2d(3, 32)
        self.conv2 = Double_Conv2d(32, 64)
        self.conv3 = Double_Conv2d(64, 128)
        self.conv4 = Double_Conv2d(128, 256)
        self.conv5 = Double_Conv2d(256, 512)                
        
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.TA5 =TripletAttention()
        self.CBAM5 =CBAM(512)
        self.CA5 =CoordAtt(512,512)
        self.conv6 = Double_Conv2d(512, 256)
        
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.TA4 =TripletAttention()
        self.CBAM4 =CBAM(256)
        self.CA4 =CoordAtt(256,256)
        self.conv7 = Double_Conv2d(256, 128)
        
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.TA3 =TripletAttention()
        self.CBAM3 =CBAM(128)
        self.CA3 =CoordAtt(128,128)
        self.conv8 = Double_Conv2d(128, 64)
        
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.TA2 =TripletAttention()
        self.CBAM2 =CBAM(64)
        self.CA2 =CoordAtt(64,64)
        self.conv9 = Double_Conv2d(64, 32)

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)#enhancement head
        self.seg_out= nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)#segmentation head

    def forward(self, x, gray):
    
        gray2 = F.max_pool2d(gray, kernel_size=2)
        gray3 = F.max_pool2d(gray2, kernel_size=2)
        gray4 = F.max_pool2d(gray3, kernel_size=2)
        gray5 = F.max_pool2d(gray4, kernel_size=2)    
    
        conv1 = self.conv1(x)
        # print(conv1.shape)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        # print(conv2.shape)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        # print(conv3.shape)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        # print(conv4.shape)
        pool4 = F.max_pool2d(conv4, kernel_size=2)


        conv5 = self.conv5(pool4)
        conv5 = conv5 * gray5
        #print('conv5.shape : ',conv5.shape)# torch.Size([8, 512, 32, 32])

        up6 = self.up6(conv5)
        conv4 = conv4 * gray4
        #print('up6.shape : ',up6.shape)#torch.Size([8, 256, 64, 64])
        up6 = torch.cat([up6, conv4], 1)
        #print('up62.shape : ',up6.shape) #torch.Size([8, 512, 64, 64])
        up6 = self.TA5(up6)
        up6 = self.CBAM5(up6)
        up6 = self.CA5(up6)
        conv6 = self.conv6(up6)
        # print(conv6.shape)

        up7 = self.up7(conv6)
        conv3 = conv3 * gray3
        up7 = torch.cat([up7, conv3], 1)
        up7 = self.TA4(up7)
        up7 = self.CBAM4(up7)
        up7 = self.CA4(up7)
        conv7 = self.conv7(up7)
        # print(conv7.shape)
        
        up8 = self.up8(conv7)
        conv2 = conv2 * gray2
        up8 = torch.cat([up8, conv2], 1)
        up8 = self.TA3(up8)
        up8 = self.CBAM3(up8)
        up8 = self.CA3(up8)
        conv8 = self.conv8(up8)
        # print(conv8.shape)

        up9 = self.up9(conv8)
        conv1 = conv1 * gray
        up9 = torch.cat([up9, conv1], 1)
        up9 = self.TA2(up9)
        up9 = self.CBAM2(up9)
        up9 = self.CA2(up9)
        conv9 = self.conv9(up9)
        # print(conv9.shape)
       
        conv10 = self.conv10(conv9)
        # print(conv10.shape)
        # out = F.pixel_shuffle(conv10, 2)
        enh_out = F.pixel_shuffle(conv10, 1)
        seg_out = self.seg_out(conv9)

        return enh_out,seg_out
        
class CBAM_UNet(nn.Module):
    def __init__(self):
        super(CBAM_UNet, self).__init__()
        self.conv1 = Double_Conv2d(3, 32)
        self.conv2 = Double_Conv2d(32, 64)
        self.conv3 = Double_Conv2d(64, 128)
        self.conv4 = Double_Conv2d(128, 256)
        self.conv5 = Double_Conv2d(256, 512)       
        
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.CBAM5 =CBAM(512)
        self.conv6 = Double_Conv2d(512, 256)
        
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.CBAM4 =CBAM(256)
        self.conv7 = Double_Conv2d(256, 128)
        
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.CBAM3 =CBAM(128)
        self.conv8 = Double_Conv2d(128, 64)
        
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.CBAM2 =CBAM(64)
        self.conv9 = Double_Conv2d(64, 32)
        
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)


    def forward(self, x):
        conv1 = self.conv1(x)
        # print(conv1.shape)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        # print(conv2.shape)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        # print(conv3.shape)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        # print(conv4.shape)
        pool4 = F.max_pool2d(conv4, kernel_size=2)

        conv5 = self.conv5(pool4)
        #print('conv5.shape : ',conv5.shape)# torch.Size([8, 512, 32, 32])

        up6 = self.up6(conv5)
        #print('up6.shape : ',up6.shape)#torch.Size([8, 256, 64, 64])
        up6 = torch.cat([up6, conv4], 1)
        #print('up62.shape : ',up6.shape) #torch.Size([8, 512, 64, 64])
        up6 = self.CBAM5(up6)
        conv6 = self.conv6(up6)
        # print(conv6.shape)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        up7 = self.CBAM4(up7)
        conv7 = self.conv7(up7)
        # print(conv7.shape)
        
        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        up8 = self.CBAM3(up8)
        conv8 = self.conv8(up8)
        # print(conv8.shape)

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        up9 = self.CBAM2(up9)
        conv9 = self.conv9(up9)
        # print(conv9.shape)

        conv10 = self.conv10(conv9)
        # print(conv10.shape)
        # out = F.pixel_shuffle(conv10, 2)
        out = F.pixel_shuffle(conv10, 1)
        # print(out.shape)

        return out


