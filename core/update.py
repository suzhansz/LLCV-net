import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.net_utils import conv2d, deconv_2d
from torch.nn.init import kaiming_normal_, constant_
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, dilation=(1, 1), kernel_size=3):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, dilation=dilation[0])
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, dilation=dilation[1])
        self.relu = nn.ReLU(inplace=True)

        self.projector = nn.Conv2d(in_planes, planes, kernel_size=1)

    def forward(self, x):
        y = x
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))

        if self.projector is not None:
            x = self.projector(x)

        return self.relu(x + y)


class BasicMotionEncoder(nn.Module):
    def __init__(self,  input_dim=128):
        super().__init__()
        self.convc1 = nn.Conv2d(input_dim, 256, 1)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(192+64, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, input_dim=input_dim)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)


        mask = .25 * self.mask(net)
        return net, mask, delta_flow


class ResBlocks(nn.Module):
    def __init__(self, res_block_num,use_bn=False):
        super(ResBlocks, self).__init__()
        self.res_block_num = res_block_num
        for res_block_idx in range(self.res_block_num):
            conv_layer_1 = conv2d(128, 128, kernel_size=3, stride=1, use_bn=use_bn)
            conv_layer_2 = conv2d(128, 128, kernel_size=3, stride=1, activation=False,
                                  use_bn=use_bn)

            self.add_module('%d' % (res_block_idx), nn.Sequential(conv_layer_1, conv_layer_2))

    def __getitem__(self, index):
        if index < 0 or index >= len(self._modules):
            raise IndexError('index %d is out of range' % (index))

        return (self._modules[str(index)])

    def __len__(self):
        return self.res_block_num


class SharedWeightsBlock(nn.Module):
    def __init__(self, input_channels=128, base_dim=32, res_block_num=2, use_bn=False, residual=True):
        super(SharedWeightsBlock, self).__init__()

        self.residual = residual
        self.pool = nn.MaxPool2d(1)

        self.conv_1_1 = conv2d(input_channels, base_dim, kernel_size=3, stride=1, use_bn=use_bn)
        self.conv_1_2 = conv2d(base_dim, base_dim, kernel_size=3, stride=1, use_bn=use_bn)

        self.conv_2_1 = conv2d(base_dim, base_dim * 2, kernel_size=3, stride=1, use_bn=use_bn)
        self.conv_2_2 = conv2d(base_dim * 2, base_dim * 2, kernel_size=3, stride=1, use_bn=use_bn)

        self.conv_3_1 = conv2d(base_dim * 2, base_dim * 4, kernel_size=3, stride=1, use_bn=use_bn)
        self.conv_3_2 = conv2d(base_dim * 4, base_dim * 4, kernel_size=3, stride=1, use_bn=use_bn)

        self.res_block_list = ResBlocks(res_block_num, base_dim * 16)

        self.conv_after_res_block = conv2d(128, 128, kernel_size=3, stride=1, use_bn=use_bn)

        self.deconv_4 = deconv_2d(base_dim * 16, base_dim * 8, use_bn=use_bn)
        self.conv_4_1 = conv2d(base_dim * 16, base_dim * 8, kernel_size=3, stride=1, use_bn=use_bn)
        self.conv_4_2 = conv2d(base_dim * 8, base_dim * 8, kernel_size=3, stride=1, use_bn=use_bn)

        self.deconv_5 = deconv_2d(base_dim * 8, base_dim * 4, use_bn=use_bn)
        self.conv_5_1 = conv2d(base_dim * 8, base_dim * 4, kernel_size=3, stride=1, use_bn=use_bn)
        self.conv_5_2 = conv2d(base_dim * 4, base_dim * 4, kernel_size=3, stride=1, use_bn=use_bn)

        if self.residual:
            self.conv_6 = conv2d(base_dim, 12, kernel_size=1, stride=1, use_bn=use_bn)
            self.depth_to_space = nn.PixelShuffle(2)

        else:
            self.conv_6 = conv2d(base_dim, 3, kernel_size=1, stride=1, use_bn=use_bn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, input_tensor):
        conv_1 = self.conv_1_1(input_tensor)
        conv_1 = self.conv_1_2(conv_1)
        conv_1_pool = self.pool(conv_1)

        conv_2 = self.conv_2_1(conv_1_pool)
        conv_2 = self.conv_2_2(conv_2)
        conv_2_pool = self.pool(conv_2)

        conv_3 = self.conv_3_1(conv_2_pool)
        conv_3 = self.conv_3_2(conv_3)
        # res block
        conv_feature = conv_3
        for res_block_idx in range(len(self.res_block_list)):
            conv_feature = self.res_block_list[res_block_idx](conv_feature) + conv_feature

        conv_feature = self.conv_after_res_block(conv_feature)

        conv_feature = conv_feature + conv_3

        return conv_feature


class BasicUpdateBlockQuarter(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlockQuarter, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, input_dim=input_dim)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(input_dim=hidden_dim, hidden_dim=256)
        self.SharedWeights = SharedWeightsBlock()

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16*9, 1, padding=0))

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        motion_features = self.SharedWeights(motion_features)

        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow


class GMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))


    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)
        net = self.gru(net, inp_cat)
        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


