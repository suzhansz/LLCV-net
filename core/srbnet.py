import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from utils.net_utils import conv2d, deconv_2d
from extractor import BasicEncoder
from update import GMAUpdateBlock
import torch.nn.functional as F
from utils.utils import coords_grid

class Discriminator(nn.Module):
    def __init__(self, F1, F2):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(F1)), 512),
            nn.Linear(int(np.prod(F2)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, F1, F2):
        F1_flat = F1.view(F1.size(0), -1)
        F2_flat = F2.view(F2.size(0), -1)

        validity1 = self.model(F1_flat)
        validity2 = self.model(F2_flat)

        return validity1, validity2


class Discrimator_Loss(torch.nn.Module):
    def __init__(self, alpha=0.25, reduction='mean'):
        super(Discrimator_Loss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, validity1, validity2):
        pt = validity1
        Drloss = -((1 - self.alpha) * validity2 * torch.log(pt + 1e-5) + self.alpha * (1 - validity2) * torch.log(
            1 - pt + 1e-5))
        if self.reduction == 'mean':
            Drloss = torch.mean(Drloss)
        elif self.reduction == 'sum':
            Drloss = torch.sum(Drloss)

        return Drloss


class Discriminator_Contrastive_Loss(torch.nn.Module):
    def __init__(self, batch_size = 1, norm_fn='batch', dropout=0.0,device='cuda'):
        super().__init__()
        self.batch_size = batch_size
        self.norm_fn = norm_fn
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        self.register_buffer("negatives_mask", (
            torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, F1, F2):
        print(F1.shape, F2.shape)

        z_feature1 = F1
        z_feature2 = F2


        representations = torch.cat([z_feature1, z_feature2], dim=3)

        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=1)

        sim_feature1feature2 = torch.cosine_similarity(similarity_matrix, self.batch_size)
        sim_feature2feature1 = torch.cosine_similarity(similarity_matrix, -self.batch_size)

        projections_1 = nn.math.l2_normalize(sim_feature1feature2, axis=1)
        projections_2 = nn.math.l2_normalize(sim_feature2feature1, axis=1)
        positives = torch.cat([projections_1, projections_2], dim=256)
        similarities = (
                nn.matmul(projections_1, projections_2, transpose_b=True) / positives
        )

        batch_size = nn.shape(projections_1)[0]
        contrastive_labels = nn.range(batch_size)
        Lloss = torch.losses.sparse_categorical_crossentropy(
            nn.concat([contrastive_labels, contrastive_labels], axis=0),
            nn.concat([similarities, nn.transpose(similarities)], axis=0),
            from_logits=True,
        )

        return Lloss


class ResBlocks(nn.Module):
    def __init__(self, res_block_num, conv_channels, use_bn=False):
        super(ResBlocks, self).__init__()
        self.res_block_num = res_block_num
        for res_block_idx in range(self.res_block_num):
            conv_layer_1 = conv2d(conv_channels, conv_channels, kernel_size=3, stride=1, use_bn=use_bn)
            conv_layer_2 = conv2d(conv_channels, conv_channels, kernel_size=3, stride=1, activation=False,
                                  use_bn=use_bn)
            self.add_module('%d' % (res_block_idx), nn.Sequential(conv_layer_1, conv_layer_2))

    def __getitem__(self, index):
        if index < 0 or index >= len(self._modules):
            raise IndexError('index %d is out of range' % (index))

        return (self._modules[str(index)])

    def __len__(self):
        return self.res_block_num


class SharedWeightsBlock(nn.Module):
    def __init__(self, input_channels=3, base_dim=32, res_block_num=16, use_bn=False, residual=True):
        super(SharedWeightsBlock, self).__init__()

        self.residual = residual
        self.pool = nn.MaxPool2d(2)

        self.conv_1_1 = conv2d(input_channels, base_dim, kernel_size=3, stride=1, use_bn=use_bn)
        self.conv_1_2 = conv2d(base_dim, base_dim, kernel_size=3, stride=1, use_bn=use_bn)

        self.conv_2_1 = conv2d(base_dim, base_dim * 2, kernel_size=3, stride=1, use_bn=use_bn)
        self.conv_2_2 = conv2d(base_dim * 2, base_dim * 2, kernel_size=3, stride=1, use_bn=use_bn)

        self.conv_3_1 = conv2d(base_dim * 2, base_dim * 4, kernel_size=3, stride=1, use_bn=use_bn)
        self.conv_3_2 = conv2d(base_dim * 4, base_dim * 4, kernel_size=3, stride=1, use_bn=use_bn)

        self.res_block_list = ResBlocks(res_block_num, base_dim * 16)

        self.conv_after_res_block = conv2d(base_dim * 16, base_dim * 16, kernel_size=3, stride=1, use_bn=use_bn)

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

        if self.residual:
            conv_7 = self.conv_5(self.conv_6)
            out = self.depth_to_space(conv_7)
        else:
            out = self.conv_5(self.conv_6)

        return out, conv_feature


class Feature_Context(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, input_tensor):
        conv_1 = self.conv_1_1(input_tensor)
        conv_1 = self.conv_1_2(conv_1)
        conv_1_pool = self.pool(conv_1)

        conv_2 = self.conv_2_1(conv_1_pool)
        conv_2 = self.conv_2_2(conv_2)
        conv_2_pool = self.pool(conv_2)

        conv_3 = self.conv_3_1(conv_2_pool)
        conv_3 = self.conv_3_2(conv_3)

        conv_feature = conv_3
        for res_block_idx in range(len(self.res_block_list)):
            conv_feature = self.res_block_list[res_block_idx](conv_feature) + conv_feature

        conv_feature = self.conv_after_res_block(conv_feature)

        conv_feature = conv_feature + conv_3

        if self.residual:
            conv_7 = self.conv_5(self.conv_6)
            out = self.depth_to_space(conv_7)
        else:
            out = self.conv_5(self.conv_6)

        return out,conv_feature


