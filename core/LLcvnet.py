import torch
import torch.nn as nn
import torch.nn.functional as F

from extractor import BasicEncoder, BasicEncoderQuarter
from update import BasicUpdateBlock, BasicUpdateBlockQuarter
from utils.utils import coords_grid, coords_grid_y_first,\
    upflow4, compute_interpolation_weights
from knn import knn_faiss_raw
from srbnet import SharedWeightsBlock
from srbnet import Discriminator_Contrastive_Loss
try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self):
            pass
        def __enter__(self):
            pass
        def  __exit__(self):
            pass

class RAFTGMA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.shared = SharedWeightsBlock
        self.dcloss = Discriminator_Contrastive_Loss
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0


        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlockQuarter(self.args, hidden_dim=hdim)
        self.dcloss = Discriminator_Contrastive_Loss(output_dim=256, norm_fn='instance', dropout=args.dropout)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def compute_sparse_corr(self, out1, out2, k=32):

        B, C, H1, W1 = self.shape
        H1, W1 = out1.shape[2:]
        H2, W2 = out2.shape[2:]
        N1 = H1 * W1
        N2 = H2 * W2

        self, out1, out2 = self.view(B, C, -1), out1.view(B, C, -1), out2.view(B, C, -1)

        with torch.no_grad():
            _, indices = knn_faiss_raw(self, out1, out2, k)

            indices_coord = indices.unsqueeze(1).expand(-1, 2, -1, -1)
            coords0 = coords_grid_y_first(B, H2, W2).view(B, 2, 1, -1).expand(-1, -1, k, -1).to(self.device)
            coords1 = coords0.gather(3, indices_coord)
            coords1 = coords1 - coords0


            batch_index = torch.arange(B).view(B, 1, 1, 1).expand(-1, -1, k, N1).type_as(coords1)


        out1 = out1.gather(2, indices.view(B, 1, -1).expand(-1, C, -1)).view(B, C, k, N1)
        out2 = out2.gather(2, indices.view(B, 1, -1).expand(-1, C, -1)).view(B, C, k, N2)


        corr_sp = torch.einsum('bcn,bckn->bkn', self, out1, out2).contiguous() / torch.sqrt(torch.tensor(C).float())
        return corr_sp, coords0, coords1, batch_index

class FlowHead(nn.Module):
    def __init__(self, input_dim=256, batch_norm=True):
        super().__init__()
        if batch_norm:
            self.flowpredictor = nn.Sequential(
                nn.Conv2d(input_dim, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 2, 3, padding=1)
            )
        else:
            self.flowpredictor = nn.Sequential(
                nn.Conv2d(input_dim, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 2, 3, padding=1)
            )

    def forward(self, x):
        return self.flowpredictor(x)

class LLcvNetEighth(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=False)
        self.cnet = BasicEncoder(output_dim=256, norm_fn='batch', dropout=False)

        self.update_block = BasicUpdateBlock(self.args, hidden_dim=128, input_dim=405)

    def initialize_flow(self, img):

        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)


        return coords0, coords1

    def upsample_flow(self, flow, mask):

        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, iters, flow_init=None, test_mode=False):

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [128, 128], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        B, _, H1, W1 = fmap1.shape
        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init


        corr_val, coords0_cv, coords1_cv, batch_index_cv = compute_sparse_corr(fmap1, fmap2, k=self.args.num_k)
        delta_flow = torch.zeros_like(coords0)
        flow_predictions = []
        search_range = 4
        corr_val = corr_val.repeat(1, 4, 1)

        for itr in range(iters):
            with torch.no_grad():


                coords1_cv = coords1_cv - delta_flow[:, [1, 0], :, :].view(B, 2, 1, -1)
                mask_pyramid = []
                weights_pyramid = []
                coords_sparse_pyramid = []

                for i in range(5):
                    coords1_sp = coords1_cv * 0.5**i
                    weights, coords1_sp = compute_interpolation_weights(coords1_sp)
                    mask = (coords1_sp[:, 0].abs() <= search_range) & (coords1_sp[:, 1].abs() <= search_range)
                    batch_ind = batch_index_cv.permute(0, 2, 3, 1).repeat(1, 4, 1, 1)[mask]
                    coords0_sp = coords0_cv.permute(0, 2, 3, 1).repeat(1, 4, 1, 1)[mask]
                    coords1_sp = coords1_sp.permute(0, 2, 3, 1)[mask]
                    coords1_sp = coords1_sp + search_range
                    coords_sp = torch.cat([batch_ind, coords0_sp, coords1_sp], dim=1)
                    coords_sparse_pyramid.append(coords_sp)
                    mask_pyramid.append(mask)
                    weights_pyramid.append(weights)

            corr_val_pyramid = []
            for mask, weights in zip(mask_pyramid, weights_pyramid):
                corr_masked = (weights * corr_val)[mask].unsqueeze(1)
                corr_val_pyramid.append(corr_masked)
            sparse_tensor_pyramid = [torch.sparse.FloatTensor(coords_sp.t().long(), corr_resample, torch.Size([B, H1, W1, 9, 9, 1])).coalesce()
                                     for coords_sp, corr_resample in zip(coords_sparse_pyramid, corr_val_pyramid)]
            corr = torch.cat([sp.to_dense().view(B, H1, W1, -1) for sp in sparse_tensor_pyramid], dim=3).permute(0, 3, 1, 2)
            coords1 = coords1.detach()

            flow = coords1 - coords0


            with autocast(enabled=self.args.mixed_precision):

                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)


            coords1 = coords1 + delta_flow

            if up_mask is None:
                flow_up = upflow4(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)
        if test_mode:
            return coords1 - coords0,flow_up
        return flow_predictions

def compute_sparse_corr(fmap1, fmap2, k=32):

    B, C, H1, W1 = fmap1.shape
    H2, W2 = fmap2.shape[2:]
    N = H1 * W1

    fmap1, fmap2 = fmap1.view(B, C, -1), fmap2.view(B, C, -1)

    with torch.no_grad():
        _, indices = knn_faiss_raw(fmap1, fmap2, k)

        indices_coord = indices.unsqueeze(1).expand(-1, 2, -1, -1)
        coords0 = coords_grid_y_first(B, H2, W2).view(B, 2, 1, -1).expand(-1, -1, k, -1).to(fmap1.device)
        coords1 = coords0.gather(3, indices_coord)
        coords1 = coords1 - coords0

        batch_index = torch.arange(B).view(B, 1, 1, 1).expand(-1, -1, k, N).type_as(coords1)

    fmap2 = fmap2.gather(2, indices.view(B, 1, -1).expand(-1, C, -1)).view(B, C, k, N)

    corr_sp = torch.einsum('bcn,bckn->bkn', fmap1, fmap2).contiguous() / torch.sqrt(torch.tensor(C).float())
    return corr_sp, coords0, coords1, batch_index



class LLcvNet(nn.Module):
    def __init__(self, args):
        super(LLcvNet,self).__init__()
        self.args = args

        self.fnet = BasicEncoderQuarter(output_dim=256, norm_fn='instance', dropout=False)
        self.cnet = BasicEncoderQuarter(output_dim=256, norm_fn='batch', dropout=False)


        self.update_block = BasicUpdateBlockQuarter(self.args, hidden_dim=128, input_dim=405)

    def initialize_flow(self, img):

        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//4, W//4).to(img.device)
        coords1 = coords_grid(N, H//4, W//4).to(img.device)

        return coords0, coords1

    def upsample_flow_quarter(self, flow, mask):

        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)


        up_flow = F.unfold(4 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4*H, 4*W)

    def forward(self, image1, image2, iters, flow_init=None, test_mode=False):

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()


        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [128, 128], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        B, _, H1, W1 = fmap1.shape


        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init


        corr_val, coords0_cv, coords1_cv, batch_index_cv = compute_sparse_corr(fmap1, fmap2, k=self.args.num_k)

        delta_flow = torch.zeros_like(coords0)

        flow_predictions = []

        search_range = 4
        corr_val = corr_val.repeat(1, 4, 1)

        for itr in range(iters):
            with torch.no_grad():


                coords1_cv = coords1_cv - delta_flow[:, [1, 0], :, :].view(B, 2, 1, -1)

                mask_pyramid = []
                weights_pyramid = []
                coords_sparse_pyramid = []

                for i in range(5):
                    coords1_sp = coords1_cv * 0.5**i
                    weights, coords1_sp = compute_interpolation_weights(coords1_sp)
                    mask = (coords1_sp[:, 0].abs() <= search_range) & (coords1_sp[:, 1].abs() <= search_range)
                    batch_ind = batch_index_cv.permute(0, 2, 3, 1).repeat(1, 4, 1, 1)[mask]
                    coords0_sp = coords0_cv.permute(0, 2, 3, 1).repeat(1, 4, 1, 1)[mask]
                    coords1_sp = coords1_sp.permute(0, 2, 3, 1)[mask]

                    coords1_sp = coords1_sp + search_range
                    coords_sp = torch.cat([batch_ind, coords0_sp, coords1_sp], dim=1)
                    coords_sparse_pyramid.append(coords_sp)

                    mask_pyramid.append(mask)
                    weights_pyramid.append(weights)

            corr_val_pyramid = []
            for mask, weights in zip(mask_pyramid, weights_pyramid):
                corr_masked = (weights * corr_val)[mask].unsqueeze(1)
                corr_val_pyramid.append(corr_masked)

            sparse_tensor_pyramid = [torch.sparse.FloatTensor(coords_sp.t().long(), corr_resample, torch.Size([B, H1, W1, 9, 9, 1])).coalesce()
                                     for coords_sp, corr_resample in zip(coords_sparse_pyramid, corr_val_pyramid)]

            corr = torch.cat([sp.to_dense().view(B, H1, W1, -1) for sp in sparse_tensor_pyramid], dim=3).permute(0, 3, 1, 2)


            coords1 = coords1.detach()

            flow = coords1 - coords0


            with autocast(enabled=self.args.mixed_precision):

                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            coords1 = coords1 + delta_flow

            if up_mask is None:
                flow_up = upflow4(coords1 - coords0)
            else:
                flow_up = self.upsample_flow_quarter(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0,flow_up

        return flow_predictions, fmap1, fmap2



























