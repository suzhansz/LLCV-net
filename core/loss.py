import torch
from torch import nn
import torch.nn.functional as F
from LLcvNet import compute_sparse_corr



class Aggregate(nn.Module):
    def __init__(
        self,
        args,
        dim,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, fmap1, fmap2):
        heads, b, c, h, w = self.heads, *fmap1.shape
        heads, b, c, h, w = self.heads, *fmap2.shape


        q1, k1 = self.to_qk(fmap1).chunk(2, dim=1)
        q2, k2 = self.to_qk(fmap2).chunk(2, dim=1)

        q1 = self.scale * q1
        q2 = self.scale * q2

        if self.args.position_only:
            sim1 = self.pos_emb(q1)
            sim2 = self.pos_emb(q2)

        elif self.args.position_and_content:

            sim_pos1 = self.pos_emb(q1)
            sim_pos2 = self.pos_emb(q2)

        attn1 = sim_pos1.softmax(dim=-1)
        attn2 = sim_pos2.softmax(dim=-1)

        if self.project is not None:
            out1 = self.project(attn1)

        elif self.project is not None:
            out2 = self.project(attn2)

        out1 = fmap1 + self.gamma * out1
        out2 = fmap2 + self.gamma * out2

        return out1, out2


def PerceptualLoss(blocks, weights, device):
    return FeatureLoss(perceptual_loss, blocks, weights, device)

def flow_loss_func(flow_preds, flow_gt, valid,
                   gamma=0.9,
                   max_flow=400,

                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        i_loss = (flow_preds[i] - flow_gt).abs()

        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    if valid.max() < 0.5:
        pass

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return flow_loss, metrics


class FeatureLoss(nn.Module):
    def __init__(self, loss, blocks, weights, device):
        super().__init__()
        self.feature_loss = loss
        assert all(isinstance(w, (int, float)) for w in weights)
        assert len(weights) == len(blocks)

        self.weights = torch.tensor(weights).to(device)

        assert len(blocks) <= 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks

        srb = Aggregate.forward(self,map)
        srb.eval()

        for param in srb.parameters():
            param.requires_grad = False

        srb = srb.to(device)

        bns = [i - 2 for i, m in enumerate(srb) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(srb[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureLoss(srb[bns[i]]) for i in blocks]
        self.features = srb[0: bns[blocks[-1]] + 1]

    def forward(self, fmap1, fmap2):

        llfeatures, nmfeatures = compute_sparse_corr(fmap1, fmap2)
        inputs = F.normalize(fmap1, llfeatures, nmfeatures)
        targets = F.normalize(fmap2, llfeatures, nmfeatures)


        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        features_loss = 0.0


        for llf, nmf, w in zip(input_features, target_features, self.weights):
            llf = llf.view(llf.size(0), -1)
            nmf = nmf.view(nmf.size(0), -1)
            features_loss += self.feature_loss(llf, nmf) * w

        return features_loss


class DiscriminatorContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (
            torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, feature1, feature2):
        z_feature1 = F.normalize(feature1, dim=1)
        z_feature2 = F.normalize(feature2, dim=1)

        representations = torch.cat([z_feature1, z_feature2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)

        sim_feature1feature2 = torch.diag(similarity_matrix, self.batch_size)
        sim_feature2feature1 = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_feature1feature2, sim_feature2feature1], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        dcloss = torch.sum(loss_partial) / (2 * self.batch_size)
        return dcloss


def perceptual_loss(x, y):
    F.mse_loss(x, y)


