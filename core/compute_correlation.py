import torch
from knn import knn_faiss_raw
from utils.utils import  coords_grid_y_first,bilinear_sampler
import torch.nn.functional as F

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []


        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2),
                            fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())
    def normalize_coords(coords, H, W):

        one = coords.new_tensor(1)
        size = torch.stack([one*W, one*H])[None]
        center = size / 2
        scaling = size.max(1, keepdim=True).values * 0.5
        return (coords - center[:, :, None]) / scaling[:, :, None]


    def compute_sparse_corr_init(fmap1, fmap2, k=32):

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
        me_corr = torch.einsum('bcn,bckn->bkn', fmap1, fmap2).contiguous() / torch.sqrt(torch.tensor(C).float())

        return me_corr, coords0, coords1, batch_index


    if __name__ == "__main__":
        torch.manual_seed(0)

        for _ in range(100):
            fmap1 = torch.randn(8, 256, 92, 124).cuda()
            fmap2 = torch.randn(8, 256, 92, 124).cuda()
            corr_me = compute_sparse_corr_init(fmap1, fmap2, k=16)



