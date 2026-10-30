"""Pure-torch re-implementation of pytorch3d.loss.chamfer_distance (default args).

Semantics mirrored from pytorch3d 0.7.4:
  cham_x = mean over points of x of min_j ||x_i - y_j||^2 ; cham_y symmetric ;
  cham_dist = batch-mean(cham_x + cham_y)   (point_reduction='mean', batch_reduction='mean', norm=2)
  cham_normals = batch-mean( mean_i(1 - |cos(nx_i, ny_nn(i))|) + mean_j(1 - |cos(ny_j, nx_nn(j))|) )
Nearest neighbours are brute force, chunked so a 10k x 10k cloud fits in memory.
"""
import torch

def _nn(a, b, chunk=4096):
    # a: [N,3], b: [M,3] -> (min squared dist [N], argmin [N])
    d_all, i_all = [], []
    for s in range(0, a.shape[0], chunk):
        d = torch.cdist(a[s:s+chunk], b) ** 2
        m, i = d.min(dim=1)
        d_all.append(m); i_all.append(i)
    return torch.cat(d_all), torch.cat(i_all)

def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None,
                     weights=None, batch_reduction="mean", point_reduction="mean", norm=2,
                     single_directional=False, abs_cosine=True):
    assert norm == 2 and point_reduction == "mean" and not single_directional
    assert x_lengths is None and y_lengths is None and weights is None
    if x.dim() == 2: x = x[None]
    if y.dim() == 2: y = y[None]
    B = x.shape[0]
    dists, nloss = [], []
    for b in range(B):
        dxy, ixy = _nn(x[b], y[b])
        dyx, iyx = _nn(y[b], x[b])
        dists.append(dxy.mean() + dyx.mean())
        if x_normals is not None and y_normals is not None:
            nx, ny = x_normals[b], y_normals[b]
            cx = torch.nn.functional.cosine_similarity(nx, ny[ixy], dim=1, eps=1e-6)
            cy = torch.nn.functional.cosine_similarity(ny, nx[iyx], dim=1, eps=1e-6)
            if abs_cosine: cx, cy = cx.abs(), cy.abs()
            nloss.append((1 - cx).mean() + (1 - cy).mean())
    cham = torch.stack(dists)
    cham_n = torch.stack(nloss) if nloss else None
    if batch_reduction == "mean":
        cham = cham.mean(); cham_n = cham_n.mean() if cham_n is not None else None
    elif batch_reduction == "sum":
        cham = cham.sum(); cham_n = cham_n.sum() if cham_n is not None else None
    return cham, cham_n
