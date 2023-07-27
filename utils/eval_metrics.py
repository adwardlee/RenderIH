import torch
import numpy as np
from pytorch3d.ops import knn_points

def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

def compute_dist_mano_to_obj(batch_mano_v, batch_v, batch_v_len):
    knn_dists, knn_idx, _ = knn_points(
        batch_mano_v, batch_v, None, None, K=1, return_nn=True
    )
    knn_dists = knn_dists.sqrt()[:, :, 0]

    #knn_dists = torch.clamp(knn_dists, dist_min, dist_max)
    return knn_dists, knn_idx[:, :, 0]

def compute_mrrpe(root_r_gt, root_l_gt, root_r_pred, root_l_pred, is_valid):
    rel_vec_gt = root_l_gt - root_r_gt
    rel_vec_pred = root_l_pred - root_r_pred

    invalid_idx = torch.nonzero((1 - is_valid).long()).view(-1)
    mrrpe = ((rel_vec_pred - rel_vec_gt) ** 2).sum(dim=1).sqrt()
    mrrpe[invalid_idx] = float("nan")
    mrrpe = mrrpe.cpu().numpy()
    return mrrpe

def compute_idx(gt_right, gt_left, thresh=3e-3):### b,778, 3
    dist, idx = compute_dist_mano_to_obj(gt_right, gt_left, 778)
    return dist, idx


def compute_cdev(pred_v3d_o, pred_v3d_r, gt_left, gt_right):

    dist_ro, idx_ro = compute_idx(gt_right, gt_left)

    contact_dist = 3 * 1e-3  # 3mm considered in contact
    vo_r_corres = torch.gather(pred_v3d_o, 1, idx_ro[:, :, None].repeat(1, 1, 3))

    # displacement vector H->O
    disp_ro = vo_r_corres - pred_v3d_r  # batch, num_v, 3
    disp_ro[dist_ro > contact_dist] = float("nan")
    cd = (disp_ro ** 2).sum(dim=2).sqrt()
    err_ro = nanmean(cd, axis=1)  # .cpu().numpy()  # m
    print('error shape', err_ro.shape, flush=True)
    print('error value', err_ro, flush=True)
    return (err_ro)