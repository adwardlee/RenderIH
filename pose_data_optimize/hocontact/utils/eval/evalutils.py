import warnings

import torch

from hocontact.hodatasets.hoquery import TransQueries, BaseQueries
from hocontact.utils.eval.zimeval import EvalUtil


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class Evaluator:
    def __init__(self):
        self.eval_meters = {
            "joints2d_base": EvalUtil(),
            "corners2d_base": EvalUtil(),
            "verts2d_base": EvalUtil(),
            "joints3d_cent": EvalUtil(),
            "joints3d": EvalUtil(),
        }
        self.loss_meters = {}

    def add_loss_value(self, loss_name, loss_val, n=1):
        if loss_name not in self.loss_meters:
            self.loss_meters[loss_name] = AverageMeter()
        self.loss_meters[loss_name].update(loss_val, n=n)

    def feed_loss_meters(self, sample, results):
        if "obj_verts2d" in results and results["obj_verts2d"] is not None and BaseQueries.OBJ_VERTS_2D in sample:
            obj_verts2d_gt = sample[TransQueries.OBJ_VERTS_2D]
            affinetrans = sample[TransQueries.AFFINETRANS]
            or_verts2d = sample[BaseQueries.OBJ_VERTS_2D]
            rec_pred = recover_back(results["obj_verts2d"].detach().cpu(), affinetrans)
            rec_gt = recover_back(obj_verts2d_gt, affinetrans)
            # Sanity check, this should be ~1pixel
            gt_err = (rec_gt - or_verts2d).norm(2, -1).mean()
            if gt_err > 1:
                warnings.warn(f"Back to orig error on gt {gt_err} > 1 pixel")
            verts2d_dists = (rec_pred - or_verts2d.cpu()).norm(2, -1)
            self.add_loss_value("obj_verts2d_mepe", verts2d_dists.mean(-1).sum().item(), n=or_verts2d.shape[0])

        if (
            "recov_hand_verts3d" in results
            and results["recov_hand_verts3d"] is not None
            and BaseQueries.HAND_VERTS_3D in sample
        ):
            hand_verts3d = sample[BaseQueries.HAND_VERTS_3D].cpu()
            verts3d_dists = (results["recov_hand_verts3d"].cpu() - hand_verts3d).norm(2, -1)
            self.add_loss_value("recov_hand_verts3d_mepe", verts3d_dists.mean(-1).sum().item(), n=hand_verts3d.shape[0])

        if (
            "recov_obj_verts3d" in results
            and results["recov_obj_verts3d"] is not None
            and BaseQueries.OBJ_VERTS_3D in sample
        ):
            or_verts3d = sample[BaseQueries.OBJ_VERTS_3D].cpu()
            verts3d_dists = (results["recov_obj_verts3d"].cpu() - or_verts3d).norm(2, -1)
            self.add_loss_value("recov_obj_verts3d_mepe", verts3d_dists.mean(-1).sum().item(), n=or_verts3d.shape[0])

        if (
            "recov_obj_corners3d" in results
            and results["recov_obj_corners3d"] is not None
            and BaseQueries.OBJ_CORNERS_3D in sample
        ):
            or_corners3d = sample[BaseQueries.OBJ_CORNERS_3D].cpu()
            corners3d_dists = (results["recov_obj_corners3d"].cpu() - or_corners3d).norm(2, -1)
            self.add_loss_value(
                "recov_obj_corners3d_mepe", corners3d_dists.mean(-1).sum().item(), n=or_corners3d.shape[0]
            )

        if "recov_joints3d" in results and results["recov_joints3d"] is not None and BaseQueries.JOINTS_3D in sample:
            or_joints3d = sample[BaseQueries.JOINTS_3D].cpu()
            joints3d_dists = (results["recov_joints3d"].cpu() - or_joints3d).norm(2, -1)
            self.add_loss_value("recov_joints3d_mepe", joints3d_dists.mean(-1).sum().item(), n=or_joints3d.shape[0])

    def feed_eval_meters(self, sample, results, center_idx=9):
        if "joints2d" in results and BaseQueries.JOINTS_2D in sample:
            gt_joints2d = sample[TransQueries.JOINTS_2D]
            affinetrans = sample[TransQueries.AFFINETRANS]
            or_joints2d = sample[BaseQueries.JOINTS_2D]
            rec_pred = recover_back(results["joints2d"].detach().cpu(), affinetrans)
            rec_gt = recover_back(gt_joints2d, affinetrans)
            # Sanity check, this should be ~1pixel
            gt_err = (rec_gt - or_joints2d).norm(2, -1).mean()
            if gt_err > 1:
                warnings.warn(f"Back to orig error on gt {gt_err} > 1 pixel")
            for gt_joints, pred_joints in zip(rec_pred.numpy(), or_joints2d.cpu().numpy()):
                self.eval_meters["joints2d_base"].feed(gt_joints, pred_joints)

        # Object 2d metric
        if "obj_corners2d" in results and results["obj_corners2d"] is not None and BaseQueries.OBJ_CORNERS_2D in sample:
            obj_corners2d_gt = sample[TransQueries.OBJ_CORNERS_2D]
            affinetrans = sample[TransQueries.AFFINETRANS]
            or_corners2d = sample[BaseQueries.OBJ_CORNERS_2D]
            rec_pred = recover_back(results["obj_corners2d"].detach().cpu(), affinetrans)
            rec_gt = recover_back(obj_corners2d_gt, affinetrans)
            # Sanity check, this should be ~1pixel
            gt_err = (rec_gt - or_corners2d).norm(2, -1).mean()
            if gt_err > 1:
                warnings.warn(f"Back to orig error on gt {gt_err} > 1 pixel")
            for gt_corners, pred_corners in zip(rec_pred.numpy(), or_corners2d.cpu().numpy()):
                self.eval_meters["corners2d_base"].feed(gt_corners, pred_corners)

        # Object 3d metric
        # if (
        #     "obj_corners3d" in results
        #     and results["obj_corners3d"] is not None
        #     and BaseQueries.OBJCORNERS3D in sample
        # ):
        #     if "recov_objcorners3d" in results:
        #         or_corners3d = sample[BaseQueries.OBJCORNERS3D]
        #         obj_corners3d_pred = results["recov_objcorners3d"].detach().cpu()
        #         for gt_corners, pred_corners in zip(obj_corners3d_pred.numpy(), or_corners3d.cpu().numpy()):
        #             self.eval_meters["corners3d_base"].feed(gt_corners, pred_corners)

        # Centered 3D hand metric
        if BaseQueries.JOINTS_3D in sample:

            if "joints3d" in results:
                gt_joints3d = sample[TransQueries.JOINTS_3D]
                pred_joints3d = results["joints3d"].cpu().detach()
                # if center_idx is not None:
                gt_joints3d_cent = gt_joints3d - gt_joints3d[:, center_idx : center_idx + 1]
                pred_joints3d_cent = pred_joints3d - pred_joints3d[:, center_idx : center_idx + 1]
                for gt_joints, pred_joints in zip(gt_joints3d_cent.numpy(), pred_joints3d_cent.numpy()):
                    self.eval_meters["joints3d_cent"].feed(gt_joints, pred_joints)

            if "recov_joints3d" in results:
                joints3d_pred = results["recov_joints3d"].detach().cpu()
                for gt_joints, pred_joints in zip(sample[BaseQueries.JOINTS_3D].cpu(), joints3d_pred):
                    self.eval_meters["joints3d"].feed(gt_joints, pred_joints)

    def parse_evaluators(self, config=None):
        """
        Parse evaluators for which PCK curves and other statistics
        must be computed
        """
        if config is None:
            config = {
                "joints2d_base": [0, 100, 100],
                "corners2d_base": [0, 100, 100],
                "verts2d_base": [0, 100, 100],
                "joints3d_cent": [0, 0.2, 20],
                "joints3d": [0, 0.5, 20],
            }
        eval_results = {}
        for eval_key, eval_meter in self.eval_meters.items():
            if eval_meter.empty():
                continue

            # we need to select way of dealing with eval results, judging by its instance type
            if isinstance(eval_meter, EvalUtil):
                start, end, steps = [config[eval_key][idx] for idx in range(3)]
                (epe_mean, epe_mean_joints, epe_median, auc, pck_curve, thresholds) = eval_meter.get_measures(
                    start, end, steps
                )

                eval_results[eval_key] = {
                    "epe_mean": epe_mean,
                    "epe_mean_joints": epe_mean_joints,
                    "epe_median": epe_median,
                    "auc": auc,
                    "thresholds": thresholds,
                    "pck_curve": pck_curve,
                }
        return eval_results


def recover_back(joints_trans, affinetrans):
    """
    Given 2d point coordinates and an affine transform, recovers original pixel points
    (locations before translation, rotation, crop, scaling... are applied during data
    augmentation)
    """
    batch_size = joints_trans.shape[0]
    point_nb = joints_trans.shape[1]
    hom2d = torch.cat([joints_trans, joints_trans.new_ones(batch_size, point_nb, 1)], -1)
    rec2d = torch.inverse(affinetrans).bmm(hom2d.transpose(1, 2).float()).transpose(1, 2)[:, :, :2]
    return rec2d
