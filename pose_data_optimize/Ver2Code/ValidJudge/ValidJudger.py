from manopth.manolayer import ManoLayer
from HandPoseConverter import HandPoseConverter
import numpy as np
import torch
class ValidJudger:
    def __init__(self, mano_root, device='cuda:0'):
        # layers and loss utils
        self.device = device
        self.mano_layer = ManoLayer(
            joint_rot_mode="quat",
            root_rot_mode="quat",
            use_pca=False,
            mano_root=mano_root,
            center_idx=0,
            flat_hand_mean=True,
            return_transf=True,
            return_full_pose=True,
            side='right'
        ).to(self.device)

        self.sub_mano_layer = ManoLayer(
            joint_rot_mode="quat",
            root_rot_mode="quat",
            use_pca=False,
            mano_root=mano_root,
            center_idx=0,
            flat_hand_mean=True,
            return_transf=True,
            return_full_pose=True,
            side='left'
        ).to(self.device)
        self.rhpc = HandPoseConverter(side='right', root=mano_root,
                                      data_type='tensor')
        zero_vec_pose = torch.from_numpy(np.asarray([[1.0, 0.0, 0.0, 0.0]] * 16, dtype=np.float32)).unsqueeze(0).to(
            self.device)
        self.zero_mat_ = self.rhpc.mano_quat_2_mat_tensor(zero_vec_pose)

        self.lhpc = HandPoseConverter(side='left', root=mano_root, data_type='tensor')
        self.sub_zero_mat_ = self.lhpc.mano_quat_2_mat_tensor(zero_vec_pose)

    def ValidationJudge(self, joint_axis, zero_mat, side):

        #         14-13-12-\
        #                   \
        #    2-- 1 -- 0 -----*
        #   5 -- 4 -- 3 ----/
        #   11 - 10 - 9 ---/
        #    8-- 7 -- 6 --/
        # relative_trans = ja.transpose(3,2) @ zero_ja
        res = torch.zeros(joint_axis.shape[0], dtype=joint_axis.dtype).to(self.device)
        relative_trans = zero_mat.transpose(3,2) @ joint_axis
        x_axis = relative_trans[:, :, :, 0]
        y_axis = relative_trans[:, :, :, 1]
        z_axis = relative_trans[:, :, :, 2]
        norm_x_vec = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0).unsqueeze(0).repeat(x_axis.shape[0], x_axis.shape[1], 1).to(x_axis.device)
        norm_y_vec = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).unsqueeze(0).repeat(x_axis.shape[0], x_axis.shape[1], 1).to(x_axis.device)
        norm_z_vec = torch.tensor([0.0, 0.0, 1.0]).unsqueeze(0).unsqueeze(0).repeat(x_axis.shape[0], x_axis.shape[1], 1).to(x_axis.device)

        m = (relative_trans - relative_trans.transpose(-1, -2)) / 2
        rot_vec = torch.zeros_like(x_axis)
        rot_vec[..., 0] = m[..., 2, 1]
        rot_vec[..., 1] = m[..., 0, 2]
        rot_vec[..., 2] = m[..., 1, 0]
        # step 1: these finger should not have twist nor splank
        forbid_ts_idx = [1, 2, 4, 5, 10, 11, 7, 8, 13, 14]
        x_cos = torch.einsum("bik,bik->bi", x_axis[:, forbid_ts_idx, :], norm_z_vec[:, forbid_ts_idx, :])
        y_cos = torch.einsum("bik,bik->bi", y_axis[:, forbid_ts_idx, :], norm_z_vec[:, forbid_ts_idx, :])
        z_cos = torch.einsum("bik,bik->bi", z_axis[:, forbid_ts_idx, :], norm_z_vec[:, forbid_ts_idx, :])
        # res += (torch.pow(x_cos, 2) + torch.pow(y_cos, 2) + torch.pow(z_cos - 1, 2)).sum(-1)
        # step 2: these finger should not have twist
        forbid_t_idx = [0, 3, 9, 6, 12]
        rot_vec_cos = torch.einsum("bik,bik->bi", rot_vec[:, forbid_t_idx, :], norm_x_vec[:, forbid_t_idx, :])
        # res += torch.sum(torch.pow(rot_vec_cos[:, 0:4], 2), -1)
        # res += torch.sum(torch.pow(torch.max(rot_vec_cos[:, 4] - 0.5, rot_vec_cos[:, 4] * 0.0), 2), -1)

        # step 3: limit the finger angle
        bend_angle = torch.atan2(x_axis[..., 1], x_axis[..., 0]) * 180/np.pi
        splank_angle = torch.atan2(-x_axis[..., 2], x_axis[..., 0]) * 180/np.pi
           # splank limit
        finger_ids = [0, 3, 9, 6, 12]
        finger_range = [[-25, 15], [-15, 15], [-25, 15], [-20, 30], [-30, 30]]
        s_max = torch.zeros(joint_axis.shape[0], dtype=joint_axis.dtype).to(self.device)
        for i in range(len(finger_ids)):
            finger_idx = finger_ids[i]
            angle_range = finger_range[i]
            sa = splank_angle[:, finger_idx]
            upper_loss = torch.relu(sa - angle_range[1])
            lower_loss = torch.relu(angle_range[0] - sa)
            splank_loss = torch.max(upper_loss, lower_loss)/180 * np.pi
            s_max = torch.max(s_max, splank_loss)
        res += s_max

        s_max *= 0.0
            # bend_Limit
        # finger_ids = [[1,4, 10, 7], [12], [13], [0, 2, 3, 5, 9, 11, 6, 8, 14]]
        # finger_range = [[-8, 120], [-8, 35], [-8, 65], [-8, 90]]
        finger_ids = [[0], [1], [3], [4], [9], [10], [6], [7], [12], [13], [14], [2, 5, 11, 8]]
        finger_range = [[-25, 70], [-4, 110], [-25, 80], [-7, 100], [-25, 70], [-10, 100], [-22, 70], [-8, 90], [-20, 40], [-35, 50], [-10, 100], [-8, 90]]
        for i in range(len(finger_ids)):
            angle_range = finger_range[i]
            for finger_id in finger_ids[i]:
                ba = bend_angle[:, finger_id]
                upper_loss = torch.relu(ba - angle_range[1])
                lower_loss = torch.relu(angle_range[0] - ba)
                bend_loss = torch.max(upper_loss, lower_loss) / 180 * np.pi
                # if(finger_id == 13):
                #     print(ba[1])
                #     print(bend_loss[1])
                s_max = torch.max(bend_loss, s_max)
        res = torch.max(res, s_max)
        return res

    def ComputeValidation(self, left_pose, right_pose):
        threshhold = 0.1 / 180 * np.pi
        cur_joint_axis = self.rhpc.mano_quat_2_mat_tensor(right_pose)
        right_res = self.ValidationJudge(cur_joint_axis[:, 1:, :, :], self.zero_mat_[:, 1:, :, :], 'r')

        cur_joint_axis = self.lhpc.mano_quat_2_mat_tensor(left_pose)
        left_res = self.ValidationJudge(cur_joint_axis[:, 1:, :, :], self.sub_zero_mat_[:, 1:, :, :], 'l')

        rate = (((right_res > threshhold) + left_res > threshhold) > 0).sum() / left_pose.shape[0]
        print(rate)
        return rate