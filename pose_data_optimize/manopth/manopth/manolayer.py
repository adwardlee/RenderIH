import os

import numpy as np
import torch
from torch.nn import Module
import copy

from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments
from manopth import rodrigues_layer, rotproj, rot6d
from manopth.tensutils import th_posemap_axisang, th_with_zeros, th_pack, subtract_flat_id, make_list
from manopth.quatutils import (
    quaternion_to_angle_axis,
    quaternion_inv,
    quaternion_mul,
    quaternion_to_rotation_matrix,
    normalize_quaternion,
)


class ManoLayer(Module):
    __constants__ = [
        "use_pca",
        "rot",
        "ncomps",
        "ncomps",
        "kintree_parents",
        "check",
        "side",
        "center_idx",
        "joint_rot_mode",
    ]

    def __init__(
        self,
        center_idx=None,
        flat_hand_mean=True,
        ncomps=6,
        side="right",
        mano_root="mano/models",
        use_pca=True,
        root_rot_mode="axisang",
        joint_rot_mode="axisang",
        robust_rot=False,
        return_transf=False,
        return_full_pose=False,
    ):
        """
        Args:
            center_idx: index of center joint in our computations,
                if -1 centers on estimate of palm as middle of base
                of middle finger and wrist
            flat_hand_mean: if True, (0, 0, 0, ...) pose coefficients match
                flat hand, else match average hand pose
            mano_root: path to MANO pkl files for left and right hand
            ncomps: number of PCA components form pose space (<45)
            side: 'right' or 'left'
    mano_layer = ManoLayer(
        mano_root="ass
            use_pca: Use PCA decomposition for pose space.
            root_rot_mode: 'axisang' or 'rotmat' or 'quat',
            joint_rot_mode: 'axisang' or 'rotmat' or 'quat', ignored if use_pca
        """
        super().__init__()

        self.center_idx = center_idx
        self.robust_rot = robust_rot

        # check root_rot_mode feasible, and set self.rot
        if root_rot_mode == "axisang":
            self.rot = 3
        elif root_rot_mode == "rotmat":
            self.rot = 6
        elif root_rot_mode == "quat":
            self.rot = 4
        else:
            raise KeyError(
                "root_rot_mode not found. shoule be one of 'axisang' or 'rotmat' or 'quat'. got {}".format(root_rot_mode)
            )

        # todo: flat_hand_mean have issues
        self.flat_hand_mean = flat_hand_mean

        # toggle extra return information
        self.return_transf = return_transf
        self.return_full_pose = return_full_pose

        # record side of hands
        self.side = side

        # check use_pca and joint_rot_mode
        if use_pca and joint_rot_mode != "axisang":
            raise TypeError("if use_pca, joint_rot_mode must be 'axisang'. got {}".format(joint_rot_mode))
        # record use_pca flag and joint_rot_mode
        self.use_pca = use_pca
        self.joint_rot_mode = joint_rot_mode
        # self.ncomps only work in axisang mode
        if use_pca:
            self.ncomps = ncomps
        else:
            self.ncomps = 45

        # do more checks on root_rot_mode, in case mode error
        if self.joint_rot_mode == "axisang":
            # add restriction to root_rot_mode
            if root_rot_mode not in ["axisang", "rotmat"]:
                err_msg = "rot_mode not compatible, "
                err_msg += "when joint_rot_mode is 'axisang', root_rot_mode should be one of "
                err_msg += "'axisang' or 'rotmat', got {}".format(root_rot_mode)
                raise KeyError(err_msg)
        else:
            # for 'rotmat' or 'quat', there rot_mode must be same for joint and root
            if root_rot_mode != self.joint_rot_mode:
                err_msg = "rot_mode not compatible, "
                err_msg += "should get the same rot mode for joint and root, "
                err_msg += "got {} for root and {} for joint".format(root_rot_mode, self.joint_rot_mode)
                raise KeyError(err_msg)
        # record root_rot_mode
        self.root_rot_mode = root_rot_mode

        # load model according to side flag
        self.side = side
        if side == "right":
            self.mano_path = os.path.join(mano_root, "MANO_RIGHT.pkl")
        elif side == "left":
            self.mano_path = os.path.join(mano_root, "MANO_LEFT.pkl")

        # parse and register stuff
        smpl_data = ready_arguments(self.mano_path)

        hands_components = smpl_data["hands_components"]

        self.smpl_data = smpl_data

        self.register_buffer("th_betas", torch.Tensor(np.array(smpl_data["betas"].r)).unsqueeze(0))
        self.register_buffer("th_shapedirs", torch.Tensor(np.array(smpl_data["shapedirs"].r)))
        self.register_buffer("th_posedirs", torch.Tensor(np.array(smpl_data["posedirs"].r)))
        self.register_buffer("th_v_template", torch.Tensor(np.array(smpl_data["v_template"].r)).unsqueeze(0))
        self.register_buffer("th_J_regressor", torch.Tensor(np.array(smpl_data["J_regressor"].toarray())))
        self.register_buffer("th_weights", torch.Tensor(np.array(smpl_data["weights"].r)))
        self.register_buffer("th_faces", torch.Tensor(np.array(smpl_data["f"]).astype(np.int32)).long())

        # Get hand mean
        hands_mean = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data["hands_mean"]
        hands_mean = hands_mean.copy()
        th_hands_mean = torch.Tensor(hands_mean).unsqueeze(0)
        if self.use_pca or self.joint_rot_mode == "axisang":
            # Save as axis-angle
            self.register_buffer("th_hands_mean", th_hands_mean)
            selected_components = hands_components[:ncomps]
            self.register_buffer("th_selected_comps", torch.Tensor(selected_components))
        elif self.joint_rot_mode == "rotmat":
            th_hands_mean_rotmat = rodrigues_layer.batch_rodrigues(th_hands_mean.view(15, 3)).reshape(15, 3, 3)
            self.register_buffer("th_hands_mean_rotmat", th_hands_mean_rotmat)
        elif self.joint_rot_mode == "quat":
            # TODO deal with flat hand mean
            self.register_buffer("th_hands_mean_quat", None)
        else:
            raise KeyError(
                "joint_rot_mode not found. shoule be one of 'axisang' or 'rotmat' or 'quat'. got {}".format(
                    self.joint_rot_mode
                )
            )

        # Kinematic chain params
        self.kintree_table = smpl_data["kintree_table"]
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents

        if self.side == 'left':
            self.th_shapedirs[:, 0, :] *= -1

    def forward(
        self, th_pose_coeffs, th_betas=None, th_trans=None, root_palm=None, share_betas=None,
    ):
        """
        Args:
        th_trans (Tensor (batch_size x ncomps)): if provided, applies trans to joints and vertices
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters for hand shape
        else centers on root joint (9th joint)
        root_palm: return palm as hand root instead of wrist
        """
        # device transger
        device = th_pose_coeffs.device
        if th_betas is None:
            th_betas = torch.zeros(1).to(device)
        if th_trans is None:
            th_trans = torch.zeros(1).to(device)
        if root_palm is None:
            root_palm = torch.Tensor([0]).to(device)
        if share_betas is None:
            share_betas = torch.Tensor([0]).to(device)

        # if len(th_pose_coeffs) == 0:
        #     return th_pose_coeffs.new_empty(0), th_pose_coeffs.new_empty(0)

        batch_size = th_pose_coeffs.shape[0]
        # Get axis angle from PCA components and coefficients
        if self.use_pca or self.joint_rot_mode == "axisang":
            # Remove global rot coeffs
            th_hand_pose_coeffs = th_pose_coeffs[:, self.rot : self.rot + self.ncomps]
            if self.use_pca:
                # PCA components --> axis angles
                th_full_hand_pose = th_hand_pose_coeffs.mm(self.th_selected_comps)
            else:
                th_full_hand_pose = th_hand_pose_coeffs

            # Concatenate back global rot
            th_full_pose = torch.cat([th_pose_coeffs[:, : self.rot], self.th_hands_mean + th_full_hand_pose], 1)
            if self.root_rot_mode == "axisang":
                # compute rotation matrixes from axis-angle while skipping global rotation
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
                root_rot = th_rot_map[:, :9].view(batch_size, 3, 3)
                th_rot_map = th_rot_map[:, 9:]
                th_pose_map = th_pose_map[:, 9:]
            else:
                # th_posemap offsets by 3, so add offset or 3 to get to self.rot=6
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose[:, 6:])
                if self.robust_rot:
                    root_rot = rot6d.robust_compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
                else:
                    root_rot = rot6d.compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
        elif self.joint_rot_mode == "rotmat":
            assert (
                th_pose_coeffs.dim() == 4
            ), "When using rot mode 'rotmat', " "th_pose_coeffs should have 4 dims, got {}".format(th_pose_coeffs.dim())
            assert th_pose_coeffs.shape[2:4] == (3, 3), (
                "When using rot mode 'rotmat', th_pose_coeffs have 3x3 matrix for two"
                "last dims, got {}".format(th_pose_coeffs.shape[2:4])
            )
            th_full_pose = th_pose_coeffs  # ! Dummy Assignment
            th_pose_rots = rotproj.batch_rotprojs(th_pose_coeffs)
            th_rot_map = th_pose_rots[:, 1:].view(batch_size, -1)
            th_pose_map = subtract_flat_id(th_rot_map)
            root_rot = th_pose_rots[:, 0]
        elif self.joint_rot_mode == "quat":
            # we need th_rot_map, th_pose_map, root_rot
            # though do no assertion
            # th_pose_coeffs should be [B, 4 + 15 * 4] = [B, 64]
            th_full_pose = th_pose_coeffs  # ! Dummy Assignment
            batch_size = th_pose_coeffs.shape[0]
            th_pose_coeffs = th_pose_coeffs.view((batch_size, 16, 4))  # [B. 16, 4]
            th_pose_coeffs_ = th_pose_coeffs.clone()
            if self.side == 'left':
                th_pose_coeffs_[:, :, 2] = -th_pose_coeffs_[:, :, 2]
                th_pose_coeffs_[:, :, 3] = -th_pose_coeffs_[:, :, 3]
            all_rots = quaternion_to_rotation_matrix(th_pose_coeffs_)  # [B, 16, 3, 3]
            # flatten things out
            root_rot = all_rots[:, 0, :, :]  # [B, 3, 3]
            th_rot_map = all_rots[:, 1:, :].view((batch_size, -1))  # [B, 15 * 9]
            th_pose_map = subtract_flat_id(th_rot_map)
        else:
            raise KeyError(
                "joint_rot_mode not found. shoule be one of 'axisang' or 'rotmat' or 'quat'. got {}".format(
                    self.joint_rot_mode
                )
            )

        # Full axis angle representation with root joint
        if th_betas is None or th_betas.numel() == 1:
            th_v_shaped = (
                torch.matmul(self.th_shapedirs, self.th_betas.transpose(1, 0)).permute(2, 0, 1) + self.th_v_template
            )
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(batch_size, 1, 1)

        else:
            if share_betas:
                th_betas = th_betas.mean(0, keepdim=True).expand(th_betas.shape[0], 10)
            th_v_shaped = torch.matmul(self.th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)
            # th_pose_map should have shape 20x135

        th_v_posed = th_v_shaped + torch.matmul(self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        # Final T pose with transformation done !

        # Global rigid transformation

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        root_trans = th_with_zeros(torch.cat([root_rot, root_j], 2))

        all_rots = th_rot_map.view(th_rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = th_j[:, lev1_idxs]
        lev2_j = th_j[:, lev2_idxs]
        lev3_j = th_j[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_trans.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_trans.unsqueeze(1).repeat(1, 5, 1, 1).view(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(all_rots.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        th_results = torch.cat(all_transforms, 1)[:, reorder_idxs]
        th_results_global = th_results

        joint_js = torch.cat([th_j, th_j.new_zeros(th_j.shape[0], 16, 1)], 2)
        tmp2 = torch.matmul(th_results, joint_js.unsqueeze(3))
        th_results2 = (th_results - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).permute(0, 2, 3, 1)

        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))

        th_rest_shape_h = torch.cat(
            [
                th_v_posed.transpose(2, 1),
                torch.ones((batch_size, 1, th_v_posed.shape[1]), dtype=th_T.dtype, device=th_T.device),
            ],
            1,
        )

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = th_results_global[:, :, :3, 3]
        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        if self.side == "right":
            tips = th_verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = th_verts[:, [745, 317, 445, 556, 673]]
        if bool(root_palm):
            palm = (th_verts[:, 95] + th_verts[:, 22]).unsqueeze(1) / 2
            th_jtr = torch.cat([palm, th_jtr[:, 1:]], 1)
        th_jtr = torch.cat([th_jtr, tips], 1)

        # Reorder joints to match visualization utilities
        th_jtr = th_jtr[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]

        if th_trans is None or bool(torch.norm(th_trans) == 0):
            if self.center_idx is not None:
                center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
            else:  # ! Dummy Center Joint (B, 1, 3)
                center_joint = torch.zeros_like(th_jtr[:, 0].unsqueeze(1))

            th_jtr = th_jtr - center_joint
            th_verts = th_verts - center_joint
        else:
            # ! SHADOW THE EFFECT OF CENTER_IDX & CENTER_JOINT
            center_joint = torch.zeros_like(th_jtr[:, 0].unsqueeze(1))
            th_jtr = th_jtr + th_trans.unsqueeze(1)
            th_verts = th_verts + th_trans.unsqueeze(1)

        global_rot = th_results_global[:, :, :3, :3]  # (B, 16, 3, 3)
        global_t = th_results_global[:, :, :3, 3:]  # (B, 16, 3, 1)
        global_t = global_t - center_joint.unsqueeze(-1) + th_trans.unsqueeze(1).unsqueeze(-1)  # (B, [16], 3, 1)
        th_transf_global = torch.cat([global_rot, global_t], dim=3)  # (B, 16, 3, 4)
        th_transf_global = th_with_zeros(th_transf_global.view(-1, 3, 4))
        th_transf_global = th_transf_global.view(batch_size, 16, 4, 4)

        # Scale to milimeters
        # th_verts = th_verts * 1000
        # th_jtr = th_jtr * 1000
        results = [th_verts, th_jtr]  # (V, J)

        if self.return_transf:
            results = results + [th_transf_global]  # (V, J, T)
            if self.return_full_pose:
                results = results + [th_full_pose]  # (V, J, T, so3)
        elif self.return_full_pose:
            results = results + [th_full_pose]  # (V, J, so3)

        return tuple(results)
