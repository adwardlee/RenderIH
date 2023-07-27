import torch
import torch.nn as nn
import torch.nn.functional as F


HAND_ANCESTOR_INDEX = [
    [],
    [0],
    [0, 1],
    [0, 1, 2],
    [0],
    [0, 4],
    [0, 4, 5],
    [0],
    [0, 7],
    [0, 7, 8],
    [0],
    [0, 10],
    [0, 10, 11],
    [0],
    [0, 13],
    [0, 13, 14]
]


class KTD(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=1024, **kwargs):
        super(KTD, self).__init__()

        self.feat_dim = feat_dim

        npose_per_joint = 6
        nshape = 10
        ncam = 3

        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout()

        self.joint_regs = nn.ModuleList()
        for joint_idx, ancestor_idx in enumerate(HAND_ANCESTOR_INDEX):
            regressor = nn.Linear(hidden_dim + npose_per_joint * len(ancestor_idx), npose_per_joint)
            nn.init.xavier_uniform_(regressor.weight, gain=0.01)
            self.joint_regs.append(regressor)

        self.decshape = nn.Linear(hidden_dim, nshape)
        self.deccam = nn.Linear(hidden_dim, ncam)
        # self.trans_pred = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Linear(hidden_dim // 2, hidden_dim // 8),
        #                                 nn.Linear(hidden_dim // 8, 3))
        # self.deccam = nn.Linear(hidden_dim, hidden_dim // 4)
        # self.deccam1 = nn.Linear(hidden_dim // 4, ncam)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

    def rot6d_to_rotmat(self, x):
        x = x.view(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        return torch.stack((b1, b2, b3), dim=-1)

    def forward(self, x, seqlen, mano_model=None,
                return_shape_cam=False, **kwargs):
        nt = x.shape[0]
        N = nt // seqlen

        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        pred_shape = self.decshape(x)
        pred_cam = self.deccam(x)
        # pred_trans = self.trans_pred(x)
        # pred_cam = self.deccam1(pred_cam)
        # pred_cam = self.deccam(x)

        pose = []
        for ancestor_idx, reg in zip(HAND_ANCESTOR_INDEX, self.joint_regs):
            ances = torch.cat([x] + [pose[i] for i in ancestor_idx], dim=1)
            pose.append(reg(ances))

        pred_pose = torch.cat(pose, dim=1)

        if return_shape_cam:
            return pred_shape, pred_cam
        output_regress = self.get_output(pred_pose, pred_shape, pred_cam, mano_model)

        return output_regress

    def get_output(self, pred_pose, pred_shape, pred_cam, mano_model):
        output = {}

        nt = pred_pose.shape[0]
        pred_rotmat = rot6d_to_rotmat(pred_pose).reshape(nt, -1, 3, 3)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(nt, -1)
        pred_vertices, pred_joints = mano_model(root_rotation=pred_rotmat[:, 0], pose=pose[:, 3:], shape=pred_shape)
        pred_vertices = pred_vertices[:nt]
        pred_joints = pred_joints[:nt]

        pred_keypoints_2d = projection(pred_joints, pred_cam)
        # pred_keypoints_2d = orthographic_projection(pred_joints, pred_cam)

        output['theta'] = torch.cat([pred_cam, pose, pred_shape], dim=1)
        output['verts'] = pred_vertices
        output['kp_2d'] = pred_keypoints_2d
        output['kp_3d'] = pred_joints
        output['rotmat'] = pred_rotmat

        return output
