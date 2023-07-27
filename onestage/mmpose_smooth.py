from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F

from onestage.mmpose.one_euro_filter import OneEuroFilter
from onestage.mmpose_utils import Bbox
from onestage.mmpose.converter import axis2euler, euler2axis

def mat2quat(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector"""
    def convert_points_to_homogeneous(points):
        if not torch.is_tensor(points):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                type(points)))
        if len(points.shape) < 2:
            raise ValueError("Input must be at least a 2D tensor. Got {}".format(
                points.shape))

        return F.pad(points, (0, 1), "constant", 1.0)

    if rotation_matrix.shape[1:] == (3, 3):
        rotation_matrix = convert_points_to_homogeneous(rotation_matrix)

    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

def aa2rotmat(axis):
    # axis : bs * 3
    # return: bs * 3 * 3
    bs = axis.shape[0]
    Imat = torch.eye(3, dtype=axis.dtype, device=axis.device).repeat(bs, 1, 1)  # bs * 3 * 3
    angle = torch.norm(axis, p=2, dim=1, keepdim=True) + 1e-8  # bs * 1
    axes = axis / angle  # bs * 3
    sin = torch.sin(angle).unsqueeze(2)  # bs * 1 * 1
    cos = torch.cos(angle).unsqueeze(2)  # bs * 1 * 1
    L = torch.zeros((bs, 3, 3), dtype=axis.dtype, device=axis.device)
    L[:, 2, 1] = axes[:, 0]
    L[:, 1, 2] = -axes[:, 0]
    L[:, 0, 2] = axes[:, 1]
    L[:, 2, 0] = -axes[:, 1]
    L[:, 1, 0] = axes[:, 2]
    L[:, 0, 1] = -axes[:, 2]
    return Imat + sin * L + (1 - cos) * L.bmm(L)

def quat2aa(quaternion):
    """Convert quaternion vector to angle axis of rotation."""
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

def euler2quat(euler):
    batch = len(euler)
    yaw = euler[:, 0]
    pitch = euler[:, 1]
    roll = euler[:, 2]
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    output = torch.cat((w.reshape(batch, -1), x.reshape(batch, -1), y.reshape(batch, -1), z.reshape(batch, -1)), dim=1)
    return output

def quat2euler(qua):
    batch = len(qua)
    L = (qua[:, 0] ** 2 + qua[:, 1] ** 2 + qua[:, 2] ** 2 + qua[:, 3] ** 2) ** 0.5
    w = qua[:,0] / L
    x = qua[:,1] / L
    y = qua[:,2] / L
    z = qua[:,3] / L
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    # print("*")
    # print(w * y - z * x)
    # temp = w * y - z * x
    # if temp >= 0.5:
    #     temp = 0.5
    # elif temp <= -0.5:
    #     temp = -0.5
    # else:
    #     pass
    pitch = 2 * torch.atan2((1 + 2 * (w * y - x * z)), (1 - 2 * (w * y - x * z))) - torch.pi / 2
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    euler = torch.cat((yaw.reshape(batch, -1), pitch.reshape(batch, -1), roll.reshape(batch,-1)), dim=1)
    return euler

def euler2axisangle(euler):
    quat = euler2quat(euler)
    aa = quat2aa(quat)
    aa[torch.isnan(aa)] = 0.0
    return aa

def axisangle2euler(aa):### b, 3*n
    '''
    out: [yaw, pitch, roll]
    '''
    rotmat = aa2rotmat(aa)
    quat = mat2quat(rotmat)
    euler = quat2euler(quat)
    return euler

@dataclass
class SmoothConfig:
    keypoints2d_beta: float = 5
    keypoints2d_min_cutoff: float = 1e-4
    keypoints2d_enable: bool = False

    keypoints3d_beta: float = 5
    keypoints3d_min_cutoff: float = 1e-4
    keypoints3d_enable: bool = False

    global_orient_beta: float = 0.15
    global_orient_min_cutoff: float = 1e-3
    global_orient_enable: bool = True

    poses_beta: float = 0.05
    poses_min_cutoff: float = 1e-3
    poses_enable: bool = True

    shape_beta: float = 0.5
    shape_min_cutoff: float = 1e-6
    shape_enable: bool = True

    global_translation_beta: float = 20
    global_translation_min_cutoff: float = 1e-5
    global_translation_enable: bool = False

    focals_beta: float = 1
    focals_min_cutoff: float = 1e-4
    focals_enable: bool = False

    camera_scale: float = 0.5
    camera_scale_min_cutoff: float = 1e-6
    camera_scale_enable: bool = True

    camera_tran: float = 1
    camera_tran_min_cutoff: float = 1e-5
    camera_tran_enable: bool = True

    vertices_beta: float = 3
    vertices_min_cutoff: float = 1e-4
    vertices_enable: bool = False

    scalelength_beta: float = 0.0001
    scalelength_min_cutoff: float = 1e-6
    scalelength_enable: bool = True

    rightrel_beta: float = 0.6
    rightrel_min_cutoff: float = 1e-6
    rightrel_enable: bool = True

@dataclass
class Hand3dResult:
    bbox: Bbox = None
    keypoints2d: torch.tensor = None
    keypoints3d: torch.tensor = None
    hand_type: np.ndarray = None
    gesture_prob:  np.ndarray = None
    global_orient: torch.tensor = None
    poses: torch.tensor = None
    betas: torch.tensor = None
    camera_scale: torch.tensor = None
    camera_tran: torch.tensor = None
    focals: np.ndarray = None
    principle_point: np.ndarray = None
    vertices: torch.tensor = None
    mask: np.ndarray = None
    scalelength: torch.tensor=None
    rightrel: torch.tensor=None




def warp_result(data, bbox, left, top, image_width, image_height):
    # mano parameters
    global_orient = data.global_orient.cpu().detach().numpy().reshape(3)
    poses = data.poses.cpu().detach().numpy().reshape(45)
    betas = data.betas.cpu().detach().numpy().reshape(10)
    global_translation = data.global_translation.cpu().detach().numpy().reshape(3)




    # keypoints2d = data.keypoints2d.cpu().detach().numpy().reshape(21, 2)
    keypoints2d = data.keypoints2d_projected.cpu().detach().numpy().reshape(21, 2)
    keypoints2d = keypoints2d + np.array([left, top])
    keypoints3d = data.keypoints3d.cpu().detach().numpy().reshape(21, 3)
    keypoints3d = keypoints3d + np.array([left, top, 0])
    # hand_type = torch.softmax(data.hand_type, dim=-1)
    hand_type = data.hand_type.cpu().detach().numpy().reshape(-1)
    # gesture = data.gesture.cpu().detach().numpy()
    gesture = None

    return Hand3dResult(
        keypoints2d=keypoints2d,
        keypoints3d=keypoints3d,
        hand_type=hand_type,
        global_orient=global_orient,
        poses=poses,
        betas=betas,
        global_translation=global_translation,
        vertices=data.vertices,
    )

@dataclass
class HandTrackerConfig:
    # detection parameters
    dt_input_size: int = 160
    dt_vertical: bool = False
    dt_model_path: str = None
    dt_threshold: float = 0.5

    # keypoints parameters
    kp_input_size: int = 224
    kp_model_path: str = None
    kp_mano_path: str = None
    kp_debug: bool = False #1008
    kp_threshold: float = 0.4
    kp_palm_distance: bool = True

    # tracking parameters
    tk_detect_interval: int = 5
    tk_iou_threshold: float = 0.0
    tk_min_size: int = 100  #100
    tk_min_hits: int = 1
    tk_max_age: int = 5
    tk_smooth_config: SmoothConfig = SmoothConfig()

    debug: bool = True #1008

class SmoothCallback:
    def __init__(self, config: SmoothConfig):
        self.filter_keypoints2d = None
        self.filter_keypoints3d = None
        self.filter_global_orient = None
        self.filter_poses = None
        self.filter_shape = None
        self.filter_global_translation = None
        self.filter_focals = None
        self.filter_vertices = None
        self.filter_scalelength = None
        self.filter_rightrel = None

        if config.keypoints2d_enable:
            self.filter_keypoints2d = OneEuroFilter(beta=config.keypoints2d_beta, min_cutoff=config.keypoints2d_min_cutoff)

        if config.keypoints3d_enable:
            self.filter_keypoints3d = OneEuroFilter(beta=config.keypoints3d_beta, min_cutoff=config.keypoints3d_min_cutoff)

        if config.global_orient_enable:
            self.filter_global_orient = OneEuroFilter(beta=config.global_orient_beta, min_cutoff=config.global_orient_min_cutoff)

        if config.poses_enable:
            self.filter_poses = OneEuroFilter(beta=config.poses_beta, min_cutoff=config.poses_min_cutoff)

        if config.shape_enable:
            self.filter_shape = OneEuroFilter(beta=config.shape_beta, min_cutoff=config.shape_beta)

        if config.global_translation_enable:
            self.filter_global_translation = OneEuroFilter(beta=config.global_translation_beta,
                                                           min_cutoff=config.global_translation_min_cutoff)

        if config.camera_scale_enable:
            self.filter_scale = OneEuroFilter(beta=config.camera_scale, min_cutoff=config.camera_scale_min_cutoff)
        if config.camera_tran_enable:
            self.filter_tran = OneEuroFilter(beta=config.camera_tran, min_cutoff=config.camera_tran_min_cutoff)

        if config.focals_enable:
            self.filter_focals = OneEuroFilter(beta=config.focals_beta,
                                                           min_cutoff=config.focals_min_cutoff)

        if config.vertices_enable:
            self.filter_vertices = OneEuroFilter(beta=config.vertices_beta,
                                                           min_cutoff=config.vertices_min_cutoff)
        if config.scalelength_enable:
            self.filter_scalelength = OneEuroFilter(beta=config.scalelength_beta,
                                                           min_cutoff=config.scalelength_min_cutoff)

        if config.rightrel_enable:
            self.filter_rightrel = OneEuroFilter(beta=config.rightrel_beta,
                                                           min_cutoff=config.rightrel_min_cutoff)

    def __call__(self, result, result_prev=None):
        device = result.global_orient.device
        if result_prev is None:
            return result
        wh = np.array([result.bbox.x2 - result.bbox.x1, result.bbox.y2 - result.bbox.y1])

        if self.filter_keypoints2d is not None:
            result.keypoints2d = self.filter_keypoints2d(result.keypoints2d / wh,
                                                                   result_prev.keypoints2d / wh) * wh


        if self.filter_keypoints3d is not None:
            result.keypoints3d = self.filter_keypoints3d(result.keypoints3d / wh,
                                                                   result_prev.keypoints3d / wh) * wh

        if self.filter_vertices is not None:
            cur_vertices = self.filter_vertices(result.vertices.cpu().numpy(),
                                                                   result_prev.vertices.cpu().numpy())
            result.vertices = torch.tensor((cur_vertices)).to(device)

        if self.filter_global_orient is not None:
            global_orient = result.global_orient
            prev_global_orient = result_prev.global_orient
            cur_global = self.filter_global_orient(global_orient.cpu().numpy(), prev_global_orient.cpu().numpy())
            result.global_orient = torch.tensor((cur_global)).to(device)
            # result.hand_info.global_orient = self.filter_global_orient(result.hand_info.global_orient,
            #                                                        result_prev.hand_info.global_orient)

        if self.filter_poses is not None:
            poses = result.poses
            prev_poses = result_prev.poses
            cur_pose = self.filter_poses(poses.cpu().numpy(), prev_poses.cpu().numpy())
            result.poses = torch.tensor((cur_pose)).to(device)
            # result.hand_info.poses = self.filter_poses(result.hand_info.poses,
            #                                                        result_prev.hand_info.poses)


        if self.filter_shape is not None:
            betas = result.betas
            prev_betas = result_prev.betas
            out_betas = self.filter_shape(betas.cpu().numpy(), prev_betas.cpu().numpy())
            result.betas = torch.tensor(out_betas).to(device)
            # result.hand_info.betas = self.filter_shape(result.hand_info.betas,
            #                                                        result_prev.hand_info.betas)

        if self.filter_scale is not None:
            scale = result.camera_scale
            prev_scale = result_prev.camera_scale
            out_scale = self.filter_scale(scale.cpu().numpy(), prev_scale.cpu().numpy())
            result.camera_scale = torch.tensor(out_scale).to(device)

        if self.filter_tran is not None:
            tran = result.camera_tran
            prev_tran = result_prev.camera_tran
            out_tran = self.filter_tran(tran.cpu().numpy(), prev_tran.cpu().numpy())
            result.camera_tran = torch.tensor(out_tran).to(device)

        if self.filter_global_translation is not None:
            # result.hand_info.focals /= result.hand_info.global_translation[2]
            result.global_translation = self.filter_global_translation(
                result.global_translation, result_prev.global_translation)

        if self.filter_scalelength is not None:
            scale = result.scalelength
            prev_scale = result_prev.scalelength
            out_scale = self.filter_scalelength(scale.cpu().numpy(), prev_scale.cpu().numpy())
            result.scalelength = torch.tensor(out_scale).to(device)

        if self.filter_rightrel is not None and result.rightrel is not None:
            rightrel = result.rightrel
            prev_rightrel = result_prev.rightrel
            out_rightrel = self.filter_rightrel(rightrel.cpu().numpy(), prev_rightrel.cpu().numpy())
            result.rightrel = torch.tensor(out_rightrel).to(device)

        # print("focals", result.focals, result.focals / result.hand_info.global_translation[2],
        #       result.hand_info.global_translation, result.hand_info.betas)

            #np.array([result.hand_info.keypoints2d.min(0), result.hand_info.keypoints2d.max(0)], np.float32).reshape(-1)

        # todo: filter focals, global_orient, poses, betas, global_translation
        return result

if __name__=='__main__':
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    axisangle = np.array([[-0.5, -1.48, 2.28], [1, 1, 1]]).reshape(2, 3)
    tmp = R.from_rotvec(axisangle)
    aa = tmp.as_euler('xyz', degrees=True)
    tmp1 = R.from_euler('xyz', aa, degrees=True)
    tmp2 = tmp1.as_rotvec()
    # R.from_euler(aa)
    #
    # euler = axisangle2euler(axisangle)
    # output = euler2axisangle(euler)
