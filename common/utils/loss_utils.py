import torch.nn.functional as F

import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


from sdf import SDF

from common.nets.mano_head import batch_rodrigues

class NewLoss(nn.Module):

    def __init__(self, right_faces, left_faces, grid_size=32, robustifier=None):
        super(NewLoss, self).__init__()
        self.sdf = SDF()
        seg_idx = np.load('part_vert.npy', allow_pickle=True)
        seg_idx = seg_idx[()]
        right_face = np.load('all_face.npy', allow_pickle=True)
        right_face = right_face[()]
        for key in seg_idx:
            seg_idx[key] = list(seg_idx[key])
        self.seg = seg_idx
        left_face = {}
        for one in right_face.keys():
            right_face[one] = torch.tensor(right_face[one].astype(np.int32)).cuda()
            left_face[one] = right_face[one][:,[1, 0, 2]]
        self.right_face = right_face
        self.left_face = left_face
        self.right_faces = right_faces.to(torch.int32)
        self.left_faces = left_faces.to(torch.int32)
        # self.register_buffer('right_face', torch.tensor(right_faces.astype(np.int32)))
        # self.register_buffer('left_face', torch.tensor(left_faces.astype(np.int32)))
        self.grid_size = grid_size
        self.robustifier = robustifier

    @torch.no_grad()
    def get_bounding_boxes1(self, vertices):
        bs = vertices.shape[0]
        boxes = torch.zeros(bs, 2, 2, 3, device=vertices.device)
        boxes[:, :, 0, :] = vertices.min(dim=2)[0]
        boxes[:, :, 1, :] = vertices.max(dim=2)[0]
        return boxes

    @torch.no_grad()
    def get_bounding_boxes(self, vertices):
        bs = vertices.shape[0]
        boxes = torch.zeros(bs, 2, 3, device=vertices.device)
        boxes[:, 0, :] = vertices.min(dim=1)[0]
        boxes[:,  1, :] = vertices.max(dim=1)[0]
        return boxes

    # def forward(self, left_vert, right_vert, scale_factor=0.2, return_per_vert_loss=False, return_origin_scale_loss=False):
    def forward(self, vertices, scale_factor=0.1, return_per_vert_loss=False,
                    return_origin_scale_loss=False):

        # vertices: (bs, 2, 778, 3)
        left_vert = vertices[:, 1]
        right_vert = vertices[:, 0]
        bs = left_vert.shape[0]
        output_left = torch.zeros((bs, 778)).to(dtype=torch.float32, device=left_vert.device)
        output_right = torch.zeros((bs, 778)).to(dtype=torch.float32, device=left_vert.device)
        left_oriscale = torch.zeros((bs, 778)).to(dtype=torch.float32, device=left_vert.device)
        right_oriscale = torch.zeros((bs, 778)).to(dtype=torch.float32, device=left_vert.device)
        num_hand = 2
        left_boxes = []
        right_boxes = []
        left_vertices = []
        right_vertices = []
        left_phi = [[] for i in range(16)]
        right_phi = [[] for i in range(16)]
        left_box = self.get_bounding_boxes(left_vert)# (bs, 2, 3)
        right_box = self.get_bounding_boxes(right_vert)
        left_center = left_box.mean(dim=1).unsqueeze(dim=1)  # (bs, 1, 3)
        left_scale = (1 + scale_factor) * 0.5 * (left_box[:, 1] - left_box[:, 0]).max(dim=-1)[0][:, None,
                                                 None]  # (bs, 1, 1)
        right_center = right_box.mean(dim=1).unsqueeze(dim=1)  # (bs, 1, 3)
        right_scale = (1 + scale_factor) * 0.5 * (right_box[:, 1] - right_box[:, 0]).max(dim=-1)[0][:,  None,
                                                None]  # (bs, 1, 1)
        boxes = self.get_bounding_boxes1(vertices)
        boxes_center = boxes.mean(dim=2).unsqueeze(dim=2)  # (bs, 2, 1, 3)
        boxes_scale = (1 + scale_factor) * 0.5 * (boxes[:, :, 1] - boxes[:, :, 0]).max(dim=-1)[0][:, :, None,
                                                 None]  # (bs, 2, 1, 1)

        for i in range(16):
            left_vertices.append(left_vert[:, self.seg[i]])
            right_vertices.append(right_vert[:, self.seg[i]])
            left_boxes.append(self.get_bounding_boxes(left_vert[:, self.seg[i]]))
            right_boxes.append(self.get_bounding_boxes(right_vert[:, self.seg[i]]))
        left_boxes = torch.stack(left_boxes) ### 16, bs, 2, 3
        right_boxes = torch.stack(right_boxes)### 16, bs, 2, 3
        left_boxes = left_boxes.permute((1, 0, 2, 3))###  bs, 16, 2, 3
        right_boxes = right_boxes.permute((1, 0, 2, 3))###  bs, 16, 2, 3
        left_boxes_center = left_boxes.mean(dim=2).unsqueeze(dim=2)###  bs, 16, 1, 3
        left_boxes_scale = (1 + scale_factor) * 0.5 * (left_boxes[:, :, 1] - left_boxes[:, :, 0]).max(dim=-1)[0][:, :, None,
                                                 None]  # (bs, 16, 1, 1)
        right_boxes_center = right_boxes.mean(dim=2).unsqueeze(dim=2)  ###  bs, 16, 1, 3
        right_boxes_scale = (1 + scale_factor) * 0.5 * (right_boxes[:, :, 1] - right_boxes[:, :, 0]).max(dim=-1)[0][:, :,
                                                      None,
                                                      None]  # (bs, 16, 1, 1)
        with torch.no_grad():
            vertices_centered = vertices - boxes_center
            vertices_centered_scaled = vertices_centered / boxes_scale
            assert (vertices_centered_scaled.min() >= -1)
            assert (vertices_centered_scaled.max() <= 1)
            right_verts = vertices_centered_scaled[:, 0].contiguous()
            left_verts = vertices_centered_scaled[:, 1].contiguous()
            right_bigphi = self.sdf(self.right_faces, right_verts, self.grid_size)
            left_bigphi = self.sdf(self.left_faces, left_verts, self.grid_size)
            assert (right_bigphi.min() >= 0)  # (bs, 32, 32, 32)
            assert (left_bigphi.min() >= 0)  # (bs, 32, 32, 32)

        losses = list()
        losses_origin_scale = list()
        for i in range(16):
            vertices_local_left = (left_vertices[i] - boxes_center[:, 0]) / boxes_scale[:, 0]
            # vertices_grid: (bs, 778, 1, 1, 3)
            vertices_grid = vertices_local_left.view(bs, -1, 1, 1, 3)
            phi_val = nn.functional.grid_sample(
                right_bigphi.unsqueeze(dim=1), vertices_grid, align_corners=True).view(bs, -1)
            output_left[:, self.seg[i]] += phi_val
            left_oriscale[:, self.seg[i]] += phi_val * boxes_scale[:, 0, 0]

            vertices_local_right = (right_vertices[i] - boxes_center[:, 1]) / boxes_scale[:, 1]
            # vertices_grid: (bs, 778, 1, 1, 3)
            vertices_grid = vertices_local_right.view(bs, -1, 1, 1, 3)
            phi_val = nn.functional.grid_sample(
                left_bigphi.unsqueeze(dim=1), vertices_grid, align_corners=True).view(bs, -1)
            output_right[:, self.seg[i]] += phi_val
            right_oriscale[:, self.seg[i]] += phi_val * boxes_scale[:, 1, 0]

        cur_loss_bp = output_left / num_hand ** 2
        # cur_loss_os = output_left * left_scale[:, 0]
        losses.append(cur_loss_bp)
        losses_origin_scale.append(left_oriscale)

        cur_loss_bp = output_right / num_hand ** 2
        # cur_loss_os = output_right * right_scale[:, 0]
        losses.append(cur_loss_bp)
        losses_origin_scale.append(right_oriscale)

        loss_per_vert = torch.cat((losses[0], losses[1]), dim=1)
        # loss_origin_scale = torch.cat((losses_origin_scale[0], losses_origin_scale[1]), dim=1)
        loss = (losses[0] + losses[1]).sum(dim=1)


        if not return_per_vert_loss:
            return loss
        else:
            if not return_origin_scale_loss:
                return loss, losses[0], losses[1]
            else:
                return loss, loss_per_vert, losses_origin_scale

class LossCollision:
    def __init__(self, threshold=0.002, resolution=6) -> None:
        seg_idx = np.load('part_vert.npy', allow_pickle=True)
        seg_idx = seg_idx[()]
        for key in seg_idx:
            seg_idx[key] = list(seg_idx[key])
        self.verts_seg_dict = seg_idx
        self.threshold = threshold
        self.rotmat = self.generate_rotmat(resolution)

        self.collision_check_dict = {
            0: [0],#list(range(16)),
            1: [0],#list(range(16)),
            2: [0],#list(range(16)),
            3: [0],#list(range(16)),
            4: [0],#list(range(16)),
            5: [0],#list(range(16)),
            6: [0],#list(range(16)),
            7: [0],#list(range(16)),
            8: [0],#list(range(16)),
            9: [0],#list(range(16)),
            10: [0],#list(range(16)),
            11: [0],#list(range(16)),
            12: [0],#list(range(16)),
            13: [0],#list(range(16)),
            14: [0],#list(range(16)),
            15: [0],#list(range(16)),
        }

    def __call__(self, vert, **kwargs):
        vert_left = vert[:, 1]
        vert_right = vert[:, 0]
        loss_left = torch.zeros((1, 778)).to(vert.device)
        loss_right = torch.zeros((1, 778)).to(vert.device)
        count = 0
        for verts_i, verts_j_list in self.collision_check_dict.items():
            for verts_j in verts_j_list:
                loss_left[:, self.verts_seg_dict[verts_i]] += self.collision_loss(vert_left, vert_right, verts_i, verts_j)
                loss_right[:, self.verts_seg_dict[verts_i]] += self.collision_loss(vert_right, vert_left, verts_i,
                                                                                  verts_j)
                count += 1
        return loss_left, loss_right

    def collision_loss(self, vert_left, vert_right, i, j):
        verts_i = vert_left[:, self.verts_seg_dict[i]]
        verts_j = vert_right
        verts_i = torch.einsum('mij,bnj->bmni', self.rotmat.to(vert_left.device), verts_i)
        verts_j = torch.einsum('mij,bnj->bmni', self.rotmat.to(vert_left.device), verts_j)

        # bounding_max_min_x = torch.max(torch.min(verts_i[:, :, :, 0], -1)[0], torch.min(verts_j[:, :, :, 0], -1)[0])
        # bounding_min_max_x = torch.min(torch.max(verts_i[:, :, :, 0], -1)[0], torch.max(verts_j[:, :, :, 0], -1)[0])
        # loss = torch.sum((bounding_min_max_x - bounding_max_min_x - self.threshold)[(bounding_min_max_x - bounding_max_min_x - self.threshold > 0).all(-1)])
        # if loss != 0:
        #     loss /= (bounding_min_max_x - bounding_max_min_x - self.threshold > 0).all(-1).sum()

        # loss = self._sdf(verts_i, verts_j)

        max_indices = torch.max(verts_i[:, :, :, 0], -1)[0] < torch.max(verts_j[:, :, :, 0], -1)[0]
        mask = torch.ones_like(max_indices).unsqueeze(-1)
        mask[max_indices] = 0

        loss_j_max2i = torch.max(verts_j[:, :, :, 0], -1)[0].unsqueeze(-1) - verts_i[:, :, :, 0] - self.threshold
        loss_i2j_min = verts_i[:, :, :, 0] - torch.min(verts_j[:, :, :, 0], -1)[0].unsqueeze(-1) - self.threshold

        j_max2i = (loss_j_max2i > 0)### b, 216, 778
        i2j_min = (loss_i2j_min > 0)
        collision_indices = j_max2i.clone()
        collision_indices[max_indices] = i2j_min[max_indices]
        collision_indices = collision_indices.permute(0, 2, 1).all(-1)

        loss = loss_j_max2i * mask + loss_i2j_min * ~mask
        loss = loss.permute(0, 2, 1)

        loss = torch.sum(loss * collision_indices.unsqueeze(-1), dim=-1)
        # loss = torch.sum(loss[collision_indices]) / loss.shape[-1]

        return loss

    def generate_rotmat(self, resolution):
        rotmat = torch.zeros(resolution ** 3, 3, 3)

        def rot_x(i):
            theta = i * np.pi / (resolution * 1)
            return torch.tensor([[1., 0., 0.],
                                 [0., np.cos(theta), -np.sin(theta)],
                                 [0., np.sin(theta), np.cos(theta)]])

        def rot_y(i):
            theta = i * np.pi / (resolution * 1)
            return torch.tensor([[np.cos(theta), 0., np.sin(theta)],
                                 [0., 1., 0.],
                                 [-np.sin(theta), 0., np.cos(theta)]])

        def rot_z(i):
            theta = i * np.pi / (resolution * 1)
            return torch.tensor([[np.cos(theta), -np.sin(theta), 0.],
                                 [np.sin(theta), np.cos(theta), 0.],
                                 [0., 0., 1.]])

        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    rotmat[i * (resolution ** 2) + j * resolution + k] = torch.mm(torch.mm(rot_x(i), rot_y(j)),
                                                                                  rot_z(k))
        return rotmat

    def __str__(self) -> str:
        return 'Loss function for collision'


class SDFLoss(nn.Module):

    def __init__(self, right_faces, left_faces, grid_size=32, robustifier=None):
        super(SDFLoss, self).__init__()
        self.sdf = SDF()
        self.right_faces = right_faces.to(torch.int32)
        self.left_faces = left_faces.to(torch.int32)
        # self.register_buffer('right_face', torch.tensor(right_faces.astype(np.int32)))
        # self.register_buffer('left_face', torch.tensor(left_faces.astype(np.int32)))
        self.grid_size = grid_size
        self.robustifier = robustifier

    @torch.no_grad()
    def get_bounding_boxes(self, vertices):
        bs = vertices.shape[0]
        boxes = torch.zeros(bs, 2, 2, 3, device=vertices.device)
        boxes[:, :, 0, :] = vertices.min(dim=2)[0]
        boxes[:, :, 1, :] = vertices.max(dim=2)[0]
        return boxes

    def forward(self, vertices, scale_factor=0.2, return_per_vert_loss=False, return_origin_scale_loss=False):
        # assert not (return_origin_scale_loss and (not return_per_vert_loss))

        # vertices: (bs, 2, 778, 3)
        bs = vertices.shape[0]
        num_hand = 2
        boxes = self.get_bounding_boxes(vertices)  # (bs, 2, 2, 3)
        loss = torch.tensor(0., device=vertices.device)

        # re-scale the input vertices
        boxes_center = boxes.mean(dim=2).unsqueeze(dim=2)  # (bs, 2, 1, 3)
        boxes_scale = (1 + scale_factor) * 0.5 * (boxes[:, :, 1] - boxes[:, :, 0]).max(dim=-1)[0][:, :, None,
                                                 None]  # (bs, 2, 1, 1)

        with torch.no_grad():
            vertices_centered = vertices - boxes_center
            vertices_centered_scaled = vertices_centered / boxes_scale
            assert (vertices_centered_scaled.min() >= -1)
            assert (vertices_centered_scaled.max() <= 1)
            right_verts = vertices_centered_scaled[:, 0].contiguous()
            left_verts = vertices_centered_scaled[:, 1].contiguous()
            right_phi = self.sdf(self.right_faces, right_verts, self.grid_size)
            left_phi = self.sdf(self.left_faces, left_verts, self.grid_size)
            assert (right_phi.min() >= 0)  # (bs, 32, 32, 32)
            assert (left_phi.min() >= 0)  # (bs, 32, 32, 32)

        # concat left & right phi
        # be aware of the order, input vertices the order is right, left
        phi = [right_phi, left_phi]
        losses = list()
        losses_origin_scale = list()

        for i in [0, 1]:
            # vertices_local: (bs, 1, 778, 3)
            vertices_local = (vertices[:, i:i + 1] - boxes_center[:, 1 - i].unsqueeze(dim=1)) / boxes_scale[:,
                                                                                                i].unsqueeze(dim=1)
            # vertices_grid: (bs, 778, 1, 1, 3)
            vertices_grid = vertices_local.view(bs, -1, 1, 1, 3)
            # Sample from the phi grid
            phi_val = nn.functional.grid_sample(
                phi[1 - i].unsqueeze(dim=1), vertices_grid, align_corners=True).view(bs, -1)
            cur_loss = phi_val  # (10, 778)

            cur_loss_bp = cur_loss / num_hand ** 2
            cur_loss_os = cur_loss * boxes_scale[:, i, 0]
            losses.append(cur_loss_bp)
            losses_origin_scale.append(cur_loss_os)

        loss = (losses[0] + losses[1])
        loss = loss.sum(dim=1)
        loss_per_vert = torch.cat((losses[0], losses[1]), dim=1)
        loss_origin_scale = torch.cat((losses_origin_scale[0], losses_origin_scale[1]), dim=1)

        if not return_per_vert_loss:
            return loss.mean()
        else:
            if not return_origin_scale_loss:
                return loss, losses[0], losses[1]
            else:
                return loss, loss_per_vert, loss_origin_scale


def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwargs):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den

def dice_loss(pred,
              target,
              valid_mask,
              smooth=1,
              exponent=2,
              class_weight=None,
              ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i],
                target[:, i],
                valid_mask=valid_mask,#[:, i],
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes


class DiceLoss(nn.Module):
    """DiceLoss.
    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 smooth=1e-3,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_dice',
                 **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        #pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        one_hot_target = one_hot_target.permute(0, 3, 1, 2)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


class TryLoss(nn.Module):

    def __init__(self, grid_size=32, robustifier=None):
        super(TryLoss, self).__init__()
        self.sdf = SDF()
        seg_idx = np.load('part_vert.npy', allow_pickle=True)
        seg_idx = seg_idx[()]
        right_face = np.load('all_face.npy', allow_pickle=True)
        right_face = right_face[()]
        for key in seg_idx:
            seg_idx[key] = list(seg_idx[key])
        self.seg = seg_idx
        left_face = {}
        for one in right_face.keys():
            right_face[one] = torch.tensor(right_face[one].astype(np.int32)).cuda()
            left_face[one] = right_face[one][:,[1, 0, 2]]
        self.right_face = right_face
        self.left_face = left_face
        # self.register_buffer('right_face', torch.tensor(right_faces.astype(np.int32)))
        # self.register_buffer('left_face', torch.tensor(left_faces.astype(np.int32)))
        self.grid_size = grid_size
        self.robustifier = robustifier

    @torch.no_grad()
    def get_bounding_boxes(self, vertices):
        bs = vertices.shape[0]
        boxes = torch.zeros(bs, 2, 3, device=vertices.device)
        boxes[:, 0, :] = vertices.min(dim=1)[0]
        boxes[:,  1, :] = vertices.max(dim=1)[0]
        return boxes

    # def forward(self, left_vert, right_vert, scale_factor=0.2, return_per_vert_loss=False, return_origin_scale_loss=False):
    def forward(self, vertices, scale_factor=0.1, return_per_vert_loss=False,
                    return_origin_scale_loss=False):

        # vertices: (bs, 2, 778, 3)
        left_vert = vertices[:, 1]
        right_vert = vertices[:, 0]
        bs = left_vert.shape[0]
        output_left = torch.zeros((bs, 778)).to(dtype=torch.float32, device=left_vert.device)
        output_right = torch.zeros((bs, 778)).to(dtype=torch.float32, device=left_vert.device)
        left_oriscale = torch.zeros((bs, 778)).to(dtype=torch.float32, device=left_vert.device)
        right_oriscale = torch.zeros((bs, 778)).to(dtype=torch.float32, device=left_vert.device)
        num_hand = 2
        left_boxes = []
        right_boxes = []
        left_vertices = []
        right_vertices = []
        left_phi = [[] for i in range(16)]
        right_phi = [[] for i in range(16)]
        left_box = self.get_bounding_boxes(left_vert)# (bs, 2, 3)
        right_box = self.get_bounding_boxes(right_vert)
        # left_center = left_box.mean(dim=1).unsqueeze(dim=1)  # (bs, 1, 3)
        # left_scale = (1 + scale_factor) * 0.5 * (left_box[:, 1] - left_box[:, 0]).max(dim=-1)[0][:, None,
        #                                          None]  # (bs, 1, 1)
        # right_center = right_box.mean(dim=1).unsqueeze(dim=1)  # (bs, 1, 3)
        # right_scale = (1 + scale_factor) * 0.5 * (right_box[:, 1] - right_box[:, 0]).max(dim=-1)[0][:,  None,
        #                                         None]  # (bs, 1, 1)

        for i in range(16):
            left_vertices.append(left_vert[:, self.seg[i]])
            right_vertices.append(right_vert[:, self.seg[i]])
            left_boxes.append(self.get_bounding_boxes(left_vert[:, self.seg[i]]))
            right_boxes.append(self.get_bounding_boxes(right_vert[:, self.seg[i]]))
        left_boxes = torch.stack(left_boxes) ### 16, bs, 2, 3
        right_boxes = torch.stack(right_boxes)### 16, bs, 2, 3
        left_boxes = left_boxes.permute((1, 0, 2, 3))###  bs, 16, 2, 3
        right_boxes = right_boxes.permute((1, 0, 2, 3))###  bs, 16, 2, 3
        left_boxes_center = left_boxes.mean(dim=2).unsqueeze(dim=2)###  bs, 16, 1, 3
        left_boxes_scale = (1 + scale_factor) * 0.5 * (left_boxes[:, :, 1] - left_boxes[:, :, 0]).max(dim=-1)[0][:, :, None,
                                                 None]  # (bs, 16, 1, 1)
        right_boxes_center = right_boxes.mean(dim=2).unsqueeze(dim=2)  ###  bs, 16, 1, 3
        right_boxes_scale = (1 + scale_factor) * 0.5 * (right_boxes[:, :, 1] - right_boxes[:, :, 0]).max(dim=-1)[0][:, :,
                                                      None,
                                                      None]  # (bs, 16, 1, 1)
        with torch.no_grad():
            for i in range(16):
                vertices_centered = left_vertices[i] - left_boxes_center[:, i]
                vertices_centered_scaled = vertices_centered / left_boxes_scale[:, i]
                vertices_centered_scaled = vertices_centered_scaled.contiguous()#.to(torch.float32)
                assert (vertices_centered_scaled.min() >= -1)
                assert (vertices_centered_scaled.max() <= 1)
                left_phi[i] = self.sdf(self.left_face[i], vertices_centered_scaled, self.grid_size)

                vertices_centered = right_vertices[i] - right_boxes_center[:, i]
                vertices_centered_scaled = vertices_centered / right_boxes_scale[:, i]
                vertices_centered_scaled = vertices_centered_scaled.contiguous()#.to(torch.float32)
                assert (vertices_centered_scaled.min() >= -1)
                assert (vertices_centered_scaled.max() <= 1)
                right_phi[i] = self.sdf(self.right_face[i], vertices_centered_scaled, self.grid_size)

                assert (right_phi[i].min() >= 0)  # (bs, 32, 32, 32)
                assert (left_phi[i].min() >= 0)  # (bs, 32, 32, 32)

        losses = list()
        losses_origin_scale = list()
        for i in range(16):
            for j in range(16):
                vertices_local_left = (left_vertices[i] - right_boxes_center[:, j]) / right_boxes_scale[:,j]
                # vertices_grid: (bs, 778, 1, 1, 3)
                vertices_grid = vertices_local_left.view(bs, -1, 1, 1, 3)
                phi_val = nn.functional.grid_sample(
                    right_phi[j].unsqueeze(dim=1), vertices_grid, align_corners=True).view(bs, -1)
                output_left[:, self.seg[i]] += phi_val
                left_oriscale[:, self.seg[i]] += phi_val * right_boxes_scale[:, j, 0]

                vertices_local_right = (right_vertices[i] - left_boxes_center[:, j]) / left_boxes_scale[:,j]
                # vertices_grid: (bs, 778, 1, 1, 3)
                vertices_grid = vertices_local_right.view(bs, -1, 1, 1, 3)
                phi_val = nn.functional.grid_sample(
                    left_phi[j].unsqueeze(dim=1), vertices_grid, align_corners=True).view(bs, -1)
                output_right[:, self.seg[i]] += phi_val
                right_oriscale[:, self.seg[i]] += phi_val * left_boxes_scale[:, j, 0]

        cur_loss_bp = output_left / num_hand ** 2
        # cur_loss_os = output_left * left_scale[:, 0]
        losses.append(cur_loss_bp)
        losses_origin_scale.append(left_oriscale)

        cur_loss_bp = output_right / num_hand ** 2
        # cur_loss_os = output_right * right_scale[:, 0]
        losses.append(cur_loss_bp)
        losses_origin_scale.append(right_oriscale)

        loss_per_vert = torch.cat((losses[0], losses[1]), dim=1)
        # loss_origin_scale = torch.cat((losses_origin_scale[0], losses_origin_scale[1]), dim=1)
        loss = (losses[0] + losses[1]).sum(dim=1)


        if not return_per_vert_loss:
            return loss
        else:
            if not return_origin_scale_loss:
                return loss, losses[0], losses[1]
            else:
                return loss, loss_per_vert, losses_origin_scale

class LossUtil(object):

    def __init__(self, opt, mano_models):
        self.inputSize = opt.inputSize
        self.pose_params_dim = opt.pose_params_dim // 2  # pose_params_dim for two hand
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.use_hand_rotation = opt.use_hand_rotation
        else:
            self.use_hand_rotation = False
        self.batch_size = opt.batchSize
        """
        if self.isTrain:
            self.shape_reg_loss_format = opt.shape_reg_loss_format
        else:
            self.shape_reg_loss_format = 'l2'
        """

        faces_right = mano_models['right'].faces
        faces_left = mano_models['left'].faces
        robustifier = opt.sdf_robustifier if self.isTrain else None
        assert robustifier is None or robustifier > 0.0
        self.sdf_loss = SDFLoss(faces_right, faces_left, robustifier=robustifier).cuda()

    def _hand_type_loss(self, gt_hand_type, pred_hand_type, hand_type_valid):
        loss = F.binary_cross_entropy(pred_hand_type, gt_hand_type, reduction='none')
        loss = loss * hand_type_valid
        return torch.mean(loss)

    def _mano_pose_loss(self, mano_pose, pred_mano_pose, mano_params_weight):
        # change pose parameters to rodrigues matrix
        pose_dim = pred_mano_pose.size(1)
        assert pose_dim in [45, 48]

        pose_rodrigues = batch_rodrigues(mano_pose.contiguous().view(-1, 3)).view(
            self.batch_size, pose_dim // 3, 3, 3)

        pred_pose_rodrigues = batch_rodrigues(pred_mano_pose.contiguous().view(-1, 3)).view( \
            self.batch_size, pose_dim // 3, 3, 3)

        if not self.use_hand_rotation and pose_dim == 48:  # pose-params contain global orient
            pose_params = pose_rodrigues[:, 1:, :, :].view(self.batch_size, -1)
            pred_pose_params = pred_pose_rodrigues[:, 1:, :, :].view(self.batch_size, -1)
        else:
            pose_params = pose_rodrigues.view(self.batch_size, -1)
            pred_pose_params = pred_pose_rodrigues.view(self.batch_size, -1)

        # square loss
        params_diff = pose_params - pred_pose_params
        square_loss = torch.mul(params_diff, params_diff)
        square_loss = square_loss * mano_params_weight
        loss = torch.mean(square_loss)

        return loss

    def _mano_shape_loss(self, mano_shape, pred_mano_shape, mano_params_weight):
        # abs loss
        shape_diff = torch.abs(mano_shape - pred_mano_shape)
        abs_loss = shape_diff * mano_params_weight
        loss = torch.mean(abs_loss)
        return loss

    def _joints_2d_loss(self, gt_keypoint, pred_keypoint, keypoint_weights):
        abs_loss = torch.abs((gt_keypoint - pred_keypoint))
        weighted_loss = abs_loss * keypoint_weights
        loss_batch = weighted_loss.reshape(self.batch_size, -1).mean(1)
        loss = torch.mean(weighted_loss)
        return loss, loss_batch

    def __align_by_root(self, joints_3d, joints_3d_weight):
        # right
        has_right_idxs = joints_3d_weight[:, 0, 0] > 0.5  # has right wrist
        joints_3d[has_right_idxs, :, :] = \
            joints_3d[has_right_idxs, :, :] - joints_3d[has_right_idxs, 0:1, :]
        # left
        no_right_idxs = joints_3d_weight[:, 0, 0] < 1e-7  # has no right wrist
        joints_3d[no_right_idxs, :, :] = \
            joints_3d[no_right_idxs, :, :] - joints_3d[no_right_idxs, 21:22, :]

    def _joints_3d_loss(self, gt_joints_3d, pred_joints_3d, joints_3d_weight):
        # align the root by default
        self.__align_by_root(gt_joints_3d, joints_3d_weight)
        self.__align_by_root(pred_joints_3d, joints_3d_weight)

        # calc squared loss
        joints_diff = gt_joints_3d - pred_joints_3d
        square_loss = torch.mul(joints_diff, joints_diff)
        square_loss = square_loss * joints_3d_weight
        loss_batch = square_loss.reshape(self.batch_size, -1).mean(1)
        loss = torch.mean(square_loss)
        return loss, loss_batch

    def _hand_trans_loss(self, gt_hand_trans, pred_hand_trans, hand_trans_weight):
        diff = gt_hand_trans - pred_hand_trans
        square_loss = diff * diff * hand_trans_weight
        loss = torch.mean(square_loss)
        return loss

    def _shape_reg_loss(self, shape_params):
        right_hand_shape = shape_params[:, :10]
        left_hand_shape = shape_params[:, 10:]
        diff = right_hand_shape - left_hand_shape
        losses = diff * diff  # l2
        loss_batch = losses.reshape(self.batch_size, -1).mean(1)
        loss = torch.mean(losses)
        return loss

    def _shape_residual_loss(self, pred_shape_params, init_shape_params):
        diff = pred_shape_params - init_shape_params
        loss = torch.abs(diff)
        loss = torch.mean(loss)
        return loss

    def _finger_reg_loss(self, joints_3d):
        joint_idxs = [
            [1, 2, 3, 17],  # index
            [4, 5, 6, 18, ],  # middle
            [7, 8, 9, 20, ],  # little
            [10, 11, 12, 19],  # ring
            [13, 14, 15, 16],  # thumb
        ]
        joint_idxs = np.concatenate(np.array(joint_idxs))
        joint_idxs = np.concatenate([joint_idxs, (joint_idxs + 21)])
        joint_idxs = torch.from_numpy(joint_idxs).long().cuda()

        bs = joints_3d.size(0)
        joints_3d = joints_3d[:, joint_idxs, :]
        joints_3d = joints_3d.view(bs, 10, 4, 3)
        joints_3d = joints_3d.view(bs * 10, 4, 3)

        fingers = torch.zeros(bs * 10, 3, 3).float()
        for i in range(3):
            fingers[:, i, :] = joints_3d[:, i, :] - joints_3d[:, i + 1, :]

        cross_value1 = torch.cross(fingers[:, 0, :], fingers[:, 1, :], dim=1)
        C1 = torch.sum(fingers[:, 2, :] * cross_value1, dim=1)

        cross_value2 = torch.cross(fingers[:, 1, :], fingers[:, 2, :], dim=1)
        C2 = torch.sum(cross_value1 * cross_value2, dim=1)
        zero_pad = torch.zeros(C2.size()).float()
        loss = torch.abs(C1) - torch.min(zero_pad, C2)
        loss = loss.view(bs, 10)

        loss_batch = torch.sum(loss, dim=1)
        loss = torch.mean(loss_batch)

        return loss, loss_batch

    def _collision_loss(self, right_hand_verts, left_hand_verts, hand_type_array):
        bs = self.batch_size

        right_hand_verts = right_hand_verts.unsqueeze(dim=1)  # (bs, 1, 778, 3)
        left_hand_verts = left_hand_verts.unsqueeze(dim=1)  # (bs, 1, 778, 3)
        hand_verts = torch.cat([right_hand_verts, left_hand_verts], dim=1)

        losses, _, losses_origin_scale = self.sdf_loss(
            hand_verts, return_per_vert_loss=True, return_origin_scale_loss=True)
        losses = losses.reshape(bs, 1)

        # weights
        weights = torch.sum(hand_type_array, dim=1) > 1.5
        weights = weights.type(torch.float32).view(bs, 1)
        losses = losses * weights
        # losses_origin_scale = losses_origin_scale * weights

        loss = torch.mean(losses)
        loss_batch = losses.view(-1, )
        return loss, loss_batch, losses_origin_scale
