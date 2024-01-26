import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from sdf import SDF

def batch_rodrigues(theta):
    # theta N x 3
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)
	
class NewLoss(nn.Module):

    def __init__(self, grid_size=32, robustifier=None):
        super(NewLoss, self).__init__()
        self.sdf = SDF()
        seg_idx = np.load('code_sdf/part_vert.npy', allow_pickle=True)
        seg_idx = seg_idx[()]

        for key in seg_idx:
            seg_idx[key] = list(seg_idx[key])
        self.seg = seg_idx

        right_face = np.load('code_sdf/right.npy', allow_pickle=True)
        left_face = np.load('code_sdf/right.npy', allow_pickle=True)

        self.right_faces = torch.tensor(right_face, dtype=torch.int32).to('cuda')
        self.left_faces = torch.tensor(left_face, dtype=torch.int32).to('cuda')
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
if __name__ == '__main__':
    NewLoss()