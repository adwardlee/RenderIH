from pprint import pprint

import numpy as np
import open3d as o3d
import torch
from manopth.anchorlayer import AnchorLayer
from manopth.axislayer import AxisLayer
from manopth.manolayer import ManoLayer
from manopth.quatutils import normalize_quaternion, quaternion_to_angle_axis
from manopth.rodrigues_layer import batch_rodrigues
from termcolor import colored
from tqdm import trange

from hocontact.postprocess.geo_loss import FieldLoss, ObjectLoss, HandLoss
from hocontact.utils import netutils
from hocontact.visualize.vis_contact_info import create_vertex_color
from scripts.HandPoseConverter import HandPoseConverter
from code_sdf.sdf_template import NewLoss
from Ver2Code.Discriminator.discrim import Pos2dDiscriminator
from torch.autograd import Variable
shape_dim = 10
class GeOptimizer:
    def __init__(
        self,
        device,
        lr=1e-2,
        n_iter=2500,
        verbose=False,
        mano_root="assets/mano",
        anchor_path="assets/anchor",
        fhb=False,
        load_fhb_path="assets/mano/fhb_skel_centeridx9.pkl",
        compensate_tsl=False,
        # values to initialize coef_val
        lambda_contact_loss=10.0,
        lambda_repulsion_loss=0.5,
        repulsion_query=0.030,
        repulsion_threshold=0.080,
            mode='both'
    ):
        self.mode = mode
        self.device = device
        self.lr = lr
        self.n_iter = n_iter

        self.anti_penetration_loss_fun = NewLoss()

        # options
        self.verbose = verbose
        self.runtime_vis = None
        self.compensate_tsl = compensate_tsl

        # layers and loss utils
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
        if mode == 'both':
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

        self.anchor_layer = AnchorLayer(anchor_path).to(self.device)
        self.axis_layer = AxisLayer().to(self.device)

        # opt val dict, const val dict
        self.opt_val = {}
        self.const_val = {}
        self.ctrl_val = {}
        self.coef_val = {
            "lambda_contact_loss": lambda_contact_loss,
            "lambda_repulsion_loss": lambda_repulsion_loss,
            "repulsion_query": repulsion_query,
            "repulsion_threshold": repulsion_threshold,
        }

        # creating slots for optimizer and scheduler
        self.optimizer = None
        self.optimizing = True
        self.scheduler = None

        # get hand init axis pose
        self.rhpc = HandPoseConverter(side='right', root='assets/mano', data_type='tensor')
        zero_vec_pose = torch.from_numpy(np.asarray([[1.0, 0.0, 0.0, 0.0]] * 16, dtype=np.float32)).unsqueeze(0).to(self.device)
        self.zero_mat_ = self.rhpc.mano_quat_2_mat_tensor(zero_vec_pose)

        if mode == 'both':
            self.lhpc = HandPoseConverter(side='left', root='assets/mano', data_type='tensor')
            self.sub_zero_mat_ = self.lhpc.mano_quat_2_mat_tensor(zero_vec_pose)

        self.disc = Pos2dDiscriminator(num_joints=15, hid_dim=512, dropout=0.05).to(self.device).eval()
        dis_state_dict = torch.load('Ver2Code/Discriminator/discrim.pth')
        print(self.disc.load_state_dict(dis_state_dict), flush=True)

    def NatureLoss(self, right_pose, left_pose):
        # right_pose, left_pose B*15*4
        bs = right_pose.size(0)
        right_pose_flattern = right_pose.view(bs, -1)
        left_pose_flattern = left_pose.view(bs, -1)
        out = self.disc(right_pose_flattern)
        sub_out = self.disc(left_pose_flattern)

        real_lable = Variable(torch.cat((torch.zeros((bs, 1)), torch.ones((bs, 1))), dim=1)).to(self.device)

        nature_loss = torch.tensor(0.0).to(self.device)

        nature_mask = out[:, 1] < 1.5 * out[:, 0]
        if nature_mask.sum() > 0:
            nature_loss = torch.nn.functional.binary_cross_entropy(out[nature_mask], real_lable[nature_mask])

        nature_mask = sub_out[:, 1] < 1.5 * sub_out[:, 0]
        if nature_mask.sum() > 0:
            nature_loss += torch.nn.functional.binary_cross_entropy(sub_out[nature_mask], real_lable[nature_mask])
        #
        # nature_loss = torch.nn.functional.binary_cross_entropy(out, real_lable)
        # nature_loss += torch.nn.functional.binary_cross_entropy(sub_out, real_lable)
        return nature_loss.mean()
    def set_opt_val(
        self,
        # static val
        vertex_contact,  # TENSOR[NVERT, ] {0, 1}
        contact_region,  # TENSOR[NVERT, 1], int
        anchor_id,  # TENSOR[NVERT, 4]: int
        anchor_elasti,  # TENSOR[NVERT, 4]
        anchor_padding_mask,  # TENSOR[NVERT, 4] {0, 1}
        hand_region_assignment=None,  # TENSOR[NHANDVERT, ]
        hand_palm_vertex_mask=None,  # TENSOR[NHANDVERT, ] {0, 1}
        # dynamic val: hand
        hand_shape_gt=None,  # TENSOR[10, ]
        hand_tsl_gt=None,  # TENSOR[3, ]
        hand_pose_gt=None,  # (LIST[NPROV, ]: int {0..16}, TENSOR[NPROV, 4])
        hand_shape_init=None,  # TENSOR[10, ]
        hand_tsl_init=None,  # TENSOR[3, ]
        hand_pose_init=None,  # (LIST[NPROV, ]: int {0..16}, TENSOR[NPROV, 4])
        # dynamic val: obj
        # hand tsl compensate
        hand_compensate_root=None,  # TENSOR[3, ]
        # runtime viz
        runtime_vis=None,
        obj_anchors=None, # if the target is sub hand, this is the anchors. if the target is a obj, this is the full vertice
        obj_normals=None,

        obj_tsl_gt=None,  # TENSOR[3, ]
        obj_pose_gt=None,  # (LIST[NPROV, ]: int {0..16}, TENSOR[NPROV, 4])
        obj_tsl_init=None,  # TENSOR[3, ]
        obj_pose_init=None,

        optimize_it=None, # B * 1
        batch_size=None,
        consistent_mask=None
    ):
        self.zero_mat = self.zero_mat_.repeat((batch_size, 1, 1, 1))
        if self.mode == 'both':
            self.sub_zero_mat = self.sub_zero_mat_.repeat((batch_size, 1, 1, 1))
        # ====== clear memory
        self.opt_val = {}
        self.const_val = {'batch_size':batch_size,
                          'optimize_it':optimize_it,
                          'consistent_mask':consistent_mask}
        self.ctrl_val = {
            "optimize_hand_shape": False,
            "optimize_hand_tsl": False,
            "optimize_hand_pose": False,
            "optimize_obj": False,
            "fhb": False,
        }

        # ============ process static values >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        vertex_contact = vertex_contact.long()
        anchor_id = anchor_id.long()
        anchor_padding_mask = anchor_padding_mask.long()

        # # boolean index contact_region, anchor_id, anchor_elasti && anchor_padding_mask
        # # obj_contact_region = contact_region[vertex_contact == 1]  # TENSOR[NCONT, ]
        # anchor_id = anchor_id[vertex_contact == 1, :]  # TENSOR[NCONT, 4]
        # anchor_elasti = anchor_elasti[vertex_contact == 1, :]  # TENSOR[NCONT, 4]
        # anchor_padding_mask = anchor_padding_mask[vertex_contact == 1, :]  # TENSOR[NCONT, 4]
        #
        # # boolean mask indexing anchor_id, anchor_elasti && obj_vert_id
        # indexed_anchor_id = anchor_id[anchor_padding_mask == 1]  # TENSOR[NVALID, ]
        # self.const_val["indexed_anchor_id"] = indexed_anchor_id
        # self.const_val["indexed_anchor_elasti"] = anchor_elasti[anchor_padding_mask == 1]  # TENSOR[NVALID, ]
        #
        # vertex_id = torch.arange(anchor_id.shape[0])[:, None].repeat_interleave(
        #     anchor_padding_mask.shape[1], dim=1
        # )  # TENSOR[NCONT, 4]
        # self.const_val["indexed_vertex_id"] = vertex_id[anchor_padding_mask == 1]  # TENSOR[NVALID, ]

        self.const_val['anchor_padding_mask'] = anchor_padding_mask
        self.const_val['anchor_elastic'] = anchor_elasti
        self.const_val['anchor_id'] = anchor_id
        self.const_val['indexed_id'] = torch.arange(anchor_id.shape[1])[:,None].repeat_interleave(anchor_id.shape[2], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1)


        # tip_anchor_mask = torch.zeros(indexed_anchor_id.shape[0]).bool().to(self.device)
        # tip_anchor_list = [2, 3, 4, 9, 10, 11, 15, 16, 17, 22, 23, 24, 29, 30, 31]
        # for tip_anchor_id in tip_anchor_list:
        #     tip_anchor_mask = tip_anchor_mask | (self.const_val["indexed_anchor_id"] == tip_anchor_id)
        # self.const_val["indexed_elasti_k"] = torch.where(
        #     tip_anchor_mask, torch.Tensor([1.0]).to(self.device), torch.Tensor([1.0]).to(self.device)
        # ).to(self.device)

        # hand faces & edges
        self.const_val["hand_faces"] = self.mano_layer.th_faces
        self.const_val["static_verts"] = self.get_static_hand_verts()
        self.const_val["hand_edges"] = HandLoss.get_edge_idx(self.const_val["hand_faces"])
        self.const_val["static_edge_len"] = HandLoss.get_edge_len(
            self.const_val["static_verts"], self.const_val["hand_edges"]
        )
        if self.mode == 'both':
            self.const_val["sub_hand_faces"] = self.sub_mano_layer.th_faces
            self.const_val["static_sub_verts"] = self.get_static_sub_hand_verts()
            self.const_val["sub_hand_edges"] = HandLoss.get_edge_idx(self.const_val["hand_faces"])
            self.const_val["static_sub_edge_len"] = HandLoss.get_edge_len(
                self.const_val["static_sub_verts"], self.const_val["sub_hand_edges"]
        )
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============ dynamic val: hand >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        # ====== hand_shape
        if hand_shape_gt is not None and hand_shape_init is not None:
            raise RuntimeError("both hand_shape gt and init are provided")
        elif hand_shape_gt is not None and hand_shape_init is None:
            self.const_val["hand_shape_gt"] = hand_shape_gt
            self.ctrl_val["optimize_hand_shape"] = False
        elif hand_shape_gt is None and hand_shape_init is not None:  # here here here here
            self.opt_val["hand_shape_var"] = hand_shape_init.detach().clone().requires_grad_(True)
            self.const_val["hand_shape_init"] = hand_shape_init
            self.const_val["hand_shape_gt"] = hand_shape_init
            self.ctrl_val["optimize_hand_shape"] = False
        else:
            # hand_tsl_gt is None and hand_tsl_init is None:
            self.opt_val["hand_shape_var"] = torch.zeros(10, dtype=torch.float, requires_grad=True, device=self.device)
            self.const_val["hand_shape_init"] = torch.zeros(10, dtype=torch.float, device=self.device)
            self.ctrl_val["optimize_hand_shape"] = True

        # ====== hand_tsl
        if hand_tsl_gt is not None and hand_tsl_init is not None:
            raise RuntimeError("both hand_tsl gt and init are provided")
        elif hand_tsl_gt is not None and hand_tsl_init is None:
            self.const_val["hand_tsl_gt"] = hand_tsl_gt
            self.ctrl_val["optimize_hand_tsl"] = False
        elif hand_tsl_gt is None and hand_tsl_init is not None:  # here here here here
            self.opt_val["hand_tsl_var"] = hand_tsl_init.detach().clone().requires_grad_(True)
            self.const_val["hand_tsl_init"] = hand_tsl_init
            self.ctrl_val["optimize_hand_tsl"] = True
        else:
            # hand_tsl_gt is None and hand_tsl_init is None:
            self.opt_val["hand_tsl_var"] = torch.zeros(3, dtype=torch.float, requires_grad=True, device=self.device)
            self.const_val["hand_tsl_init"] = torch.zeros(3, dtype=torch.float, device=self.device)
            self.ctrl_val["optimize_hand_tsl"] = True

        ### ====== obj_tsl
        if self.mode == 'both':
            self.opt_val["obj_tsl_var"] = obj_tsl_init.detach().clone().requires_grad_(True)
            self.const_val["obj_tsl_init"] = obj_tsl_init


        # ====== hand pose
        # this is complex! need special care
        if hand_pose_gt is not None and hand_pose_init is not None: # here here here here
            # full gt and init provided
            gt_pose_idx, gt_pose_val = hand_pose_gt
            init_pose_idx, init_pose_val = hand_pose_init
            if len(set(gt_pose_idx).intersection(set(init_pose_idx))) > 0:
                raise RuntimeError("repeat hand_pose gt & init provided")
            if set(gt_pose_idx).union(set(init_pose_idx)) != set(range(16)):
                raise RuntimeError("hand_pose: not enough gt & init")
            self.const_val["hand_pose_gt_idx"] = gt_pose_idx
            self.const_val["hand_pose_gt_val"] = gt_pose_val
            self.const_val["hand_pose_var_idx"] = init_pose_idx
            self.opt_val["hand_pose_var_val"] = init_pose_val.detach().clone().requires_grad_(True)
            self.const_val["hand_pose_init_val"] = init_pose_val
            self.ctrl_val["optimize_hand_pose"] = True
        elif hand_pose_gt is not None and hand_pose_init is None:
            gt_pose_idx, gt_pose_val = hand_pose_gt
            self.const_val["hand_pose_gt_idx"] = gt_pose_idx
            self.const_val["hand_pose_gt_val"] = gt_pose_val
            if set(gt_pose_idx) == set(range(16)):
                # full gt provided
                self.const_val["hand_pose_var_idx"] = []
                self.opt_val["hand_pose_var_val"] = torch.zeros((0, 4), dtype=torch.float, device=self.device)
                self.ctrl_val["optimize_hand_pose"] = False
            else:
                # partial gt provided
                var_pose_idx = self.get_var_pose_idx(gt_pose_idx)
                n_var_pose = len(var_pose_idx)
                init_val = np.array([[0.9999, 0.0, -0.0101, 0.0]] * n_var_pose).astype(np.float32)
                self.const_val["hand_pose_var_idx"] = var_pose_idx
                self.opt_val["hand_pose_var_val"] = torch.tensor(
                    init_val, dtype=torch.float, requires_grad=True, device=self.device
                )
                init_val_true = np.array([[1.0, 0.0, 0.0, 0.0]] * n_var_pose).astype(np.float32)
                self.const_val["hand_pose_init_val"] = torch.tensor(
                    init_val_true, dtype=torch.float, device=self.device
                )
                self.ctrl_val["optimize_hand_pose"] = True
        elif hand_pose_gt is None and hand_pose_init is not None:
            # full init provided
            init_pose_idx, init_pose_val = hand_pose_init
            if set(init_pose_idx) != set(range(16)):
                raise RuntimeError("hand_pose: not enough init")
            self.const_val["hand_pose_gt_idx"] = []
            self.const_val["hand_pose_gt_val"] = torch.zeros((0, 4), dtype=torch.float).to(self.device)
            self.const_val["hand_pose_var_idx"] = init_pose_idx
            self.opt_val["hand_pose_var_val"] = init_pose_val.detach().clone().requires_grad_(True)
            self.const_val["hand_pose_init_val"] = init_pose_val
            self.ctrl_val["optimize_hand_pose"] = True
        else:
            # hand_pose_gt is None and hand_pose_init is None:
            # nothing provided
            self.const_val["hand_pose_gt_idx"] = []
            self.const_val["hand_pose_gt_val"] = torch.zeros((0, 4), dtype=torch.float).to(self.device)
            self.const_val["hand_pose_var_idx"] = list(range(16))
            n_var_pose = 16
            init_val = np.array([[0.9999, 0.0, -0.0101, 0.0]] * n_var_pose).astype(np.float32)
            self.opt_val["hand_pose_var_val"] = torch.tensor(
                init_val, dtype=torch.float, requires_grad=True, device=self.device
            )
            init_val_true = np.array([[1.0, 0.0, 0.0, 0.0]] * n_var_pose).astype(np.float32)
            self.const_val["hand_pose_init_val"] = torch.tensor(init_val_true, dtype=torch.float, device=self.device)
            self.ctrl_val["optimize_hand_pose"] = True

        ### ====== obj_pose
        if self.mode == 'both':
            gt_pose_idx, gt_pose_val = obj_pose_gt
            init_pose_idx, init_pose_val = obj_pose_init
            if len(set(gt_pose_idx).intersection(set(init_pose_idx))) > 0:
                raise RuntimeError("repeat obj_pose gt & init provided")
            if set(gt_pose_idx).union(set(init_pose_idx)) != set(range(16)):
                raise RuntimeError("obj_pose: not enough gt & init")
            self.const_val["obj_pose_gt_idx"] = gt_pose_idx
            self.const_val["obj_pose_gt_val"] = gt_pose_val
            self.const_val["obj_pose_var_idx"] = init_pose_idx
            self.opt_val["obj_pose_var_val"] = init_pose_val.detach().clone().requires_grad_(True)
            self.const_val["obj_pose_init_val"] = init_pose_val

        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.ctrl_val["optimize_obj"] = False
        self.const_val["obj_verts_3d_gt"] = obj_anchors[vertex_contact == 1, :]
        self.const_val['vertex_contact'] = vertex_contact
        self.const_val["full_obj_verts_3d"] = obj_anchors
        self.const_val["full_obj_normals"] = obj_normals
        # # ============ dynamic val: obj >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # # region
        # if obj_verts_3d_gt is not None and obj_verts_3d_can is not None:
        #     raise RuntimeError("optimize obj mode conflict: both provided")
        # elif obj_verts_3d_gt is None and obj_verts_3d_can is None:
        #     raise RuntimeError("optimize obj mode conflict: neither proided")
        # elif obj_verts_3d_gt is not None and obj_verts_3d_can is None:
        #     self.const_val["obj_verts_3d_gt"] = obj_verts_3d_gt[vertex_contact == 1, :]
        #     self.const_val["obj_normals_gt"] = obj_normals_gt[vertex_contact == 1, :]
        #     self.ctrl_val["optimize_obj"] = False
        #     self.const_val["full_obj_verts_3d"] = obj_verts_3d_gt
        #     self.const_val["full_obj_normals"] = obj_normals_gt
        # else:
        #     # obj_verts_3d_gt is None and obj_verts_3d_can is not None
        #     self.const_val["obj_verts_3d_can"] = obj_verts_3d_can[vertex_contact == 1, :]
        #     self.const_val["obj_normals_can"] = obj_normals_can[vertex_contact == 1, :]
        #     self.ctrl_val["optimize_obj"] = True
        #     self.const_val["full_obj_verts_3d"] = obj_verts_3d_can
        #     self.const_val["full_obj_normals"] = obj_normals_can
        #     # check if init value is provided
        #     if obj_rot_init is None and obj_tsl_init is None:
        #         # both not provided
        #         self.opt_val["obj_rot_var"] = torch.tensor(
        #             [0.001, 0.001, 0.001], dtype=torch.float, requires_grad=True, device=self.device
        #         )
        #         self.opt_val["obj_tsl_var"] = torch.zeros(3, dtype=torch.float, requires_grad=True, device=self.device)
        #         self.const_val["obj_rot_init"] = torch.tensor(
        #             [0.000, 0.000, 0.000], dtype=torch.float, device=self.device
        #         )
        #         self.const_val["obj_tsl_init"] = torch.zeros(3, dtype=torch.float, device=self.device)
        #     elif obj_rot_init is not None and obj_tsl_init is not None:
        #         self.opt_val["obj_rot_var"] = obj_rot_init.detach().clone().requires_grad_(True)
        #         self.opt_val["obj_tsl_var"] = obj_tsl_init.detach().clone().requires_grad_(True)
        #         self.const_val["obj_rot_init"] = obj_rot_init
        #         self.const_val["obj_tsl_init"] = obj_tsl_init
        #     else:
        #         raise RuntimeError("incomplete init for optimize obj")
        # # endregion
        # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============ construct optimizer & scheduler >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        # ====== optimizer
        if (
            self.ctrl_val["optimize_hand_shape"]
            or self.ctrl_val["optimize_hand_tsl"]
            or self.ctrl_val["optimize_hand_pose"]
            or self.ctrl_val["optimize_obj"]
        ):
            # dispatch lr to different param
            param = []
            if self.ctrl_val["optimize_hand_shape"]:
                param.append({"params": [self.opt_val["hand_shape_var"]]})
            if self.ctrl_val["optimize_hand_tsl"]:
                param.append({"params": [self.opt_val["hand_tsl_var"]], "lr": 0.01 * self.lr}) # 0.25
                if self.mode == 'both':
                    param.append({"params": [self.opt_val["obj_tsl_var"]], "lr": 0.01 * self.lr})
            if self.ctrl_val["optimize_hand_pose"]:
                param.append({"params": [self.opt_val["hand_pose_var_val"]], "lr": 1 * self.lr})
                if self.mode == 'both':
                    param.append({"params": [self.opt_val["obj_pose_var_val"]], "lr": 1 * self.lr})
            if self.ctrl_val["optimize_obj"]:
                param.append({"params": [self.opt_val["obj_rot_var"]]})
                param.append({"params": [self.opt_val["obj_tsl_var"]], "lr": 0.1 * self.lr})

            self.optimizer = torch.optim.Adam(param, lr=self.lr)
            self.optimizing = True
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, min_lr=1e-5, mode="min", factor=0.5, patience=20, verbose=False
            )
        else:
            self.optimizing = False
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ====== runtime viz
        self.runtime_vis = runtime_vis

        # ====== verbose
        if self.verbose:
            print("Optimizing: ", self.optimizing)
            pprint(self.ctrl_val)
            pprint(list(self.opt_val.keys()))
            pprint(list(self.const_val.keys()))
            pprint(self.coef_val)

    @staticmethod
    def get_var_pose_idx(sel_pose_idx):
        # gt has 16 pose
        all_pose_idx = set(range(16))
        sel_pose_idx_set = set(sel_pose_idx)
        var_pose_idx = all_pose_idx.difference(sel_pose_idx_set)
        return list(var_pose_idx)

    def get_static_hand_verts(self):
        init_val_pose = np.array([[1.0, 0.0, 0.0, 0.0]] * 16).astype(np.float32)
        vec_pose = torch.tensor(init_val_pose).reshape(-1).unsqueeze(0).float().to(self.device)
        vec_shape = torch.zeros(1, 10).float().to(self.device)
        v, j, t, _ = self.mano_layer(vec_pose, vec_shape)
        v = v.squeeze(0)
        return v
    def get_static_sub_hand_verts(self):
        init_val_pose = np.array([[1.0, 0.0, 0.0, 0.0]] * 16).astype(np.float32)
        vec_pose = torch.tensor(init_val_pose).reshape(-1).unsqueeze(0).float().to(self.device)
        vec_shape = torch.zeros(1, 10).float().to(self.device)
        v, j, t, _ = self.sub_mano_layer(vec_pose, vec_shape)
        v = v.squeeze(0)
        return v

    @staticmethod
    def assemble_pose_vec(gt_idx, gt_pose, var_idx, var_pose):
        idx_tensor = torch.cat((torch.Tensor(gt_idx).long(), torch.Tensor(var_idx).long()))
        pose_tensor = torch.cat((gt_pose, var_pose), dim=0)
        pose_tensor = pose_tensor[torch.argsort(idx_tensor)]
        return pose_tensor

    @staticmethod
    def assemble_batch_pose_vec(gt_idx, gt_pose, var_idx, var_pose):
        # batch_size = gt_pose.shape[0]
        idx_tensor = torch.cat((torch.Tensor(gt_idx).long(), torch.Tensor(var_idx).long()))
        pose_tensor = torch.cat((gt_pose, var_pose), dim=1)
        pose_tensor = pose_tensor[:, torch.argsort(idx_tensor), :]
        return pose_tensor

    @staticmethod
    def transf_vectors(vectors, tsl, rot):
        """
        vectors: [K, 3], tsl: [3, ], rot: [3, ]
        return: [K, 3]
        """
        rot_matrix = batch_rodrigues(rot.unsqueeze(0)).squeeze(0).reshape((3, 3))
        vec = (rot_matrix @ vectors.T).T
        vec = vec + tsl
        return vec

    def loss_fn(self, opt_val, const_val, ctrl_val, coef_val):
        bts = const_val['batch_size']
        dim = const_val['anchor_id'].shape[2]
        var_hand_pose_assembled = self.assemble_batch_pose_vec(
            const_val["hand_pose_gt_idx"],
            const_val["hand_pose_gt_val"],
            const_val["hand_pose_var_idx"],
            opt_val["hand_pose_var_val"],
        )
        if self.mode == 'both':
            var_sub_hand_pose_assembled = self.assemble_batch_pose_vec(
                const_val["obj_pose_gt_idx"],
                const_val["obj_pose_gt_val"],
                const_val["obj_pose_var_idx"],
                opt_val["obj_pose_var_val"],
            )
        else:
            var_sub_hand_pose_assembled = None
        self.opt_val['optimized_hand_pose'] = var_hand_pose_assembled
        self.opt_val['optimized_sub_hand_pose'] = var_sub_hand_pose_assembled
        # print(opt_val["hand_pose_var_val"].shape)
        # print(opt_val["hand_pose_var_val"])
        # print(var_hand_pose_assembled)
        # dispatch hand var

        vec_pose = var_hand_pose_assembled


        if ctrl_val["optimize_hand_shape"]:
            vec_shape = opt_val["hand_shape_var"].unsqueeze(0)
        else:
            # vec_shape = const_val["hand_shape_gt"].unsqueeze(0).repeat(const_val['batch_size'],1)
            vec_shape = const_val["hand_shape_gt"]
        if ctrl_val["optimize_hand_tsl"]:
            vec_tsl = opt_val["hand_tsl_var"]
            if self.mode == 'both':
                vec_sub_tsl = opt_val["obj_tsl_var"]
        else:
            vec_tsl = const_val["hand_tsl_gt"].unsqueeze(0)

        # rebuild hand
        rebuild_verts, rebuild_joints, rebuild_transf, rebuild_full_pose = self.mano_layer(vec_pose, vec_shape[:, :shape_dim])
        if self.mode == 'both':
            vec_sub_pose = var_sub_hand_pose_assembled
            rebuild_sub_verts, rebuild_sub_joints, rebuild_sub_transf, rebuild_sub_full_pose = self.sub_mano_layer(
                vec_sub_pose, vec_shape[:, shape_dim:])
            rebuild_sub_joints = rebuild_sub_joints + vec_sub_tsl.unsqueeze(1).repeat(1, rebuild_sub_joints.shape[1], 1)
            rebuild_sub_verts = rebuild_sub_verts + vec_sub_tsl.unsqueeze(1).repeat(1, rebuild_sub_verts.shape[1], 1)
            rebuild_sub_transf = rebuild_sub_transf + torch.cat(
                [
                    torch.cat([torch.zeros(bts, 3, 3).to(self.device), vec_sub_tsl.unsqueeze(-1)], dim=2),
                    torch.zeros(bts, 1, 4).to(self.device),
                ],
                dim=1,
            ).unsqueeze(1).repeat(1, 16, 1, 1)
            rebuild_sub_verts_squeezed = rebuild_sub_verts
        # skel adaption

        rebuild_joints = rebuild_joints + vec_tsl.unsqueeze(1).repeat(1, rebuild_sub_joints.shape[1], 1)
        rebuild_verts = rebuild_verts + vec_tsl.unsqueeze(1).repeat(1, rebuild_sub_verts.shape[1], 1)
        rebuild_transf = rebuild_transf + torch.cat(
                [
                    torch.cat([torch.zeros(bts, 3, 3).to(self.device), vec_tsl.unsqueeze(-1)], dim=2),
                    torch.zeros(bts, 1, 4).to(self.device),
                ],
                dim=1,
            ).unsqueeze(1).repeat(1, 16, 1, 1)
        rebuild_verts_squeezed = rebuild_verts

        # rebuild anchor
        rebuild_anchor = self.anchor_layer(rebuild_verts)
        rebuild_anchor = rebuild_anchor.contiguous()  # TENSOR[N, 32, 3]
        # rebuild_anchor = rebuild_anchor.squeeze(0)  # TENSOR[N, 32, 3]
        # anchor_pos = rebuild_anchor[const_val["indexed_anchor_id"]]  # TENSOR[NVALID, 3]

        if self.mode == 'both':
            rebuild_sub_anchor = self.anchor_layer(rebuild_sub_verts)
            rebuild_sub_anchor = rebuild_sub_anchor.contiguous()
            # rebuild_sub_anchor = rebuild_sub_anchor.squeeze(0)
            # sub_anchor_pos = rebuild_sub_anchor[const_val["vertex_contact"] == 1, :]

        # dispatch obj var
        if ctrl_val["optimize_obj"]:
            obj_verts = self.transf_vectors(
                const_val["obj_verts_3d_can"],
                opt_val["obj_tsl_var"],
                opt_val["obj_rot_var"],
            )
            full_obj_verts = self.transf_vectors(
                const_val["full_obj_verts_3d"],
                opt_val["obj_tsl_var"],
                opt_val["obj_rot_var"],
            )
            full_obj_normals = self.transf_vectors(
                const_val["full_obj_normals"],
                torch.zeros(3, dtype=torch.float, device=self.device),
                opt_val["obj_rot_var"],
            )
        else:
            obj_verts = const_val["obj_verts_3d_gt"]
            full_obj_verts = const_val["full_obj_verts_3d"]
            full_obj_normals = const_val["full_obj_normals"]

        # contact loss
        if self.mode == 'both':
            indexed_anchor = rebuild_anchor.unsqueeze(2).repeat(1, 1, dim, 1).gather(1,
                                                                  const_val['anchor_id'].unsqueeze(-1).repeat(1, 1, 1,
                                                                                                              3))
            padding_sub_anchor = rebuild_sub_anchor.unsqueeze(2).repeat(1, 1, dim, 1)
            contact_loss = FieldLoss.batch_contact_loss(
                indexed_anchor,
                padding_sub_anchor,
                const_val["anchor_padding_mask"],
                const_val["anchor_elastic"],
            )
            fvi = self.anchor_layer.face_vert_idx.squeeze(0)
            obj_normals = torch.cross((rebuild_sub_verts_squeezed[:, fvi[:, 1], :] - rebuild_sub_verts_squeezed[:, fvi[:, 0], :]),
                                   (rebuild_sub_verts_squeezed[:, fvi[:, 2], :] - rebuild_sub_verts_squeezed[:, fvi[:, 0], :]))
            obj_normals = -obj_normals / obj_normals.norm(dim=2).unsqueeze(-1).repeat(1,1,3)
            # _obj_normals = torch.ones(rebuild_sub_verts_squeezed.shape, dtype=rebuild_sub_verts_squeezed.dtype).to(
            #     "cuda")
            # repulsion_loss = FieldLoss.batch_full_repulsion_loss(
            #     rebuild_verts_squeezed,
            #     rebuild_sub_verts_squeezed,
            #     _obj_normals,
            #     query=coef_val["repulsion_query"],
            #     threshold=coef_val["repulsion_threshold"],
            # )

            # repulsion_loss = FieldLoss.batch_full_repulsion_loss(
            #     rebuild_verts_squeezed,
            #     rebuild_sub_anchor,
            #     obj_normals,
            #     query=coef_val["repulsion_query"],
            #     threshold=coef_val["repulsion_threshold"],
            # )
            combine_verts = torch.cat((rebuild_verts_squeezed.unsqueeze(1),
                                      rebuild_sub_verts_squeezed.unsqueeze(1)), dim=1)
            anti_penetration_loss = self.anti_penetration_loss_fun(combine_verts).mean()
            # anti_penetration_loss = torch.tensor(0.0).to(self.device)
            repulsion_loss = torch.Tensor([0.0]).to(self.device)
        else:
            contact_loss = FieldLoss.contact_loss(
                anchor_pos,
                obj_verts[const_val["indexed_vertex_id"]],
                const_val["indexed_anchor_elasti"],
                const_val["indexed_elasti_k"],
            )
            repulsion_loss = FieldLoss.full_repulsion_loss(
                rebuild_verts_squeezed,
                full_obj_verts,
                full_obj_normals,
                query=coef_val["repulsion_query"],
                threshold=coef_val["repulsion_threshold"],
            )
        # repulsion loss

        # if ctrl_val["optimize_hand_pose"]:
        #     # get hand loss
        #     quat_norm_loss = HandLoss.batch_pose_quat_norm_loss(var_hand_pose_assembled)
        #     var_hand_pose_normalized = normalize_quaternion(var_hand_pose_assembled)
        #     pose_reg_loss = HandLoss.pose_reg_loss(
        #         var_hand_pose_normalized[:, const_val["hand_pose_var_idx"]], const_val["hand_pose_init_val"]
        #     )
        #
        #     b_axis, u_axis, l_axis = self.axis_layer(rebuild_joints, rebuild_transf)
        #
        #     # var_hand_pose_normalized = var_hand_pose_normalized.reshape((16, 4))
        #     angle_axis = quaternion_to_angle_axis(var_hand_pose_normalized)
        #     angle_axis = angle_axis[:, 1:, :]  # ignore global rot [15, 3]
        #     axis = angle_axis / torch.norm(angle_axis, dim=-1, keepdim=True)
        #     angle = torch.norm(angle_axis, dim=-1, keepdim=False)
        #     # limit angle
        #     angle_limit_loss = HandLoss.rotation_angle_loss(angle=angle, batch=True)
        #
        #     joint_b_axis_loss = HandLoss.joint_b_axis_loss(b_axis, axis, batch=True)
        #     joint_u_axis_loss = HandLoss.joint_u_axis_loss(u_axis, axis, batch=True)
        #     joint_l_limit_loss = HandLoss.joint_l_limit_loss(l_axis, axis, batch=True)
        #
        #     edge_loss = HandLoss.edge_len_loss(
        #         rebuild_verts_squeezed, const_val["hand_edges"], const_val["static_edge_len"]
        #     )
        #     if self.mode == 'both':
        #         quat_norm_loss += HandLoss.batch_pose_quat_norm_loss(var_sub_hand_pose_assembled)
        #         var_sub_hand_pose_normalized = normalize_quaternion(var_sub_hand_pose_assembled)
        #         pose_reg_loss += HandLoss.pose_reg_loss(
        #             var_sub_hand_pose_normalized[:, const_val["obj_pose_var_idx"]], const_val["obj_pose_init_val"]
        #         )
        #         b_axis, u_axis, l_axis = self.axis_layer(rebuild_sub_joints, rebuild_sub_transf, 'left')
        #
        #         var_sub_hand_pose_normalized[..., 2] = -var_sub_hand_pose_normalized[..., 2]
        #         var_sub_hand_pose_normalized[..., 3] = -var_sub_hand_pose_normalized[..., 3]
        #
        #         angle_axis = quaternion_to_angle_axis(var_sub_hand_pose_normalized)
        #         angle_axis = angle_axis[:, 1:, :]  # ignore global rot [15, 3]
        #         axis = angle_axis / torch.norm(angle_axis, dim=-1, keepdim=True)
        #         angle = torch.norm(angle_axis, dim=-1, keepdim=False)
        #         # limit angle
        #         angle_limit_loss += HandLoss.rotation_angle_loss(angle, batch=True)
        #
        #         joint_b_axis_loss += HandLoss.joint_b_axis_loss(b_axis, axis, batch=True)
        #         joint_u_axis_loss += HandLoss.joint_u_axis_loss(u_axis, axis, batch=True)
        #         joint_l_limit_loss += HandLoss.joint_l_limit_loss(l_axis, axis, batch=True)
        #
        #         edge_loss += HandLoss.edge_len_loss(
        #             rebuild_sub_verts_squeezed, const_val["sub_hand_edges"], const_val["static_sub_edge_len"]
        #         )
        if ctrl_val["optimize_hand_pose"]:
            # get hand loss
            quat_norm_loss = HandLoss.batch_pose_quat_norm_loss(var_hand_pose_assembled)
            var_hand_pose_normalized = normalize_quaternion(var_hand_pose_assembled)
            pose_reg_loss = HandLoss.pose_reg_loss(
                var_hand_pose_normalized[:, const_val["hand_pose_var_idx"]], const_val["hand_pose_init_val"]
            )

            b_axis, u_axis, l_axis = self.axis_layer(rebuild_joints, rebuild_transf)
            cur_joint_axis = self.rhpc.mano_quat_2_mat_tensor(var_hand_pose_normalized)

            # var_hand_pose_normalized = var_hand_pose_normalized.reshape((16, 4))
            angle_axis = quaternion_to_angle_axis(var_hand_pose_normalized)
            angle_axis = angle_axis[:, 1:, :]  # ignore global rot [15, 3]
            axis = angle_axis / torch.norm(angle_axis, dim=-1, keepdim=True)
            angle = torch.norm(angle_axis, dim=-1, keepdim=False)
            # limit angle
            angle_limit_loss = torch.Tensor([0.0]).to(self.device)

            joint_b_axis_loss = torch.Tensor([0.0]).to(self.device)
            joint_u_axis_loss = torch.Tensor([0.0]).to(self.device)
            joint_l_limit_loss = torch.Tensor([0.0]).to(self.device)

            ergonic_loss = HandLoss.hand_pose_ergonomics_loss(cur_joint_axis[:, 1:, :, :], self.zero_mat[:, 1:, :, :], 'r')
            # ergonic_loss = HandLoss.hand_pose_ergonomics_loss(var_hand_pose_normalized, b_axis, u_axis, l_axis, axis)

            edge_loss = HandLoss.edge_len_loss(
                rebuild_verts_squeezed, const_val["hand_edges"], const_val["static_edge_len"]
            )
            if self.mode == 'both':
                quat_norm_loss += HandLoss.batch_pose_quat_norm_loss(var_sub_hand_pose_assembled)
                var_sub_hand_pose_normalized = normalize_quaternion(var_sub_hand_pose_assembled)
                pose_reg_loss += HandLoss.pose_reg_loss(
                    var_sub_hand_pose_normalized[:, const_val["obj_pose_var_idx"]], const_val["obj_pose_init_val"]
                )
                # b_axis, u_axis, l_axis = self.axis_layer(rebuild_sub_joints, rebuild_sub_transf, 'left')
                #
                # var_sub_hand_pose_normalized[..., 2] = -var_sub_hand_pose_normalized[..., 2]
                # var_sub_hand_pose_normalized[..., 3] = -var_sub_hand_pose_normalized[..., 3]
                #
                # angle_axis = quaternion_to_angle_axis(var_sub_hand_pose_normalized)
                # angle_axis = angle_axis[:, 1:, :]  # ignore global rot [15, 3]
                # axis = angle_axis / torch.norm(angle_axis, dim=-1, keepdim=True)
                # angle = torch.norm(angle_axis, dim=-1, keepdim=False)

                cur_joint_axis = self.lhpc.mano_quat_2_mat_tensor(var_sub_hand_pose_normalized)
                ergonic_loss += HandLoss.hand_pose_ergonomics_loss(cur_joint_axis[:, 1:, :, :],
                                                                  self.sub_zero_mat[:, 1:, :, :], 'l')

                edge_loss += HandLoss.edge_len_loss(
                    rebuild_sub_verts_squeezed, const_val["sub_hand_edges"], const_val["static_sub_edge_len"]
                )


            ## batch consistent
            if bts >= 2:
                prev_idx = torch.arange(bts) - 2
                prev_idx = torch.clamp_min_(prev_idx, 0)
                pose_consistent_loss = HandLoss.pose_consisitent_loss(var_hand_pose_normalized,
                                                                      var_hand_pose_normalized[prev_idx],
                                                                      const_val['consistent_mask'][0])
                if self.mode == 'both':
                    pose_consistent_loss += HandLoss.pose_consisitent_loss(var_sub_hand_pose_normalized,
                                                                           var_sub_hand_pose_normalized[prev_idx],
                                                                           const_val['consistent_mask'][1])
            else:
                pose_consistent_loss = torch.Tensor([0.0]).to(self.device)
        else:
            quat_norm_loss = torch.Tensor([0.0]).to(self.device)
            pose_reg_loss = torch.Tensor([0.0]).to(self.device)
            angle_limit_loss = torch.Tensor([0.0]).to(self.device)
            joint_b_axis_loss = torch.Tensor([0.0]).to(self.device)
            joint_u_axis_loss = torch.Tensor([0.0]).to(self.device)
            joint_l_limit_loss = torch.Tensor([0.0]).to(self.device)
            edge_loss = torch.Tensor([0.0]).to(self.device)
            pose_consistent_loss = torch.Tensor([0.0]).to(self.device)
            # pose_reg_loss_to_zero = torch.Tensor([0.0]).to(self.device)

        if ctrl_val["optimize_hand_shape"]:
            shape_reg_loss = HandLoss.shape_reg_loss(opt_val["hand_shape_var"], const_val["hand_shape_init"])
        else:
            shape_reg_loss = torch.Tensor([0.0]).to(self.device)

        if ctrl_val["optimize_hand_tsl"]:
            hand_tsl_loss = HandLoss.hand_tsl_loss(opt_val["hand_tsl_var"], const_val["hand_tsl_init"])
            if self.mode == 'both':
                hand_tsl_loss += HandLoss.hand_tsl_loss(opt_val["obj_tsl_var"], const_val["obj_tsl_init"])
        else:
            hand_tsl_loss = torch.Tensor([0.0]).to(self.device)

        if ctrl_val["optimize_obj"]:
            obj_transf_loss = ObjectLoss.obj_transf_loss(
                opt_val["obj_tsl_var"], opt_val["obj_rot_var"], const_val["obj_tsl_init"], const_val["obj_rot_init"]
            )
        else:
            obj_transf_loss = torch.Tensor([0.0]).to(self.device)

        nature_loss = self.NatureLoss(var_hand_pose_normalized[:, 1:], var_sub_hand_pose_normalized[:, 1:])
        print('\r' + 'nature_loss: {}, ergonic_loss: {}, anti_penetration_loss: {}, contact_loss: {}'.format(nature_loss, ergonic_loss, anti_penetration_loss,
                                                                                                100 * coef_val["lambda_contact_loss"] * contact_loss),end="")
        loss = (
            # ============= HAND ANATOMICAL LOSS
            1.0 * quat_norm_loss
            # + 1.0 * angle_limit_loss
            + 1.0 * edge_loss
            # + 0.1 * joint_b_axis_loss * 0.1
            # + 0.1 * joint_u_axis_loss
            # + 0.05 * joint_l_limit_loss
            # ============= ELAST POTENTIONAL ENERGY
            + 1 * coef_val["lambda_contact_loss"] * contact_loss
            # + coef_val["lambda_repulsion_loss"] * repulsion_loss
            +  0.02 * coef_val["lambda_repulsion_loss"]* anti_penetration_loss # 0.02
            # ==c=========== OFFSET LOSS
            # + 10.0 * pose_reg_loss
            + 1.0 * shape_reg_loss
            + 0.0 * hand_tsl_loss #3.0
            + 0.1 * obj_transf_loss #3.0
            # + 100.0 * pose_consistent_loss
            + 1. * ergonic_loss
            + 1 * nature_loss
        )
        # debug: runtime viz
        if self.runtime_vis != 9527:
            if self.ctrl_val["optimize_obj"]:
                full_obj_verts = self.transf_vectors(
                    self.const_val["full_obj_verts_3d"][0],
                    self.opt_val["obj_tsl_var"].detach()[0],
                    self.opt_val["obj_rot_var"].detach()[0],
                )
            else:
                if self.mode == 'both':
                    full_obj_verts = rebuild_sub_verts
                else:
                    full_obj_verts = self.const_val["full_obj_verts_3d"]

            if not ctrl_val["optimize_hand_pose"]:
                b_axis, u_axis, l_axis = self.axis_layer(rebuild_joints, rebuild_transf)  # mend this up
            self.runtime_show(rebuild_verts[0], b_axis[0], u_axis[0], l_axis[0], rebuild_transf[0], full_obj_verts[0])

        return (
            loss,
            {
                "quat_norm_loss": quat_norm_loss.detach().cpu().item(),
                "angle_limit_loss": angle_limit_loss.detach().cpu().item(),
                "edge_loss": edge_loss.detach().cpu().item(),
                "joint_b_axis_loss": joint_b_axis_loss.detach().cpu().item(),
                "joint_u_axis_loss": joint_u_axis_loss.detach().cpu().item(),
                "joint_l_limit_loss": joint_l_limit_loss.detach().cpu().item(),
                "contact_loss": contact_loss.detach().cpu().item(),
                "repulsion_loss": repulsion_loss.detach().cpu().item(),
                "pose_reg_loss": pose_reg_loss.detach().cpu().item(),
                "hand_tsl_loss": hand_tsl_loss.detach().cpu().item(),
                "obj_transf_loss": obj_transf_loss.detach().cpu().item(),
            },
        )

    def optimize(self, progress=False):
        if progress:
            bar = trange(self.n_iter, position=3)
            bar_hand = trange(0, position=2, bar_format="{desc}")
            bar_contact = trange(0, position=1, bar_format="{desc}")
            bar_axis = trange(0, position=0, bar_format="{desc}")
        else:
            bar = range(self.n_iter)

        loss = torch.Tensor([1000.0]).to(self.device)
        loss_dict = {}
        for _ in bar:
            if self.optimizing:
                self.optimizer.zero_grad()
            loss, loss_dict = self.loss_fn(self.opt_val, self.const_val, self.ctrl_val, self.coef_val)

            if self.optimizing:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)
            # with torch.autograd.profiler.profile() as prof:
            #
            # print(prof.key_averages().table(sort_by='cuda_time_total'))

            if progress:
                bar.set_description("TOTAL LOSS {:4e}".format(loss.item()))
                try:
                    bar_hand.set_description(
                        colored("HAND_REGUL_LOSS: ", "yellow")
                        + "QN={:.3e} PR={:.3e} EG={:.3e}".format(
                            loss_dict["quat_norm_loss"],  # QN
                            loss_dict["pose_reg_loss"],  # PR
                            loss_dict["edge_loss"],  # Edge
                        )
                    )
                except:
                    pass
                try:
                    bar_contact.set_description(
                        colored("HO_CONTACT_LOSS: ", "blue")
                        + "Conta={:.3e}, Repul={:.3e}, OT={:.3e}".format(
                            loss_dict["contact_loss"],  # Conta
                            loss_dict["repulsion_loss"],  # Repul
                            loss_dict["obj_transf_loss"],  # OT
                        )
                    )
                except:
                    pass
                try:
                    bar_axis.set_description(
                        colored("ANGLE_LOSS: ", "cyan")
                        + "AL={:.3e} JB={:.3e} JU={:.3e} JL={:.3e}".format(
                            loss_dict["angle_limit_loss"],  # AL
                            loss_dict["joint_b_axis_loss"],  # JB
                            loss_dict["joint_u_axis_loss"],  # JU
                            loss_dict["joint_l_limit_loss"],  # JL
                        )
                    )
                except:
                    pass

        out_dict = dict(optimized_hand_pose=self.opt_val['optimized_hand_pose'].cpu().detach(),
                        optimized_hand_tsl=self.opt_val['hand_tsl_var'].cpu().detach(),
                        optimized_sub_hand_pose=None,
                        optimized_sub_hand_tsl=None
                        )
        if self.mode == 'both':
            out_dict['optimized_sub_hand_pose'] = self.opt_val['optimized_sub_hand_pose'].cpu().detach()
            out_dict['optimized_sub_hand_tsl'] = self.opt_val['obj_tsl_var'].cpu().detach()
        return out_dict
        # if self.mode == 'both':
        #     return loss.item(), loss_dict, self.opt_val['optimized_hand_pose'].cpu().detach(), self.opt_val[
        #         'hand_tsl_var'].cpu().detach(), self.opt_val['optimized_sub_hand_pose'].cpu().detach(), self.opt_val['obj_tsl_var'].cpu().detach()
        # else:
        #     return loss.item(), loss_dict, self.opt_val['optimized_hand_pose'].cpu().detach(), self.opt_val['hand_tsl_var'].cpu().detach()

    def recover_hand(self, squeeze_out=True):
        vars_hand_pose_assembled = self.assemble_pose_vec(
            self.const_val["hand_pose_gt_idx"],
            self.const_val["hand_pose_gt_val"],
            self.const_val["hand_pose_var_idx"],
            self.opt_val["hand_pose_var_val"],
        ).detach()
        vars_hand_pose_normalized = normalize_quaternion(vars_hand_pose_assembled)
        vec_pose = vars_hand_pose_normalized.unsqueeze(0)
        if self.ctrl_val["optimize_hand_shape"]:
            vec_shape = self.opt_val["hand_shape_var"].detach().unsqueeze(0)
        else:
            vec_shape = self.const_val["hand_shape_gt"].unsqueeze(0)
        if self.ctrl_val["optimize_hand_tsl"]:
            vec_tsl = self.opt_val["hand_tsl_var"].detach().unsqueeze(0)
        else:
            vec_tsl = self.const_val["hand_tsl_gt"].unsqueeze(0)

        device = vec_pose.device
        rebuild_verts, rebuild_joints, rebuild_transf, rebuild_full_pose = self.mano_layer(vec_pose, vec_shape)
        # skel adaption
        if self.ctrl_val["fhb"]:
            adapt_joints, _ = self.adaptor(rebuild_verts)
            adapt_joints = adapt_joints.transpose(1, 2)
            rebuild_joints = rebuild_joints - adapt_joints[:, 9].unsqueeze(1)
            rebuild_verts = rebuild_verts - adapt_joints[:, 9].unsqueeze(1)
        rebuild_verts = rebuild_verts + vec_tsl
        rebuild_joints = rebuild_joints + vec_tsl
        rebuild_transf = rebuild_transf + torch.cat(
            [
                torch.cat((torch.zeros((3, 3), device=device), vec_tsl.T), dim=1),
                torch.zeros((1, 4), device=device),
            ],
            dim=0,
        )
        if squeeze_out:
            rebuild_verts, rebuild_joints, rebuild_transf = (
                rebuild_verts.squeeze(0),
                rebuild_joints.squeeze(0),
                rebuild_transf.squeeze(0),
            )
        return rebuild_verts, rebuild_joints, rebuild_transf

    def recover_hand_pose(self):
        vars_hand_pose_assembled = self.assemble_pose_vec(
            self.const_val["hand_pose_gt_idx"],
            self.const_val["hand_pose_gt_val"],
            self.const_val["hand_pose_var_idx"],
            self.opt_val["hand_pose_var_val"],
        ).detach()
        vars_hand_pose_normalized = normalize_quaternion(vars_hand_pose_assembled)
        return vars_hand_pose_normalized

    def recover_obj(self):
        if self.ctrl_val["optimize_obj"]:
            obj_verts = self.transf_vectors(
                self.const_val["full_obj_verts_3d"],
                self.opt_val["obj_tsl_var"].detach(),
                self.opt_val["obj_rot_var"].detach(),
            )
        else:
            obj_verts = self.const_val["full_obj_verts_3d"]
        return obj_verts

    def obj_rot_np(self):
        if self.ctrl_val["optimize_obj"]:
            res = self.opt_val["obj_rot_var"].detach().cpu().numpy()
            return res
        else:
            raise RuntimeError("not optimizing obj, cannot get obj_rot")

    def obj_tsl_np(self):
        if self.ctrl_val["optimize_obj"]:
            res = self.opt_val["obj_tsl_var"].detach().cpu().numpy()
            return res
        else:
            raise RuntimeError("not optimizing obj, cannot get obj_tsl")

    def runtime_show(self, hand_verts, b_axis, u_axis, l_axis, hand_transf, obj_verts):
        has_rot = False
        b_axis = b_axis.detach().cpu().squeeze(0).numpy()
        u_axis = u_axis.detach().cpu().squeeze(0).numpy()
        l_axis = l_axis.detach().cpu().squeeze(0).numpy()
        hand_transf = hand_transf.detach().cpu().squeeze(0).numpy()
        b_rot_ms = []
        u_rot_ms = []
        l_rot_ms = []

        while True:
            self.runtime_vis["hand_mesh"].vertices = o3d.utility.Vector3dVector(
                np.array(hand_verts.detach().cpu().squeeze(0))
            )
            self.runtime_vis["hand_mesh"].compute_vertex_normals()
            self.runtime_vis["obj_mesh"].vertices = o3d.utility.Vector3dVector(
                np.array(obj_verts.detach().cpu().squeeze(0))
            )
            self.runtime_vis["obj_mesh"].compute_vertex_normals()
            if not has_rot:
                for i in range(16):
                    if not i:
                        continue
                    b_rot = caculate_align_mat(b_axis[i - 1])
                    b_rot_ms.append(b_rot)
                    self.runtime_vis["b_axis"][i] = self.runtime_vis["b_axis"][i].rotate(b_rot, center=(0, 0, 0))
                    self.runtime_vis["b_axis"][i] = self.runtime_vis["b_axis"][i].rotate(
                        hand_transf[i][:3, :3], center=(0, 0, 0)
                    )
                    self.runtime_vis["b_axis"][i] = self.runtime_vis["b_axis"][i].translate(hand_transf[i][:3, 3].T)
                    # self.runtime_vis["window"].update_geometry(self.runtime_vis["b_axis"][i])

                    u_rot = caculate_align_mat(u_axis[i - 1])
                    u_rot_ms.append(u_rot)
                    self.runtime_vis["up_axis"][i] = self.runtime_vis["up_axis"][i].rotate(u_rot, center=(0, 0, 0))
                    self.runtime_vis["up_axis"][i] = self.runtime_vis["up_axis"][i].rotate(
                        hand_transf[i][:3, :3], center=(0, 0, 0)
                    )
                    self.runtime_vis["up_axis"][i] = self.runtime_vis["up_axis"][i].translate(hand_transf[i][:3, 3].T)
                    # self.runtime_vis["window"].update_geometry(self.runtime_vis["up_axis"][i])

                    l_rot = caculate_align_mat(l_axis[i - 1])
                    l_rot_ms.append(l_rot)
                    self.runtime_vis["l_axis"][i] = self.runtime_vis["l_axis"][i].rotate(l_rot, center=(0, 0, 0))
                    self.runtime_vis["l_axis"][i] = self.runtime_vis["l_axis"][i].rotate(
                        hand_transf[i][:3, :3], center=(0, 0, 0)
                    )
                    self.runtime_vis["l_axis"][i] = self.runtime_vis["l_axis"][i].translate(hand_transf[i][:3, 3].T)
                    # self.runtime_vis["window"].update_geometry(self.runtime_vis["l_axis"][i])

                has_rot = True
            self.runtime_vis["window"].update_geometry(self.runtime_vis["hand_mesh"])
            self.runtime_vis["window"].update_geometry(self.runtime_vis["obj_mesh"])
            self.runtime_vis["window"].update_renderer()
            if not self.runtime_vis["window"].poll_events():
                break

        for i in range(16):
            if not i:
                continue
            self.runtime_vis["b_axis"][i] = self.runtime_vis["b_axis"][i].translate(-hand_transf[i][:3, 3].T)
            self.runtime_vis["b_axis"][i] = self.runtime_vis["b_axis"][i].rotate(
                hand_transf[i][:3, :3].T, center=(0, 0, 0)
            )
            self.runtime_vis["b_axis"][i] = self.runtime_vis["b_axis"][i].rotate(b_rot_ms[i - 1].T, center=(0, 0, 0))

            self.runtime_vis["up_axis"][i] = self.runtime_vis["up_axis"][i].translate(-hand_transf[i][:3, 3].T)
            self.runtime_vis["up_axis"][i] = self.runtime_vis["up_axis"][i].rotate(
                hand_transf[i][:3, :3].T, center=(0, 0, 0)
            )
            self.runtime_vis["up_axis"][i] = self.runtime_vis["up_axis"][i].rotate(u_rot_ms[i - 1].T, center=(0, 0, 0))

            self.runtime_vis["l_axis"][i] = self.runtime_vis["l_axis"][i].translate(-hand_transf[i][:3, 3].T)
            self.runtime_vis["l_axis"][i] = self.runtime_vis["l_axis"][i].rotate(
                hand_transf[i][:3, :3].T, center=(0, 0, 0)
            )
            self.runtime_vis["l_axis"][i] = self.runtime_vis["l_axis"][i].rotate(l_rot_ms[i - 1].T, center=(0, 0, 0))
        return


def caculate_align_mat(vec):
    vec = vec / np.linalg.norm(vec)
    z_unit_Arr = np.array([0, 0, 1])

    z_mat = np.array(
        [
            [0, -z_unit_Arr[2], z_unit_Arr[1]],
            [z_unit_Arr[2], 0, -z_unit_Arr[0]],
            [-z_unit_Arr[1], z_unit_Arr[0], 0],
        ]
    )

    z_c_vec = np.matmul(z_mat, vec)
    z_c_vec_mat = np.array(
        [
            [0, -z_c_vec[2], z_c_vec[1]],
            [z_c_vec[2], 0, -z_c_vec[0]],
            [-z_c_vec[1], z_c_vec[0], 0],
        ]
    )

    if np.dot(z_unit_Arr, vec) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, vec) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, vec))

    return qTrans_Mat


def init_runtime_viz_hand(
        hand_verts,
        obj_verts,
        hand_faces,
        obj_faces,
        contact_info
):
    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
    hand_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]] * len(hand_verts)))
    hand_mesh.compute_vertex_normals()

    hand_mesh_cur = o3d.geometry.TriangleMesh()
    hand_mesh_cur.triangles = o3d.utility.Vector3iVector(hand_faces)

    obj_mesh = o3d.geometry.TriangleMesh()
    obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)
    obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts)
    obj_mesh.compute_vertex_normals()
    obj_mesh.vertex_colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]] * len(obj_verts))

    vis_cur = o3d.visualization.VisualizerWithKeyCallback()
    vis_cur.create_window(window_name="Runtime Hand", width=1080, height=1080)
    vis_cur.add_geometry(hand_mesh_cur)
    # vis_cur.add_geometry(hand_mesh)
    vis_cur.add_geometry(obj_mesh)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])  # 添加坐标系
    # vis_cur.add_geometry(mesh_frame)
    back_axis_list = []
    up_axis_list = []
    left_axis_list = []

    for i in range(16):
        b = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0015,
            cone_radius=0.002,
            cylinder_height=0.05,
            cone_height=0.008,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
        b.paint_uniform_color([45 / 255.0, 220 / 255.0, 190 / 255.0])
        b.compute_vertex_normals()
        vis_cur.add_geometry(b)
        back_axis_list.append(b)

        u = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0015,
            cone_radius=0.002,
            cylinder_height=0.04,
            cone_height=0.008,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
        u.paint_uniform_color([250 / 255.0, 100 / 255.0, 100 / 255.0])
        u.compute_vertex_normals()
        vis_cur.add_geometry(u)
        up_axis_list.append(u)

        l = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0015,
            cone_radius=0.002,
            cylinder_height=0.04,
            cone_height=0.008,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
        l.paint_uniform_color([230 / 255.0, 120 / 255.0, 60 / 255.0])
        l.compute_vertex_normals()
        vis_cur.add_geometry(l)
        left_axis_list.append(l)

    vis_cur.poll_events()
    runtime_vis = {
        "hand_mesh_gt": hand_mesh,
        "hand_mesh": hand_mesh_cur,
        "obj_mesh": obj_mesh,
        "b_axis": back_axis_list,
        "up_axis": up_axis_list,
        "l_axis": left_axis_list,
        "window": vis_cur,
    }
    return runtime_vis

def update_runtime_viz_hand(
    runtime_vis,
        hand_verts,
        obj_verts,
        hand_faces,
        obj_faces,
):
    runtime_vis["hand_mesh"].vertices = o3d.utility.Vector3dVector(hand_verts)
    runtime_vis["hand_mesh"].triangles = o3d.utility.Vector3iVector(hand_faces)
    runtime_vis["hand_mesh"].compute_vertex_normals()
    runtime_vis["hand_mesh_gt"].vertices = o3d.utility.Vector3dVector(hand_verts)
    runtime_vis["hand_mesh_gt"].triangles = o3d.utility.Vector3iVector(hand_faces)
    runtime_vis["hand_mesh_gt"].vertex_colors = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 1.0]] * len(hand_verts))
    )
    runtime_vis["hand_mesh_gt"].compute_vertex_normals()

    runtime_vis["obj_mesh"].vertices = o3d.utility.Vector3dVector(obj_verts)
    runtime_vis["obj_mesh"].triangles = o3d.utility.Vector3iVector(obj_faces)
    runtime_vis["obj_mesh"].vertex_colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 0.0, 0.0]] * len(obj_verts))
    )
    runtime_vis["obj_mesh"].compute_vertex_normals()

    runtime_vis["window"].update_geometry(runtime_vis["hand_mesh"])
    runtime_vis["window"].update_geometry(runtime_vis["hand_mesh_gt"])
    # runtime_vis["window"].update_geometry(runtime_vis["obj_mesh"])
    runtime_vis["window"].poll_events()
    runtime_vis["window"].update_renderer()
    runtime_vis["window"].reset_view_point(True)

def init_runtime_viz(
    hand_verts_gt,
    hand_verts_init,
    obj_verts_gt,
    hand_faces,
    obj_verts_cur,
    obj_faces_cur,
    contact_info,
    cam_extr=None,
):
    hand_mesh_gt = o3d.geometry.TriangleMesh()
    hand_mesh_gt.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh_gt.vertices = o3d.utility.Vector3dVector(hand_verts_gt)
    hand_mesh_gt.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]] * len(hand_verts_gt)))
    hand_mesh_gt.compute_vertex_normals()
    obj_mesh_gt = o3d.geometry.TriangleMesh()
    obj_mesh_gt.triangles = o3d.utility.Vector3iVector(obj_faces_cur)
    obj_mesh_gt.vertices = o3d.utility.Vector3dVector(obj_verts_gt)
    obj_mesh_gt.vertex_colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]] * len(obj_verts_gt)))
    obj_mesh_gt.compute_vertex_normals()

    hand_mesh_cur = o3d.geometry.TriangleMesh()
    hand_mesh_cur.triangles = o3d.utility.Vector3iVector(hand_faces)
    obj_mesh = o3d.geometry.TriangleMesh()
    obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces_cur)
    obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts_cur)
    obj_colors = create_vertex_color(contact_info, "contact_region")
    obj_mesh.compute_vertex_normals()
    obj_mesh.vertex_colors = o3d.utility.Vector3dVector(obj_colors)
    vis_cur = o3d.visualization.VisualizerWithKeyCallback()
    vis_cur.create_window(window_name="Runtime Hand", width=1080, height=1080)
    vis_cur.add_geometry(obj_mesh)
    vis_cur.add_geometry(hand_mesh_cur)
    vis_cur.add_geometry(hand_mesh_gt)
    vis_cur.add_geometry(obj_mesh_gt)
    back_axis_list = []
    up_axis_list = []
    left_axis_list = []
    for i in range(16):
        b = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0015,
            cone_radius=0.002,
            cylinder_height=0.05,
            cone_height=0.008,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
        b.paint_uniform_color([45 / 255.0, 220 / 255.0, 190 / 255.0])
        b.compute_vertex_normals()
        vis_cur.add_geometry(b)
        back_axis_list.append(b)

        u = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0015,
            cone_radius=0.002,
            cylinder_height=0.04,
            cone_height=0.008,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
        u.paint_uniform_color([250 / 255.0, 100 / 255.0, 100 / 255.0])
        u.compute_vertex_normals()
        vis_cur.add_geometry(u)
        up_axis_list.append(u)

        l = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0015,
            cone_radius=0.002,
            cylinder_height=0.04,
            cone_height=0.008,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
        l.paint_uniform_color([230 / 255.0, 120 / 255.0, 60 / 255.0])
        l.compute_vertex_normals()
        vis_cur.add_geometry(l)
        left_axis_list.append(l)

    vis_cur.poll_events()
    runtime_vis = {
        "hand_mesh_gt": hand_mesh_gt,
        "hand_mesh": hand_mesh_cur,
        "obj_mesh": obj_mesh,
        "obj_mesh_gt": obj_mesh_gt,
        "b_axis": back_axis_list,
        "up_axis": up_axis_list,
        "l_axis": left_axis_list,
        "window": vis_cur,
    }
    if cam_extr is not None:
        ctl = runtime_vis["window"].get_view_control()
        parameters = ctl.convert_to_pinhole_camera_parameters()
        parameters.extrinsic = cam_extr
        ctl.convert_from_pinhole_camera_parameters(parameters)
        # ro = runtime_vis["window"].get_render_option()
        # ro.load_from_json("./ro.json")

    def hide_gt(vis):
        vis.remove_geometry(hand_mesh_gt, reset_bounding_box=False)
        vis.remove_geometry(obj_mesh_gt, reset_bounding_box=False)
        vis.add_geometry(obj_mesh, reset_bounding_box=False)
        vis.add_geometry(hand_mesh_cur, reset_bounding_box=False)

        vis.update_renderer()
        vis.poll_events()

    def show_gt(vis):
        vis.add_geometry(hand_mesh_gt, reset_bounding_box=False)
        vis.add_geometry(obj_mesh_gt, reset_bounding_box=False)
        vis.remove_geometry(obj_mesh, reset_bounding_box=False)
        vis.remove_geometry(hand_mesh_cur, reset_bounding_box=False)

        vis.update_renderer()
        vis.poll_events()

    def axis_on(vis):
        for i in range(16):
            vis.add_geometry(back_axis_list[i], reset_bounding_box=False)
            vis.add_geometry(up_axis_list[i], reset_bounding_box=False)
            vis.add_geometry(left_axis_list[i], reset_bounding_box=False)
        vis.update_renderer()
        vis.poll_events()

    def axis_off(vis):
        for i in range(16):
            vis.remove_geometry(back_axis_list[i], reset_bounding_box=False)
            vis.remove_geometry(up_axis_list[i], reset_bounding_box=False)
            vis.remove_geometry(left_axis_list[i], reset_bounding_box=False)
        vis.update_renderer()
        vis.poll_events()

    vis_cur.register_key_callback(ord("A"), hide_gt)
    vis_cur.register_key_callback(ord("Z"), show_gt)
    vis_cur.register_key_callback(ord("U"), axis_on)
    vis_cur.register_key_callback(ord("I"), axis_off)

    return runtime_vis


def update_runtime_viz(
    runtime_vis,
    hand_verts_gt,
    hand_verts_curr,
    obj_verts_gt,
    obj_verts_curr,
    hand_faces,
    obj_faces_cur,
):
    runtime_vis["hand_mesh"].vertices = o3d.utility.Vector3dVector(hand_verts_curr)
    runtime_vis["hand_mesh"].triangles = o3d.utility.Vector3iVector(hand_faces)
    runtime_vis["hand_mesh"].compute_vertex_normals()
    runtime_vis["hand_mesh_gt"].vertices = o3d.utility.Vector3dVector(hand_verts_gt)
    runtime_vis["hand_mesh_gt"].triangles = o3d.utility.Vector3iVector(hand_faces)
    runtime_vis["hand_mesh_gt"].vertex_colors = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 1.0]] * len(hand_verts_gt))
    )
    runtime_vis["hand_mesh_gt"].compute_vertex_normals()
    runtime_vis["obj_mesh"].vertices = o3d.utility.Vector3dVector(obj_verts_curr)
    runtime_vis["obj_mesh"].triangles = o3d.utility.Vector3iVector(obj_faces_cur)
    runtime_vis["obj_mesh"].vertex_colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 1.0, 0.0]] * len(obj_verts_curr))
    )
    runtime_vis["obj_mesh"].compute_vertex_normals()
    runtime_vis["obj_mesh_gt"].vertices = o3d.utility.Vector3dVector(obj_verts_gt)
    runtime_vis["obj_mesh_gt"].triangles = o3d.utility.Vector3iVector(obj_faces_cur)
    runtime_vis["obj_mesh_gt"].vertex_colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 0.0, 0.0]] * len(obj_verts_gt))
    )
    runtime_vis["obj_mesh_gt"].compute_vertex_normals()
    runtime_vis["window"].update_geometry(runtime_vis["hand_mesh"])
    runtime_vis["window"].update_geometry(runtime_vis["hand_mesh_gt"])
    runtime_vis["window"].update_geometry(runtime_vis["obj_mesh"])
    runtime_vis["window"].update_geometry(runtime_vis["obj_mesh_gt"])
    runtime_vis["window"].poll_events()
    runtime_vis["window"].update_renderer()
    runtime_vis["window"].reset_view_point(True)
