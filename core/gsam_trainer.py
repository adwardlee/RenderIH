import sys
import os
from tkinter.messagebox import NO
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import json
import cv2 as cv
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
import random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer

from common.myhand.lijun_model_graph import load_graph_model
from common.utils.transforms import get_alignMesh
from common.utils.intag_eval import eval_hand
# from models.model import load_model

from utils.tb_utils import tbUtils
from utils.lr_sc import StepLR_withWarmUp
from utils.DataProvider import DataProvider
from utils.vis_utils import mano_two_hands_renderer
from utils.manoutils import get_mano_path

from core.loader import handDataset
from dataset.interhand import fix_shape, InterHand_dataset
from core.Loss import GraphLoss, calc_loss_GCN
from core.vis_train import tb_vis_train_gcn
from dataset.dataset_utils import IMG_SIZE, BLUR_KERNEL
from dataset.inference import get_final_preds2
from gsam.gsam import gsam_loss

class Jr():
    def __init__(self, J_regressor,
                 device='cuda'):
        self.device = device
        self.process_J_regressor(J_regressor)

    def process_J_regressor(self, J_regressor):
        J_regressor = J_regressor.clone().detach()
        tip_regressor = torch.zeros_like(J_regressor[:5])
        tip_regressor[0, 745] = 1.0
        tip_regressor[1, 317] = 1.0
        tip_regressor[2, 444] = 1.0
        tip_regressor[3, 556] = 1.0
        tip_regressor[4, 673] = 1.0
        J_regressor = torch.cat([J_regressor, tip_regressor], dim=0)
        new_order = [0, 13, 14, 15, 16,
                     1, 2, 3, 17,
                     4, 5, 6, 18,
                     10, 11, 12, 19,
                     7, 8, 9, 20]
        self.J_regressor = J_regressor[new_order].contiguous().to(self.device)

    def __call__(self, v):
        return torch.matmul(self.J_regressor, v)

def freeze_model(model):
    for (name, params) in model.named_parameters():
        params.requires_grad = False


def train_gcn(rank=0, world_size=1, cfg=None, dist_training=False):
    if dist_training:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(cfg.TRAIN.DIST_PORT)
        print("Init distributed training on local rank {}".format(rank))
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    mano_path = get_mano_path()


    outfile = open(cfg.LOG_DIR, 'a')

    # -------------------------------------------------
    # | 1. load model/optimizer/scheduler/tensorboard |
    # -------------------------------------------------
    # load network
    network = load_graph_model(cfg)
    network.to(rank)

    if cfg.MODEL.freeze_upsample:
        freeze_model(network.decoder.unsample_layer)

    converter = {}
    for hand_type in ['left', 'right']:
        converter[hand_type] = network.decoder.converter[hand_type]

    if dist_training:
        network = DDP(
            network, device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )
    # print('local rank {}: init model, done'.format(rank))

    # load optimizer
    optim_params = list(filter(lambda p: p.requires_grad, network.parameters()))
    if dist_training:
        optimizer = ZeroRedundancyOptimizer(
            optim_params,
            optimizer_class=torch.optim.Adam,
            lr=cfg.TRAIN.LR
        )
    else:
        optimizer = torch.optim.Adam(optim_params, lr=cfg.TRAIN.LR)

    # print('local rank {}: init optimizer, done'.format(rank))

    # load learning rate scheduler
    lr_scheduler = StepLR_withWarmUp(optimizer,
                                     last_epoch=-1 if cfg.TRAIN.current_epoch == 0 else cfg.TRAIN.current_epoch,
                                     init_lr=1e-3 * cfg.TRAIN.LR,
                                     warm_up_epoch=cfg.TRAIN.warm_up,
                                     gamma=cfg.TRAIN.lr_decay_gamma,
                                     step_size=cfg.TRAIN.lr_decay_step,
                                     min_thres=0.05)
    from gsam.scheduler import ProportionScheduler
    from gsam.gsam import GSAM
    rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=cfg.TRAIN.LR, min_lr=1e-3 * cfg.TRAIN.LR,
                                        max_value=1, min_value=1)
    optimizer = GSAM(params=optim_params, base_optimizer=optimizer, model=network, gsam_alpha=0.1,
                     rho_scheduler=rho_scheduler, adaptive=False)
    # print('local rank {}: init lr_scheduler, done'.format(rank))

    if rank == 0:
        # tensorboard
        writer = SummaryWriter(cfg.TB.SAVE_DIR)
        renderer = mano_two_hands_renderer(img_size=IMG_SIZE, device='cuda:{}'.format(rank))

    # --------------------------
    # | 2. load dataset & Loss |
    # --------------------------
    aux_lambda = 2**(6 - len(cfg.MODEL.DECONV_DIMS))### '/mnt/user/E-shenfei.llj-356552/data/dataset/interhand_5fps/interhand_data/'
    trainDataset = handDataset(mano_path=mano_path,
                               interPath=cfg.DATASET.INTERHAND_PATH,
                               theta=[-cfg.DATA_AUGMENT.THETA, cfg.DATA_AUGMENT.THETA],
                               scale=[1 - cfg.DATA_AUGMENT.SCALE, 1 + cfg.DATA_AUGMENT.SCALE],
                               uv=[-cfg.DATA_AUGMENT.UV, cfg.DATA_AUGMENT.UV],
                               aux_size=IMG_SIZE // aux_lambda)
    evalDataset = handDataset(mano_path=mano_path,
                               interPath=cfg.DATASET.INTERHAND_PATH,
                               theta=[-cfg.DATA_AUGMENT.THETA, cfg.DATA_AUGMENT.THETA],
                               scale=[1 - cfg.DATA_AUGMENT.SCALE, 1 + cfg.DATA_AUGMENT.SCALE],
                               uv=[-cfg.DATA_AUGMENT.UV, cfg.DATA_AUGMENT.UV],
                               aux_size=IMG_SIZE // aux_lambda, train=False, flip=False)
    # print('local rank {}: init dataset, done'.format(rank))

    provider_train = DataProvider(dataset=trainDataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=4, dist=dist_training)

    provider_eval = DataProvider(dataset=evalDataset, batch_size=128,
                                 num_workers=4, dist=dist_training)
    train_batch_per_epoch = provider_train.batch_per_epoch
    eval_batch_per_epoch = provider_eval.batch_per_epoch
    # print('local rank {}: init data loader, done'.format(rank))

    Loss = {}
    faces = {}
    J_regressor_layer = {}
    for hand_type in ['left', 'right']:
        with open(mano_path[hand_type], 'rb') as file:
            manoData = pickle.load(file, encoding='latin1')
        J_regressor = manoData['J_regressor'].tocoo(copy=False)
        location = []
        data = []
        for i in range(J_regressor.data.shape[0]):
            location.append([J_regressor.row[i], J_regressor.col[i]])
            data.append(J_regressor.data[i])
        i = torch.LongTensor(location)
        v = torch.FloatTensor(data)
        J_regressor = torch.sparse.FloatTensor(i.t(), v, torch.Size([16, 778])).to_dense()
        Loss[hand_type] = GraphLoss(J_regressor, manoData['f'],
                                    level=4,
                                    device=rank)
        # device='cuda:{}'.format(rank))
        faces[hand_type] = manoData['f']
        J_regressor_layer[hand_type] = Jr(J_regressor)

    # print('local rank {}: init training loss, done'.format(rank))

    # ------------
    # | 3. train |
    # ------------
    # print('local rank {}: strat training'.format(rank))
    for epoch in range(cfg.TRAIN.current_epoch, cfg.TRAIN.EPOCHS):
        network.train()
        train_bar = range(train_batch_per_epoch)
        if rank == 0:
            train_bar = tqdm(train_bar)
        for bIdx in train_bar:
            total_idx = epoch * train_batch_per_epoch + bIdx

            # ------------
            # | training |
            # ------------
            label_list = provider_train.next()
            label_list_out = []
            for label in label_list:
                if label is not None:
                    label_list_out.append(label.to(rank))
            # mask, dense, hms,
            [ori_img,
             imgTensors,
             v2d_l, j2d_l, v2d_r, j2d_r,
             v3d_l, j3d_l, v3d_r, j3d_r,
             root_rel] = label_list_out
            # result, paramsDict, handDictList, otherInfo = network(imgTensors)

            if cfg.MODEL.freeze_upsample:
                upsample_weight = None
            else:
                if dist_training:
                    upsample_weight = network.module.decoder.get_upsample_weight()
                else:
                    upsample_weight = network.decoder.get_upsample_weight()

            # loss, aux_lost_dict, mano_loss_dict, coarsen_loss_dict = \
            #     calc_loss_GCN(cfg, epoch,
            #                   Loss['left'], Loss['right'],
            #                   converter['left'], converter['right'],
            #                   result, paramsDict, handDictList, otherInfo,
            #                   None, None, None,
            #                   v2d_l, j2d_l, v2d_r, j2d_r,
            #                   v3d_l, j3d_l, v3d_r, j3d_r,
            #                   root_rel, img_size=imgTensors.shape[-1],
            #                   upsample_weight=upsample_weight)
            #mask, dense, hms,
            optimizer.set_closure(gsam_loss, imgTensors, [v2d_l, j2d_l, v2d_r, j2d_r, v3d_l, j3d_l, v3d_r, j3d_r,root_rel],cfg=cfg,
                              epoch=epoch, graph_loss_left=Loss['left'], graph_loss_right=Loss['right'],
                              converter_left=converter['left'], converter_right=converter['right'],img_size=imgTensors.shape[-1],
                              upsample_weight=upsample_weight)
            pred, loss = optimizer.step()



            # ---------------
            # | tensorboard |
            # ---------------
            if rank == 0:
                writer.add_scalar('learning_rate', lr_scheduler.get_lr()[0], total_idx)
                writer.add_scalar('train/total_loss', loss.item(), total_idx)
                if total_idx % 50 == 0:
                    outfile.write(' epoch {} , iter {},  loss {} \n'.format(epoch, total_idx, loss.item()))
                    outfile.flush()
                # for k, v in mano_loss_dict.items():
                #     if k != 'total_loss':
                #         writer.add_scalar('train/mano_{}'.format(k), v.item(), total_idx)
                # for k, v in aux_lost_dict.items():
                #     if k != 'total_loss':
                #         writer.add_scalar('train/aux_{}'.format(k), v.item(), total_idx)
                # for k, v in coarsen_loss_dict.items():
                #     if k != 'total_loss':
                #         for t in range(len(v)):
                #             writer.add_scalar('train/coarsen_{}_{}'.format(k, t), v[t].item(), total_idx)

                # --------
                # | tqdm |
                # --------
                train_bar.set_description('train, epoch:{}'.format(epoch))
                train_bar.set_postfix(totalLoss=loss.item())

        lr_scheduler.step()
        optimizer.update_rho_t()

        if (epoch + 1) % cfg.SAVE.SAVE_GAP == 0:
            if rank == 0:  # save checkpoint in main process
                torch.save(network.state_dict(), os.path.join(cfg.SAVE.SAVE_DIR, str(epoch + 1) + '.pth'))

        if (epoch + 1) % 5 == 0 or epoch+1 == cfg.TRAIN.EPOCHS or epoch == 0:
            network.eval()

            gt_mesh_left = []
            gt_3djoint_left = []
            mesh_output_save_left = []
            joint_output_save_left = []
            gt_mesh_right = []
            gt_3djoint_right = []
            mesh_output_save_right = []
            joint_output_save_right = []
            joints_loss = {'left': [], 'right': []}
            verts_loss = {'left': [], 'right': []}

            pajoints_loss = {'left': [], 'right': []}
            paverts_loss = {'left': [], 'right': []}

            double_jointloss = {'left': [], 'right': []}
            double_pajointloss = {'left': [], 'right': []}
            double_vertloss = {'left': [], 'right': []}
            double_pavertloss = {'left': [], 'right': []}
            double_predvert = []
            double_predjoint = []
            double_gtvert = []
            double_gtjoint = []

            eval_bar = range(eval_batch_per_epoch)
            if rank == 0:
                eval_bar = tqdm(eval_bar)
            for bIdx in eval_bar:
                # ------------
                # | training |
                # ------------
                label_list = provider_eval.next()
                label_list_out = []
                for label in label_list:
                    if label is not None:
                        label_list_out.append(label.to(rank))

                # mask, dense, hms,
                [ori_img,
                 imgTensors,
                 v2d_l, j2d_l, v2d_r, j2d_r,
                 v3d_l, j3d_l, v3d_r, j3d_r,
                 root_rel] = label_list_out

                # forward
                with torch.no_grad():
                    result, paramsDict, handDictList, otherInfo = network(imgTensors)
                result['j3d'] = {}
                # save output
                j3d_l = J_regressor_layer['left'](v3d_l)
                j3d_r = J_regressor_layer['right'](v3d_r)

                gt_vertices_left_ori = v3d_l
                gt_3d_joints_left_ori = j3d_l

                # normalize gt based on hand's wrist
                gt_3d_root = gt_3d_joints_left_ori[:, 0, :]
                gt_vertices_left = gt_vertices_left_ori - gt_3d_root[:, None, :]
                gt_3d_joints_left = gt_3d_joints_left_ori - gt_3d_root[:, None, :]
                ### right ####
                gt_vertices_right_ori = v3d_r
                gt_3d_joints_right_ori = j3d_r

                # normalize gt based on hand's wrist
                gt_3d_root = gt_3d_joints_right_ori[:, 0, :]
                gt_vertices_right = gt_vertices_right_ori - gt_3d_root[:, None, :]
                gt_3d_joints_right = gt_3d_joints_right_ori - gt_3d_root[:, None, :]


                pred_vertices_leftori = result['verts3d']['left']
                pred_3djoints_left = (J_regressor_layer['left'](pred_vertices_leftori)).cpu().numpy()
                result['j3d']['left'] = J_regressor_layer['left'](pred_vertices_leftori)
                pred_vertices_leftori = pred_vertices_leftori.cpu().numpy()
                pred_3d_pelvis = pred_3djoints_left[:, 0, :]
                pred_3d_joints_from_mesh = pred_3djoints_left - pred_3d_pelvis[:, None, :]
                pred_vertices_left = pred_vertices_leftori - pred_3d_pelvis[:, None, :]
                mesh_output_save_left.append(pred_vertices_left)
                joint_output_save_left.append(pred_3d_joints_from_mesh)
                gt_mesh_left.append(gt_vertices_left.cpu().numpy())
                gt_3djoint_left.append(gt_3d_joints_left.cpu().numpy())
                ### right ###
                pred_vertices_right = result['verts3d']['right']
                pred_3djoints_right = (J_regressor_layer['right'](pred_vertices_right)).cpu().numpy()
                result['j3d']['right'] = J_regressor_layer['right'](pred_vertices_right)
                pred_vertices_right = pred_vertices_right.cpu().numpy()
                pred_3d_pelvis = pred_3djoints_right[:, 0, :]
                pred_3d_joints_from_mesh = pred_3djoints_right - pred_3d_pelvis[:, None, :]
                pred_vertices_right = pred_vertices_right - pred_3d_pelvis[:, None, :]
                mesh_output_save_right.append(pred_vertices_right)
                joint_output_save_right.append(pred_3d_joints_from_mesh)
                gt_mesh_right.append(gt_vertices_right.cpu().numpy())
                gt_3djoint_right.append(gt_3d_joints_right.cpu().numpy())

                eval_hand(gt_vertices_left_ori, gt_vertices_right_ori, gt_3d_joints_left_ori, gt_3d_joints_right_ori,
                          result['verts3d']['left'], result['verts3d']['right'], result['j3d']['left'],
                          result['j3d']['right'],
                          joints_loss, verts_loss, pajoints_loss, paverts_loss)

                ### left relative to right root ###
                length_trans = pred_3djoints_left[:, 9:10] - pred_3d_pelvis[:, None, :]
                length_left = pred_3djoints_left[:, 9:10] - pred_3djoints_left[:, 0:1, :]
                gt_length_left = gt_3d_joints_left_ori[:, 9:10, :] - gt_3d_joints_left_ori[:, 0:1, :]
                gt_trans_left = gt_3d_joints_left_ori[:, 9:10, :] - gt_3d_joints_right_ori[:, 0:1, :]
                double_leftvert = (pred_vertices_leftori - pred_3d_pelvis[:, None, :]) / (length_trans + 1e-8) * length_left
                double_leftjoint = (pred_3djoints_left - pred_3d_pelvis[:, None, :]) / (length_trans + 1e-8) * length_left
                double_predvert.append(np.concatenate((double_leftvert, pred_vertices_right), axis=1))
                double_predjoint.append(np.concatenate((double_leftjoint, pred_3d_joints_from_mesh), axis=1))
                double_gtvert.append(np.concatenate(( ((gt_vertices_left + gt_3d_joints_left_ori[:, 0:1, :]- gt_3d_root[:, None, :])/(gt_trans_left + 1e-8)*gt_length_left).cpu().numpy(), gt_vertices_right.cpu().numpy()), axis=1))
                double_gtjoint.append(np.concatenate(( ((gt_3d_joints_left + gt_3d_joints_left_ori[:, 0:1, :]- gt_3d_root[:, None, :])/(gt_trans_left + 1e-8)* gt_length_left).cpu().numpy(), gt_3d_joints_right.cpu().numpy()), axis=1))


            gt_mesh = np.array(np.concatenate(gt_mesh_left, axis=0))
            gt_3djoint = np.array(np.concatenate(gt_3djoint_left, axis=0))
            pred_mesh = np.array(np.concatenate(mesh_output_save_left, axis=0))
            pred_3djoint = np.array(np.concatenate(joint_output_save_left, axis=0))
            pa_mesh_error, _, _ = get_alignMesh(pred_mesh, gt_mesh)
            pa_joint_error, _, _ = get_alignMesh(pred_3djoint, gt_3djoint)
            if rank == 0:
                outfile.write('PAMPJPE: {} \n'.format(1000 * np.mean(pa_joint_error)))
                outfile.write('PAMPVPE: {} \n'.format(1000 * np.mean(pa_mesh_error)))
            error_mpjpe = np.mean(np.sqrt(((pred_3djoint - gt_3djoint) ** 2).sum(axis=-1)).mean(axis=-1))
            if rank == 0:
                outfile.write('MPJPE: {} \n'.format(1000 * error_mpjpe))
            error_mpjpe = np.mean(np.sqrt(((pred_mesh - gt_mesh) ** 2).sum(axis=-1)).mean(axis=-1))
            if rank == 0:
                outfile.write('MPVPE: {} \n'.format(1000 * error_mpjpe))

            gt_mesh = np.array(np.concatenate(gt_mesh_right, axis=0))
            gt_3djoint = np.array(np.concatenate(gt_3djoint_right, axis=0))
            pred_mesh = np.array(np.concatenate(mesh_output_save_right, axis=0))
            pred_3djoint = np.array(np.concatenate(joint_output_save_right, axis=0))
            pa_mesh_error, _, _ = get_alignMesh(pred_mesh, gt_mesh)
            pa_joint_error, _, _ = get_alignMesh(pred_3djoint, gt_3djoint)
            if rank == 0:
                outfile.write('PAMPJPE: {} \n'.format(1000 * np.mean(pa_joint_error)))
                outfile.write('PAMPVPE: {} \n'.format(1000 * np.mean(pa_mesh_error)))
            error_mpjpe = np.mean(np.sqrt(((pred_3djoint - gt_3djoint) ** 2).sum(axis=-1)).mean(axis=-1))
            if rank == 0:
                outfile.write('MPJPE: {} \n'.format(1000 * error_mpjpe))
            error_mpjpe = np.mean(np.sqrt(((pred_mesh - gt_mesh) ** 2).sum(axis=-1)).mean(axis=-1))
            if rank == 0:
                outfile.write('MPVPE: {} \n'.format(1000 * error_mpjpe))

            gt_mesh = np.array(np.concatenate(double_gtvert, axis=0))
            gt_3djoint = np.array(np.concatenate(double_gtjoint, axis=0))
            pred_mesh = np.array(np.concatenate(double_predvert, axis=0))
            pred_3djoint = np.array(np.concatenate(double_predjoint, axis=0))
            nan_num = np.sum(np.isnan(pred_mesh))
            nan_num1 = np.sum(np.isnan(gt_mesh))
            if nan_num != 0 or nan_num1 != 0:
                if rank == 0:
                    outfile.write('pred mesh {} , gt_mesh {} \n'.format(nan_num, nan_num1), flush=True)
                idx = (pred_mesh == pred_mesh)[0]
                pred_mesh = pred_mesh[idx]
                gt_mesh = gt_mesh[idx]
                pred_3djoint = pred_3djoint[idx]
                gt_3djoint = gt_3djoint[idx]
            try:
                error_mpjpe = np.mean(np.sqrt(((pred_3djoint - gt_3djoint) ** 2).sum(axis=-1)).mean(axis=-1))
                if rank == 0:
                    outfile.write('double MPJPE: {} \n'.format(1000 * error_mpjpe))
                error_mpjpe = np.mean(np.sqrt(((pred_mesh - gt_mesh) ** 2).sum(axis=-1)).mean(axis=-1))
                if rank == 0:
                    outfile.write('double MPVPE: {} \n'.format(1000 * error_mpjpe))
                pa_mesh_error, _, _ = get_alignMesh(pred_mesh, gt_mesh)
                pa_joint_error, _, _ = get_alignMesh(pred_3djoint, gt_3djoint)
                if rank == 0:
                    outfile.write('double PAMPJPE: {} \n'.format(1000 * np.mean(pa_joint_error)))
                    outfile.write('double PAMPVPE: {} \n'.format(1000 * np.mean(pa_mesh_error)))
            except:
                continue
            #### intag eval ####
            joints_loss['left'] = np.concatenate(joints_loss['left'], axis=0)
            joints_loss['right'] = np.concatenate(joints_loss['right'], axis=0)
            verts_loss['left'] = np.concatenate(verts_loss['left'], axis=0)
            verts_loss['right'] = np.concatenate(verts_loss['right'], axis=0)

            pajoints_loss['left'] = np.concatenate(pajoints_loss['left'], axis=0)
            pajoints_loss['right'] = np.concatenate(pajoints_loss['right'], axis=0)
            paverts_loss['left'] = np.concatenate(paverts_loss['left'], axis=0)
            paverts_loss['right'] = np.concatenate(paverts_loss['right'], axis=0)

            joints_mean_loss_left = joints_loss['left'].mean() * 1000
            joints_mean_loss_right = joints_loss['right'].mean() * 1000
            verts_mean_loss_left = verts_loss['left'].mean() * 1000
            verts_mean_loss_right = verts_loss['right'].mean() * 1000

            if rank == 0:
                outfile.write('joint mean error: \n')
                outfile.write('    left: {} mm, right: {} mm \n'.format(joints_mean_loss_left, joints_mean_loss_right))
                outfile.write('    all: {} mm \n'.format((joints_mean_loss_left + joints_mean_loss_right) / 2))
                outfile.write('vert mean error: \n')
                outfile.write('    left: {} mm, right: {} mm \n'.format(verts_mean_loss_left, verts_mean_loss_right))
                outfile.write('    all: {} mm \n'.format((verts_mean_loss_left + verts_mean_loss_right) / 2))

            joints_mean_loss_left = pajoints_loss['left'].mean() * 1000
            joints_mean_loss_right = pajoints_loss['right'].mean() * 1000
            verts_mean_loss_left = paverts_loss['left'].mean() * 1000
            verts_mean_loss_right = paverts_loss['right'].mean() * 1000

            if rank == 0:
                outfile.write('joint pa _ mean error: \n')
                outfile.write('    left_pa: {} mm, right: {} mm \n'.format(joints_mean_loss_left, joints_mean_loss_right))
                outfile.write('    all_pa: {} mm \n'.format((joints_mean_loss_left + joints_mean_loss_right) / 2))
                outfile.write('vert mean error: \n')
                outfile.write('    left_pa: {} mm, right: {} mm \n'.format(verts_mean_loss_left, verts_mean_loss_right))
                outfile.write('    all_pa: {} mm \n'.format((verts_mean_loss_left + verts_mean_loss_right) / 2))
                outfile.flush()
            if rank == 0:
                writer.add_scalar('learning_rate', lr_scheduler.get_lr()[0], total_idx)
            network.train()
    if dist_training:
        dist.barrier()
        dist.destroy_process_group()
