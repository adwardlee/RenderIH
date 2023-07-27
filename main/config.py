import os
import os.path as osp
import sys
import numpy as np

class Config:
    
    ## dataset
    # HO3D, DEX_YCB, FREIHAND, INTERHAND
    trainset = 'INTERHAND'#'INTERHAND'
    testset = 'INTERHAND'#'INTERHAND'
    model_name = 'graph'### 'newgraph' ##'graph' #'intag_regress' # 'myhand' ## 'regressor'## adapt### nohms ### graph ## real2d ########### change ###########

    schedule = 'cosine'
    num_workers = 8
    data_dir = '/mnt/user/E-shenfei.llj-356552/data/dataset/freihand1/'
    train_yaml = '/mnt/user/E-shenfei.llj-356552/data/dataset/freihand1/train.yaml'
    test_yaml = '/mnt/user/E-shenfei.llj-356552/data/dataset/freihand1/test.yaml'
    ## input, output
    input_img_shape = (256,256)
    
    ## training config
    if trainset == 'HO3D':
        lr_dec_epoch = [10*i for i in range(1,7)]
        end_epoch = 70
        lr = 1e-4
        lr_dec_factor = 0.7
        lambda_mano_verts = 1e4
        lambda_mano_joints = 1e4
        lambda_mano_pose = 10
        lambda_mano_shape = 0.1
        lambda_joints_img = 100
        ckpt_freq = 10
        train_batch_size = 64
        test_batch_size = 128
        data_dir = '/mnt/user/E-shenfei.llj-356552/data/dataset/ho3d/'
        print_freq = 200
        cam = False
    elif trainset == 'DEX_YCB':
        lr_dec_epoch = [i for i in range(1,25)]
        end_epoch = 25
        lr = 1e-4
        lr_dec_factor = 0.9
        lambda_mano_verts = 1e4
        lambda_mano_joints = 1e4
        lambda_mano_pose = 10
        lambda_mano_shape = 0.1
        lambda_joints_img = 100
        ckpt_freq = 10
        train_batch_size = 64
        test_batch_size = 256
        data_dir = '/mnt/user/E-shenfei.llj-356552/data/dataset/'
        print_freq = 200
        cam = False
    elif trainset == 'FREIHAND':
        lr_dec_epoch = [10 * i for i in [3, 6, 8, 10]]
        end_epoch = 120
        lr = 1e-4
        lr_dec_factor = 0.7
    elif trainset == 'INTERHAND':
        cam = True
        data_dir = '/mnt/user/E-shenfei.llj-356552/data/dataset/interhand_5fps/interhand_data/'
        data_type = 'interhand_dataaug' ######## for data augmentation ##### interhand_xinchuan ## interhand_dataaug ### interhand_subset ### interhand_adapt#####
        ## xinchuan_subset ## #  # interhand_sdf ###
        lr_dec_epoch = [10*i for i in range(2,7,2)]
        end_epoch = 90
        seed = 2453
        lr = 1.3e-4####### 1.3e-4
        lr_dec_factor = 0.7###0.7
        train_batch_size = 32 # per GPU ######### change ###########
        lambda_mano_verts = 1e4
        lambda_mano_joints = 1e4
        lambda_mano_pose = 10
        lambda_mano_shape = 1
        lambda_joints_img = 100
        lambda_hms = 10
        lambda_mask = 10
        lambda_dp = 10
        ckpt_freq = 5#### no change !!!!
        print_freq = 50
        mano_flag = True #### change ###############################
        decay = 1e-6

        sdf = False
        render = False ####### change
        normal = True
        edge = True
        vert2d = True
        dice = False ### use focal or dice loss
        lambda_sdf = 1000000
        lambda_render = 100
        lambda_normal = 10
        lambda_edge = 100
        sdf_thresh = 0.01
        sdf_train = 'sdf10mm_train.npy'#'sdf10mm_train.npy' , 'sdfloss.npy'
        sdf_test = 'sdf10mm_test.npy'#'sdf10mm_test.npy', 'sdfloss_test.npy'

        ## testing config
        test_batch_size = 384

        ### for new graph ###
        reverse = False

    ## others
    num_thread = 20
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False#### change #######
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    datadir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'scratch', 'train_scratch_flip')#'tune_sdf10mm', 'lljsdf1e5_sdf10mm_thresh10mm_lr4e-5')######change ############  trainsubset10_aug_gannodis_pretrain ##trainsubset30_aug_gannodis_pretrain
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    mano_path = osp.join('misc', 'mano')
    
    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from common.utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.datadir))
add_pypath(osp.join(cfg.datadir, cfg.trainset))
add_pypath(osp.join(cfg.datadir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
