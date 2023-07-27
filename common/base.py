import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from main.config import cfg

from main.model import get_model
from common.myhand.lijun_model import load_nohms_model
from common.myhand.lijun_model_graph import load_graph_model
from common.myhand.lijun_model_newgraph import load_new_model
from common.myhand.model import load_model
from common.myhand.model_real2d import load_real2d
from common.myhand.model_adapt import load_adapt
from main.double_model import get_double_model
from common.net_inverse import ParamRegressor
from main.intag_regress_model import fullModel
from common.utils.warmup_scheduler import GradualWarmupScheduler
from data.FREIHAND.build import make_hand_data_loader
from data.INTERHAND.interhand import InterHand_dataset
from data.INTERHAND.interhand_adapt import InterHand_adapt
from data.INTERHAND.interhand_dataaug import InterHandaug_dataset
from data.INTERHAND.interhand_sdf import InterHandsdf_dataset
from data.INTERHAND.interhand_subset import InterHand_subset
from data.INTERHAND.interhand_xinchuang_aug import InterHandxinchuan_dataset
from data.INTERHAND.interhand_xinchuang_subset import InterHandxinchuan_subset
from data.INTERHAND.diff_dataaug import Diffaug_dataset
from data.HO3D.HO3D import HO3D
from data.DEX_YCB.DEX_YCB import DEX_YCB
# dynamic dataset import
# exec('from ' + cfg.trainset + ' import ' + cfg.trainset)
# exec('from ' + cfg.testset + ' import ' + cfg.testset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self, args=None):
        return

    @abc.abstractmethod
    def _make_model(self, cfg, epoch=0):
        return

class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_encoder_optimizer(self, model):
        model_params = list(filter(lambda p: p.requires_grad, model.encoder.parameters())) + list(filter(lambda p: p.requires_grad, model.mid_model.parameters()))
        optimizer = torch.optim.AdamW(model_params, lr=cfg.lr, weight_decay=cfg.decay)
        return optimizer

    def get_optimizer(self, model):
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(model_params, lr=cfg.lr, weight_decay=cfg.decay)
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path) 
        start_epoch = ckpt['epoch'] + 1
        self.logger.info(model.load_state_dict(ckpt['network'], strict=False))
        # optimizer.load_state_dict(ckpt['optimizer']) ###uncomment for continue training

        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            if cfg.model_name == 'adapt':
                for g in self.encoder_optimizer.param_groups:
                    g['lr'] = cfg.lr * (cfg.lr_dec_factor ** idx)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr * (cfg.lr_dec_factor ** idx)
        else:
            if cfg.model_name == 'adapt':
                for g in self.encoder_optimizer.param_groups:
                    g['lr'] = cfg.lr * (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr * (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        if cfg.model_name == 'adapt':
            for g in self.encoder_optimizer.param_groups:
                cur_lr = g['lr']
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self, args=None):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        # train_dataset = eval(cfg.trainset)(transforms.ToTensor(), "train")
        if args.trainset == 'FREIHAND':
            self.batch_generator, dataset_len = make_hand_data_loader(args, args.train_yaml,
                                                False, is_train=True, scale_factor=1)#DataLoader(dataset=train_dataset, batch_size=cfg.num_gpus*cfg.train_batch_size, shuffle=True, num_workers=cfg.num_thread, pin_memory=True)
        elif args.trainset == 'INTERHAND':
            if cfg.model_name == 'adapt' or cfg.data_type == 'interhand_adapt': ## or cfg.model_name == 'real2d'
                dataset = InterHand_adapt(args.data_dir, split='train', img_size=args.input_img_shape)
            elif cfg.data_type == 'interhand_subset':
                dataset = InterHand_subset(args.data_dir, split='train', img_size=args.input_img_shape)
            elif cfg.data_type == 'interhand_sdf':
                dataset = InterHandsdf_dataset(args.data_dir, split='train', img_size=args.input_img_shape, input_cfg=args)
            elif cfg.data_type == 'interhand_dataaug':
                dataset = InterHandaug_dataset(args.data_dir, split='train', img_size=args.input_img_shape)
            elif cfg.data_type == 'interhand_xinchuan':
                dataset = InterHandxinchuan_dataset(args.data_dir, split='train', img_size=args.input_img_shape)
            elif cfg.data_type == 'xinchuan_subset':
                dataset = InterHandxinchuan_subset(args.data_dir, split='train', img_size=args.input_img_shape)
            elif cfg.data_type == 'diffaug':
                dataset = Diffaug_dataset(args.data_dir, split='train', img_size=args.input_img_shape)
            else:
                dataset = InterHand_dataset(args.data_dir, split='train', img_size=args.input_img_shape)
            self.batch_generator = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True,
                         num_workers=args.num_workers, pin_memory=True)
            dataset_len = len(dataset)
        elif args.trainset == 'HO3D':
            train_dataset = HO3D(transforms.ToTensor(), "train")
            self.itr_per_epoch = math.ceil(len(train_dataset) / cfg.num_gpus / cfg.train_batch_size)
            self.batch_generator = DataLoader(dataset=train_dataset, batch_size=cfg.num_gpus * cfg.train_batch_size,
                                              shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
            dataset_len = len(train_dataset)
        elif args.trainset == 'DEX_YCB':
            train_dataset = DEX_YCB(transforms.ToTensor(), "train")
            self.itr_per_epoch = math.ceil(len(train_dataset) / cfg.num_gpus / cfg.train_batch_size)
            self.batch_generator = DataLoader(dataset=train_dataset, batch_size=cfg.num_gpus * cfg.train_batch_size,
                                              shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
            dataset_len = len(train_dataset)
        self.logger.info("dataset length {}".format(dataset_len))
        self.itr_per_epoch = math.ceil(dataset_len / args.num_gpus / args.train_batch_size)

    def _make_model(self, cfg, epoch=0):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        if cfg.trainset == 'INTERHAND':
            if cfg.model_name == 'myhand':
                model = load_model('common/myhand/defaults.yaml')
            elif cfg.model_name == 'regressor':
                model = ParamRegressor()
            elif cfg.model_name == 'intag_regress':
                model = fullModel()
            elif cfg.model_name == 'cliff':
                model = load_model('common/myhand/defaults.yaml', cliff=True)
            elif cfg.model_name == 'adapt':
                model = load_adapt('common/myhand/defaults.yaml')
                # target_model = load_adapt('common/myhand/defaults.yaml')
            elif cfg.model_name == 'real2d':
                model = load_real2d('common/myhand/defaults.yaml')
            elif cfg.model_name == 'nohms':
                model = load_nohms_model('common/myhand/defaults.yaml')
            elif cfg.model_name == 'graph':
                model = load_graph_model('common/myhand/defaults.yaml')
            elif cfg.model_name == 'newgraph':
                model = load_new_model('common/myhand/defaults.yaml')
            else:
                model = get_double_model('train')
        else:
            model = get_model('train')

        if len(cfg.gpu_ids) > 1:
            model = DataParallel(model).cuda()
            # if cfg.model_name == 'adapt':
            #     target_model = DataParallel(target_model).cuda()
        else:
            # if cfg.model_name == 'adapt':
                # target_model.cuda()
            model.cuda()

        optimizer = self.get_optimizer(model)
        if cfg.model_name == 'adapt': #and cfg.continue_train:
            # self.target_optimizer = self.get_optimizer(target_model)
            self.encoder_optimizer = self.get_encoder_optimizer(model)
            # _, target_model, _ = self.load_model(target_model, optimizer)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
            self.logger.info('continue training ')
        else:
            start_epoch = 0
        model.train()
        if cfg.model_name == 'adapt':
            # self.target_model = target_model
            # self.target_model.eval()
            model.eval()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

class Tester(Base):
    def __init__(self):
        super(Tester, self).__init__(log_name = 'test_logs.txt')

    def _make_batch_generator(self, args=None):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        self.test_dataset = None#eval(cfg.testset)(transforms.ToTensor(), "test")
        if args.trainset == 'FREIHAND':
            self.batch_generator, dataset_len = make_hand_data_loader(args, args.test_yaml,
                                            False, is_train=False, scale_factor=1)#DataLoader(dataset=self.test_dataset, batch_size=cfg.num_gpus*cfg.test_batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
        elif args.trainset == 'INTERHAND':
            # if cfg.model_name == 'adapt' or cfg.data_type == 'interhand_adapt': ### for double discriminator
            #     dataset = InterHand_adapt(args.data_dir, split='test', img_size=args.input_img_shape)
            # else:
            if cfg.data_type == 'interhand_dataaug':
                dataset = InterHandaug_dataset(args.data_dir, split='test', img_size=args.input_img_shape)
            elif cfg.data_type == 'interhand_sdf':
                dataset = InterHandsdf_dataset(args.data_dir, split='test', img_size=args.input_img_shape, input_cfg=args)
            elif cfg.data_type == 'interhand_xinchuan':
                dataset = InterHandxinchuan_dataset(args.data_dir, split='test', img_size=args.input_img_shape)
            else:
                dataset = InterHand_dataset(args.data_dir, split='test', img_size=args.input_img_shape)
            self.batch_generator = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
            dataset_len = len(dataset)
        elif args.testset == 'HO3D':
            self.test_dataset = HO3D(transforms.ToTensor(), "test")
            self.batch_generator = DataLoader(dataset=self.test_dataset, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                              shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
            dataset_len = len(self.test_dataset)
        elif args.testset == 'DEX_YCB':
            self.test_dataset = DEX_YCB(transforms.ToTensor(), "test")
            self.itr_per_epoch = math.ceil(len(self.test_dataset) / cfg.num_gpus / cfg.test_batch_size)
            self.batch_generator = DataLoader(dataset=self.test_dataset, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                              shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
            dataset_len = len(self.test_dataset)
        self.logger.info("dataset length {} ".format(dataset_len))
        self.itr_per_epoch = math.ceil(dataset_len / args.num_gpus / args.test_batch_size)

    def _make_model(self, cfg, epoch=0, load_ckpt=True):
        model_path = os.path.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(epoch))
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        if cfg.testset == 'INTERHAND':
            if cfg.model_name == 'myhand':
                model = load_model('common/myhand/defaults.yaml')
            elif cfg.model_name == 'regressor':
                model = ParamRegressor()
            elif cfg.model_name == 'intag_regress':
                model = fullModel()
            elif cfg.model_name == 'cliff':
                model = load_model('common/myhand/defaults.yaml', cliff=True)
            elif cfg.model_name == 'adapt':
                model = load_adapt('common/myhand/defaults.yaml')
                # target_model = load_adapt('common/myhand/defaults.yaml')
            elif cfg.model_name == 'real2d':
                model = load_real2d('common/myhand/defaults.yaml')
            elif cfg.model_name == 'nohms':
                model = load_nohms_model('common/myhand/defaults.yaml')
            elif cfg.model_name == 'graph':
                model = load_graph_model('common/myhand/defaults.yaml')
            elif cfg.model_name == 'newgraph':
                model = load_new_model('common/myhand/defaults.yaml')
            else:
                model = get_double_model('test')
        else:
            model = get_model('test')
        if len(cfg.gpu_ids) > 1 or cfg.testset == 'HO3D' or cfg.testset == 'DEX_YCB':
            # if cfg.model_name == 'adapt':
                # target_model = DataParallel(target_model).cuda()
            model = DataParallel(model).cuda()
        else:
            # if cfg.model_name == 'adapt':
            #     target_model.cuda()
            model.cuda()
        if load_ckpt == True:
            ckpt = torch.load(model_path)
            self.logger.info(model.load_state_dict(ckpt['network'], strict=False))
        model.eval()
        # if cfg.model_name == 'adapt':
            # target_model.eval()
            # self.target_model = target_model

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.test_dataset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, test_epoch):
        self.test_dataset.print_eval_result(test_epoch)