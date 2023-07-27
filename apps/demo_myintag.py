import argparse
import torch
from tqdm import tqdm
import numpy as np
import random
import time
import cv2
import os

from common.utils.mano import MANO

from core.graph_model import GraphRender
from utils.vis_utils import mano_two_hands_renderer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_vert_obj(path, verts, faces):
    with open(path, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--model_path', type=str, default='0', dest='model_path')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():
    # argument parse and create log
    args = parse_args()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    model = GraphRender('utils/defaults.yaml', args.model_path)


    start = time.time()
    # img_path = '/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/rgb2hands/RGB2HANDS_Benchmark/data/seq01_crossed/color/'## seq02_occlusion  seq03_shuffle seq04_scratch
    # img_path = '/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/tziona/original/cropimg/'
    ## seq01_crossed seq02_occlusion  seq03_shuffle  seq04_scratch
    img_path = '/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/rgb2hands/RGB2HANDS_Benchmark/data/cropimg/'
    imgs = [img_path + x for x in os.listdir(img_path) if ('.jpg' in x) or ('.png' in x)]
    for imgname in imgs:
        print('img name ', imgname, flush=True)
        img = cv2.imread(imgname)

        # forward
        start = time.time()
        with torch.no_grad():
            out = model.run_model(img)
        end = time.time()
        img_overlap = model.render(out, bg_img=img)
        print('cur time ', end - start , flush=True)
        cv2.imwrite(os.path.join('rebut/ori_rgb2hands/', imgname.split('/')[-1].split('.jpg')[0] + '_output.jpg'), img_overlap)


if __name__ == "__main__":
    main()
