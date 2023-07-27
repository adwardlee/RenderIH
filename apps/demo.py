import numpy as np
import torch
import cv2 as cv
import glob
import os
import argparse


import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import load_model
from utils.config import load_cfg
from utils.manoutils import get_mano_path, imgUtils
from dataset.dataset_utils import IMG_SIZE
from core.test_utils import InterRender


def cut_img(img, bbox):
    cut = img[max(int(bbox[2]), 0):min(int(bbox[3]), img.shape[0]),
              max(int(bbox[0]), 0):min(int(bbox[1]), img.shape[1])]
    cut = cv.copyMakeBorder(cut,
                            max(int(-bbox[2]), 0),
                            max(int(bbox[3] - img.shape[0]), 0),
                            max(int(-bbox[0]), 0),
                            max(int(bbox[1] - img.shape[1]), 0),
                            borderType=cv.BORDER_CONSTANT,
                            value=(0, 0, 0))
    return cut


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='misc/model/config.yaml')
    parser.add_argument("--model", type=str, default='misc/model/wild_demo.pth')
    parser.add_argument("--live_demo", action='store_true')
    parser.add_argument("--img_path", type=str, default='demo/')
    parser.add_argument("--save_path", type=str, default='demo/')
    parser.add_argument("--render_size", type=int, default=256)
    opt = parser.parse_args()

    model = InterRender(cfg_path=opt.cfg,
                        model_path=opt.model,
                        render_size=opt.render_size)

    if not opt.live_demo:
        data_path = '/home/admin/workspace/dataset/interhand_5fps/interhand_data/'
        split = 'test'
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).reshape(3, 1, 1)
        std = torch.tensor(std).reshape(3, 1, 1)
        for idx in [100373, 102133, 107929, 104934, 104809, 103767, 150967, 151031,150217,150556,151993, 156266, 157848,158571,158580,158584,159081,159097,161209, 151160]:#[161209, 151160, 150217, 158571, 159097]:#tqdm(range(151000, 251000)):#[100373, 102133, 107929, 104934, 104809, 103767, 150967, 151031,150217,150556,151993, 156266, 157848,158571,158580,158584,159081,159097]:
            inputs = {}
            targets = {}
            thearray = np.load(os.path.join(data_path, split, 'all', '{}.npy'.format(idx)), allow_pickle=True)
            tmpimg = thearray[()]['img'].reshape(1, 3, 256, 256)[0]
            img = tmpimg * std + mean
            img = (img.permute(1, 2, 0) * 255).numpy().astype(np.uint8)[:,:,::-1]  ### chw 0-1  -> hwc 0-255
            params = model.run_model(img)
            img_overlap = model.render(params, bg_img=img)
            cv.imwrite(os.path.join(opt.save_path, str(idx) + '_output.jpg'), img_overlap)



        # img_path_list = glob.glob(os.path.join(opt.img_path, '*.jpg')) + glob.glob(os.path.join(opt.img_path, '*.png'))
        # for img_path in img_path_list:
        #     img_name = os.path.basename(img_path)
        #     if img_name.find('output.jpg') != -1:
        #         continue
        #     img_name = img_name[:img_name.find('.')]
        #     img = cv.imread(img_path)
        #     params = model.run_model(img)
        #     img_overlap = model.render(params, bg_img=img)
        #     cv.imwrite(os.path.join(opt.save_path, img_name + '_output.jpg'), img_overlap)
    else:
        video_reader = cv.VideoCapture(0)
        fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_reader.set(cv.CAP_PROP_FOURCC, fourcc)

        smooth = False
        params_last = None
        params_last_v = None
        params_v = None
        params_a = None

        fIdx = 0
        with torch.no_grad():
            while True:
                fIdx = fIdx + 1
                _, img = video_reader.read()
                if img is None:
                    exit()
                w = min(img.shape[1], img.shape[0]) / 2 * 0.6
                left = int(img.shape[1] / 2 - w)
                top = int(img.shape[0] / 2 - w)
                size = int(2 * w)
                bbox = [left, left + size, top, top + size]
                bbox = np.array(bbox).astype(np.int32)
                crop_img = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]

                params = model.run_model(crop_img)
                if smooth and params_last is not None and params_v is not None and params_a is not None:
                    for k in params.keys():
                        if isinstance(params[k], torch.Tensor):
                            pred = params_last[k] + params_v[k] + 0.5 * params_a[k]
                            params[k] = (0.7 * params[k] + 0.3 * pred)

                img_out = model.render(params, bg_img=crop_img)
                img[bbox[2]:bbox[3], bbox[0]:bbox[1]] = cv.resize(img_out, (size, size))
                cv.line(img, (int(bbox[0]), int(bbox[2])), (int(bbox[0]), int(bbox[3])), (0, 0, 255), 2)
                cv.line(img, (int(bbox[1]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), (0, 0, 255), 2)
                cv.line(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[2])), (0, 0, 255), 2)
                cv.line(img, (int(bbox[0]), int(bbox[3])), (int(bbox[1]), int(bbox[3])), (0, 0, 255), 2)
                cv.imshow('cap', img)

                if params_last is not None:
                    params_v = {}
                    for k in params.keys():
                        if isinstance(params[k], torch.Tensor):
                            params_v[k] = (params[k] - params_last[k])
                if params_last_v is not None and params_v is not None:
                    params_a = {}
                    for k in params.keys():
                        if isinstance(params[k], torch.Tensor):
                            params_a[k] = (params_v[k] - params_last_v[k])
                params_last = params
                params_last_v = params_v

                key = cv.waitKey(1)

                if key == 27:
                    exit()
