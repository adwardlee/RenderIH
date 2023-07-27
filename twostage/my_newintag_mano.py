import os, sys
import cv2
import json
import torch
from operator import imod
import numpy as np
from tqdm import tqdm
from os.path import join
import mmcv
from mmpose.core.post_processing.temporal_filters.savizky_golay_filter import SavizkyGolayFilter


from twostage.smplmodel import load_model, merge_params, select_nf
from twostage.handocc_api import init_with_spin
from twostage.pipeline.mirror import multi_stage_optimize, multi_stage_optimize_single
from twostage.smplmodel.manolayer import ManoLayer

from models.manolayer import ManoLayer
from twostage.mediapipe_2d import img_to_2djoint

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def bbox_revise(bbox):
    return [bbox[1], bbox[0], bbox[1] + bbox[2], bbox[0] + bbox[3]]

def demo_1v1pmf_smpl_mirror(path, body_model, spin_model, args, combine_model=None):
    subs = args.sub
    assert len(subs) > 0

    sg_filter = SavizkyGolayFilter(33)
    hands = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.12,
        min_tracking_confidence=0.3)
    # 遍历所有文件夹
    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Faild to load video file {args.video_path}'
    frame_idx = 0
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        frame_idx += 1
        image_name = 'image_' + str(frame_idx).zfill(6) + '.jpg'
        ori_image = cur_frame
        img_size = ori_image.shape[:2]
        joints2d = img_to_2djoint(hands, ori_image)



    for sub in subs:
        dataset = ImageFolderSingle(path, subs=[sub], out=args.out, kpts_type=args.body)
        start, end = args.start, min(args.end, len(dataset))
        frames = list(range(start, end, args.step))
        nFrames = len(frames)
        pids = [0]
        body_params_left = {pid:[None for nf in frames] for pid in pids}
        body_params_right = {pid: [None for nf in frames] for pid in pids}
        bboxes = {pid:[None for nf in frames] for pid in pids}
        keypoints2d = {pid:[None for nf in frames] for pid in pids}
        vertices_left = {pid: [None for nf in frames] for pid in pids}
        vertices_right = {pid: [None for nf in frames] for pid in pids}
        for nf in tqdm(frames, desc='loading'):
            image, annots = dataset[nf]
            # 这个时候如果annots不够 不能够跳过了，需要进行补全
            camera = dataset.camera(nf)
            # 初始化每个人的SMPL参数
            for i, annot in enumerate(annots):
                if 'single_flag' not in annot:
                    annot['single_flag'] = 0
                pid = annot['id']
                if pid not in pids:
                    continue

                if annot['single_flag']:### if two single hand ###
                    result = init_with_spin(body_model, spin_model, image,
                                            [annot['left_box'], annot['right_box']],annot['twohand'], camera, annot['single_flag'], combine_model, args.input_res)
                else:
                    result = init_with_spin(body_model, spin_model, image,
                        annot['rectbbox'], annot['twohand'], camera, annot['single_flag'], combine_model, args.input_res)

                body_params_left[pid][nf-start] = result['body_params_left']
                body_params_right[pid][nf - start] = result['body_params_right']
                bboxes[pid][nf-start] = annot['bbox']
                keypoints2d[pid][nf-start] = annot['twohand']
                vertices_left[pid][nf - start] = result['vertices_left']
                vertices_right[pid][nf - start] = result['vertices_right']


        # stack [p1f1, p1f2, p1f3, ..., p1fn, p2f1, p2f2, p2f3, ..., p2fn]
        # TODO:for missing bbox
        body_params_left = merge_params([merge_params(body_params_left[pid], share_shape=False) for pid in pids[:1]])
        body_params_right = merge_params([merge_params(body_params_right[pid], share_shape=False) for pid in pids[:1]])
        body_params_left['Th'] = sg_filter(body_params_left['Th'].reshape(-1, 1, 3)).reshape(-1, 3)
        body_params_left['shapes'] = sg_filter(body_params_left['shapes'].reshape(-1, 1, 10)).reshape(-1, 10).mean(axis=0, keepdims=True)
        body_params_right['Th'] = sg_filter(body_params_right['Th'].reshape(-1, 1, 3)).reshape(-1, 3)
        body_params_right['shapes'] = sg_filter(body_params_right['shapes'].reshape(-1, 1, 10)).reshape(-1, 10).mean(axis=0, keepdims=True)

        # bboxes: (nViews, nFrames, 5)
        bboxes = np.stack([np.stack(bboxes[pid]) for pid in pids[:1]])
        # keypoints: (nViews, nFrames, nJoints, 3)
        keypoints2d = np.stack([np.stack(keypoints2d[pid]) for pid in pids[:1]])
        vertices_left = np.stack([np.stack(vertices_left[pid]) for pid in pids[:1]])
        vertices_right = np.stack([np.stack(vertices_right[pid]) for pid in pids[:1]])

        np.save('body_left_jieyin', body_params_left)
        np.save('body_right_jieyin', body_params_right)
        np.save('bboxes_jieyin', bboxes)
        np.save('keypoints2d_jieyin', keypoints2d)
        np.save('vertices_left', vertices_left)
        np.save('keypoints_right', vertices_right)
        #
        body_params_left = np.load('body_left_jieyin.npy', allow_pickle=True)[()]
        body_params_right = np.load('body_right_jieyin.npy', allow_pickle=True)[()]
        bboxes = np.load('bboxes_jieyin.npy', allow_pickle=True)
        keypoints2d = np.load('keypoints2d_jieyin.npy', allow_pickle=True)

        ################## get first 200 frames ###################


        # optimize
        P = dataset.camera(start)['P']
        if args.normal:
            normal = dataset.normal_all(start=start, end=end)
        else:
            normal = None
        body_model['left'].to('cuda')
        body_model['right'].to('cuda')
        body_params_left = multi_stage_optimize_single(body_model['left'], body_params_left, bboxes, keypoints2d[:, :, :21], Pall=P, normal=normal, args=args)
        body_params_right = multi_stage_optimize_single(body_model['right'], body_params_right, bboxes, keypoints2d[:, :, 21:],
                                                  Pall=P, normal=normal, args=args)

        ########## get ori results ##############
        # body_params_left = ori_left
        # body_params_right = ori_right
        # for key in body_params_left.keys():
        #     body_params_left[key] = body_params_left[key]
        # for key in body_params_right.keys():
        #     body_params_right[key] = body_params_right[key]
        ###########################################
        body_model['left'].to('cpu')
        body_model['right'].to('cpu')
        # write
        vertices_left, keypoints_left = body_model['left'](body_params_left)
        vertices_right, keypoints_right = body_model['right'](body_params_right)
        vertices_left = vertices_left.numpy()
        keypoints_left = keypoints_left.numpy()
        vertices_right = vertices_right.numpy()
        keypoints_right = keypoints_right.numpy()
        dataset.no_img = not args.vis_smpl
        for nf in tqdm(frames, desc='rendering'):
            idx = nf - start
            write_data = [{'id': pids[i], 'keypoints3d_left': keypoints_left[i*nFrames+idx], 'keypoints3d_right': keypoints_right[i*nFrames+idx]} for i in range(len(pids[:1]))]
            dataset.write_keypoints3d(write_data, nf)
            write_data = [{'id': pids[i], 'vertices_left': vertices_left[i*nFrames+idx], 'vertices_right': vertices_right[i*nFrames+idx]} for i in range(len(pids[:1]))]
            dataset.write_vertices(write_data, nf)
            for i in range(len(pids[:1])):
                write_data[i].update(select_nf(body_params_left, i*nFrames+idx))
            dataset.write_smpl(write_data, nf)
            # 保存结果
            if args.vis_smpl:
                image, annots = dataset[nf]
                camera = dataset.camera(nf)
                render_data = {pids[i]: {
                    'vertices_left': vertices_left[i*nFrames+idx],
                    'faces_left': body_model['left'].faces,
                    'vertices_right': vertices_right[i * nFrames + idx],
                    'faces_right': body_model['right'].faces,
                    'vid': 0, 'name': 'human_{}'.format(pids[i])} for i in range(len(pids[:1]))}
                dataset.vis_double_smpl(render_data, image, camera, nf)
        raise

if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    parser = load_parser()
    parser.add_argument('--skel', type=str, default=None,
        help='path to keypoints3d')
    parser.add_argument('--input_res', type=int, default=256,
                        help='input resolution')
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--video', action='store_false')
    parser.add_argument('--gtK', action='store_true')
    parser.add_argument('--normal', action='store_true',
        help='set to use the normal of the mirror')
    parser.add_argument('--double', type=int, default=1)
    args = parse_parser(parser)

    helps = '''
  Demo code for single view and one person with mirror:

    - Input : {}: [{}]
    - Output: {}
    - Body  : {} => {}, {}
    '''.format(args.path, ', '.join(args.sub), args.out,
        args.model, args.gender, args.body)
    print(helps)
    with Timer('Loading {}, {}'.format(args.model, args.gender)):
        body_model = {'right': ManoLayer('handoccnet/MANO_RIGHT.pkl'),
                      'left': ManoLayer('handoccnet/MANO_LEFT.pkl')}
        if torch.sum(torch.abs(body_model['left'].shapedirs[:, 0, :] - body_model['right'].shapedirs[:, 0, :])) < 1:
            print('Fix shapedirs bug of MANO')
            body_model['left'].shapedirs[:, 0, :] *= -1
        # body_model['right']
        # body_model['left']
    with Timer('Loading SPIN'):
        spin_model = get_intag_model()

        mano_model = MANO().cuda()
        mano_model.layer = mano_model.layer.cuda()
        mano_model.eval()
    demo_1v1pmf_smpl_mirror(args.path, body_model, spin_model, args, mano_model)

