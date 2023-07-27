import os
import warnings
from argparse import ArgumentParser
import numpy as np
import json
import cv2
import pickle
import mmcv
import math
from main.config import cfg
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
import imageio
import time
from torchvision.transforms import Normalize

from mmpose.apis import (collect_multi_frames, get_track_id,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_tracking_result)
from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo

from onestage.mmpose_utils import process_image,get_graph_model,get_color,rendering, Bbox
from onestage.vis_utils import mano_two_hands_renderer
from onestage.mmpose.sort import Sort, iou_batch
from onestage.mmpose_smooth import HandTrackerConfig, SmoothCallback, Hand3dResult
from common.utils.mano import MANO
from common.utils.manolayer import rodrigues_batch
from core.graph_model import GraphRender
from utils.manoutils import projection_batch

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def bbox_intersection(bbox1, bbox2):
    one_x1, one_y1, one_x2, one_y2 = bbox1.x1, bbox1.y1, bbox1.x2, bbox1.y2
    two_x1, two_y1, two_x2, two_y2 = bbox2.x1, bbox2.y1, bbox2.x2, bbox2.y2
    output = Bbox(max(one_x1, two_x1), max(one_y1, two_y1), min(one_x2, two_x2), max(one_y2, two_y2))
    return output

def classbox_iou(bbox1, bbox2):
    left, top, right, bottom = bbox1.x1, bbox1.y1, bbox1.x2, bbox1.y2
    min_x, min_y, max_x, max_y = bbox2.x1, bbox2.y1, bbox2.x2, bbox2.y2
    intersection = max(min(right, max_x) - max(left, min_x), 0) * max(min(bottom, max_y) - max(top, min_y), 0)
    union = (max(right, max_x) - min(left, min_x)) * (max(bottom, max_y) - min(top, min_y))
    return intersection / union

def bbox_dist_ratio(bbox1, bbox2):
    ratio = 1.4
    left, top, right, bottom = bbox1
    width1 = (right - left)/2
    height1 = (bottom - top)/2
    center1 = np.array([(left+right)/2, (top+bottom)/2])
    min_x, min_y, max_x, max_y = bbox2
    width2 = (max_x - min_x)/2
    height2 = (max_y - min_y)/2
    center2 = np.array([(min_x+max_x)/2, (min_y+max_y)/2])
    intersection = max(min(right, max_x) - max(left, min_x), 0) * max(min(bottom, max_y) - max(top, min_y), 0)
    union = (max(right, max_x) - min(left, min_x)) * (max(bottom, max_y) - min(top, min_y))
    iou = intersection / union
    if iou > 0:
        return 1
    if (np.abs(center1[0] - center2[0]) < ratio * (width1 + width2)) and \
            (np.abs(center1[1] - center2[1]) < ratio * (height1 + height2)):
        return 1

    return 0

def bbox_iou(bbox1, bbox2):
    left, top, right, bottom = bbox1
    tmphand = bbox2### 21 x 2
    min_x, min_y, max_x, max_y = tmphand
    intersection = max(min(right, max_x) - max(left, min_x), 0) * max(min(bottom, max_y) - max(top, min_y), 0)
    union = (max(right, max_x) - min(left, min_x)) * (max(bottom, max_y) - min(top, min_y))

    return intersection / union

def get_bbox_from_pose(pose_2d, height=None, width=None, rate = 0.15):
    # this function returns bounding box from the 2D pose
    # here use pose_2d[:, -1] instead of pose_2d[:, 2]
    # because when vis reprojection, the result will be (x, y, depth, conf)
    validIdx = pose_2d[:, -1] > 0
    if validIdx.sum() == 0:
        return [0, 0, 100, 100, 0]
    y_min = int(min(pose_2d[validIdx, 1]))
    y_max = int(max(pose_2d[validIdx, 1]))
    x_min = int(min(pose_2d[validIdx, 0]))
    x_max = int(max(pose_2d[validIdx, 0]))
    # length = max(y_max - y_min, x_max - x_min)
    # center_x = (x_min + x_max) // 2
    # center_y = (y_min + y_max) // 2
    # y_min = center_y - length // 2
    # y_max = center_y + length // 2
    # x_min = center_x - length // 2
    # x_max = center_x + length // 2
    dx = (x_max - x_min)*rate
    dy = (y_max - y_min)*rate

    # 后面加上类别这些
    bbox = [x_min-dx, y_min-dy, x_max+dx, y_max+dy, 1]
    if height is not None and width is not None:
        bbox = [max(0, bbox[0]), max(0, bbox[1]), min(width - 1, bbox[2]), min(height - 1, bbox[3])]
    return bbox

def cropimg(cur_frame, bbox, save_reso=256):
    scale_rate = 1

    frame_h, frame_w, _ = cur_frame.shape

    bbox_w = math.floor(bbox[2]) - math.ceil(bbox[0])
    bbox_h = math.floor(bbox[3]) - math.ceil(bbox[1])
    crop_size = min(int(max(bbox_w, bbox_h) * scale_rate), max(frame_h, frame_w))
    background = np.zeros((crop_size, crop_size, 3))
    vis_img = np.zeros((crop_size * 2, crop_size * 2, 3))


    # x1 = max(int(int(bbox[0]) - (crop_size - bbox_w) / 2), 0)
    # x2 = min(int(int(bbox[0]) + (crop_size + bbox_w) / 2), frame_w - 1)
    # y1 = max(int(int(bbox[1]) - (crop_size - bbox_h) / 2), 0)
    # y2 = min(int(int(bbox[1]) + (crop_size + bbox_h) / 2), frame_h - 1)

    x1 = max(math.ceil(bbox[0]), 0)
    x2 = min(math.floor(bbox[2]), frame_w - 1)
    y1 = max(math.ceil(bbox[1]), 0)
    y2 = min(math.floor(bbox[3]), frame_h - 1)

    bx1 = max(int((crop_size - bbox_w) // 2), 0)
    bx2 = min(bx1 + (x2 - x1), crop_size)
    by1 = max(int((crop_size - bbox_h) // 2), 0)
    by2 = min(by1 + (y2 - y1), crop_size)

    length_w = bbox_w
    length_h = bbox_h
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    vis_curx1 = max(center_x - length_w, 0)
    vis_curx2 = min(center_x + length_w, frame_w - 1)
    vis_cury1 = max(center_y - length_h, 0)
    vis_cury2 = min(center_y + length_h, frame_h - 1)
    vis_width = min(vis_curx2 - vis_curx1, 2 * crop_size)
    vis_height = min(vis_cury2 - vis_cury1, 2 * crop_size)

    vis_bgx1 = max(crop_size - vis_width // 2, 0)
    vis_bgx2 = min(vis_bgx1 + vis_width, 2 * crop_size)
    vis_bgy1 = max(crop_size - vis_height // 2, 0)
    vis_bgy2 = min(vis_bgy1 + vis_height, 2 * crop_size)



    try:
        background[by1:by2, bx1:bx2, :] = cur_frame[y1:y2, x1:x2, :]
        vis_img[vis_bgy1:vis_bgy2, vis_bgx1:vis_bgx2] = cur_frame[vis_cury1:vis_cury2, vis_curx1:vis_curx2]
    except:
        print(' frame_h {}, frame_w {}, bbox {}, x1 {}, x2 {}, y1 {}, y2 {}'.format(frame_h, frame_w, bbox, x1, x2, y1, y2), flush=True)
        print(' crop_size {},  bbox {}, x1 {}, x2 {}, y1 {}, y2 {}'.format(crop_size, bbox, vis_bgx1, vis_bgx2, vis_bgy1,
                                                                                    vis_bgy2), flush=True)
    try:
        crop = cv2.resize(background, (save_reso, save_reso))
        vis_img = cv2.resize(vis_img, (save_reso * 2, save_reso * 2))
    except:
        print(' frame_h {}, frame_w {}, bbox {}, x1 {}, x2 {}, y1 {}, y2 {}'.format(frame_h, frame_w, bbox, x1, x2, y1, y2), flush=True)
        exit()
    cur_box = Bbox(x1,y1,x2,y2)
    bg_box = Bbox(bx1, by1, bx2, by2)
    s2_cur_box = Bbox(vis_curx1, vis_cury1, vis_curx2, vis_cury2)
    s2_bg_box = Bbox(vis_bgx1, vis_bgy1, vis_bgx2, vis_bgy2)
    return cur_box, crop_size, bg_box, crop, vis_img, s2_cur_box, s2_bg_box

def mkdir_with_check(path):
    if not os.path.exists(path):
        os.mkdir(path)


def yxyx2xywh(yxyx):
    return [yxyx[1], yxyx[0], yxyx[2] - yxyx[0], yxyx[3] - yxyx[1]]

def main():
    """Visualize the demo images.
    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('--det_config', type=str, default='/mnt/workspace/code/hand/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py', help='Config file for detection')
    parser.add_argument('--det_checkpoint', type=str, default='/mnt/workspace/code/hand/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', help='Checkpoint file for detection')
    parser.add_argument('--pose_config', type=str, default='/mnt/workspace/code/hand/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py', help='Config file for pose')
    parser.add_argument('--pose_checkpoint', type=str, default='/mnt/workspace/code/hand/mmpose/demo/mmdetection_cfg/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth', help='Checkpoint file for pose')
    parser.add_argument('--mesh_checkpoint', type=str,
                        default='/mnt/workspace/code/hand/ori_intag/output/mano/realsyn122w_pretrain/model/exp/25.pth',
                        help='Checkpoint file for pose')

    parser.add_argument('--video-path', type=str, default='videos/20230323.mp4',help='Video path')
    parser.add_argument('--vis', type=int, default=1, help='Video path')### save 3d hand pose or not ###
    parser.add_argument('--save', type=int, default=1, help='Video path')### save 2d pose or not ###
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='video/',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.5, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='(Deprecated, please use --smooth and --smooth-filter-cfg) '
        'Using One_Euro_Filter for smoothing.')
    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Apply a temporal filter to smooth the pose estimation results. '
        'See also --smooth-filter-cfg.')
    parser.add_argument(
        '--smooth-filter-cfg',
        type=str,
        default='/mnt/workspace/code/hand/mmpose/configs/_base_/filters/one_euro.py',
        help='Config file of the filter to smooth the pose estimation '
        'results. See also --smooth.')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the pose'
        'estimation stage. Default: False.')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    print('Initializing model...')
    ### init mano left/right model ###
    mano_left = MANO('left').cuda()
    mano_left_layer = mano_left.layer.cuda()
    mano_right = MANO('right').cuda()
    mano_right_layer = mano_right.layer.cuda()
    if args.vis:
        render = mano_two_hands_renderer(img_size=256, device='cuda')
        s2_render = mano_two_hands_renderer(img_size=512, device='cuda')
    track_cfg = HandTrackerConfig()
    left_smoothcall = SmoothCallback(track_cfg.tk_smooth_config)
    right_smoothcall = SmoothCallback(track_cfg.tk_smooth_config)
    mesh_predictor = GraphRender('utils/defaults.yaml', args.mesh_checkpoint)
    prev_left_result = None
    prev_right_result = None
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # read video
    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = video.fps
        size = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_2d_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)
        video3dWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_3d_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)
        orivideoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'ori_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # frame index offsets for inference, used in multi-frame inference setting
    if args.use_multi_frames:
        assert 'frame_indices_test' in pose_model.cfg.data.test.data_cfg
        indices = pose_model.cfg.data.test.data_cfg['frame_indices_test']

    # build pose smoother for temporal refinement
    if args.euro:
        warnings.warn(
            'Argument --euro will be deprecated in the future. '
            'Please use --smooth to enable temporal smoothing, and '
            '--smooth-filter-cfg to set the filter config.',
            DeprecationWarning)
        smoother = Smoother(
            filter_cfg='configs/_base_/filters/one_euro.py', keypoint_dim=2)
    elif args.smooth:
        smoother = Smoother(filter_cfg=args.smooth_filter_cfg, keypoint_dim=2)
    else:
        smoother = None

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    next_id = 0
    pose_results = []
    print('Running inference...')


    ### llj mkdir save ###
    # data_to_json = []
    #
    # base_path = os.path.join(args.out_video_root, os.path.basename(args.video_path).split('.')[0])
    # images_path = os.path.join(base_path, 'images')
    # rgb_path = os.path.join(images_path, 'rgb')
    # crop_path = os.path.join(images_path, 'crop')
    # crop_ori_left = os.path.join(images_path, 'crop_ori_left')
    # crop_ori_right = os.path.join(images_path, 'crop_ori_right')
    # results_path = os.path.join(images_path, 'results')
    #
    # mkdir_with_check(base_path)
    # mkdir_with_check(images_path)
    # mkdir_with_check(rgb_path)
    # mkdir_with_check(crop_path)
    # mkdir_with_check(crop_ori_left)
    # mkdir_with_check(crop_ori_right)
    # mkdir_with_check(results_path)
    #####################
    outnumpy = {}
    outleftvert = {}
    outrightvert = {}
    root_right = torch.zeros((1,1,3)).cuda()
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        ### save img name ###
        image_name = 'image_{:06d}.jpg'.format(frame_id)
        out = None
        v3d_left = None
        v3d_right = None


        pose_results_last = pose_results

        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, cur_frame)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        if args.use_multi_frames:
            frames = collect_multi_frames(video, frame_id, indices,
                                          args.online)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            frames if args.use_multi_frames else cur_frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr)

        # post-process the pose results with smoother
        if smoother:
            pose_results = smoother.smooth(pose_results)


        ### left hand : 92 -> 112; right hand : 113 -> 133
        single_flag = 0
        left_flag = 0
        right_flag = 0
        idx = 0
        total_time = 0
        end_time = start_time = 0
        v_color = get_color()
        for pose_tmp in pose_results:
            pose_tmp['keypoints'][0:5] = 0
            pose_tmp['keypoints'][23:91] = 0
            bbox = pose_tmp['bbox']
            kp2d = pose_tmp['keypoints']
            track_id = pose_tmp['track_id']

            height, width, _ = cur_frame.shape
            if track_id != 0:
                 continue

            left_points = kp2d[91:112].reshape(21,3)#[:, :2]
            left_score = kp2d[91:112].reshape(21,3)[:, 2].mean()
            left_box = get_bbox_from_pose(left_points, height, width, rate=0.35)
            # left_box = [np.clip(min(left_points[:, 0]), 0, width), np.clip(min(left_points[:, 1]), 0, height),
            #             np.clip(max(left_points[:, 0]), 0, width), np.clip(max(left_points[:, 1]), 0, height)]
            right_points = kp2d[112:133].reshape(21, 3)#[:, :2]
            right_box = get_bbox_from_pose(right_points, height, width, rate=0.35)
            right_score = kp2d[112:133].reshape(21, 3)[:, 2].mean()
            # right_box = [np.clip(min(right_points[:, 0]), 0, width), np.clip(min(right_points[:, 1]), 0, height),
            #              np.clip(max(right_points[:, 0]), 0, width), np.clip(max(right_points[:, 1]), 0, height)]

            if left_score >= args.kpt_thr:
                left_flag = True
            if right_score >= args.kpt_thr:
                right_flag = True

            if left_score == False and right_flag == False:
                continue

            out_img = cur_frame.copy()

            start_time = time.time()

            if left_flag and right_flag:
                if bbox_dist_ratio(left_box, right_box) == 0:
                    single_flag = 1
                else:
                    union_box = get_bbox_from_pose(np.concatenate((left_points, right_points), axis=0), height, width, rate=0.3)

                    # track_path = 'track_{}'.format(track_id)
                    # mkdir_with_check(os.path.join(crop_path, track_path))
                    # mkdir_with_check(os.path.join(crop_ori_left, track_path))
                    # mkdir_with_check(os.path.join(crop_ori_right, track_path))

                    # if args.vis:
                    #     cv2.imwrite(os.path.join(rgb_path, image_name), cur_frame)
                    union_curbox, scale_union, union_bgbox, union_img, s2_union_img, s2_union_curbox, s2_union_bgbox = cropimg(
                        cur_frame,
                        union_box, )
                    # os.path.join(
                    #     crop_path,
                    #     track_path,
                    #     image_name))

            if left_flag:
                left_curbox, scale_left, left_bgbox, left_img, s2_left_img, s2_left_curbox, s2_left_bgbox = cropimg(cur_frame,
                                                                                                          left_box,)
                                                                                                          # os.path.join(
                                                                                                          #     crop_ori_left,
                                                                                                          #     track_path,
                                                                                                          #     image_name))

            if right_flag:
                right_curbox, scale_right, right_bgbox, right_img, s2_right_img, s2_right_curbox, s2_right_bgbox = cropimg(
                    cur_frame, right_box,) #os.path.join(crop_ori_right, track_path, image_name))

            if left_flag and right_flag:
                if single_flag == 1:
                    # if bbox_iou(s2_left_curbox, s2_right_curbox) > 0.02:
                    #     overlap_box = bbox_intersection(s2_left_curbox, s2_right_curbox)
                    #     left_overlap = Bbox(overlap_box.x1 - s2_left_curbox.x1 + s2_left_bgbox.x1,
                    #                         overlap_box.y1 - s2_left_curbox.y1 + s2_left_bgbox.y1,
                    #                         overlap_box.x2 - s2_left_curbox.x1 + s2_left_bgbox.x1,
                    #                         overlap_box.y2 - s2_left_curbox.y1 + s2_left_bgbox.y1)###### overlap_box
                    #     right_overlap = Bbox(overlap_box.x1 - s2_right_curbox.x1 + s2_right_bgbox.x1,
                    #                         overlap_box.y1 - s2_right_curbox.y1 + s2_right_bgbox.y1,
                    #                         overlap_box.x2 - s2_right_curbox.x1 + s2_right_bgbox.x1,
                    #                         overlap_box.y2 - s2_right_curbox.y1 + s2_right_bgbox.y1)###### overlap_box

                    proc_left = process_image(left_img)
                    proc_right = process_image(right_img)
                    inputs = {'img': torch.cat([proc_left, proc_right], dim=0).cuda()}
                    out = mesh_predictor.run_mymodel(inputs['img'])
                    pose_param_left = out['mano_pose_left'][0:1]
                    pose_param_right = out['mano_pose_right'][1:2]
                    shape_param_left = out['mano_shape_left'][0:1]
                    shape_param_right = out['mano_shape_right'][1:2]
                    v3d_left = out['v3d_left'][0:1]
                    v3d_right = out['v3d_right'][1:2]
                    outscale_left = out['scale_left'][0:1]
                    outtrans2d_left = out['trans2d_left'][0:1]
                    outscale_right = out['scale_right'][1:2]
                    outtrans2d_right = out['trans2d_right'][1:2]
                    scalelength_left = out['scalelength_left'][0:1]
                    scalelength_right = out['scalelength_right'][1:2]

                    ### smooth left params #################################################
                    cur_left = Hand3dResult(
                        bbox=left_curbox,
                        global_orient=pose_param_left[0, :3],
                        poses=pose_param_left[0, 3:],
                        betas=shape_param_left[0],
                        camera_scale=outscale_left,
                        camera_tran=outtrans2d_left[0],
                        vertices=v3d_left[0],
                        scalelength=scalelength_left,
                    )
                    new_left = left_smoothcall(cur_left, prev_left_result)
                    prev_left_result = new_left

                    outscale_left = new_left.camera_scale
                    outtrans2d_left = new_left.camera_tran[None, :]
                    scalelength_left = new_left.scalelength
                    #v3d_left = new_left.vertices[None, :]
                    pose_param_left = torch.cat((new_left.global_orient[None, :],new_left.poses[None, :]), axis=1)
                    shape_param_left = new_left.betas[None, :]
                    v3d_left, j3d_left = mano_left_layer(rodrigues_batch(pose_param_left[:, :3]), pose_param_left[:, 3:], shape_param_left)
                    v3d_left /= 1000
                    v3d_left = v3d_left - j3d_left[:, 0:1, :] / 1000
                    ####################################################################
                    v3d_right, j3d_right = mano_right_layer(rodrigues_batch(pose_param_right[:, :3]),
                                                            pose_param_right[:, 3:],
                                                            shape_param_right)
                    j3d_right = j3d_right - j3d_right[:, 0:1, :]
                    j3d_right /= 1000
                    j2d_right = projection_batch(outscale_right, outtrans2d_right, j3d_right)
                    pixel_length = (torch.tensor(
                        [right_curbox.x1 - left_curbox.x1, right_curbox.y1 - left_curbox.y1]).cuda().reshape(1, 2))
                    root_rel = (((j2d_right[:, 0] /256 * scale_right + pixel_length) * 256 / scale_left - (outtrans2d_left * 128 + 128)) / (
                                outscale_left * 256)).reshape(1, 1, 2)
                    # pixel_length = (torch.tensor(
                    #     [right_prevbox.x1 - right_curbox.x1, right_prevbox.y1 - right_curbox.y1]).cuda().reshape(1, 2))
                    # root_rel = (((j2d_rightprev + pixel_length) / scale_right * 256 - (outtrans2d_right * 128 + 128))/(outscale_right * 256)).reshape(1,1,2)
                    #############################################
                    # root_rel = (((cur_2dright_root - prev_2dright_root + j2d_rightprev + pixel_length) * 256 / scale_right - (
                    #             outtrans2d_right * 128 + 128))/ (outscale_right * 256)).reshape(1,1,2)
                    # prev_2dright_root = cur_2dright_root
                    # right_prevbox = right_curbox

                    root_right[:,:,:2] = root_rel
                    ### smooth right params ###########################################
                    cur_right = Hand3dResult(
                        bbox=right_curbox,
                        global_orient=pose_param_right[0, :3],
                        poses=pose_param_right[0, 3:],
                        betas=shape_param_right[0],
                        camera_scale=outscale_right,
                        camera_tran=outtrans2d_right[0],
                        vertices=v3d_right[0],
                        scalelength=scalelength_right,
                        rightrel=root_right[0,0]
                    )
                    new_right = right_smoothcall(cur_right, prev_right_result)
                    prev_right_result = new_right

                    outscale_right = new_right.camera_scale
                    outtrans2d_right = new_right.camera_tran[None, :]
                    scalelength_right = new_right.scalelength
                    # v3d_right = new_right.vertices[None, :]
                    pose_param_right = torch.cat((new_right.global_orient[None, :], new_right.poses[None, :]), axis=1)
                    shape_param_right = new_right.betas[None, :]
                    v3d_right, j3d_right = mano_right_layer(rodrigues_batch(pose_param_right[:, :3]), pose_param_right[:, 3:],
                                                  shape_param_right)
                    root_right = new_right.rightrel[None, None, :]
                    v3d_right /= 1000
                    v3d_right = v3d_right - j3d_right[:, 0:1, :] / 1000
                    j3d_right = j3d_right - j3d_right[:, 0:1, :]
                    j3d_right /= 1000
                    j2d_right = projection_batch(outscale_right, outtrans2d_right, j3d_right) * scale_right / 256
                    j2d_rightprev = j2d_right[0, 0:1, :2]
                    ###############################################################################
                    if args.vis:
                        # img_left, img_right = rendering(render, outscale_left, outtrans2d_left, outscale_right, outtrans2d_right, v3d_left,
                        #                                 v3d_right, left_img=left_img, right_img=right_img, two=1, single=1)
                        # resize_img = cv2.resize(img_left, (scale_left, scale_left))
                        # out_img[left_curbox.y1:left_curbox.y2, left_curbox.x1:left_curbox.x2] = resize_img[left_bgbox.y1:left_bgbox.y2,
                        #                                                                         left_bgbox.x1:left_bgbox.x2, ]
                        #
                        # resize_img = cv2.resize(img_right, (scale_right, scale_right))
                        # out_img[right_curbox.y1:right_curbox.y2, right_curbox.x1:right_curbox.x2] = resize_img[right_bgbox.y1:right_bgbox.y2,
                        #                                                                             right_bgbox.x1:right_bgbox.x2, ]
                        img_left, img_right, left_hand, mask_left = rendering(s2_render, outscale_left* 0.5, outtrans2d_left* 0.5, outscale_right* 0.5,
                                                        outtrans2d_right* 0.5, v3d_left,
                                                        v3d_right, left_img=s2_left_img, right_img=s2_right_img, two=1,
                                                        single=1)
                        resize_img = cv2.resize(img_left, (2 * scale_left, 2 * scale_left))
                        resize_lefthand = cv2.resize(left_hand, (2 * scale_left, 2 * scale_left))
                        resize_leftmask = 1 - cv2.resize(mask_left, (2 * scale_left, 2 * scale_left))
                        out_img[s2_left_curbox.y1:s2_left_curbox.y2, s2_left_curbox.x1:s2_left_curbox.x2] = resize_img[
                                                                                                s2_left_bgbox.y1:s2_left_bgbox.y2,
                                                                                                s2_left_bgbox.x1:s2_left_bgbox.x2, ]

                        resize_img = cv2.resize(img_right, (2 * scale_right, 2 * scale_right))
                        out_img[s2_right_curbox.y1:s2_right_curbox.y2, s2_right_curbox.x1:s2_right_curbox.x2] = resize_img[
                                                                                                    s2_right_bgbox.y1:s2_right_bgbox.y2,
                                                                                                    s2_right_bgbox.x1:s2_right_bgbox.x2, ]

                        if classbox_iou(s2_left_curbox, s2_right_curbox) > 0.02:
                            out_img = out_img.astype(np.float32)
                            out_img[s2_left_curbox.y1:s2_left_curbox.y2, s2_left_curbox.x1:s2_left_curbox.x2] *= resize_leftmask[
                                                                                                    s2_left_bgbox.y1:s2_left_bgbox.y2,
                                                                                                    s2_left_bgbox.x1:s2_left_bgbox.x2, None]
                            out_img[s2_left_curbox.y1:s2_left_curbox.y2, s2_left_curbox.x1:s2_left_curbox.x2] += resize_lefthand[
                                                                                                    s2_left_bgbox.y1:s2_left_bgbox.y2,
                                                                                                    s2_left_bgbox.x1:s2_left_bgbox.x2, ]
                            out_img = out_img.astype(np.uint8)

                    v3d_right = v3d_right + root_right.reshape(1, 1, 3)


                    v3d_left *= scalelength_left
                    v3d_right *= scalelength_right

                else:
                    proc_union = process_image(union_img)
                    inputs = {'img': proc_union.cuda()}
                    targets = {}
                    meta_info = {}
                    out = mesh_predictor.run_mymodel(inputs['img'])
                    pose_param_left = out['mano_pose_left'][0:1]
                    pose_param_right = out['mano_pose_right'][0:1]
                    shape_param_left = out['mano_shape_left'][0:1]
                    shape_param_right = out['mano_shape_right'][0:1]
                    v3d_left = out['v3d_left'][0:1]
                    v3d_right = out['v3d_right'][0:1]
                    outscale_left = out['scale_left']
                    outtrans2d_left = out['trans2d_left']
                    outscale_right = out['scale_right']
                    outtrans2d_right = out['trans2d_right']
                    scalelength_left = out['scalelength_left']
                    scalelength_right = out['scalelength_right']
                    right_rel = out['root_rel'].reshape(-1, 1, 3)
                    root_right = right_rel

                    ### smooth left params #################################################
                    cur_left = Hand3dResult(
                        bbox=left_curbox,
                        global_orient=pose_param_left[0, :3],
                        poses=pose_param_left[0, 3:],
                        betas=shape_param_left[0],
                        camera_scale=outscale_left,
                        camera_tran=outtrans2d_left[0],
                        vertices=v3d_left[0],
                        scalelength=scalelength_left,
                    )
                    new_left = left_smoothcall(cur_left, prev_left_result)
                    prev_left_result = new_left

                    outscale_left = new_left.camera_scale
                    outtrans2d_left = new_left.camera_tran[None, :]
                    scalelength_left = new_left.scalelength
                    # v3d_left = new_left.vertices[None, :]
                    pose_param_left = torch.cat((new_left.global_orient[None, :], new_left.poses[None, :]), axis=1)
                    shape_param_left = new_left.betas[None, :]
                    v3d_left, j3d_left = mano_left_layer(rodrigues_batch(pose_param_left[:, :3]), pose_param_left[:, 3:],
                                                  shape_param_left)
                    v3d_left /= 1000
                    v3d_left = v3d_left - j3d_left[:, 0:1, :] / 1000
                    ####################################################################

                    ### smooth right params ###########################################
                    cur_right = Hand3dResult(
                        bbox=right_curbox,
                        global_orient=pose_param_right[0, :3],
                        poses=pose_param_right[0, 3:],
                        betas=shape_param_right[0],
                        camera_scale=outscale_right,
                        camera_tran=outtrans2d_right[0],
                        vertices=v3d_right[0],
                        scalelength=scalelength_right,
                        rightrel=root_right[0,0],
                    )
                    new_right = right_smoothcall(cur_right, prev_right_result)
                    prev_right_result = new_right

                    outscale_right = new_right.camera_scale
                    outtrans2d_right = new_right.camera_tran[None, :]
                    scalelength_right = new_right.scalelength
                    root_right = new_right.rightrel[None, None, :]
                    # v3d_right = new_right.vertices[None, :]
                    pose_param_right = torch.cat((new_right.global_orient[None, :], new_right.poses[None, :]), axis=1)
                    shape_param_right = new_right.betas[None, :]
                    v3d_right, j3d_right = mano_right_layer(rodrigues_batch(pose_param_right[:, :3]), pose_param_right[:, 3:],
                                                    shape_param_right)
                    v3d_right /= 1000
                    v3d_right = v3d_right - j3d_right[:, 0:1, :] / 1000

                    j3d_right = j3d_right - j3d_right[:, 0:1, :]
                    j3d_right /= 1000
                    j2d_right = projection_batch(outscale_right, outtrans2d_right, j3d_right) * scale_right / 256
                    j2d_rightprev = j2d_right[0, 0:1, :2]
                    ###############################################################################
                    if args.vis:
                        # img = rendering(render, outscale_left, outtrans2d_left, outscale_right, outtrans2d_right,
                        #                                 v3d_left, v3d_right, union_img=union_img, two=1, single=0)
                        # resize_img = cv2.resize(img, (scale_union, scale_union))
                        # out_img[union_curbox.y1:union_curbox.y2, union_curbox.x1:union_curbox.x2] = resize_img[union_bgbox.y1:union_bgbox.y2, union_bgbox.x1:union_bgbox.x2,]

                        img = rendering(s2_render, outscale_left* 0.5, outtrans2d_left* 0.5, outscale_right* 0.5, outtrans2d_right* 0.5,
                                                        v3d_left, v3d_right, union_img=s2_union_img, two=1, single=0)
                        resize_img = cv2.resize(img, (2 * scale_union, 2 * scale_union))
                        out_img[s2_union_curbox.y1:s2_union_curbox.y2, s2_union_curbox.x1:s2_union_curbox.x2] = resize_img[s2_union_bgbox.y1:s2_union_bgbox.y2, s2_union_bgbox.x1:s2_union_bgbox.x2,]
                    v3d_left *= scalelength_left
                    v3d_right += root_right
                    v3d_right *= scalelength_right
                    right_prevbox = right_curbox

            elif left_flag:
                proc_left = process_image(left_img)
                inputs = {'img': proc_left.cuda()}
                out = mesh_predictor.run_mymodel(inputs['img'])
                pose_param_left = out['mano_pose_left'][0:1]
                shape_param_left = out['mano_shape_left'][0:1]
                v3d_left = out['v3d_left']
                outscale_left = out['scale_left']
                outtrans2d_left = out['trans2d_left']
                scalelength_left = out['scalelength_left'][0:1]

                ### smooth left params #################################################
                cur_left = Hand3dResult(
                    bbox=left_curbox,
                    global_orient=pose_param_left[0, :3],
                    poses=pose_param_left[0, 3:],
                    betas=shape_param_left[0],
                    camera_scale=outscale_left,
                    camera_tran=outtrans2d_left[0],
                    vertices=v3d_left[0],
                    scalelength=scalelength_left,
                )
                new_left = left_smoothcall(cur_left, prev_left_result)
                prev_left_result = new_left

                outscale_left = new_left.camera_scale
                outtrans2d_left = new_left.camera_tran[None, :]
                scalelength_left = new_left.scalelength
                # v3d_left = new_left.vertices[None, :]
                pose_param_left = torch.cat((new_left.global_orient[None, :], new_left.poses[None, :]), axis=1)
                shape_param_left = new_left.betas[None, :]
                v3d_left, j3d_left = mano_left_layer(rodrigues_batch(pose_param_left[:, :3]), pose_param_left[:, 3:],
                                              shape_param_left)
                v3d_left /= 1000
                v3d_left = v3d_left - j3d_left[:, 0:1, :] / 1000
                ####################################################################

                if args.vis:
                    # img = rendering(render, outscale_left, outtrans2d_left, None, None,
                    #                 v3d_left, None, left_img=left_img, two=0, single=0, left=1)
                    # resize_img = cv2.resize(img, (scale_left, scale_left))
                    # out_img[left_curbox.y1:left_curbox.y2, left_curbox.x1:left_curbox.x2] = resize_img[
                    #                                                                         left_bgbox.y1:left_bgbox.y2,
                    #                                                                         left_bgbox.x1:left_bgbox.x2, ]
                    img = rendering(s2_render, outscale_left* 0.5, outtrans2d_left* 0.5,
                                                    None,
                                                    None, v3d_left,
                                                    None, left_img=s2_left_img, two=0,
                                                    single=0, left=1)
                    resize_img = cv2.resize(img, (2 * scale_left, 2 * scale_left))
                    out_img[s2_left_curbox.y1:s2_left_curbox.y2, s2_left_curbox.x1:s2_left_curbox.x2] = resize_img[
                                                                                                        s2_left_bgbox.y1:s2_left_bgbox.y2,
                                                                                                        s2_left_bgbox.x1:s2_left_bgbox.x2, ]
                v3d_left *= scalelength_left

            elif right_flag:
                proc_right = process_image(right_img)
                inputs = {'img': proc_right.cuda()}
                out = mesh_predictor.run_mymodel(inputs['img'])
                pose_param_right = out['mano_pose_right']
                shape_param_right = out['mano_shape_right']
                v3d_right = out['v3d_right']
                # v3d_right = out['mesh_coord_cam_right']
                outscale_right = out['scale_right']
                outtrans2d_right = out['trans2d_right']
                scalelength_right = out['scalelength_right']
                ### smooth right params ###########################################
                cur_right = Hand3dResult(
                    bbox=right_curbox,
                    global_orient=pose_param_right[0, :3],
                    poses=pose_param_right[0, 3:],
                    betas=shape_param_right[0],
                    camera_scale=outscale_right,
                    camera_tran=outtrans2d_right[0],
                    vertices=v3d_right[0],
                    scalelength=scalelength_right,
                    rightrel=root_right[0, 0],
                )
                new_right = right_smoothcall(cur_right, prev_right_result)
                prev_right_result = new_right

                outscale_right = new_right.camera_scale
                outtrans2d_right = new_right.camera_tran[None, :]
                scalelength_right = new_right.scalelength
                # v3d_right = new_right.vertices[None, :]
                pose_param_right = torch.cat((new_right.global_orient[None, :], new_right.poses[None, :]), axis=1)
                shape_param_right = new_right.betas[None, :]
                root_right = new_right.rightrel[None, None, :]
                v3d_right, j3d_right = mano_right_layer(rodrigues_batch(pose_param_right[:, :3]), pose_param_right[:, 3:],
                                                shape_param_right)
                v3d_right /= 1000
                v3d_right = v3d_right - j3d_right[:, 0:1, :] / 1000

                j3d_right = j3d_right - j3d_right[:, 0:1, :]
                j3d_right /= 1000
                j2d_right = projection_batch(outscale_right, outtrans2d_right, j3d_right) * scale_right / 256
                j2d_rightprev = j2d_right[0, 0:1, :2]
                ###############################################################################
                if args.vis:
                    # img = rendering(render, None, None, outscale_right, outtrans2d_right,
                    #                 None, v3d_right, right_img=right_img, two=0, single=0, left=0, right=1)
                    # resize_img = cv2.resize(img, (scale_right, scale_right))
                    # out_img[right_curbox.y1:right_curbox.y2, right_curbox.x1:right_curbox.x2] = resize_img[
                    #                                                                             right_bgbox.y1:right_bgbox.y2,
                    #                                                                             right_bgbox.x1:right_bgbox.x2, ]
                    img1 = rendering(s2_render, None, None, outscale_right* 0.5, outtrans2d_right* 0.5,
                                    None, v3d_right, right_img=s2_right_img, two=0, single=0, left=0, right=1)
                    resize_img = cv2.resize(img1, (2 * scale_right, 2 * scale_right))
                    # tmp = cur_frame.copy()
                    out_img[s2_right_curbox.y1:s2_right_curbox.y2, s2_right_curbox.x1:s2_right_curbox.x2] = resize_img[
                                                                                                s2_right_bgbox.y1:s2_right_bgbox.y2,
                                                                                          s2_right_bgbox.x1:s2_right_bgbox.x2, ]
                v3d_right += root_right
                v3d_right *= scalelength_right
                right_prevbox = right_curbox
            if args.vis:
                video3dWriter.write(out_img)
            end_time = time.time()
            idx += 1
            total_time += (end_time - start_time)
            print(' ; avg total time ', total_time / (idx+0.001), flush=True)


            # data_to_json.append({'image_id': image_name,
            #                      'leftbox': [left_x1, left_y1, left_x2, left_y2, scale_left, cl_x1, cl_y1, cl_x2, cl_y2],
            #                      'rightbox': [right_x1, right_y1, right_x2, right_y2, scale_right, cr_x1, cr_y1, cr_x2, cr_y2],
            #                      'unionbox': [union_x1, union_y1, union_x2, union_y2, scale_union, cunion_x1, cunion_y1, cunion_x2, cunion_y2],
            #                      'leftpoints': left_points.tolist(),
            #                      'rightpoints': right_points.tolist()})


        # show the results
        if args.vis:
            vis_frame = vis_pose_tracking_result(
                pose_model,
                cur_frame,
                pose_results,
                radius=args.radius,
                thickness=args.thickness,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                show=False)
            if left_flag and right_flag:
                if single_flag:
                    cv2.rectangle(vis_frame, (left_curbox.x1, left_curbox.y1), (left_curbox.x2, left_curbox.y2), (0, 0, 255), 3)
                    cv2.rectangle(vis_frame, (right_curbox.x1, right_curbox.y1), (right_curbox.x2, right_curbox.y2), (0, 0, 255), 3)
                else:
                    cv2.rectangle(vis_frame, (union_curbox.x1, union_curbox.y1), (union_curbox.x2, union_curbox.y2), (0, 0, 255), 3)
            elif left_flag:
                cv2.rectangle(vis_frame, (left_curbox.x1, left_curbox.y1), (left_curbox.x2, left_curbox.y2), (0, 0, 255), 3)
            elif right_flag:
                cv2.rectangle(vis_frame, (right_curbox.x1, right_curbox.y1), (right_curbox.x2, right_curbox.y2), (0, 0, 255), 3)
    #
    #     if args.show:
    #         cv2.imshow('Frame', vis_frame)
    #
        if out:
            for key in out:
                try:
                    out[key] = out[key].cpu().numpy()
                except:
                    continue
            outnumpy[frame_id] = out
            try:
                outleftvert[frame_id] = v3d_left.cpu().numpy().reshape(778, 3)
            except:
                pass
            try:
                outrightvert[frame_id] = v3d_right.cpu().numpy().reshape(778,3)
            except:
                pass

        if save_out_video and args.save:
            videoWriter.write(vis_frame)
            orivideoWriter.write(cur_frame)
    #
    #     if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    if args.vis:
        video3dWriter.release()
    if save_out_video and args.save:
        videoWriter.release()
    np.save(args.out_video_root + '/' + os.path.basename(args.video_path), outnumpy)
    np.save(args.out_video_root + '/' + os.path.basename(args.video_path) + '_left', outleftvert)
    np.save(args.out_video_root + '/' + os.path.basename(args.video_path) + '_right', outrightvert)


if __name__ == '__main__':
    main()
