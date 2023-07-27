import cv2
import numpy as np

def media_to_np(landmark, img_size, score):
    output = np.zeros((21, 3))
    for i in range(21):
        output[i, 0] = landmark[i].x * img_size[1]
        output[i, 1] = landmark[i].y * img_size[0]
        output[i, 2] = score
    return output

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

def img_to_2djoint(hands, ori_image):
    joints2d = {'left': None, 'right': None}
    ori_image.flags.writeable = False
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    if results.multi_hand_landmarks is None:
        if joints2d['left'] is None or joints2d['right'] is None:
            return
        joints2d['left'][:, 2] *= 0.1
        joints2d['right'][:, 2] *= 0.1
        return


    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    handnums = len(results.multi_hand_landmarks)
    out_joints = []
    for i in range(handnums):
        label = results.multi_handedness[i].classification[0].label  ### 'Left', 'Right'
        score = results.multi_handedness[i].classification[0].score
        onejoint = media_to_np(results.multi_hand_landmarks[i].landmark, img_size, score)
        out_joints.append(onejoint)

        if label == 'Right':
            joints2d['left'] = onejoint
            if handnums == 1:
                if joints2d['right'] is None:
                    joints2d['right'] = joints2d['left'].copy()
                joints2d['right'][:, 2] *= 0.1
        elif label == 'Left':
            joints2d['right'] = onejoint
            if handnums == 1:
                if joints2d['left'] is None:
                    joints2d['left'] = joints2d['right'].copy()
                joints2d['left'][:, 2] *= 0.1
    return joints2d

def save_json_annot(joint2d, image):
    height, width = image.shape[:2]
    annots = {'bbox': None,
              'personID': 0,
              'keypoints': None,
              'isKeyframe': False,
              'face2d': np.zeros((70, 3)).tolist(),
              'handl2d': np.zeros((21, 3)).tolist(),
              'handr2d': np.zeros((21, 3)).tolist(),
              'twohand': np.zeros((42, 3)).tolist(),
              }
    left_flag = False
    right_flag = False
    if joint2d['left'] is not None:
        annots['handl2d'] = joint2d['left'].tolist()
        left_flag = True
    if joint2d['right'] is not None:
        annots['handr2d'] = joint2d['right'].tolist()
        right_flag = True

    if left_flag and right_flag:
        bbox_left = get_bbox_from_pose(joint2d['left'], height, width, rate=0.3)
        bbox_right = get_bbox_from_pose(joint2d['right'], height, width, rate=0.3)
        if bbox_iou(bbox_left, bbox_right) < 0.02:
            annots['single_flag'] = 1
        twohand = np.concatenate((joint2d['left'], joint2d['right']), axis=0)
        annots['bbox'] = get_bbox_from_pose(twohand, height, width)
        annots['twohand'] = twohand.tolist()
    elif left_flag:
        annots['bbox'] = get_bbox_from_pose(joint2d['left'], height, width)
    elif right_flag:
        annots['bbox'] = get_bbox_from_pose(joint2d['right'], height, width)
    return annots