# This script is used to demonstrate the pose argue of the converted data
from scripts.HandPoseArguer import HandPoseArguer
import numpy as np
import os
import copy
origin_data_path = 'data/template_converted_data.npy'
argued_data_dir = 'data'
add_origin = False
splice_data_length = 5000
argued_data_path = origin_data_path.replace('.npy', '_argued.npy')
argue_size = 5
start_id = 0
end_id = -1
data = np.load(origin_data_path, allow_pickle=True).item()
if add_origin:
    raw_data = copy.deepcopy(data)
rhpa = HandPoseArguer(side='right')
lhpa = HandPoseArguer(side='left')
end_id = data['capture_frame_id'].__len__() if end_id == -1 else end_id
for side_id in list(data.keys()):
    if side_id in ['left', 'right']:
        rot = data[side_id]['rot']
        rot = rot[start_id:end_id]
        data[side_id]['loc'] = np.repeat(data[side_id]['loc'][start_id:end_id], argue_size, axis=0)
        data[side_id]['shape'] = np.repeat(data[side_id]['shape'][start_id:end_id], argue_size, axis=0)
        if side_id == 'right':
            hpa = rhpa
        else:
            hpa = lhpa
        data[side_id]['rot'] = hpa.argue_pose(rot, pose_type='euler', vis=False, argue_size=argue_size)
        if add_origin:
            for key_id in data[side_id]:
                data[side_id][key_id] = np.row_stack([data[side_id][key_id], raw_data[side_id][key_id][start_id:end_id]])
    elif side_id in ['capture_frame_id']:
        data[side_id] = np.repeat(data[side_id][start_id:end_id], argue_size)
        if add_origin:
            data[side_id]= np.append(data[side_id], raw_data[side_id][start_id:end_id])
# cut to splice
data_id = 0
total_data_length = data['capture_frame_id'].__len__()
left_idx = 0
right_idx = left_idx + splice_data_length
while left_idx < total_data_length:
    right_idx = min(total_data_length, right_idx)
    sub_dict = copy.deepcopy(data)
    for side_id in list(sub_dict.keys()):
        if side_id in ['left', 'right']:
            sub_dict[side_id]['loc'] = data[side_id]['loc'][left_idx:right_idx]
            sub_dict[side_id]['rot'] = data[side_id]['rot'][left_idx:right_idx]
            sub_dict[side_id]['shape'] = data[side_id]['shape'][left_idx:right_idx]
        elif side_id in ['capture_frame_id']:
            sub_dict[side_id] = data[side_id][left_idx:right_idx]

    data_splice_name = os.path.join(argued_data_dir, 'template_argued_data' + str(data_id) + '.npy')
    np.save(data_splice_name, sub_dict)
    left_idx += splice_data_length
    right_idx += splice_data_length
    data_id += 1


