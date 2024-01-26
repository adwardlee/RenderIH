from PoseArgue import PoseArguer
import numpy as np
from HandPoseConverter import HandPoseConverter
import os
import copy

origin_data_path = '/home/tallery/CodeSpace/myCPF/dataset/36W/data/36w_data.npy'
argued_data_dir = './data'
add_origin = True
splice_data_length = 100000000
argued_data_path = origin_data_path.replace('.npy', '_argued.npy')
argue_size = 15
start_id = 0  #41
end_id = -1
vis = True
data = np.load(origin_data_path, allow_pickle=True).item()
if add_origin:
    raw_data = copy.deepcopy(data)
rhpa = PoseArguer(side='right')
lhpa = PoseArguer(side='left')

end_id = data['capture_frame_id'].__len__() if end_id == -1 else end_id

rhpc = HandPoseConverter(side='right', root='../../assets/mano')

out_rot_data = np.asarray([])
real_rates = []
for rr in np.arange(0.0, 1.0, 0.1):
    print(rr)
    for side_id in list(data.keys()):
        if side_id in ['left', 'right']:
            rot = data[side_id]['rot']
            rot = rot[start_id:end_id]
            if side_id == 'right':
                hpa = rhpa
            else:
                hpa = lhpa
            argued_poses = hpa.argue_pose(rot, pose_type='euler', vis=False, argue_size=argue_size, random_level=1.0 - rr)

            data_length = argued_poses.__len__()

            # convet to axis angle
            argued_poses = rhpc.mano_quat_flat_2_normal(rhpc.euler_2_mano_quat(argued_poses),  return_type='aa')
            real_rates.extend([rr for i in range(data_length)])
            if out_rot_data.__len__() == 0:
                out_rot_data = argued_poses
            else:
                out_rot_data = np.row_stack([out_rot_data, argued_poses])
if add_origin:
    for side_id in ['left', 'right']:
        out_rot_data = np.row_stack([out_rot_data, data[side_id]['rot'][start_id:end_id]])
        real_rates.extend([1.0 for i in range(end_id - start_id)])

# cut to splice
data_id = 0
total_data_length = real_rates.__len__()
left_idx = 0
right_idx = left_idx + splice_data_length

while left_idx < total_data_length:
    right_idx = min(total_data_length, right_idx)

    out_dict = dict(rot = out_rot_data[left_idx:right_idx], confidence = real_rates[left_idx:right_idx])

    data_splice_name = os.path.join(argued_data_dir, 'argued_hand_pose_' + str(data_id) + '.npy')
    np.save(data_splice_name, out_dict)
    left_idx += splice_data_length
    right_idx += splice_data_length
    data_id += 1