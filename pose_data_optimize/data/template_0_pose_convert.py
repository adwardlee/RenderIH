import numpy as np
from scripts.HandPoseConverter import HandPoseConverter
import torch

# this script is used to demonstrate the conversion of standard annotations of datasets into data formats
# for data augmentation and optimization

convert_data_path = 'data/template_converted_data.npy'

lhpc = HandPoseConverter(side='left', root='assets/mano')
rhpc = HandPoseConverter(side='right',root='assets/mano')

right_dict = dict(rot=[],
                      loc=[],
                      shape=[])
left_dict = dict(rot=[],
                     loc=[],
                     shape=[])
right_aa_rot_batch = []
left_aa_rot_batch = []
right_root_loc_batch = []
left_root_loc_batch = []

right_shape = []
left_shape = []
id_dict = []

for i in range(10):
    template_data = np.load('data/template_renderih_data_sample.pkl', allow_pickle=True)

    frame_inf = template_data['mano_params']

    right_shape.append(frame_inf['right']['shape'])
    left_shape.append(frame_inf['left']['shape'])

    right_aa_rot_batch.append(frame_inf['right']['oripose'])
    right_root_loc_batch.append(frame_inf['right']['trans'][0])

    left_aa_rot_batch.append(frame_inf['left']['oripose'])
    left_root_loc_batch.append(frame_inf['left']['trans'][0])

    id_dict.append(i)

r_e_rot = rhpc.mano_quat_2_euler(rhpc.axis_angle_2_mano_quat(right_aa_rot_batch, flat_hand=False))
l_e_rot = lhpc.mano_quat_2_euler(lhpc.axis_angle_2_mano_quat(left_aa_rot_batch, flat_hand=False))

right_quat = rhpc.axis_angle_2_mano_quat(right_aa_rot_batch, flat_hand=False)
r_loc = rhpc.trans_2_loc(torch.tensor(right_quat), torch.tensor(right_shape).squeeze(1), torch.tensor(right_root_loc_batch), 0).numpy()
left_quat = lhpc.axis_angle_2_mano_quat(left_aa_rot_batch, flat_hand=False)
l_loc = lhpc.trans_2_loc(torch.tensor(left_quat), torch.tensor(left_shape).squeeze(1), torch.tensor(left_root_loc_batch), 0).numpy()

right_dict['rot'] = r_e_rot
right_dict['loc'] = r_loc * 100
left_dict['rot'] = l_e_rot
left_dict['loc'] = l_loc * 100
right_dict['shape'] = np.asarray(right_shape)
left_dict['shape'] = np.asarray(left_shape)
out_dict = dict(right=right_dict, left=left_dict, capture_frame_id=np.asarray(id_dict))
if convert_data_path is not None:
    np.save(convert_data_path, out_dict)