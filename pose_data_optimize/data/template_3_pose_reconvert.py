# This script is used to convert optimized poses to the original data format for training
from scripts.HandPoseConverter import HandPoseConverter
import numpy as np
import os
import copy
from tqdm import *
def copy_dict_struct(raw_data):
    if type(raw_data) is not dict:
        return np.asarray([])
    new_data = dict()
    for key in raw_data.keys():
        new_data[key] = copy_dict_struct(raw_data[key])
    return new_data

def data_loader(data_path, mode, scale):
    mocap_data = np.load(data_path, allow_pickle=True).item()
    left_rot = mocap_data['left']['rot']
    left_loc = mocap_data['left']['loc'] * scale / 100
    right_rot = mocap_data['right']['rot']
    right_loc = mocap_data['right']['loc'] * scale / 100

    if mode == 1:
        left_rot[:, 0, 1:3] = -left_rot[:, 0, 1:3]
        left_loc[:, 1] = -left_loc[:, 1]
        right_rot[:, 0, 1:3] = -right_rot[:, 0, 1:3]
        right_loc[:, 1] = -right_loc[:, 1]
    return right_rot, right_loc, left_rot, left_loc, mocap_data

def assign_discrete_data(hand_data, rot, trans, shape):
    hand_data['pose'] = rot
    hand_data['trans'] = trans
    hand_data['shape'] = shape

mano_asset_path = 'assets/mano'
out_put_data_dir = 'data/optimezed_argued_data'

os.makedirs(out_put_data_dir,exist_ok=True)

template_file_path = 'data/template_renderih_data_sample.pkl'
template_file = np.load(template_file_path, allow_pickle=True)
discrete_data_ = copy_dict_struct(template_file)

lhpc = HandPoseConverter(side='left', root=mano_asset_path)
rhpc = HandPoseConverter(side='right', root=mano_asset_path)

optimized_data = 'data/template_optimized_data0.npy'
data_list = [optimized_data]
# out_put_dict = dict()
for data_name in tqdm(data_list):
    right_rot, right_loc, left_rot, left_loc, raw_data = data_loader(data_name, 0, 1.0)
    bts = right_rot.shape[0]
    right_shape = raw_data['right']['shape']
    left_shape = raw_data['left']['shape']
    right_quat = rhpc.euler_2_mano_quat(right_rot)
    left_quat = lhpc.euler_2_mano_quat(left_rot)

    right_trans = rhpc.loc_2_trans(right_quat, right_shape, right_loc, 0)
    left_trans = lhpc.loc_2_trans(left_quat, left_shape, left_loc, 0)

    right_aa = rhpc.mano_quat_flat_2_normal(right_quat, return_type='aa')
    left_aa = lhpc.mano_quat_flat_2_normal(left_quat, return_type='aa')

    capture_frame_id = raw_data['capture_frame_id']

    for data_id in tqdm(range(capture_frame_id.__len__())):
        capture_id = capture_frame_id[data_id]
        if type(capture_id) is not str:
            capture_id = str(capture_id)


        discrete_data = copy.deepcopy(discrete_data_)

        assign_discrete_data(discrete_data['mano_params']['left'], left_aa[data_id], left_trans[data_id:data_id+1], left_shape[data_id:data_id+1])
        assign_discrete_data(discrete_data['mano_params']['right'], right_aa[data_id], right_trans[data_id:data_id + 1],
                             right_shape[data_id:data_id + 1])
        save_id = 0
        out_put_data_path = os.path.join(out_put_data_dir, capture_id + '_{}_'.format(save_id) + '.npy')
        while os.path.exists(out_put_data_path):
            save_id += 1
            out_put_data_path = os.path.join(out_put_data_dir, capture_id + '_{}_'.format(save_id) + '.npy')
        np.save(out_put_data_path, discrete_data)