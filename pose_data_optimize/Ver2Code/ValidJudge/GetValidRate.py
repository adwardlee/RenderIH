import torch
import os
from ValidJudger import ValidJudger
import numpy as np
from HandPoseConverter import HandPoseConverter
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

root_path = '/home/tallery/CodeSpace/23_02_02_cpf/dataset/DiscreteDataset/ArguedDataOptimizedFiltered'
mano_root = '/home/tallery/CodeSpace/23_02_02_cpf/assets/mano'
device = 'cuda:0'
rhpc = HandPoseConverter(side='right', root=mano_root)
lhpc = HandPoseConverter(side='left', root=mano_root)
data_list = os.listdir(root_path)
V = ValidJudger(mano_root, device)

total_poses = 0
valid_poses = 0
for data_name in data_list:
    data_path = os.path.join(root_path, data_name)
    right_rot, right_loc, left_rot, left_loc, raw_data = data_loader(data_path, 0, 1.0)

    left_quat = lhpc.euler_2_mano_quat(left_rot)
    right_quat = rhpc.euler_2_mano_quat(right_rot)
    left_quat = torch.from_numpy(left_quat).to(device)
    right_quat = torch.from_numpy(right_quat).to(device)

    r = V.ComputeValidation(left_quat, right_quat)
    total_poses += left_quat.shape[0]
    valid_poses += r * left_quat.shape[0]
print("valid rate: {}".format(valid_poses / total_poses))