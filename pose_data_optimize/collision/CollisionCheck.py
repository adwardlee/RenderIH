import trimesh
import trimesh.collision as collision
import numpy as np
from manopth.manolayer import ManoLayer
from HandPoseConverter import HandPoseConverter
import torch

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

root_path = '/home/tallery/CodeSpace/myCPF/dataset/DiscreteDataset'
optimized_data_path = '/home/tallery/CodeSpace/23_02_02_cpf/dataset/DiscreteDataset/ArguedDataOptimized/discrete_hand_pose_0.npy'
vis = False
right_rot, right_loc, left_rot, left_loc, raw_data = data_loader(optimized_data_path, 0, 1.0)
start_id = 0
end_id = len(right_loc)

lhpc = HandPoseConverter(side='left')
rhpc = HandPoseConverter(side='right')
left_mano_layer = ManoLayer(
        mano_root="../assets/mano",
        use_pca=False,
        ncomps=6,
        flat_hand_mean=True,
        center_idx=0,
        return_transf=False,
        side='left',
        root_rot_mode='quat',
        joint_rot_mode='quat'
    )
right_mano_layer = ManoLayer(
    mano_root="../assets/mano",
    use_pca=False,
    ncomps=6,
    flat_hand_mean=True,
    center_idx=0,
    return_transf=False,
    side='right',
    root_rot_mode='quat',
    joint_rot_mode='quat'
)

# data init
left_quat = lhpc.euler_2_mano_quat(left_rot)
right_quat = rhpc.euler_2_mano_quat(right_rot)
left_quat = torch.from_numpy(left_quat[start_id:end_id])
right_quat = torch.from_numpy(right_quat[start_id:end_id])
left_loc = torch.from_numpy(left_loc[start_id:end_id])
right_loc = torch.from_numpy(right_loc[start_id:end_id])
right_shape = torch.from_numpy(raw_data['right']['shape'].astype(np.float32)[start_id:end_id])
left_shape = torch.from_numpy(raw_data['left']['shape'].astype(np.float32)[start_id:end_id])
faces = np.array(left_mano_layer.th_faces).astype(np.long)

# my_shape = torch.tensor([0.5082395, -0.39488167, -1.7484332, 1.6630946, 0.34428665, -1.37387,
#                                        0.38293332, 1.196094, 0.6538949, -0.94331187])
# left_shape = my_shape.unsqueeze(0).repeat(len(right_loc), 1)
# right_shape = my_shape.unsqueeze(0).repeat(len(right_loc), 1)
vertex_colors = np.array([0.8, 0.8, 0.8, 0.8])
vertex_colors = np.expand_dims(vertex_colors, 0).repeat(778, axis=0)

collision_manager = collision.CollisionManager()
lv, lj = left_mano_layer(left_quat, left_shape)
rv, rj = right_mano_layer(right_quat, right_shape)
lv = lv + left_loc.unsqueeze(1).repeat(1,778,1)
rv = rv + right_loc.unsqueeze(1).repeat(1,778,1)
lv = lv.numpy()
rv = rv.numpy()

# right_tri_mesh = trimesh.Trimesh(rv[0], faces)
# left_tri_mesh = trimesh.Trimesh(lv[0], faces)
valid_count = 0
valid_id = []
for i in range(lv.__len__()):
    collision_manager = collision.CollisionManager()
    # lq = left_quat[i:i + 1]
    # rq = right_quat[i:i + 1]
    # l_vertices, l_joints = left_mano_layer(lq, left_shape[i:i+1])
    # r_vertices, r_joints = right_mano_layer(rq, right_shape[i:i+1])

    right_tri_mesh = trimesh.Trimesh(rv[i], faces, vertex_colors=vertex_colors)
    left_tri_mesh = trimesh.Trimesh(lv[i], faces, vertex_colors=vertex_colors)
    mesh_dict = dict(right=right_tri_mesh, left=left_tri_mesh)
    collision_manager.add_object('right', right_tri_mesh)
    collision_manager.add_object('left', left_tri_mesh)
    contact_able, contact_datas = collision_manager.in_collision_internal(return_data=True)
    print('{}:{}'.format(i, contact_datas.__len__()))
    if contact_datas.__len__() <= 100:
        valid_count += 1
        valid_id.append(i + start_id)
    else:
        if vis:
            for cd in contact_datas:
                # right_tri_mesh.show()
                for side in cd._inds.keys():
                    mesh_dict[side].visual.face_colors[cd._inds[side]] = trimesh.visual.random_color()
            mesh_dict['right'].show()
            mesh_dict['left'].show()
                # mesh_dict[cd.'right']
                # mesh.visual.face_colors[facet] = trimesh.visual.random_color()
print(valid_count / lv.__len__())
print(valid_id)