import torch
from manopth.manolayer import ManoLayer as mano1
# from manopth import demo
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import random
import os

def rodrigues_batch(axis):
    # axis : bs * 3
    # return: bs * 3 * 3
    bs = axis.shape[0]
    Imat = torch.eye(3, dtype=axis.dtype, device=axis.device).repeat(bs, 1, 1)  # bs * 3 * 3
    angle = torch.norm(axis, p=2, dim=1, keepdim=True) + 1e-8  # bs * 1
    axes = axis / angle  # bs * 3
    sin = torch.sin(angle).unsqueeze(2)  # bs * 1 * 1
    cos = torch.cos(angle).unsqueeze(2)  # bs * 1 * 1
    L = torch.zeros((bs, 3, 3), dtype=axis.dtype, device=axis.device)
    L[:, 2, 1] = axes[:, 0]
    L[:, 1, 2] = -axes[:, 0]
    L[:, 0, 2] = axes[:, 1]
    L[:, 2, 0] = -axes[:, 1]
    L[:, 1, 0] = axes[:, 2]
    L[:, 0, 1] = -axes[:, 2]
    return Imat + sin * L + (1 - cos) * L.bmm(L)

def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

def generate_obj(annot, filename,mano_right,mano_left, cur_idx, save_annot, mesh_path):
    batch_size = 1
    # Select number of principal components for pose space

    # annot = np.load(filename, allow_pickle=True)
    annot = annot[()]
    # print(annot)
    # ori_idx = int(filename.split('/')[-1].split('_')[2])
    mano_param=annot['mano_params']
    left_param=mano_param['left']
    right_param=mano_param['right']
    all_cam = annot['all_cam']
    camera_num = len(all_cam)
    # multiplier = camera_num // 21
    cam_idx = random.sample(range(camera_num), min(15, camera_num)) ### select how much camera
    for start_idx, one in enumerate(cam_idx):
        camera = all_cam[one]
        camera_r = camera['R'].reshape(3,3)
        camera_t = camera['t'].reshape(1,3)
        camera_in = camera['camera']
        annot['camera']['R'] = camera_r
        annot['camera']['t'] = camera_t
        annot['camera']['camera'] = camera_in
        # mano_left.shapedirs[:,0,:] *= -1

        # hand_verts_l, hand_joints_l = mano_left(root_rotation=rodrigues_batch(torch.from_numpy(left_param['pose'][0:1, :3])).to(torch.float32),
        #                                         pose=torch.from_numpy(left_param['pose'][1:]).reshape(1, -1).to(torch.float32),
        #                                         shape=torch.from_numpy(left_param['shape']), trans=torch.from_numpy(left_param['trans']))
        # hand_verts_r, hand_joints_r = mano_right(root_rotation=rodrigues_batch(torch.from_numpy(right_param['pose'][0:1, :3])).to(torch.float32),
        #                                          pose=torch.from_numpy(right_param['pose'][1:].reshape(1, -1)).to(torch.float32),
        #                                          shape=torch.from_numpy(right_param['shape']), trans=torch.from_numpy(right_param['trans']))

        hand_verts_l, hand_joints_l = mano_left(
            torch.from_numpy(left_param['pose']).reshape(1, -1).to(torch.float32),
            torch.from_numpy(left_param['shape']), torch.from_numpy(left_param['trans']))
        hand_verts_r, hand_joints_r = mano_right(
            torch.from_numpy(right_param['pose'].reshape(1, -1)).to(torch.float32),
            torch.from_numpy(right_param['shape']), torch.from_numpy(right_param['trans']))

        hand_verts_l /= 1000
        hand_verts_r /= 1000
        hand_joints_l /= 1000
        hand_joints_r /= 1000
        show=False
        if show:
            mano_faces=mano_left.faces.astype(float)
            batch_idx=0
            alpha=0.2

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in range(2):
                if i==0:
                    verts=hand_verts_r[0]
                    joints=hand_joints_r[0]
                else:
                    verts=hand_verts_l[0]
                    joints=hand_joints_l[0]
                # verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][
                #     batch_idx]
                if mano_faces is None:
                    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
                else:
                    mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
                    face_color = (141 / 255, 184 / 255, 226 / 255)
                    edge_color = (50 / 255, 50 / 255, 50 / 255)
                    mesh.set_edgecolor(edge_color)
                    mesh.set_facecolor(face_color)
                    ax.add_collection3d(mesh)
                ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
            cam_equal_aspect_3d(ax, verts.numpy())

            plt.show()

        # hand_verts_l/=1000
        # hand_verts_r/=1000
        # hand_joints_r/=1000
        # hand_joints_l/=1000

        hand_verts_l = hand_verts_l[0].numpy() @ camera_r.T + camera_t
        hand_verts_r = hand_verts_r[0].numpy() @ camera_r.T + camera_t

        left_2d = hand_verts_l @camera_in.T
        left_2d = left_2d[:, :2] / left_2d[:, 2:]

        right_2d = hand_verts_r @ camera_in.T
        right_2d = right_2d[:, :2] / right_2d[:, 2:]
        #### x: 334 y: 512 ####
        xmin = min(left_2d[:, 0].min(), right_2d[:, 0].min() )
        xmax = max(left_2d[:, 0].max(), right_2d[:, 0].max() )
        ymin = min(left_2d[:, 1].min(), right_2d[:, 1].min() )
        ymax = max(left_2d[:, 1].max(), right_2d[:, 1].max() )

        if xmin < 0 or ymin < 0 or xmax > 333 or ymax > 511:
            continue
        outannot = save_annot + str(cur_idx) + '.pkl'
        with open(outannot, 'wb') as file1:
            pickle.dump(annot, file1)
        output = True
        tag = False
        if output:
            for i in range(2):
                if i == 0:
                    hand_type = 'right'
                    hand_verts = hand_verts_r
                else:
                    hand_type = 'left'
                    hand_verts = hand_verts_l
                ## Write to an .obj file
                outmesh_path = mesh_path + '{}_{}.obj'.format(cur_idx,
                                                              hand_type)  # f'/mnt/workspace/workgroup/yunqian/render_hand/test_obj/{idx}_{hand_type}.obj'
                with open(outmesh_path, 'w') as fp:
                    for v in hand_verts:
                        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                        tag = True

                    # for f in mano_layer.th_faces+1: # Faces are 1-based, not 0-based in obj files
                    #     fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

                    f1 = open("utils/obj_tex.txt", "r")
                    for x in f1:
                        # print(x)
                        fp.write(x)
                    # print(f.read())

                ## Print message
                # print('..Output mesh saved to: ', outmesh_path)
            if tag != True:
                print('error idx {}, file {} '.format(cur_idx, filename),flush=True)
        cur_idx += 1

mano_left=mano1(mano_root='mano',center_idx=None,use_pca=False,flat_hand_mean=False, side='left')
mano_right=mano1(mano_root='mano',center_idx=None,use_pca=False,flat_hand_mean=False, side='right')
mano_left.th_shapedirs[:,0,:] *= -1
# mano_left=ManoLayer(manoPath='mano/MANO_LEFT.pkl',center_idx=None,use_pca=True)
# mano_right=ManoLayer(manoPath='mano/MANO_RIGHT.pkl',center_idx=None,use_pca=True)
# mano_left.shapedirs[:,0,:] *= -1
# generate_obj(250091,mano_right,mano_left)
n=0
path = '/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/interpolate_pose/'
# mesh_path = '/mnt/workspace/workgroup/lijun/hand_dataset/xinchuan_4895mesh/'
# ori_annot = '/mnt/workspace/dataset/interhand_5fps/interhand_data/train/single_pose/'
# save_annot = '/mnt/workspace/dataset/interhand_5fps/interhand_data/train/xinchuan_4895annot/'
# mesh_path = '/mnt/workspace/workgroup/lijun/hand_dataset/xinchuan_200wmesh/'
# ori_annot = '/mnt/workspace/dataset/interhand_5fps/interhand_data/train/single_pose/'
# save_annot = '/mnt/workspace/dataset/interhand_5fps/interhand_data/train/xinchuan_200w/oriannot/'
mesh_path = '/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/camera_pose_test/'
ori_annot = '/mnt/workspace/dataset/interhand_5fps/interhand_data/train/single_pose/'
save_annot = '/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/oriannot_test/'
if not os.path.exists(mesh_path):
    os.makedirs(mesh_path)
if not os.path.exists(save_annot):
    os.makedirs(save_annot)
files=os.listdir(path)
files.sort()
print(' length file ', len(files), flush=True)### 68737
start = 366358 + 8000 * 0#### 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100###  change ### 366358， 649493
# start = 1015851 + 8000 * 180#### 0, 15, 30, 45, , 60, 75, 90, 105, 120, 135, 150, 165, 180### 100000 change ### 366358， 649493
input_idx = 8000 * 0################## 0, 8000, 16000, 3x8_10e3,4x8_10e3, 5x8_10e3, 6x8_10e3, 7x8_10e3, 8x8_10e3, 9x8,10e3,10x8_10e3,11x8_10e3
################################# 12x8_10e3,
print('start idx', start, flush=True)
# for idx in tqdm(range(72000, 75000)): ### change ###
for idx in tqdm(range(200, 500)):  ### change ###
    # n+=1
    # if n>10:
        # break
    # print(i.split('.')[0])
    filename = files[idx]
    parts = filename.split('.npy')[0].split('_')
    oriname = '_'.join(parts[:-2]) + '.pkl'
    anno = np.load(path + filename, allow_pickle=True)
    orianno = pickle.load(open(ori_annot + oriname, 'rb'))
    # if np.array2string(orianno['mano_params']['left']['pose'][:3]) == np.array2string(anno[()]['mano_params']['left']['pose'][:3]):
    #     continue
    filename = path + filename
    generate_obj(anno, filename,mano_right,mano_left, start + input_idx * 15, save_annot, mesh_path)
    input_idx += 1
    # generate_obj(250091, mano_right, mano_left)