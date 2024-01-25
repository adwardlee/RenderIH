from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from manopth.manolayer import ManoLayer


def generate_random_hand(batch_size=1, ncomps=6, mano_root='mano/models'):
    nfull_comps = ncomps + 3  # Add global orientation dims to PCA
    random_pcapose = torch.rand(batch_size, nfull_comps)
    mano_layer = ManoLayer(mano_root=mano_root)
    verts, joints = mano_layer(random_pcapose)
    return {'verts': verts, 'joints': joints, 'faces': mano_layer.th_faces}


def display_hand(hand_info, mano_faces=None, ax=None, alpha=0.2, cam_view=False, batch_idx=0, show=True, save=''):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][
        batch_idx]
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.5)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

    ax.scatter(joints[0, 0], joints[0, 1], joints[0, 2], s=42, color='c', marker='p')

    ax.scatter(joints[1, 0], joints[1, 1], joints[1, 2], color='y', marker='s')
    ax.scatter(joints[2, 0], joints[2, 1], joints[2, 2], color='y', marker='^')
    ax.scatter(joints[3, 0], joints[3, 1], joints[3, 2], color='y', marker='o')
    ax.scatter(joints[4, 0], joints[4, 1], joints[4, 2], color='y', marker='*')

    ax.scatter(joints[5, 0], joints[5, 1], joints[5, 2], color='r', marker='s')
    ax.scatter(joints[6, 0], joints[6, 1], joints[6, 2], color='r', marker='^')
    ax.scatter(joints[7, 0], joints[7, 1], joints[7, 2], color='r', marker='o')
    ax.scatter(joints[8, 0], joints[8, 1], joints[8, 2], color='r', marker='*')

    ax.scatter(joints[9, 0], joints[9, 1], joints[9, 2], color='b', marker='s')
    ax.scatter(joints[10, 0], joints[10, 1], joints[10, 2], color='b', marker='^')
    ax.scatter(joints[11, 0], joints[11, 1], joints[11, 2], color='b', marker='o')
    ax.scatter(joints[12, 0], joints[12, 1], joints[12, 2], color='b', marker='*')

    ax.scatter(joints[13, 0], joints[13, 1], joints[13, 2], color='g', marker='s')
    ax.scatter(joints[14, 0], joints[14, 1], joints[14, 2], color='g', marker='^')
    ax.scatter(joints[15, 0], joints[15, 1], joints[15, 2], color='g', marker='o')
    ax.scatter(joints[16, 0], joints[16, 1], joints[16, 2], color='g', marker='*')

    ax.scatter(joints[17, 0], joints[17, 1], joints[17, 2], color='m', marker='s')
    ax.scatter(joints[18, 0], joints[18, 1], joints[18, 2], color='m', marker='^')
    ax.scatter(joints[19, 0], joints[19, 1], joints[19, 2], color='m', marker='o')
    ax.scatter(joints[20, 0], joints[20, 1], joints[20, 2], color='m', marker='*')

    if cam_view:
        ax.view_init(azim=-90.0, elev=-90.0)
    cam_equal_aspect_3d(ax, verts.numpy())
    if show:
        plt.show()
    if save:
        plt.savefig("{}.png".format(save))

def _display_hand(hand_info, mano_faces=None, ax=None, alpha=0.2, batch_idx=0, show=True):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][
        batch_idx]
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
    if show:
        plt.show()


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
