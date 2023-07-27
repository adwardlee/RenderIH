import os
import os.path as osp
import cv2
import numpy as np
import matplotlib
# matplotlib.use('tkagg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from main.config import cfg
from PIL import Image, ImageDraw
import pyrender, trimesh
from copy import deepcopy
import open3d as o3d


def get_keypoint_rgb(skeleton):
    rgb_dict = {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb_null'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith('index_null'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle_null'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring_null'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky_null'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)

    return rgb_dict


def get_keypoint_rgb_new(skeleton):
    rgb_dict = {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb4'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 255)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('index4'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (255, 153, 255)
        elif joint_name.endswith('middle4'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 153, 255)
        elif joint_name.endswith('ring4'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (255, 153, 255)
        elif joint_name.endswith('pinky4'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (255, 153, 255)

    return rgb_dict


def get_keypoint_rgb_fing(skeleton):
    rgb_dict = {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb4'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 255)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('index4'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (255, 153, 255)
        elif joint_name.endswith('middle4'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('ring4'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (255, 153, 255)
        elif joint_name.endswith('pinky4'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (255, 153, 255)

    return rgb_dict


def vis_keypoints(img, kps, score, skeleton, filename=None, score_thr=0.4, line_width=3, circle_rad=3, save_path=None,
                  hand_type=None):
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = Image.fromarray(img.astype('uint8'))
    draw = ImageDraw.Draw(_img)
    for i in range(21):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name],
                      width=line_width)
        if score[i] > score_thr:
            draw.ellipse(
                (kps[i][0] - circle_rad, kps[i][1] - circle_rad, kps[i][0] + circle_rad, kps[i][1] + circle_rad),
                fill=rgb_dict[joint_name])
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0] - circle_rad, kps[pid][1] - circle_rad, kps[pid][0] + circle_rad,
                          kps[pid][1] + circle_rad), fill=rgb_dict[parent_joint_name])

    return np.array(_img)
    # if save_path is None:
    #     _img.save(osp.join(cfg.vis_dir, filename))
    # else:
    #     _img.save(osp.join(save_path, filename))


def vis_3d_keypoints(kps_3d, score, skeleton, filename=None, score_thr=0.4, line_width=3, circle_rad=3, plot=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb_dict = get_keypoint_rgb(skeleton)

    for i in range(kps_3d.shape[0]):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i, 0], kps_3d[pid, 0]])
        y = np.array([kps_3d[i, 1], kps_3d[pid, 1]])
        z = np.array([kps_3d[i, 2], kps_3d[pid, 2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(x, z, -y, c=np.array(rgb_dict[parent_joint_name]) / 255., linewidth=line_width)
        if score[i] > score_thr:
            ax.scatter(kps_3d[i, 0], kps_3d[i, 2], -kps_3d[i, 1], c=np.array(rgb_dict[joint_name]).reshape(1, 3) / 255.,
                       marker='o')
        if score[pid] > score_thr and pid != -1:
            ax.scatter(kps_3d[pid, 0], kps_3d[pid, 2], -kps_3d[pid, 1],
                       c=np.array(rgb_dict[parent_joint_name]).reshape(1, 3) / 255., marker='o')

    if not plot:
        return ax
    plt.show()
    cv2.waitKey(0)

    if filename is not None:
        fig.savefig(osp.join(cfg.vis_dir, filename), dpi=fig.dpi)


def vis_keypoints_new(img, kps, score, skeleton, filename=None, score_thr=0.4, line_width=3, circle_rad=3,
                      save_path=None, hand_type='right'):
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = Image.fromarray(img.astype('uint8'))
    draw = ImageDraw.Draw(_img)
    for i in range(21):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        colr_p_in = list(rgb_dict[parent_joint_name])
        colr_in = list(rgb_dict[joint_name])
        if True:  # i in range(8-4, 12-4):
            colr_p = colr_p_in
            colr = colr_in
        else:
            colr_p = deepcopy(colr_p_in)
            colr_p[0] = colr_p[1]
            colr_p[1] = colr_p_in[0]
            colr = deepcopy(colr_in)
            colr[0] = colr[1]
            colr[1] = colr_in[0]
        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=tuple(colr_p), width=line_width)
        if score[i] > score_thr:
            draw.ellipse(
                (kps[i][0] - circle_rad, kps[i][1] - circle_rad, kps[i][0] + circle_rad, kps[i][1] + circle_rad),
                fill=tuple(colr))
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0] - circle_rad, kps[pid][1] - circle_rad, kps[pid][0] + circle_rad,
                          kps[pid][1] + circle_rad), fill=tuple(colr_p))

    return np.array(_img)
    # if save_path is None:
    #     _img.save(osp.join(cfg.vis_dir, filename))
    # else:
    #     _img.save(osp.join(save_path, filename))


def vis_3d_keypoints_new(kps_3d, score, skeleton, filename=None, score_thr=0.4, line_width=3, circle_rad=3, plot=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb_dict = get_keypoint_rgb_new(skeleton)

    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i, 0], kps_3d[pid, 0]])
        y = np.array([kps_3d[i, 1], kps_3d[pid, 1]])
        z = np.array([kps_3d[i, 2], kps_3d[pid, 2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            colr = np.array(rgb_dict[joint_name]) / 255.
            if i < 21:
                colr = colr
            else:
                colr = colr[[1, 0, 2]]
            ax.plot(x, z, -y, c=colr, linewidth=line_width)
        if score[i] > score_thr:
            colr = np.array(rgb_dict[joint_name]).reshape(1, 3) / 255.
            if i < 21:
                colr = colr
            else:
                colr = colr[:, [1, 0, 2]]
            ax.scatter(kps_3d[i, 0], kps_3d[i, 2], -kps_3d[i, 1], c=colr, marker='o')
        if score[pid] > score_thr and pid != -1:
            colr = np.array(rgb_dict[parent_joint_name]).reshape(1, 3) / 255.
            if i < 21:
                colr = colr
            else:
                colr = colr[:, [1, 0, 2]]
            ax.scatter(kps_3d[pid, 0], kps_3d[pid, 2], -kps_3d[pid, 1], c=colr, marker='o')

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    if not plot:
        return ax
    plt.show()
    cv2.waitKey(0)

    if filename is not None:
        fig.savefig(osp.join(cfg.vis_dir, filename), dpi=fig.dpi)


def vis_3d_obj_corners(kps_3d_list, filename=None, score_thr=0.4, line_width=3, circle_rad=3, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    # rgb_dict = get_keypoint_rgb(skeleton)
    cols1 = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]
    cols2 = [(255 / 2, 0, 0), (0, 255 / 2, 0), (0, 0, 255 / 2), (0, 0, 255 / 2), (0, 0, 255 / 2), (0, 0, 255 / 2)]
    cols = [cols1, cols2]

    for ii, kps_3d in enumerate(kps_3d_list):
        for i in range(8):
            ax.scatter(kps_3d[i, 0], kps_3d[i, 2], -kps_3d[i, 1],
                       # c=np.array(rgb_dict[joint_name]).reshape(1, 3) / 255.,
                       marker='o')

        jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

        for i in range(len(jointConns)):
            for j in range(len(jointConns[i]) - 1):
                jntC = jointConns[i][j]
                jntN = jointConns[i][j + 1]

                x = np.array([kps_3d[jntC, 0], kps_3d[jntN, 0]])
                y = np.array([kps_3d[jntC, 1], kps_3d[jntN, 1]])
                z = np.array([kps_3d[jntC, 2], kps_3d[jntN, 2]])

                ax.plot(x, z, -y, c=np.array(cols[ii][i]) / 255.,
                        linewidth=line_width)

    plt.show()
    cv2.waitKey(0)

    if filename is not None:
        fig.savefig(osp.join(cfg.vis_dir, filename), dpi=fig.dpi)


def vis_2d_obj_corners(imgIn, gtIn, estIn=None, filename=None, upscale=1, lineThickness=3):
    jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    jointColsGt = (255, 0, 0)
    newCol = (jointColsGt[0] + jointColsGt[1] + jointColsGt[2]) / 3
    jointColsEst = (newCol, newCol, newCol)

    # draws lines connected using jointConns
    img = np.zeros((imgIn.shape[0], imgIn.shape[1], imgIn.shape[2]), dtype=np.uint8)
    img[:, :, :] = (imgIn).astype(np.uint8)

    img = cv2.resize(img, (upscale * imgIn.shape[1], upscale * imgIn.shape[0]), interpolation=cv2.INTER_CUBIC)
    if gtIn is not None:
        gt = gtIn.copy() * upscale
    if estIn is not None:
        est = estIn.copy() * upscale

    for i in range(len(jointConns)):
        for j in range(len(jointConns[i]) - 1):
            jntC = jointConns[i][j]
            jntN = jointConns[i][j + 1]
            if gtIn is not None:
                cv2.line(img, (int(gt[jntC, 0]), int(gt[jntC, 1])), (int(gt[jntN, 0]), int(gt[jntN, 1])), jointColsGt,
                         lineThickness)
            if estIn is not None:
                cv2.line(img, (int(est[jntC, 0]), int(est[jntC, 1])), (int(est[jntN, 0]), int(est[jntN, 1])),
                         jointColsEst, lineThickness)

    if filename is not None:
        cv2.imwrite(filename, img)

    return img


class OrthographicRender():
    def __init__(self):
        self.scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[0.0, 0.0, 0.0])
        self.renderer = pyrender.OffscreenRenderer(viewport_width=cfg.output_hm_shape[2],
                                                   viewport_height=cfg.output_hm_shape[1])
        self.camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
        self.scene.add(self.camera, pose=np.eye(4) * 1.0)
        light = pyrender.SpotLight(color=np.ones(3), intensity=60.0,
                                   innerConeAngle=np.pi / 16.0,
                                   outerConeAngle=np.pi / 6.0)
        self.scene.add(light, pose=np.eye(4) * 1.0)

    def render(self, cam_param, mesh):
        # convert mesh to opengl coordinate system in (-1,1) for x and y
        trans = np.zeros((3,))
        trans[:2] = cam_param[1:]
        vertices = np.array(mesh.vertices)
        vertices[:, :2] = vertices[:, :2] * cam_param[0] + cam_param[1:]
        center = np.array([cfg.output_hm_shape[2], cfg.output_hm_shape[1]]) / 2. - np.array([0.5, 0.5])
        vertices[:, :2] = (vertices[:, :2] - center) / center
        vertices[:, 1] = -vertices[:, 1]
        vertices[:, 2] = -(vertices[:, 2] + 5.)
        # trans[2] = 5.
        # mesh.translate(trans)
        # print(vertices)

        mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=vertices, faces=np.array(mesh.triangles), ))
        # vertex_colors=(np.load('/home/shreyas/docs/vertex_colors.npy')[:,[2,1,0]])))

        mesh_node = self.scene.add(mesh, pose=np.eye(4))
        color, depth = self.renderer.render(self.scene)
        self.scene.remove_node(mesh_node)

        return color, depth > 0


def get_img_from_axis(ax, fig):
    # Image from plot
    ax.axis('off')
    # To remove the huge white borders
    ax.margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_from_plot = image_from_plot[527:983, 1031:1488]
    cv2.imwrite('test.png', image_from_plot)

    return image_from_plot


def vis_attn(weights, mask, kpt_loc, img, ax, fig):
    jid_list = [4, 10, 19, 4 + 20, 10 + 20, 19 + 20]
    jid_list = [10, 4, 8, 12, 16, 10 + 20]
    jid_list = [41, 42]
    # jid_list = [0, 40, 33+4, 34+4, 35+4, 36+4]
    # jid_list = [0, 40, 25+8, 26+8, 27+8, 28+8]
    # jid_list = [0, 40, 24-20, 25-20, 26-20, 27-20]
    jid_col_map = ['or', 'or', 'or', 'or', 'or', 'or']
    img_list = []
    for i, jid in enumerate(jid_list):
        ax.imshow(img)
        jid_col = jid_col_map[i]
        weights[jid] = weights[jid] / np.max(weights[jid])
        for i in range(weights.shape[1]):
            if mask[i] > 0:  # and int(weights[jid,i]*2) > 0:
                ax.plot(int(kpt_loc[i, 0]), int(kpt_loc[i, 1]), jid_col, markersize=weights[jid, i] * 15)
            #     img = cv2.circle(img, (int(kpt_loc[i,0]), int(kpt_loc[i,1])), radius=weights[jid,i]*2, color=(255,0,0), thickness=-1)
        plt.show()
        image_from_plot = get_img_from_axis(ax, fig)
        img_list.append(image_from_plot)
        # ax.clear()
    return img_list


def concat_imgs(img_list, dump_dir, img_name):
    new_img_list = []
    for img in img_list:
        img = cv2.resize(img, (256, 256))
        if len(img.shape) == 2:
            img = np.tile(img[:, :, None], (1, 1, 3))
        new_img_list.append(img)
    cat_img = np.concatenate(new_img_list, axis=1)
    cv2.imwrite(os.path.join(dump_dir, img_name + '.png'), cat_img[:, :, [2, 1, 0]])


def vis_3d_keypoints_with_image(visImgKps, kps_3d, skeleton, line_width=3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb_dict = get_keypoint_rgb(skeleton)

    x, y = np.ogrid[0:visImgKps.shape[0], 0:visImgKps.shape[1]]
    ax.plot_surface(y, np.atleast_2d(2.01), -x, rstride=1, cstride=1, facecolors=visImgKps, shade=False)

    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i, 0], kps_3d[pid, 0]])
        y = np.array([kps_3d[i, 1], kps_3d[pid, 1]])
        z = np.array([kps_3d[i, 2], kps_3d[pid, 2]])

        if pid != -1:
            ax.plot(x, z, -y, c=np.array(rgb_dict[parent_joint_name]) / 255., linewidth=2)
            ax.scatter(kps_3d[i, 0], kps_3d[i, 2], -kps_3d[i, 1],
                       c=np.array(rgb_dict[parent_joint_name]).reshape(1, 3) / 255., marker='o')
        else:
            ax.scatter(kps_3d[i, 0], kps_3d[i, 2], -kps_3d[i, 1], c=np.array(rgb_dict[joint_name]).reshape(1, 3) / 255.,
                       marker='o')

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    plt.subplots_adjust(left=0.06, bottom=0.18, right=0.46, top=0.88, wspace=0.2, hspace=0.2)
    plt.savefig('Test.png')
    plt.show()


class H2O3DObjects():
    def __init__(self):
        self.get_obj_id_to_mesh()

    def get_obj_id_to_mesh(self):
        YCB_models_dir = cfg.object_models_dir
        obj_names = os.listdir(YCB_models_dir)
        self.obj_id_to_name = {int(o[:3]): o for o in obj_names}
        self.obj_id_to_mesh = {}
        self.obj_id_to_dia = {}
        for id in self.obj_id_to_name.keys():
            if id not in [3, 4, 6, 10, 11, 19, 21, 25, 35, 37, 24]:
                continue
            obj_name = self.obj_id_to_name[id]
            # print(os.path.join(YCB_models_dir, obj_name, 'textured_simple_2000.obj'))
            assert os.path.exists(os.path.join(YCB_models_dir, obj_name, 'textured_simple_2000.obj'))
            o3d_mesh = o3d.io.read_triangle_mesh(os.path.join(YCB_models_dir, obj_name, 'textured_simple_2000.obj'))
            # o3d.visualization.draw_geometries([o3d_mesh])
            self.obj_id_to_mesh[id] = o3d_mesh


def getViewMat(mesh, view_mat_path=None):
    if os.path.exists(view_mat_path):
        assert os.path.exists(view_mat_path)
        return np.loadtxt(view_mat_path)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=640, height=480, left=0, top=0,
                      visible=True)  # use visible=True to visualize the point cloud
    vis.get_render_option().light_on = False

    camera_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    fx, fy = camera_param.intrinsic.get_focal_length()
    cx = camera_param.intrinsic.intrinsic_matrix[0, 2]
    cy = camera_param.intrinsic.intrinsic_matrix[1, 2]

    # camera_param.intrinsic.set_intrinsics(640, 480, intrinsics[0, 0], intrinsics[1, 1], cx, cy)
    ctr = vis.get_view_control()

    vis.add_geometry(mesh)

    vis.run()
    vis.destroy_window()

    camera_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    extrinsic = camera_param.extrinsic
    camera_param.extrinsic = extrinsic

    np.savetxt(view_mat_path, extrinsic)
    return extrinsic


def capture_view(mesh, view_mat_path):
    assert os.path.exists(view_mat_path)
    view_mat = np.loadtxt(view_mat_path)
    # view_mat = np.eye(4)
    # view_mat[1,1] = -1
    # view_mat[2, 2] = -1

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=640, height=480, left=0, top=0,
                      visible=True)  # use visible=True to visualize the point cloud
    # vis.get_render_option().light_on = False
    vis.get_render_option().mesh_show_back_face = True

    for m in mesh:
        vis.add_geometry(m)

    camera_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    fx, fy = camera_param.intrinsic.get_focal_length()
    cx = camera_param.intrinsic.intrinsic_matrix[0, 2]
    cy = camera_param.intrinsic.intrinsic_matrix[1, 2]

    camera_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    camera_param.extrinsic = view_mat

    ctr = vis.get_view_control()
    ctr.set_constant_z_far(20.)
    ctr.set_constant_z_near(-2)
    ctr.convert_from_pinhole_camera_parameters(camera_param)

    render = vis.capture_screen_float_buffer(do_render=True)

    vis.destroy_window()

    render = (np.asarray(render) * 255).astype(np.uint8)

    return render