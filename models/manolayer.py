import pickle
import numpy as np
import torch
from torch.nn import Module


def convert_mano_pkl(loadPath, savePath):
    # in original MANO pkl file, 'shapedirs' component is a chumpy object, convert it to a numpy array
    manoData = pickle.load(open(loadPath, 'rb'), encoding='latin1')
    output = {}
    manoData['shapedirs'].r
    for (k, v) in manoData.items():
        if k == 'shapedirs':
            output['shapedirs'] = v.r
        else:
            output[k] = v
    pickle.dump(output, open(savePath, 'wb'))


def vec2mat(vec):
    # vec: bs * 6
    # output: bs * 3 * 3
    x = vec[:, 0:3]
    y = vec[:, 3:6]
    x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)
    y = y - torch.sum(x * y, dim=1, keepdim=True) * x
    y = y / (torch.norm(y, p=2, dim=1, keepdim=True) + 1e-8)
    z = torch.cross(x, y)
    return torch.stack([x, y, z], dim=2)


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


def get_trans(old_z, new_z):
    # z: bs x 3
    x = torch.cross(old_z, new_z)
    x = x / torch.norm(x, dim=1, keepdim=True)
    old_y = torch.cross(old_z, x)
    new_y = torch.cross(new_z, x)
    old_frame = torch.stack((x, old_y, old_z), axis=2)
    new_frame = torch.stack((x, new_y, new_z), axis=2)
    trans = torch.matmul(new_frame, old_frame.permute(0, 2, 1))
    return trans


def build_mano_frame(skelBatch):
    # skelBatch: bs x 21 x 3
    bs = skelBatch.shape[0]
    mano_son = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]  # 15
    mano_parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]  # 16
    palm_idx = [13, 1, 4, 10, 7]
    mano_order = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]  # 21

    skel = skelBatch[:, mano_order]
    z = skel[:, mano_son] - skel[:, 1:16]  # bs x 15 x 3
    z = z / torch.norm(z, dim=2, keepdim=True)
    z = torch.cat((torch.zeros_like(z[:, 0:1]), z), axis=1)  # bs x 16 x 3
    x = torch.zeros_like(z)  # bs x 16 x 3
    x[:, :, 1] = 1.0
    y = torch.zeros_like(z)  # bs x 16 x 3

    palm = skel[:, palm_idx] - skel[:, 0:1]  # bs x 5 x 3
    n = torch.cross(palm[:, :-1], palm[:, 1:])  # bs x 4 x 3
    n = n / torch.norm(n, dim=2, keepdim=True)
    palm_x = torch.zeros((bs, 5, 3), dtype=n.dtype, device=n.device)
    palm_x[:, :-1] = palm_x[:, :-1] + n
    palm_x[:, 1:] = palm_x[:, 1:] + n
    palm_x = palm_x / torch.norm(palm_x, dim=2, keepdim=True)
    x[:, palm_idx] = palm_x

    y[:, palm_idx] = torch.cross(z[:, palm_idx], x[:, palm_idx])
    y[:, palm_idx] = y[:, palm_idx] / torch.norm(y[:, palm_idx], dim=2, keepdim=True)
    x[:, palm_idx] = torch.cross(y[:, palm_idx], z[:, palm_idx])
    frame = torch.stack((x, y, z), axis=3)  # bs x 15 x 3 x 3
    for i in range(1, 16):
        if i in palm_idx:
            continue
        trans = get_trans(z[:, mano_parent[i]], z[:, i])
        frame[:, i] = torch.matmul(trans, frame[:, mano_parent[i]])
    return frame[:, 1:]


class ManoLayer(Module):
    def __init__(self, manoPath, center_idx=9, use_pca=True, new_skel=False):
        super(ManoLayer, self).__init__()

        self.center_idx = center_idx
        self.use_pca = use_pca
        self.new_skel = new_skel

        manoData = pickle.load(open(manoPath, 'rb'), encoding='latin1')

        self.new_order = [0,
                          13, 14, 15, 16,
                          1, 2, 3, 17,
                          4, 5, 6, 18,
                          10, 11, 12, 19,
                          7, 8, 9, 20]

        # 45 * 45: PCA mat
        self.register_buffer('hands_components', torch.from_numpy(manoData['hands_components'].astype(np.float32)))
        hands_components_inv = torch.inverse(self.hands_components)
        self.register_buffer('hands_components_inv', hands_components_inv)
        # 16 * 778, J_regressor is a scipy csc matrix
        J_regressor = manoData['J_regressor'].tocoo(copy=False)
        location = []
        data = []
        for i in range(J_regressor.data.shape[0]):
            location.append([J_regressor.row[i], J_regressor.col[i]])
            data.append(J_regressor.data[i])
        i = torch.LongTensor(location)
        v = torch.FloatTensor(data)
        self.register_buffer('J_regressor', torch.sparse.FloatTensor(i.t(), v, torch.Size([16, 778])).to_dense(),
                             persistent=False)
        # 16 * 3
        self.register_buffer('J_zero', torch.from_numpy(manoData['J'].astype(np.float32)), persistent=False)
        # 778 * 16
        self.register_buffer('weights', torch.from_numpy(manoData['weights'].astype(np.float32)), persistent=False)
        # (778, 3, 135)
        self.register_buffer('posedirs', torch.from_numpy(manoData['posedirs'].astype(np.float32)), persistent=False)
        # (778, 3)
        self.register_buffer('v_template', torch.from_numpy(manoData['v_template'].astype(np.float32)), persistent=False)
        # (778, 3, 10) shapedirs is <class 'chumpy.reordering.Select'>
        if isinstance(manoData['shapedirs'], np.ndarray):
            self.register_buffer('shapedirs', torch.Tensor(manoData['shapedirs']).float(), persistent=False)
        else:
            self.register_buffer('shapedirs', torch.Tensor(manoData['shapedirs'].r.copy()).float(), persistent=False)
        # 45
        self.register_buffer('hands_mean', torch.from_numpy(manoData['hands_mean'].astype(np.float32)), persistent=False)

        self.faces = manoData['f']  # 1538 * 3: faces

        self.parent = [-1, ]
        for i in range(1, 16):
            self.parent.append(manoData['kintree_table'][0, i])

    def get_faces(self):
        return self.faces

    def train(self, mode=True):
        self.is_train = mode

    def eval(self):
        self.train(False)

    def pca2axis(self, pca):
        rotation_axis = pca.mm(self.hands_components[:pca.shape[1]])  # bs * 45
        rotation_axis = rotation_axis + self.hands_mean
        return rotation_axis  # bs * 45

    def pca2Rmat(self, pca):
        return self.axis2Rmat(self.pca2axis(pca))

    def axis2Rmat(self, axis):
        # axis: bs x 45
        rotation_mat = rodrigues_batch(axis.view(-1, 3))
        rotation_mat = rotation_mat.view(-1, 15, 3, 3)
        return rotation_mat

    def axis2pca(self, axis):
        # axis: bs x 45
        pca = axis - self.hands_mean
        pca = pca.mm(self.hands_components_inv)
        return pca

    def Rmat2pca(self, R):
        # R: bs x 15 x 3 x 3
        return self.axis2pca(self.Rmat2axis(R))

    def Rmat2axis(self, R):
        # R: bs x 3 x 3
        R = R.view(-1, 3, 3)
        temp = (R - R.permute(0, 2, 1)) / 2
        L = temp[:, [2, 0, 1], [1, 2, 0]]  # bs x 3
        sin = torch.norm(L, dim=1, keepdim=False)  # bs
        L = L / (sin.unsqueeze(-1) + 1e-8)

        temp = (R + R.permute(0, 2, 1)) / 2
        temp = temp - torch.eye((3), dtype=R.dtype, device=R.device)
        temp2 = torch.matmul(L.unsqueeze(-1), L.unsqueeze(1))
        temp2 = temp2 - torch.eye((3), dtype=R.dtype, device=R.device)
        temp = temp[:, 0, 0] + temp[:, 1, 1] + temp[:, 2, 2]
        temp2 = temp2[:, 0, 0] + temp2[:, 1, 1] + temp2[:, 2, 2]
        cos = 1 - temp / (temp2 + 1e-8)  # bs

        sin = torch.clamp(sin, min=-1 + 1e-7, max=1 - 1e-7)
        theta = torch.asin(sin)

        # prevent in-place operation
        theta2 = torch.zeros_like(theta)
        theta2[:] = theta
        idx1 = (cos < 0) & (sin > 0)
        idx2 = (cos < 0) & (sin < 0)
        theta2[idx1] = 3.14159 - theta[idx1]
        theta2[idx2] = -3.14159 - theta[idx2]
        axis = theta2.unsqueeze(-1) * L

        return axis.view(-1, 45)

    def get_local_frame(self, shape):
        # output: frame[..., [0,1,2]] = [splay, bend, twist]
        # get local joint frame at zero pose
        with torch.no_grad():
            shapeBlendShape = torch.matmul(self.shapedirs, shape.permute(1, 0)).permute(2, 0, 1)
            v_shaped = self.v_template + shapeBlendShape  # bs * 778 * 3
            j_tpose = torch.matmul(self.J_regressor, v_shaped)  # bs * 16 * 3
            j_tpose_21 = torch.cat((j_tpose, v_shaped[:, [744, 320, 444, 555, 672]]), axis=1)
            j_tpose_21 = j_tpose_21[:, self.new_order]
            frame = build_mano_frame(j_tpose_21)
        return frame  # bs x 15 x 3 x 3

    @staticmethod
    def buildSE3_batch(R, t):
        # R: bs * 3 * 3
        # t: bs * 3 * 1
        # return: bs * 4 * 4
        bs = R.shape[0]
        pad = torch.zeros((bs, 1, 4), dtype=R.dtype, device=R.device)
        pad[:, 0, 3] = 1.0
        temp = torch.cat([R, t], 2)  # bs * 3 * 4
        return torch.cat([temp, pad], 1)

    @staticmethod
    def SE3_apply(SE3, v):
        # SE3: bs * 4 * 4
        # v: bs * 3
        # return: bs * 3
        bs = v.shape[0]
        pad = torch.ones((bs, 1), dtype=v.dtype, device=v.device)
        temp = torch.cat([v, pad], 1).unsqueeze(2)  # bs * 4 * 1
        return SE3.bmm(temp)[:, :3, 0]

    def forward(self, root_rotation, pose, shape, trans=None, scale=None):
        # input
        # root_rotation : bs * 3 * 3
        # pose : bs * ncomps or bs * 15 * 3 * 3
        # shape : bs * 10
        # trans : bs * 3 or None
        # scale : bs or None
        bs = root_rotation.shape[0]

        if self.use_pca:
            rotation_mat = self.pca2Rmat(pose)
        else:
            rotation_mat = pose

        shapeBlendShape = torch.matmul(self.shapedirs, shape.permute(1, 0)).permute(2, 0, 1)
        v_shaped = self.v_template + shapeBlendShape  # bs * 778 * 3

        j_tpose = torch.matmul(self.J_regressor, v_shaped)  # bs * 16 * 3

        Imat = torch.eye(3, dtype=rotation_mat.dtype, device=rotation_mat.device).repeat(bs, 15, 1, 1)
        pose_shape = rotation_mat.view(bs, -1) - Imat.view(bs, -1)  # bs * 135
        poseBlendShape = torch.matmul(self.posedirs, pose_shape.permute(1, 0)).permute(2, 0, 1)
        v_tpose = v_shaped + poseBlendShape  # bs * 778 * 3

        SE3_j = []
        R = root_rotation
        t = (torch.eye(3, dtype=pose.dtype, device=pose.device).repeat(bs, 1, 1) - R).bmm(j_tpose[:, 0].unsqueeze(2))
        SE3_j.append(self.buildSE3_batch(R, t))
        for i in range(1, 16):
            R = rotation_mat[:, i - 1]
            t = (torch.eye(3, dtype=pose.dtype, device=pose.device).repeat(bs, 1, 1) - R).bmm(j_tpose[:, i].unsqueeze(2))
            SE3_local = self.buildSE3_batch(R, t)
            SE3_j.append(torch.matmul(SE3_j[self.parent[i]], SE3_local))
        SE3_j = torch.stack(SE3_j, dim=1)  # bs * 16 * 4 * 4

        j_withoutTips = []
        j_withoutTips.append(j_tpose[:, 0])
        for i in range(1, 16):
            j_withoutTips.append(self.SE3_apply(SE3_j[:, self.parent[i]], j_tpose[:, i]))

        # there is no boardcast matmul for sparse matrix for now (pytorch 1.6.0)
        SE3_v = torch.matmul(self.weights, SE3_j.view(bs, 16, 16)).view(bs, -1, 4, 4)  # bs * 778 * 4 * 4

        v_output = SE3_v[:, :, :3, :3].matmul(v_tpose.unsqueeze(3)) + SE3_v[:, :, :3, 3:4]
        v_output = v_output[:, :, :, 0]  # bs * 778 * 3

        jList = j_withoutTips + [v_output[:, 745], v_output[:, 317], v_output[:, 444], v_output[:, 556], v_output[:, 673]]

        j_output = torch.stack(jList, dim=1)
        j_output = j_output[:, self.new_order]

        if self.center_idx is not None:
            center = j_output[:, self.center_idx:(self.center_idx + 1)]
            v_output = v_output - center
            j_output = j_output - center

        if scale is not None:
            scale = scale.unsqueeze(1).unsqueeze(2)  # bs * 1 * 1
            v_output = v_output * scale
            j_output = j_output * scale

        if trans is not None:
            trans = trans.unsqueeze(1)  # bs * 1 * 3
            v_output = v_output + trans
            j_output = j_output + trans

        if self.new_skel:
            j_output[:, 5] = (v_output[:, 63] + v_output[:, 144]) / 2
            j_output[:, 9] = (v_output[:, 271] + v_output[:, 220]) / 2
            j_output[:, 13] = (v_output[:, 148] + v_output[:, 290]) / 2
            j_output[:, 17] = (v_output[:, 770] + v_output[:, 83]) / 2

        return v_output, j_output


if __name__ == '__main__':
    convert_mano_pkl('models/MANO_RIGHT.pkl', 'MANO_RIGHT.pkl')
    convert_mano_pkl('models/MANO_LEFT.pkl', 'MANO_LEFT.pkl')

    mano = ManoLayer(manoPath='models/MANO_RIGHT.pkl', center_idx=9, use_pca=True)
    pose = torch.rand((10, 30))
    shape = torch.rand((10, 10))
    rotation = torch.rand((10, 3))
    root_rotation = rodrigues_batch(rotation)
    trans = torch.rand((10, 3))
    scale = torch.rand((10))
    v, j = mano(root_rotation=root_rotation,
                pose=pose,
                shape=shape,
                trans=trans,
                scale=scale)
