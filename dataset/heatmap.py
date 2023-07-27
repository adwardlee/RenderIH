import numpy as np


def build_hm(x, y, sigma=4, res=64):
    xl = np.arange(0, res, 1, float)[np.newaxis, :]
    yl = np.arange(0, res, 1, float)[:, np.newaxis]
    hm = np.exp(- ((xl - x) ** 2 + (yl - y) ** 2) / (2 * sigma ** 2))
    return hm


class HeatmapGenerator():
    def __init__(self, output_res=128, sigma=-1):
        self.output_res = output_res
        if sigma < 0:
            sigma = self.output_res / 32
        self.sigma = sigma

    def __call__(self, joints, scale=1):
        if joints.ndim == 2:
            joints = joints[np.newaxis, ...]
        if joints.shape[-1] == 2:
            joints = np.concatenate([joints, np.ones_like(joints[..., :1])], -1)

        # input : joints bs x N x 3
        bs = joints.shape[0]
        num_joints = joints.shape[1]
        hms = np.zeros((bs, num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma * scale

        for bsIdx in range(bs):
            for idx, pt in enumerate(joints[bsIdx]):
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue
                    hms[bsIdx, idx] = build_hm(x, y, sigma, self.output_res)
        return hms
