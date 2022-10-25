import torch
import numpy as np

# evaluation metrics
# Rotation error worths attention due to symmetrics, especially for pure point clouds.
def rot_diff_rad(rot1, rot2, axis=None, up_and_down_sym=False):
    if axis <= 2 and 0 <= axis:
        if isinstance(rot1, np.ndarray):
            y1, y2 = rot1[..., axis], rot2[..., axis]  # [Bs, 3]
            diff = np.sum(y1 * y2, axis=-1)  # [Bs]
            diff = np.clip(diff, a_min=-1.0, a_max=1.0)
            if up_and_down_sym:
                diff = np.abs(diff)
            return np.arccos(diff)
        else:
            y1, y2 = rot1[..., axis], rot2[..., axis]  # [Bs, 3]
            diff = torch.sum(y1 * y2, dim=-1)  # [Bs]
            diff = torch.clamp(diff, min=-1.0, max=1.0)
            if up_and_down_sym:
                diff = torch.abs(diff)
            return torch.acos(diff)
    else:
        if isinstance(rot1, np.ndarray):
            mat_diff = np.matmul(rot1, rot2.swapaxes(-1, -2))
            diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
            diff = (diff - 1) / 2.0
            diff = np.clip(diff, a_min=-1.0, a_max=1.0)
            return np.arccos(diff)
        else:
            mat_diff = torch.matmul(rot1, rot2.transpose(-1, -2))
            diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
            diff = (diff - 1) / 2.0
            diff = torch.clamp(diff, min=-1.0, max=1.0)
            return torch.acos(diff)

def rot_diff_degree(rot1, rot2, axis=None, up_and_down_sym=False):
    return rot_diff_rad(rot1, rot2, axis=axis, up_and_down_sym=up_and_down_sym) / np.pi * 180.0


def trans_diff(trans1, trans2):  # [N, 3]
    return torch.norm((trans1 - trans2),p=2, dim=-1)  # [..., 3, 1] -> [..., 3] -> [...]


def scale_diff(scale1, scale2):
    return torch.abs(scale1 - scale2)


def theta_diff(theta1, theta2):
    return torch.abs(theta1 - theta2)



