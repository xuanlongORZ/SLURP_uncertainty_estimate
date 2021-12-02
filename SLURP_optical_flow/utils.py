import numpy as np
from PIL import Image

import torch
from torch import Tensor

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".flo",".npy",".ppm",".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

UNKNOWN_FLOW_THRESH = 1e7
def flow_to_image(flow):
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0
    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))
    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)
    img = compute_color(u, v)
    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0
    return np.uint8(img)
def compute_color(u, v):
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img
def make_color_wheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel



class TotalVariation(torch.nn.Module):
    """Calculate the total variation for one or batch tensor.

    The total variation is the sum of the absolute differences for neighboring
    pixel-values in the input images.

    """

    def __init__(self, *, is_mean_reduction: bool = False) -> None:
        """Constructor.

        Args:
            is_mean_reduction (bool, optional):
                When `is_mean_reduction` is True, the sum of the output will be
                divided by the number of elements those used
                for total variation calculation. Defaults to False.
        """
        super(TotalVariation, self).__init__()
        self._is_mean = is_mean_reduction

    def forward(self, tensor_: Tensor) -> Tensor:
        return self._total_variation(tensor_)

    def _total_variation(self, tensor_: Tensor) -> Tensor:
        """Calculate total variation.

        Args:
            tensor_ (Tensor): input tensor must be the any following shapes:
                - 2-dimensional: [height, width]
                - 3-dimensional: [channel, height, width]
                - 4-dimensional: [batch, channel, height, width]

        Raises:
            ValueError: Input tensor is not either 2, 3 or 4-dimensional.

        Returns:
            Tensor: the output tensor shape depends on the size of the input.
                - Input tensor was 2 or 3 dimensional
                    return tensor as a scalar
                - Input tensor was 4 dimensional
                    return tensor as an array
        """
        ndims_ = tensor_.dim()

        if ndims_ == 2:
            y_diff = tensor_[1:, :] - tensor_[:-1, :]
            x_diff = tensor_[:, 1:] - tensor_[:, :-1]
        elif ndims_ == 3:
            y_diff = tensor_[:, 1:, :] - tensor_[:, :-1, :]
            x_diff = tensor_[:, :, 1:] - tensor_[:, :, :-1]
        elif ndims_ == 4:
            y_diff = tensor_[:, :, 1:, :] - tensor_[:, :, :-1, :]
            x_diff = tensor_[:, :, :, 1:] - tensor_[:, :, :, :-1]
        else:
            raise ValueError(
                'Input tensor must be either 2, 3 or 4-dimensional.')

        sum_axis = tuple({abs(x) for x in range(ndims_ - 3, ndims_)})
        y_denominator = (
            y_diff.shape[sum_axis[0]::].numel() if self._is_mean else 1
        )
        x_denominator = (
            x_diff.shape[sum_axis[0]::].numel() if self._is_mean else 1
        )

        return (
            torch.sum(torch.abs(y_diff), dim=sum_axis) / y_denominator
            + torch.sum(torch.abs(x_diff), dim=sum_axis) / x_denominator
        )