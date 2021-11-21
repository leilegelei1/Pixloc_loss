"""
A set of geometry tools for PyTorch tensors and sometimes NumPy arrays.
"""

import torch
import numpy as np

def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1]+(1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / points[..., -1:]


@torch.jit.script
def undistort_points(pts, dist):
    '''Undistort normalized 2D coordinates
       and check for validity of the distortion model.
    '''
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]
    undist = pts
    valid = torch.ones(pts.shape[:-1], device=pts.device, dtype=torch.bool)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts**2, -1, keepdim=True)
        radial = k1*r2 + k2*r2**2
        undist = undist + pts * radial

        # The distortion model is supposedly only valid within the image
        # boundaries. Because of the negative radial distortion, points that
        # are far outside of the boundaries might actually be mapped back
        # within the image. To account for this, we discard points that are
        # beyond the inflection point of the distortion model,
        # e.g. such that d(r + k_1 r^3 + k2 r^5)/dr = 0
        limited = ((k2 > 0) & ((9*k1**2-20*k2) > 0)) | ((k2 <= 0) & (k1 > 0))
        limit = torch.abs(torch.where(
            k2 > 0, (torch.sqrt(9*k1**2-20*k2)-3*k1)/(10*k2), 1/(3*k1)))
        valid = valid & torch.squeeze(~limited | (r2 < limit), -1)

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            uv = torch.prod(pts, -1, keepdim=True)
            undist = undist + 2*p12*uv + p21*(r2 + 2*pts**2)
            # TODO: handle tangential boundaries

    return undist, valid


@torch.jit.script
def J_undistort_points(pts, dist):
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]

    J_diag = torch.ones_like(pts)
    J_cross = torch.zeros_like(pts)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts**2, -1, keepdim=True)
        uv = torch.prod(pts, -1, keepdim=True)
        radial = k1*r2 + k2*r2**2
        d_radial = (2*k1 + 4*k2*r2)
        J_diag += radial + (pts**2)*d_radial
        J_cross += uv*d_radial

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            J_diag += 2*p12*pts.flip(-1) + 6*p21*pts
            J_cross += 2*p12*pts + 2*p21*pts.flip(-1)

    J = torch.diag_embed(J_diag) + torch.diag_embed(J_cross).flip(-1)
    return J

import torch
import numpy as np
import functools

def masked_mean(x, mask, dim):
    mask = mask.float()
    return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)

def checkpointed(cls, do=True):
    '''Adapted from the DISK implementation of Michał Tyszkiewicz.'''
    assert issubclass(cls, torch.nn.Module)

    class Checkpointed(cls):
        def forward(self, *args, **kwargs):
            super_fwd = super(Checkpointed, self).forward
            if any((torch.is_tensor(a) and a.requires_grad) for a in args):
                return torch.utils.checkpoint.checkpoint(
                        super_fwd, *args, **kwargs)
            else:
                return super_fwd(*args, **kwargs)

    return Checkpointed if do else cls


def torchify(func):
    """Extends to NumPy arrays a function written for PyTorch tensors.

    Converts input arrays to tensors and output tensors back to arrays.
    Supports hybrid inputs where some are arrays and others are tensors:
    - in this case all tensors should have the same device and float dtype;
    - the output is not converted.

    No data copy: tensors and arrays share the same underlying storage.

    Warning: kwargs are currently not supported when using jit.
    """
    # TODO: switch to  @torch.jit.unused when is_scripting will work
    @torch.jit.ignore
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        device = None
        dtype = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device_ = arg.device
                if device is not None and device != device_:
                    raise ValueError(
                        'Two input tensors have different devices: '
                        f'{device} and {device_}')
                device = device_
                if torch.is_floating_point(arg):
                    dtype_ = arg.dtype
                    if dtype is not None and dtype != dtype_:
                        raise ValueError(
                            'Two input tensors have different float dtypes: '
                            f'{dtype} and {dtype_}')
                    dtype = dtype_

        args_converted = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg).to(device)
                if torch.is_floating_point(arg):
                    arg = arg.to(dtype)
            args_converted.append(arg)

        rets = func(*args_converted, **kwargs)

        def convert_back(ret):
            if isinstance(ret, torch.Tensor):
                if device is None:  # no input was torch.Tensor
                    ret = ret.cpu().numpy()
            return ret

        # TODO: handle nested struct with map tensor
        if not isinstance(rets, tuple):
            rets = convert_back(rets)
        else:
            rets = tuple(convert_back(ret) for ret in rets)
        return rets

    # BUG: is_scripting does not work in 1.6 so wrapped is always called
    if torch.jit.is_scripting():
        return func
    else:
        return wrapped