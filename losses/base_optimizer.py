"""
Implements a simple differentiable optimizer based on Levenberg-Marquardt
with a constant, scalar damping factor and a fixed number of iterations.
"""

import logging
from typing import Tuple, Dict, Optional
import torch
from torch import Tensor

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from losses.base_model import BaseModel
from geometry.utils import masked_mean,torchify
from geometry.wrappers import Camera,Pose
from geometry.optimization import optimizer_step
from geometry.interpolation import Interpolator
from geometry.costs import DirectAbsoluteCost
from geometry.losses import *
from geometry import losses  # noqa
from easydict import EasyDict

logger = logging.getLogger(__name__)

class BaseOptimizer(BaseModel):
    default_conf = dict(
        num_iters=15,
        loss_fn='scaled_barron(0, 0.1)',
        jacobi_scaling=False,
        normalize_features=True,
        lambda_=0.01,  # Gauss-Newton
        interpolation=dict(
            mode='linear',
            pad=4,
        ),
        grad_stop_criteria=1e-4,
        dt_stop_criteria=5e-3,  # in meters
        dR_stop_criteria=5e-2,  # in degrees

        # deprecated entries
        sqrt_diag_damping=False,
        bound_confidence=True,
        no_conditions=True,
        verbose=False,
    )

    logging_fn = None

    def _init(self, conf):

        print(self.default_conf)
        conf = {**conf,**self.default_conf}
        print('base optimizer: ',conf)
        conf = EasyDict(conf)

        print(conf)

        self.loss_fn = eval('losses.' + conf.loss_fn)
        self.interpolator = Interpolator(**conf.interpolation)
        self.cost_fn = DirectAbsoluteCost(self.interpolator,
                                          normalize=conf.normalize_features)
        assert conf.lambda_ >= 0.
        # deprecated entries
        assert not conf.sqrt_diag_damping
        assert conf.bound_confidence
        assert conf.no_conditions
        assert not conf.verbose

    def log(self, **args):
        if self.logging_fn is not None:
            self.logging_fn(**args)

    def early_stop(self, **args):
        stop = False
        if not self.training and (args['i'] % 10) == 0:
            T_delta, grad = args['T_delta'], args['grad']
            grad_norm = torch.norm(grad.detach(), dim=-1)
            small_grad = grad_norm < self.conf.grad_stop_criteria
            dR, dt = T_delta.magnitude()
            small_step = ((dt < self.conf.dt_stop_criteria)
                          & (dR < self.conf.dR_stop_criteria))
            if torch.all(small_step | small_grad):
                stop = True
        return stop

    def J_scaling(self, J: Tensor, J_scaling: Tensor, valid: Tensor):
        if J_scaling is None:
            J_norm = torch.norm(J.detach(), p=2, dim=(-2))
            J_norm = masked_mean(J_norm, valid[..., None], -2)
            J_scaling = 1 / (1 + J_norm)
        J = J * J_scaling[..., None, None, :]
        return J, J_scaling

    def build_system(self, J: Tensor, res: Tensor, weights: Tensor):
        grad = torch.einsum('...ndi,...nd->...ni', J, res)   # ... x N x 6
        grad = weights[..., None] * grad
        grad = grad.sum(-2)  # ... x 6

        Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)  # ... x N x 6 x 6
        Hess = weights[..., None, None] * Hess
        Hess = Hess.sum(-3)  # ... x 6 x6

        return grad, Hess

    def _forward(self, data: Dict):
        return self._run(
            data['p3D'], data['F_ref'], data['F_q'], data['T_init'],
            data['cam_q'], data['mask'], data.get('W_ref_q'))

    @torchify
    def run(self, *args, **kwargs):
        return self._run(*args, **kwargs)

    def _run(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor,
             T_init: Pose, camera: Camera, mask: Optional[Tensor] = None,
             W_ref_query: Optional[Tuple[Tensor, Tensor]] = None):

        T = T_init
        J_scaling = None
        if self.conf.normalize_features:
            F_ref = torch.nn.functional.normalize(F_ref, dim=-1)
        args = (camera, p3D, F_ref, F_query, W_ref_query)
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        for i in range(self.conf.num_iters):
            res, valid, w_unc, _, J = self.cost_fn.residual_jacobian(T, *args)
            if mask is not None:
                valid &= mask
            failed = failed | (valid.long().sum(-1) < 10)  # too few points

            # compute the cost and aggregate the weights
            cost = (res**2).sum(-1)
            cost, w_loss, _ = self.loss_fn(cost)
            weights = w_loss * valid.float()
            if w_unc is not None:
                weights *= w_unc
            if self.conf.jacobi_scaling:
                J, J_scaling = self.J_scaling(J, J_scaling, valid)

            # solve the linear system
            g, H = self.build_system(J, res, weights)
            delta = optimizer_step(g, H, self.conf.lambda_, mask=~failed)
            if self.conf.jacobi_scaling:
                delta = delta * J_scaling

            # compute the pose update
            dt, dw = delta.split([3, 3], dim=-1)
            T_delta = Pose.from_aa(dw, dt)
            T = T_delta @ T

            self.log(i=i, T_init=T_init, T=T, T_delta=T_delta, cost=cost,
                     valid=valid, w_unc=w_unc, w_loss=w_loss, H=H, J=J)
            if self.early_stop(i=i, T_delta=T_delta, grad=g, cost=cost):
                break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed

    def loss(self, pred, data):
        cam_q = data['query']['camera']

        def project(T_r2q):
            return cam_q.world2image(T_r2q * data['ref']['points3D'])

        p2D_q_gt, mask = project(data['T_r2q_gt'])
        p2D_q_i, mask_i = project(data['T_r2q_init'])
        mask = (mask & mask_i).float()

        too_few = torch.sum(mask, -1) < 10
        if torch.any(too_few):
            logger.warning(
                'Few points in batch ' + str([
                    (data['scene'][i], data['ref']['index'][i].item(),
                     data['query']['index'][i].item())
                    for i in torch.where(too_few)[0]]))

        def reprojection_error(T_r2q):
            p2D_q, _ = project(T_r2q)
            err = torch.sum((p2D_q_gt - p2D_q) ** 2, dim=-1)
            err = scaled_barron(1., 2.)(err)[0] / 4
            err = masked_mean(err, mask, -1)
            return err

        num_scales = len(self.extractor.scales)
        success = None
        losses = {'total': 0.}
        for i, T_opt in enumerate(pred['T_r2q_opt']):
            err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
            loss = err / num_scales
            if i > 0:
                loss = loss * success.float()
            thresh = self.conf.success_thresh * self.extractor.scales[-1 - i]
            success = err < thresh
            losses[f'reprojection_error/{i}'] = err
            losses['total'] += loss
        losses['reprojection_error'] = err
        losses['total'] *= (~too_few).float()

        err_init = reprojection_error(pred['T_r2q_init'][0])
        losses['reprojection_error/init'] = err_init

        return losses
    def metrics(self, pred, data):
        T_q2r_gt = data['ref']['T_w2cam'] @ data['query']['T_w2cam'].inv()

        @torch.no_grad()
        def scaled_pose_error(T_r2q):
            err_R, err_t = (T_r2q @ T_q2r_gt).magnitude()
            if self.conf.normalize_dt:
                err_t /= torch.norm(T_q2r_gt.t, dim=-1)
            return err_R, err_t

        metrics = {}
        for i, T_opt in enumerate(pred['T_r2q_opt']):
            err = scaled_pose_error(T_opt)
            metrics[f'R_error/{i}'], metrics[f't_error/{i}'] = err
        metrics['R_error'], metrics['t_error'] = err

        err_init = scaled_pose_error(pred['T_r2q_init'][0])
        metrics['R_error/init'], metrics['t_error/init'] = err_init

        return metrics
