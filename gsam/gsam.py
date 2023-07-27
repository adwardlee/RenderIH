import torch
from .util import enable_running_stats, disable_running_stats
import contextlib
from torch.distributed import ReduceOp

from core.Loss import calc_aux_loss

def gsam_loss(inputs, targets, cfg, epoch,
                  graph_loss_left, graph_loss_right,
                  converter_left, converter_right,
                  img_size,
                  upsample_weight=None):
    result, paramsDict, handDictList, otherInfo = inputs
    v2d_l, j2d_l, v2d_r, j2d_r,\
    v3d_l, j3d_l, v3d_r, j3d_r,\
    root_rel = targets
    aux_lost_dict = {}
    aux_lost_dict['total_loss'] = 0#calc_aux_loss(cfg, graph_loss_left,otherInfo,mask, dense, hms)

    v3d_r = v3d_r + root_rel.unsqueeze(1)
    j3d_r = j3d_r + root_rel.unsqueeze(1)

    v2dList = []
    v3dList = []
    for i in range(len(handDictList)):
        v2dList.append(handDictList[i]['verts2d']['left'])
        v3dList.append(handDictList[i]['verts3d']['left'])
    mano_loss_dict_left, coarsen_loss_dict_left \
        = graph_loss_left.calc_loss(converter_left,
                                    v3d_l, v2d_l,
                                    result['verts3d']['left'], result['verts2d']['left'],
                                    v3dList, v2dList,
                                    img_size)

    v2dList = []
    v3dList = []
    for i in range(len(handDictList)):
        v2dList.append(handDictList[i]['verts2d']['right'])
        v3dList.append(handDictList[i]['verts3d']['right'])
    mano_loss_dict_right, coarsen_loss_dict_right \
        = graph_loss_right.calc_loss(converter_right,
                                     v3d_r, v2d_r,
                                     result['verts3d']['right'], result['verts2d']['right'],
                                     v3dList, v2dList,
                                     img_size)

    mano_loss_dict = {}
    for k in mano_loss_dict_left.keys():
        mano_loss_dict[k] = (mano_loss_dict_left[k] + mano_loss_dict_right[k]) / 2

    coarsen_loss_dict = {}
    for k in coarsen_loss_dict_left.keys():
        coarsen_loss_dict[k] = []
        for i in range(len(coarsen_loss_dict_left[k])):
            coarsen_loss_dict[k].append((coarsen_loss_dict_left[k][i] + coarsen_loss_dict_right[k][i]) / 2)

    cfg = cfg.LOSS_WEIGHT
    alpha = 0 if epoch < 50 else 1

    if upsample_weight is not None:
        mano_loss_dict['upsample_norm_loss'] = graph_loss_left.upsample_weight_loss(upsample_weight)
    else:
        mano_loss_dict['upsample_norm_loss'] = torch.zeros_like(mano_loss_dict['vert3d_loss'])

    mano_loss = 0 \
        + cfg.DATA.LABEL_3D * mano_loss_dict['vert3d_loss'] \
        + cfg.DATA.LABEL_2D * mano_loss_dict['vert2d_loss'] \
        + cfg.DATA.LABEL_3D * mano_loss_dict['joint_loss'] \
        + cfg.GRAPH.NORM.NORMAL * mano_loss_dict['norm_loss'] \
        + alpha * cfg.GRAPH.NORM.EDGE * mano_loss_dict['edge_loss']

    coarsen_loss = 0
    for i in range(len(coarsen_loss_dict['v3d_loss'])):
        coarsen_loss = coarsen_loss \
            + cfg.DATA.LABEL_3D * coarsen_loss_dict['v3d_loss'][i]  \
            + cfg.DATA.LABEL_2D * coarsen_loss_dict['v2d_loss'][i]


    total_loss = mano_loss + coarsen_loss + cfg.NORM.UPSAMPLE * mano_loss_dict['upsample_norm_loss']

    return total_loss

class GSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, gsam_alpha, rho_scheduler, adaptive=False, perturb_eps=1e-12, grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(GSAM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps
        self.alpha = gsam_alpha
        
        # initialize self.rho_t
        self.update_rho_t()
        
        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else: # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')
    
    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm( weight_adaptive = self.adaptive )
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w
                
    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):
        # calculate inner product
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                inner_prod += torch.sum(
                    self.state[p]['old_g'] * p.grad.data
                )

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='old_g')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['old_g'] - cosine * old_grad_norm * p.grad.data / (new_grad_norm + self.perturb_eps)
                p.grad.data.add_( vertical, alpha=-alpha)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized(): # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                    torch.stack([
                        ( (torch.abs(p.data) if weight_adaptive else 1.0) *  p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        else:
            norm = torch.norm(
                torch.stack([
                    ( (torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # disable running stats for second pass
            disable_running_stats(self.model)

            # get gradient at perturbed weights
            get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()
            
        # synchronize gradients across workers
        self._sync_grad()    

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return outputs, loss_value
