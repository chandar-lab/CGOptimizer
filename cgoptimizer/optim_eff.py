"""
Implementations for _C_eff enhanced optimizers as well as their vanilla counterparts.
Vanilla algorthims are sourced from PyTorch source, and _C_eff iterations are largely
based on those as well.

https://github.com/pytorch/pytorch/tree/master/torch/optim
"""

import math
from copy import deepcopy

import torch
from torch.optim import Optimizer

from .priority_dict import TensorList


class SGD_C_eff(Optimizer):
    """
    Efficient GPU-based implementation of SGD (and optionally SGD with momentum) with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.

    Conventional SGD or SGD with momentum can be recovered by setting kappa=0.

    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=0.001, kappa=1.0, dampening=0.,
                 weight_decay=0, momentum=0.,
                 decay=0.7, topC=10, aggr='sum', 
                 synced=True, buffer_dtype=None):

        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid decay value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid topC value: {}".format(topC))

        defaults = dict(lr=lr, kappa=kappa, dampening=dampening,
                        weight_decay=weight_decay, momentum=momentum, aggr=aggr,
                        decay=decay, gradHist={}, topC=topC,
                        synced=synced, buffer_dtype=buffer_dtype)

        super(SGD_C_eff, self).__init__(params, defaults)
        self.resetOfflineStats()
        self.resetAnalysis()

    def getOfflineStats(self):
        return self.offline_grad

    def getAnalysis(self):
        return self.g_analysis

    def resetAnalysis(self):
        self.g_analysis = {'gt': 0., 'gc': 0., 'count': 0, 'gc_aggr': 0}

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def __setstate__(self, state):
        super(SGD_C_eff, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            kappa = group['kappa']
            dampening = group['dampening']
            decay = group['decay']
            momentum = group['momentum']
            topc = group['topC']
            aggr = group['aggr']
            
            synced = group['synced']
            buffer_dtype = group['buffer_dtype']

            d_p_norm = 0.0

            if synced:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    d_p_norm += torch.sqrt(torch.sum(torch.square(d_p)))

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if not synced:
                    d_p_norm = d_p.norm()
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if kappa != 0:
                    param_state = self.state[p]
                    if 'critical gradients' not in param_state:
                        crit_buf = param_state['critical gradients'] = TensorList()
                        crit_buf.setHyper(decay_rate=decay, K=topc, dtype=buffer_dtype)
                        crit_buf.addItem(d_p_norm, deepcopy(d_p))
                    else:
                        crit_buf = param_state['critical gradients']
                        aggr_mean = crit_buf.aggr_sum.div(crit_buf.size())
                        aggr_grad = torch.add(d_p, aggr_mean)
                        if crit_buf.isFull():
                            if d_p_norm > crit_buf.pokeSmallest():
                                self.offline_grad['yes'] += 1
                                crit_buf.addItem(d_p_norm, deepcopy(d_p))
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            crit_buf.addItem(d_p_norm, deepcopy(d_p))
                        d_p = aggr_grad

                    crit_buf.decay()

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(
                                d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

        return loss


class Adam_C_eff(Optimizer):
    """
    Efficient GPU-based implementation of Adam with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.

    Conventional Adam can be recovered by setting kappa=0.

    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 decay=0.7, kappa=1.0, topC=10,
                 weight_decay=0, amsgrad=False, aggr='mean', synced=True, buffer_dtype=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid decay value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid topC value: {}".format(topC))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, aggr=aggr, amsgrad=amsgrad,
                        kappa=kappa, topC=topC, decay=decay, 
                        synced=synced, buffer_dtype=buffer_dtype)

        super(Adam_C_eff, self).__init__(params, defaults)
        self.resetOfflineStats()
        self.resetAnalysis()

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def __setstate__(self, state):
        super(Adam_C_eff, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def getAnalysis(self):
        return self.g_analysis

    def resetAnalysis(self):
        self.g_analysis = {'gt': 0., 'gc': 0., 'count': 0, 'gc_aggr': 0}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            grad_norm = 0.0

            if group['synced']:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    grad_norm += torch.sqrt(torch.sum(torch.square(d_p)))

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if not group['synced']:
                    grad_norm = grad.norm()
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                kappa = group['kappa']
                decay = group['decay']
                topc = group['topC']
                aggr = group['aggr']
                buffer_dtype = group['buffer_dtype']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if kappa > 0.:
                        state['critical gradients'] = TensorList()
                        state['critical gradients'].setHyper(decay_rate=decay, K=topc, dtype=buffer_dtype)
                        state['critical gradients'].addItem(grad_norm, deepcopy(grad))
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                else:
                    if kappa > 0.:
                        aggr_mean = state['critical gradients'].aggr_sum.div(state['critical gradients'].size())
                        aggr_grad = torch.add(grad, aggr_mean)
                        if state['critical gradients'].isFull():
                            if grad_norm > state['critical gradients'].pokeSmallest():
                                self.offline_grad['yes'] += 1
                                state['critical gradients'].addItem(grad_norm, deepcopy(grad))
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            state['critical gradients'].addItem(grad_norm, deepcopy(grad))
                    grad = aggr_grad

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                state['critical gradients'].decay()

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class AdamW_C_eff(Optimizer):
    """
    Efficient GPU-based implementation of Adam with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.

    Conventional Adam can be recovered by setting kappa=0.

    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 decay=0.7, kappa=1.0, topC=10,
                 weight_decay=0.01, amsgrad=False, aggr='mean', synced=True, buffer_dtype=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid decay value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid topC value: {}".format(topC))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, aggr=aggr, amsgrad=amsgrad,
                        kappa=kappa, topC=topC, decay=decay,
                        synced=synced, buffer_dtype=buffer_dtype)

        super(AdamW_C_eff, self).__init__(params, defaults)
        self.resetOfflineStats()
        self.resetAnalysis()

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def __setstate__(self, state):
        super(AdamW_C_eff, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def getAnalysis(self):
        return self.g_analysis

    def resetAnalysis(self):
        self.g_analysis = {'gt': 0., 'gc': 0., 'count': 0, 'gc_aggr': 0}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            grad_norm = 0.0

            if group['synced']:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    grad_norm += torch.sqrt(torch.sum(torch.square(d_p)))

            for p in group['params']:
                if p.grad is None:
                    continue
                p.mul_(1 - group['lr'] * group['weight_decay'])
                grad = p.grad.data
                if not group['synced']:
                    grad_norm = grad.norm()
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                kappa = group['kappa']
                decay = group['decay']
                topc = group['topC']
                aggr = group['aggr']
                buffer_dtype = group['buffer_dtype']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if kappa > 0.:
                        state['critical gradients'] = TensorList()
                        state['critical gradients'].setHyper(decay_rate=decay, K=topc, dtype=buffer_dtype)
                        state['critical gradients'].addItem(grad_norm, deepcopy(grad))
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                else:
                    if kappa > 0.:
                        aggr_mean = state['critical gradients'].aggr_sum.div(state['critical gradients'].size())
                        aggr_grad = torch.add(grad, aggr_mean)
                        if state['critical gradients'].isFull():
                            if grad_norm > state['critical gradients'].pokeSmallest():
                                self.offline_grad['yes'] += 1
                                state['critical gradients'].addItem(grad_norm, deepcopy(grad))
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            state['critical gradients'].addItem(grad_norm, deepcopy(grad))
                    grad = aggr_grad

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                state['critical gradients'].decay()

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class RMSprop_C_eff(Optimizer):
    """
    Efficient GPU-based implementation of RMSprop with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.

    Conventional RMSprop can be recovered by setting kappa=0.

    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False, decay=0.7, kappa=1.0,
                 topC=10, aggr='mean', synced=True, buffer_dtype=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid decay value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid topC value: {}".format(topC))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps,
                        centered=centered, weight_decay=weight_decay,
                        aggr=aggr, kappa=kappa, topC=topC, decay=decay,
                        synced=synced, buffer_dtype=buffer_dtype)
        super(RMSprop_C_eff, self).__init__(params, defaults)
        self.resetOfflineStats()

    def __setstate__(self, state):
        super(RMSprop_C_eff, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grad_norm = 0.0

            if group['synced']:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    grad_norm += torch.sqrt(torch.sum(torch.square(d_p)))

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if not group['synced']:
                    grad_norm = grad.norm()
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                kappa = group['kappa']
                decay = group['decay']
                topc = group['topC']
                aggr = group['aggr']
                buffer_dtype = group['buffer_dtype']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if kappa > 0.:
                        state['critical gradients'] = TensorList()
                        state['critical gradients'].setHyper(decay_rate=decay, K=topc, dtype=buffer_dtype)
                        state['critical gradients'].addItem(grad_norm, deepcopy(grad))
                else:
                    aggr_mean = state['critical gradients'].aggr_sum.div(state['critical gradients'].size())
                    aggr_grad = torch.add(grad, aggr_mean)
                    if kappa > 0.:
                        if state['critical gradients'].isFull():
                            if grad_norm > state['critical gradients'].pokeSmallest():
                                self.offline_grad['yes'] += 1
                                state['critical gradients'].addItem(grad_norm, deepcopy(grad))
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            state['critical gradients'].addItem(grad_norm, deepcopy(grad))
                    grad = aggr_grad

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                state['critical gradients'].decay()

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-group['lr'])
                else:
                    p.addcdiv_(grad, avg, value=-group['lr'])

        return loss

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}
