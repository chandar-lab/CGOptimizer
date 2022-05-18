"""
Implementations for _C enhanced optimizers as well as their vanilla counterparts.
Vanilla algorthims are sourced from PyTorch source, and _C iterations are largely
based on those as well.

https://github.com/pytorch/pytorch/tree/master/torch/optim
"""

import math
import torch
from torch.optim import Optimizer
from .priority_dict import PriorityDict
from copy import deepcopy


def aggregate(d_p, crit_buf, func, kappa=1.0):
    """
    Reusable aggregation function to join current iteration gradient and critical
    gradients

    :param d_p: Current-iteration gradient
    :param crit_buf: Buffer of Critical Gradients
    :param func: String name of aggregation. Should be "sum", "mid", or "mean"
    :param kappa: Multiplicative factor for CG buffer
    :return: Aggregated total gradient
    """

    if "sum" == func:
        crit_buf_ = crit_buf.gradMean()
        crit_buf_.mul_(kappa)
        return torch.add(d_p, crit_buf_)
    elif "mid" == func:
        crit_buf_ = crit_buf.gradMean()
        crit_buf_.mul_(kappa)
        return torch.mul(torch.add(d_p, crit_buf_), 0.5)
    elif "mean" == func:
        crit_buf_ = crit_buf.gradSum()
        crit_buf_.mul_(kappa)
        return torch.div(torch.add(d_p, crit_buf_), crit_buf.size() + 1)
    else:
        raise ValueError("Invalid aggregation function")


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}
        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)
        self.resetOfflineStats()

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

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
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

        return loss


class SGD_C(Optimizer):
    """
    Implementation of SGD (and optionally SGD with momentum) with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.

    Conventional SGD or SGD with momentum can be recovered by setting kappa=0.

    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=0.001, kappa=1.0, dampening=0.,
                 weight_decay=0, momentum=0.,
                 decay=0.7, topC=10, aggr='sum', synced=True):

        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid alpha value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid alpha value: {}".format(topC))

        defaults = dict(lr=lr, kappa=kappa, dampening=dampening,
                        weight_decay=weight_decay, momentum=momentum,
                        aggr=aggr, decay=decay, topC=topC, synced=synced)

        super(SGD_C, self).__init__(params, defaults)
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
        super(SGD_C, self).__setstate__(state)

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
                        crit_buf = param_state['critical gradients'] = PriorityDict()
                        crit_buf.setHyper(decay_rate=decay, K=topc)
                        crit_buf[d_p_norm] = deepcopy(d_p)
                    else:
                        crit_buf = param_state['critical gradients']
                        aggr_grad = aggregate(d_p, crit_buf, aggr, kappa)
                        if crit_buf.isFull():
                            if d_p_norm > crit_buf.pokeSmallest():
                                self.offline_grad['yes'] += 1
                                crit_buf[d_p_norm] = deepcopy(d_p)
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            crit_buf[d_p_norm] = deepcopy(d_p)
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


class Adam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)
        self.resetOfflineStats()

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    #  Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class AdamW(Optimizer):
    r"""Implements AdamW algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (default: 0.01)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _AdamW\: A Method for Stochastic Optimization:
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)
        self.resetOfflineStats()

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.mul_(1 - group['lr'] * group['weight_decay'])
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    #  Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # if group['weight_decay'] != 0:
                #     grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class Adam_C(Optimizer):
    """
    Implementation of Adam with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.

    Conventional Adam can be recovered by setting kappa=0.

    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 decay=0.7, kappa=1.0, topC=10,
                 weight_decay=0, amsgrad=False, aggr='mean', synced=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid alpha value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid alpha value: {}".format(topC))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, aggr=aggr, amsgrad=amsgrad,
                        kappa=kappa, topC=topC, decay=decay, synced=synced)

        super(Adam_C, self).__init__(params, defaults)
        self.resetOfflineStats()
        self.resetAnalysis()

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def __setstate__(self, state):
        super(Adam_C, self).__setstate__(state)
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

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if kappa > 0.:
                        state['critical gradients'] = PriorityDict()
                        state['critical gradients'].setHyper(decay_rate=decay, K=topc)
                        state['critical gradients'][grad_norm] = deepcopy(grad)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                else:
                    if kappa > 0.:
                        aggr_grad = aggregate(grad, state['critical gradients'], aggr)
                        if state['critical gradients'].isFull():
                            if grad_norm > state['critical gradients'].pokeSmallest():
                                self.offline_grad['yes'] += 1
                                state['critical gradients'][grad_norm] = deepcopy(grad)
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            state['critical gradients'][grad_norm] = deepcopy(grad)
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
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group['eps'])

                step_size = group['lr'] / bias_correction1

                state['critical gradients'].decay()

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class AdamW_C(Optimizer):
    """
    Implementation of AdamW with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.

    Conventional AdamW can be recovered by setting kappa=0.

    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 decay=0.7, kappa=1.0, topC=10,
                 weight_decay=0.01, amsgrad=False, aggr='mean', synced=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid alpha value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid alpha value: {}".format(topC))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, aggr=aggr, amsgrad=amsgrad,
                        kappa=kappa, topC=topC, decay=decay, synced=synced)

        super(AdamW_C, self).__init__(params, defaults)
        self.resetOfflineStats()
        self.resetAnalysis()

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def __setstate__(self, state):
        super(AdamW_C, self).__setstate__(state)
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
                        'AdamW does not support sparse gradients, please consider SparseAdamW instead')
                amsgrad = group['amsgrad']
                kappa = group['kappa']
                decay = group['decay']
                topc = group['topC']
                aggr = group['aggr']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if kappa > 0.:
                        state['critical gradients'] = PriorityDict()
                        state['critical gradients'].setHyper(decay_rate=decay, K=topc)
                        state['critical gradients'][grad_norm] = deepcopy(grad)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                else:
                    if kappa > 0.:
                        aggr_grad = aggregate(grad, state['critical gradients'], aggr)
                        if state['critical gradients'].isFull():
                            if grad_norm > state['critical gradients'].pokeSmallest():
                                self.offline_grad['yes'] += 1
                                state['critical gradients'][grad_norm] = deepcopy(grad)
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            state['critical gradients'][grad_norm] = deepcopy(grad)
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
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group['eps'])

                step_size = group['lr'] / bias_correction1

                state['critical gradients'].decay()

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class RMSprop(Optimizer):
    r"""Implements RMSprop algorithm.
    Proposed by G. Hinton in his
    `course <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`.
    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The
    effective learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where
    :math:`\alpha` is the scheduled learning rate and :math:`v` is the weighted
    moving average of the squared gradient.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps,
                        centered=centered, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)
        self.resetOfflineStats()

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)
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
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] =\
                        torch.zeros_like(p,memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = \
                            torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = \
                            torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(
                        group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

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


class RMSprop_C(Optimizer):
    """
    Implementation of RMSprop with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.

    Conventional RMSprop can be recovered by setting kappa=0.

    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False, decay=0.7, kappa=1.0,
                 topC=10, aggr='mean', synced=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid alpha value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid alpha value: {}".format(topC))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps,
                        centered=centered, weight_decay=weight_decay,
                        aggr=aggr, kappa=kappa, topC=topC, decay=decay, synced=synced)
        super(RMSprop_C, self).__init__(params, defaults)
        self.resetOfflineStats()

    def __setstate__(self, state):
        super(RMSprop_C, self).__setstate__(state)
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
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = \
                        torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = \
                            torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = \
                            torch.zeros_like(p, memory_format=torch.preserve_format)
                    if kappa > 0.:
                        state['critical gradients'] = PriorityDict()
                        state['critical gradients'].setHyper(decay_rate=decay, K=topc)
                        state['critical gradients'][grad_norm] = deepcopy(grad)
                else:
                    aggr_grad = aggregate(grad, state['critical gradients'], aggr)
                    if kappa > 0.:
                        if state['critical gradients'].isFull():
                            if grad_norm > state['critical gradients'].pokeSmallest():
                                self.offline_grad['yes'] += 1
                                state['critical gradients'][grad_norm] = deepcopy(grad)
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            state['critical gradients'][grad_norm] = deepcopy(grad)
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
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(
                        group['eps'])
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
