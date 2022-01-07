import torch.optim

class SMORMS3(torch.optim.Optimizer):
    """
    Optimizer described by Simon Funk
    """
    def __init__(self, params, lr=0.0025, eps=1e-16):
        """
        Setup optimizer with parameters.
        lr: default learning rate
        eps: default epsilon
        """
        defaults = dict(lr=lr)
        super(SMORMS3, self).__init__(params, defaults)
        self.eps = eps

    def step(self, closure=None):
        """
        Perform a single gradient step for all parameters.
        """
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None: # skip if param has no gradient
                    continue
                grad = p.grad.data
                param_state = self.state[p]
                if 'mem' not in param_state: # setup accumulators once
                    param_state['mem'] = torch.full_like(p.data, 1.)
                    param_state['g'] = torch.full_like(p.data, 0.)
                    param_state['g2'] = torch.full_like(p.data, 0.)
                mem = param_state['mem']
                g, g2 = param_state['g'], param_state['g2']
                # g = (1-r)*g + r*g, g2 = (1-r)*g2 + r*g**2
                r = 1. / (mem + 1.)
                g.mul_(1 - r).addcmul_(r, grad)
                g2.mul_(1 - r).addcmul_(r, grad**2)
                # mem = mem * (1 - g*g/(g2 + eps)) + 1
                div = g2 + self.eps
                mem.mul_(1 - (g**2 / div)).add_(1.)
                # lrate = min(lr, g*g/(g2 + eps))
                lrate = torch.clamp(g**2 / div, max=lr)
                # p = p - lrate*grad/(sqrt(g2) + eps)
                new_grad = lrate*grad / (g2.sqrt() + self.eps)
                p.data.add_(new_grad, alpha=-1)
        return loss
