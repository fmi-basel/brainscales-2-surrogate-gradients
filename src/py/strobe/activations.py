from typing import Dict

import torch.autograd


class SuperSpike(torch.autograd.Function):
    @staticmethod
    def process_arguments(kwargs: Dict = {}):
        defaults = {
                "scale": 50.0
                }
        defaults.update(kwargs)
        return [defaults["scale"]]

    @staticmethod
    def forward(ctx, v: torch.Tensor, scale: float):
        ctx.scale = scale
        ctx.save_for_backward(v)

        return torch.gt(v, torch.as_tensor(0.0)).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        v, = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad = grad_input/(ctx.scale*torch.abs(v) + 1.0)**2
        return grad, None
