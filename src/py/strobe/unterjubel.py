import torch.autograd

__all__ = ["Unterjubel", "unterjubel"]


class Unterjubel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_prime, train_on_hardware=True):
        if train_on_hardware:
            ctx.save_for_backward(input)
            return input_prime
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


unterjubel = Unterjubel.apply
