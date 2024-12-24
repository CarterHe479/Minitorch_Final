from typing import Tuple

from .tensor import Tensor

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor_functions import Function, rand

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    input = input.permute(0, 1, 3, 2)
    input = input.contiguous()
    input = input.view(batch, channel, width, new_height, kh)
    input = input.permute(0, 1, 3, 2, 4)
    input = input.contiguous()
    input = input.view(batch, channel, new_height, new_width, kh * kw)
    return input, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to the input tensor.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, height, width = input.shape
    t, _, _ = tile(input, kernel)
    t = t.mean(dim=4)
    t = t.view(batch, channel, t.shape[2], t.shape[3])
    return t


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:  # noqa: D417
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input: batch x channel x height x width

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:  # noqa: D102
        max_reduce_prime = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, max_reduce_prime)
        return max_reduce_prime

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:  # noqa: D102
        input, max_reduce = ctx.saved_tensors
        return grad_output * (input == max_reduce), 0.0


def max(input: Tensor, dim: int) -> Tensor:  # noqa: D417
    """Compute the max as a 1-hot tensor.

    Args:
    ----
        input: batch x channel x height x width

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:  # noqa: D417
    """Compute the softmax as a tensor.

    Args:
    ----
        input: batch x channel x height x width

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    t = input.exp()
    s = t.sum(dim)
    return t / s


def logsoftmax(input: Tensor, dim: int) -> Tensor:  # noqa: D417
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input: batch x channel x height x width

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    # t = input.exp()
    # s = t.sum(dim)
    # return input - s.log()

    # using logsumexp trick, where the max is substracted then added back in
    t = input - max(input, dim)
    s = t.exp().sum(dim)
    return t - s.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to the input tensor.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, height, width = input.shape
    t, _, _ = tile(input, kernel)
    t = max(t, 4)
    t = t.view(batch, channel, t.shape[2], t.shape[3])
    return t


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:  # noqa: D417
    """Dropout positions based on random noise, include an argument to turn off.

    Args:
    ----
        input: batch x channel x height x width
        p: probability of dropping
        training: if True, apply dropout, if False, return input

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    if not ignore:
        return input * (rand(input.shape) > p)
    else:
        return input


# TODO: Implement for Task 4.3.
