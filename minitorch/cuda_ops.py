# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:  # noqa: ANN003, D103
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs) -> FakeCUDAKernel:  # noqa: ANN003, D103
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:  # noqa: D102
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(  # noqa: D102
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:  # noqa: D102
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            out[index_to_position(out_index, out_strides)] = fn(
                in_storage[index_to_position(in_index, in_strides)]
            )

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_data = a_storage[index_to_position(a_index, a_strides)]
            b_data = b_storage[index_to_position(b_index, b_strides)]
            out[index_to_position(out_index, out_strides)] = fn(a_data, b_data)

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    if i < size:
        value = float(a[i])
        cache[pos] = value
        cuda.syncthreads()
    else:
        cache[pos] = 0.0
    if i < size:
        # for stride in range(1, BLOCK_DIM):
        for stride in [1, 2, 4, 8, 16]:
            if pos % (2 * stride) == 0:
                cache[pos] += cache[pos + stride]
            cuda.syncthreads()
        if pos == 0:
            out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:  # noqa: D103
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        cache[pos] = reduce_value

        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            dim = a_shape[reduce_dim]
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos

            if out_index[reduce_dim] < dim:
                cache[pos] = a_storage[index_to_position(out_index, a_strides)]
                cuda.syncthreads()

                idx = 0
                while 2**idx < BLOCK_DIM:
                    if pos % ((2**idx) * 2) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + 2**idx])
                        cuda.syncthreads()
                    idx += 1
            if pos == 0:
                out[index_to_position(out_index, out_strides)] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Implement a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32  # Define the maximum block dimension size for shared memory.

    # Allocate shared memory arrays for storing chunks of matrices `a` and `b`.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Identify the row index (i) and column index (j) for the current thread in the grid.
    i = cuda.blockIdx.x * BLOCK_DIM + tx
    j = cuda.blockIdx.y * BLOCK_DIM + ty

    # If the indices exceed the matrix size, exit early (bounds check).
    if i >= size or j >= size:
        return

    # Load the current element of matrix `a` into shared memory for this thread.
    a_shared[i, j] = a[i * size + j]

    # Load the current element of matrix `b` into shared memory for this thread.
    b_shared[i, j] = b[i * size + j]

    # Synchronize all threads in the block to ensure shared memory is fully populated.
    cuda.syncthreads()

    # Initialize an accumulator variable for the dot product of the row and column.
    accum = 0.0

    # Perform the dot product of row `i` of `a` with column `j` of `b`.
    for k in range(size):  # Iterate over the shared dimension.
        accum += a_shared[i, k] * b_shared[k, j]  # Multiply and accumulate the result.

    # Write the computed result to the global memory location in the output matrix.
    out[i * size + j] = accum


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:  # noqa: D103
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # Calculate the batch stride for tensor `a` based on whether batching is required.
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    # Calculate the batch stride for tensor `b` similarly.
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Determine the batch index for the current thread based on the z-dimension of the block.
    batch = cuda.blockIdx.z

    # Define the block size for shared memory.
    BLOCK_DIM = 32

    # Allocate shared memory for chunks of tensor `a` and tensor `b` being processed.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Calculate the global position (i, j) of the current thread in the output matrix.
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # Row index.
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # Column index.

    # Determine the thread's local position (pi, pj) within its block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    # Initialize an accumulator for the dot product calculation.
    accum = 0.0

    # Loop over the shared dimension of matrices `a` and `b` in blocks of size `BLOCK_DIM`.
    for idx in range(
        0, a_shape[2], BLOCK_DIM
    ):  # Iterate through chunks of the shared dimension.
        # Load a block of `a` into shared memory.
        if i < a_shape[1] and (idx + pj) < a_shape[2]:  # Ensure within bounds for `a`.
            a_shared[pi, pj] = a_storage[
                a_batch_stride * batch + a_strides[1] * i + a_strides[2] * (idx + pj)
            ]
        else:
            a_shared[pi, pj] = 0.0  # Pad out-of-bounds values with 0.

        # Load a block of `b` into shared memory.
        if j < b_shape[2] and (idx + pi) < b_shape[1]:  # Ensure within bounds for `b`.
            b_shared[pi, pj] = b_storage[
                b_batch_stride * batch + b_strides[1] * (idx + pi) + b_strides[2] * j
            ]
        else:
            b_shared[pi, pj] = 0.0  # Pad out-of-bounds values with 0.

        # Synchronize all threads to ensure shared memory is fully populated before computation.
        cuda.syncthreads()

        # Compute the dot product of the row of `a` with the column of `b` for the current block.
        for k in range(BLOCK_DIM):  # Iterate through elements within the shared block.
            if (idx + k) < a_shape[
                2
            ]:  # Ensure we don't access beyond the shared dimension.
                accum += a_shared[pi, k] * b_shared[k, pj]  # Accumulate the product.

        # Synchronize all threads to ensure all threads have completed the computation.
        cuda.syncthreads()

    # Write the final accumulated value to the global memory output matrix, if within bounds.
    if (
        batch < out_shape[0] and i < out_shape[1] and j < out_shape[2]
    ):  # Adjusted to ensure `batch` respects bounds.
        out[out_strides[0] * batch + out_strides[1] * i + out_strides[2] * j] = (
            accum  # Store the computed value in `out`.
        )


# Decorate the function with `jit` for compilation and execution with CUDA.
tensor_matrix_multiply = jit(_tensor_matrix_multiply)