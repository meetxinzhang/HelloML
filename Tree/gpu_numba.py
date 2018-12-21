'''
Matrix multiplication sample, some numba and CUDA testing code
'''
import math
import numpy as np
from numba import cuda, jit, float64

TPB = 7  # thread per block


@cuda.jit
def mat_mul_naive_kernal(A, B, C):
    '''matrix multiplication on gpu, naive method using global device memory
    '''
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        summation = 0
        for k in range(A.shape[1]):
            summation += A[i, k] * B[k, j]
        C[i, j] = summation


def host_naive(A, B):
    '''host code for calling naive kernal
    '''
    A = np.array(A)
    B = np.array(B)

    m = A.shape[0]
    n = B.shape[1]
    C = np.full((m, n), 0, dtype=np.float64)

    d_A = cuda.to_device(A)  # d_ --> device
    d_B = cuda.to_device(B)
    d_C = cuda.device_array(C.shape, np.float64)

    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(A.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(B.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    mat_mul_naive_kernal[blockspergrid, threadsperblock](d_A, d_B, d_C)

    return d_C.copy_to_host()
