import numpy as np
import cupy as cp
from numba import cuda
import time

from torch import conv2d

def benchmark_cpu(n):
    a=np.empty((n, n), dtype=np.int32)
    b=np.empty((n, n), dtype=np.int32)
    
    start=time.time()
    c = conv2d(a, b)
    cpu_time=time.time()-start
    return cpu_time

def benchmark_cupy(n):
    a=cp.empty((n, n), dtype=cp.int32)
    b=cp.empty((n, n), dtype=cp.int32)
    
    # 预热GPU
    cp.cuda.Stream.null.synchronize()
    
    start=time.time()
    c = conv2d(a, b)
    cp.cuda.Stream.null.synchronize()
    cupy_time=time.time()-start
    return cupy_time

def benchmark_numba(n):
    @cuda.jit
    def conv_kernel(a, b, c):
        row, col = cuda.grid(2)
        if row < a.shape[0] and col < a.shape[1]:
            c[row, col] = conv2d(a[row, col], b[row, col])
    
    a=np.empty((n, n), dtype=np.int32)
    b=np.empty((n, n), dtype=np.int32)
    c=np.zeros_like(a)

    d_a=cuda.to_device(a)
    d_b=cuda.to_device(b)
    d_c=cuda.device_array_like(c)

    threads_per_block=(16, 16)
    blocks_per_grid=(n + threads_per_block[0] - 1) // threads_per_block[0], (n + threads_per_block[1] - 1) // threads_per_block[1]
    # 预热
    conv_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()

    start=time.time()

    conv_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()

    numba_time=time.time()-start
    return numba_time
    
    

# 测试不同大小的数组
sizes = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]

print("性能对比（秒）:")
print(f"{'数组大小':<12} {'CPU':<12} {'CuPy':<12} {'Numba':<12}")
print("-" * 50)

for n in sizes:
    cpu_t = benchmark_cpu(n)
    cupy_t = benchmark_cupy(n)
    numba_t = benchmark_numba(n)
    
    print(f"{n:<12} {cpu_t:<12.4f} {cupy_t:<12.4f} {numba_t:<12.4f}")
    print(f"加速比: CuPy vs CPU: {cpu_t/cupy_t:.1f}x, Numba vs CPU: {cpu_t/numba_t:.1f}x")
    print()