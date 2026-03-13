import numpy as np
import cupy as cp
from numba import cuda
import time

def benchmark_cpu(n):
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    
    start = time.time()
    c = a + b
    cpu_time = time.time() - start
    return cpu_time

def benchmark_cupy(n):
    a = cp.random.randn(n, dtype=cp.float32)
    b = cp.random.randn(n, dtype=cp.float32)
    
    # 预热GPU
    cp.cuda.Stream.null.synchronize()
    
    start = time.time()
    c = a + b
    cp.cuda.Stream.null.synchronize()
    cupy_time = time.time() - start
    return cupy_time

def benchmark_numba(n):
    @cuda.jit
    def add_kernel(a, b, c):
        idx = cuda.grid(1)
        if idx < a.size:
            c[idx] = a[idx] + b[idx]
    
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.zeros_like(a)
    
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(c)
    
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # 预热
    add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()
    
    start = time.time()
    add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()
    numba_time = time.time() - start
    
    return numba_time

# 测试不同大小的数组
sizes = [10000, 100000, 1000000, 10000000, 100000000]

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