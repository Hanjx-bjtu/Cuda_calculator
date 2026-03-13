#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>

__global__ void matrixAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

void matrixAdd_nonCUDA(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

void checkResult(const float *A, const float *B, const float *C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        if (C[i] != A[i] + B[i])
        {
            std::cerr << "\033[1;31m\tError at index " << i << ": " << C[i] << " != " << A[i] + B[i] << "\033[0m" << std::endl;
            return;
        }
    }
    std::cout << "\033[1;32m\tResults are correct!\033[0m" << std::endl;
}

void Test(int N)
{
    printf("Testing with N = %d\n", N);
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Perform matrix addition on CPU for comparison
    {
        struct timeval start, stop;
        gettimeofday(&start, NULL);
        matrixAdd_nonCUDA(h_A, h_B, h_C, N);
        gettimeofday(&stop, NULL);
        printf("CPU execution time: %.3f ms\n", (stop.tv_sec - start.tv_sec) * 1000.0 + (stop.tv_usec - start.tv_usec) / 1000.0);
        checkResult(h_A, h_B, h_C, N);
    }

    // Perform matrix addition on GPU
    {
        // Set time the kernel execution
        struct timeval start, stop;

        gettimeofday(&start, NULL);

        // Allocate device memory
        float *d_A;
        float *d_B;
        float *d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        // Copy data from host to device
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Launch kernel
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        matrixAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

        // Copy result back to host
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        gettimeofday(&stop, NULL);

        printf("Kernel execution time: %.3f ms\n", (stop.tv_sec - start.tv_sec) * 1000.0 + (stop.tv_usec - start.tv_usec) / 1000.0);

        // Check results
        checkResult(h_A, h_B, h_C, N);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\n");
}

int main()
{
    int element_num[21]={1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456,536870912,1073741824};
    for(int i=0;i<21;i++)
    {
        Test(element_num[i]);
    }
    return 0;
}