#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>

__global__ void matrixMul(const float *A, const float *B, float *C, int N)
{
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    float sum = 0.0f;

    // divide the matrix into sub-matrices and calculate partial sums
    for (int k = 0; k < N; k += 16)
    {
        if (row < N && k + tx < N)
            As[ty][tx] = A[row * N + k + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && k + ty < N)
            Bs[ty][tx] = B[(k + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < 16; i++)
        {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}

void matrixMul_nonCUDA(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            C[i * N + j] = 0;
            for (int k = 0; k < N; ++k)
            {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void checkResult(const float *A, const float *B, const float *C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float expected = 0.0f;
            for (int k = 0; k < N; ++k)
            {
                expected += A[i * N + k] * B[k * N + j];
            }
            if (fabs(C[i * N + j] - expected) > 1e-5)
            {
                std::cerr << "\033[1;31m\tError at index (" << i << ", " << j << "): " << C[i * N + j] << " != " << expected << "\033[0m" << std::endl;
                return;
            }
        }
    }
    std::cout << "\033[1;32m\tResults are correct!\033[0m" << std::endl;
}

void Test(int N)
{
    printf("Testing with N = %d\n", N);
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N * N; ++i)
    {

        h_A[i] = rand() / static_cast<float>(RAND_MAX);
        h_B[i] = rand() / static_cast<float>(RAND_MAX);
    }

    // Perform matrix multiplication on CPU for comparison
    {
        struct timeval start, stop;
        gettimeofday(&start, NULL);
        matrixMul_nonCUDA(h_A, h_B, h_C, N);
        gettimeofday(&stop, NULL);
        printf("CPU execution time: %.3f ms\n", (stop.tv_sec - start.tv_sec) * 1000.0 + (stop.tv_usec - start.tv_usec) / 1000.0);
        checkResult(h_A, h_B, h_C, N);
    }

    // Perform matrix multiplication on GPU
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
        int blockSize = 64;
        int numBlocks = (N + blockSize - 1) / blockSize;
        matrixMul<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

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
    int element_num[8] = {8, 16, 32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < 8; i++)
    {
        Test(element_num[i]);
    }
    return 0;
}