#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>
#include <cmath>

__constant__ float d_kernel[1024];

__global__ void matrixConv(const float *A, float *C, int input_size, int kernel_size)
{
    int output_size = input_size - kernel_size + 1;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_size && y < output_size)
    {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel_size; ky++)
        {
            for (int kx = 0; kx < kernel_size; kx++)
            {
                int input_x = x + kx;
                int input_y = y + ky;
                sum += A[input_y * input_size + input_x] *
                       d_kernel[ky * kernel_size + kx];
            }
        }
        C[y * output_size + x] = sum;
    }
}

void matrixConv_nonCUDA(const float *A, const float *B, float *C, int input_size, int kernel_size)
{
    int output_size = input_size - kernel_size + 1;
    for (int y = 0; y < output_size; y++)
    {
        for (int x = 0; x < output_size; x++)
        {
            float sum = 0.0f;
            for (int ky = 0; ky < kernel_size; ky++)
            {
                for (int kx = 0; kx < kernel_size; kx++)
                {
                    int input_x = x + kx;
                    int input_y = y + ky;
                    sum += A[input_y * input_size + input_x] *
                           B[ky * kernel_size + kx];
                }
            }
            C[y * output_size + x] = sum;
        }
    }
}

bool checkResult(const float *A, const float *B, const float *C, int input_size, int kernel_size)
{
    int output_size = input_size - kernel_size + 1;
    for (int i = 0; i < output_size; i++)
    {
        for (int j = 0; j < output_size; j++)
        {
            float expected = 0.0f;
            for (int ky = 0; ky < kernel_size; ky++)
            {
                for (int kx = 0; kx < kernel_size; kx++)
                {
                    expected += A[(i + ky) * input_size + (j + kx)] *
                                B[ky * kernel_size + kx];
                }
            }
            if (fabs(expected - C[i * output_size + j]) > 1e-5)
            {
                std::cerr << "\033[1;31m\tError at index (" << i << ", " << j << "): " << C[i * output_size + j] << " != " << expected << "\033[0m" << std::endl;
                return false;
            }
        }
    }

    printf("\033[1;32m\tResults are correct!\033[0m\n");
    return true;
}

void Test(int input_size, int kernel_size = 3)
{
    std::cout << "\nTesting input_size = " << input_size
              << ", kernel_size = " << kernel_size << "\n";

    size_t bytes_in = input_size * input_size * sizeof(float);
    int output_size = input_size - kernel_size + 1;
    size_t bytes_out = output_size * output_size * sizeof(float);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);

    float *h_A = (float *)malloc(bytes_in);
    float *h_B = (float *)malloc(kernel_bytes);
    float *h_C_cpu = (float *)malloc(bytes_out);
    float *h_C_gpu = (float *)malloc(bytes_out);

    // Initialize input matrix and kernel
    for (int i = 0; i < input_size * input_size; i++)
        h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < kernel_size * kernel_size; i++)
        h_B[i] = rand() / (float)RAND_MAX;

    // CPU convolution
    {
        struct timeval cpu_start, cpu_stop;
        gettimeofday(&cpu_start, NULL);
        matrixConv_nonCUDA(h_A, h_B, h_C_cpu, input_size, kernel_size);
        gettimeofday(&cpu_stop, NULL);
        double cpu_time = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1000.0 +
                          (cpu_stop.tv_usec - cpu_start.tv_usec) / 1000.0;

        printf("CPU execution time: %.3f ms\n", (cpu_stop.tv_sec - cpu_start.tv_sec) * 1000.0 + (cpu_stop.tv_usec - cpu_start.tv_usec) / 1000.0);
        checkResult(h_A, h_B, h_C_cpu, input_size, kernel_size);
    }

    // GPU convolution
    {
        float *d_A, *d_C;
        cudaMalloc(&d_A, bytes_in);
        cudaMalloc(&d_C, bytes_out);

        cudaMemcpy(d_A, h_A, bytes_in, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_kernel, h_B, kernel_bytes);

        dim3 blockDim(16, 16);
        dim3 gridDim((output_size + blockDim.x - 1) / blockDim.x,
                     (output_size + blockDim.y - 1) / blockDim.y);

        struct timeval gpu_start, gpu_stop;
        gettimeofday(&gpu_start, NULL);

        matrixConv<<<gridDim, blockDim>>>(d_A, d_C, input_size, kernel_size);
        cudaDeviceSynchronize();

        gettimeofday(&gpu_stop, NULL);
        printf("Kernel execution time: %.3f ms\n", (gpu_stop.tv_sec - gpu_start.tv_sec) * 1000.0 + (gpu_stop.tv_usec - gpu_start.tv_usec) / 1000.0);

        cudaMemcpy(h_C_gpu, d_C, bytes_out, cudaMemcpyDeviceToHost);

        checkResult(h_A, h_B, h_C_gpu, input_size, kernel_size);

        cudaFree(d_A);
        cudaFree(d_C);
    }

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
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