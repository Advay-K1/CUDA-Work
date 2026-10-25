/* 
C = alpha * AB + beta * C
A --> [M, K]
B --> [K, N]
C --> [M, N]
*/
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

__global__ void sgemmV1(int M, int N, int K, float alpha, const float * A, const float * B, float beta, float * C) {
    //Cij = alpha * summation from k=0 to K - 1 of (Aik * Bkj) + beta Cij
    //get element of C to compute

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int column = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && column < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + column];
        }

        C[row * N + column] = alpha * sum + beta * C[row * N + column];
    }
}

int main() {
    int M = 4096, N = 4096, K = 4096;

    float alpha = 1.0f;
    float beta = 0.0f;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < M * K; i++) {
        h_A[i] = dist(gen);
    }

    for (int i = 0; i < K * N; i++) {
        h_B[i] = dist(gen);
    }

    float *d_A, *d_B, *d_C;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), sizeC, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim(
        (M + blockDim.x - 1) / blockDim.x,
        (N + blockDim.y - 1) / blockDim.y
    );

    sgemmV1<<<gridDim, blockDim>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C
    );

    cudaError_t err = cudaGetLastError();

    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }

    if (err != cudaSuccess) {
        std::cerr << "CUDA error: "
                  << cudaGetErrorString(err)
                  << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 1;
    }

    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "First 5x5 elements of C:\n";

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}