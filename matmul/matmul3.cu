/* 
C = alpha * AB + beta * C
A --> [M, K]
B --> [K, N]
C --> [M, N]

Optimization --> Applied simple tiling logic where we load tiles of A and B into shared memory
and compute matmuls for these tiles and write back to HBM
*/

#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

__global__ void sgemmV3(int M, int N, int K, float alpha, const float * A, const float * B, float beta, float * C) {
    constexpr int BLOCKSIZE = 32;
    //32 x 32 block --> 32 * 32 * 4 = 4096 bytes --> 8KiB of shared memory per block (A and B)
    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    //local thread coordinates within the 32 x 32 tile
    int threadRow = threadIdx.x / BLOCKSIZE;
    int threadCol = threadIdx.x % BLOCKSIZE;

    //global output coorindates 
    const int row = blockIdx.y * BLOCKSIZE + threadRow;
    const int col = blockIdx.x * BLOCKSIZE + threadCol;

    float tmp = 0.0f;

    for (int kBase = 0; kBase < K; kBase += BLOCKSIZE) {
        //load A Tile

        if (row < M && kBase + threadCol < K) {
            As[threadRow][threadCol] = A[row * K + kBase + threadCol]; 
        } else {
            As[threadRow][threadCol] =0.0f;
        }

        //load B Tile
        if (kBase + threadRow < K && col < N) {
            Bs[threadRow][threadCol] = B[(kBase + threadRow) * N + col];
        } else {
            Bs[threadRow][threadCol] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; ++k) {
            tmp += As[threadRow][k] * Bs[k][threadCol];
        }

        __syncthreads();        
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
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

    dim3 gridDim(
        (M + 32 - 1) / 32,
        (N + 32 - 1) / 32
    );
    dim3 blockDim(32 * 32);


    sgemmV3<<<gridDim, blockDim>>>(
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

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}