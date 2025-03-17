#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "cluster_error.h"

__device__ float cluster_difference(Cluster *cluster, Cluster *prev_cluster) {
    return sqrt((cluster->x - prev_cluster->x) * (cluster->x - prev_cluster->x) +
        (cluster->y - prev_cluster->y) * (cluster->y - prev_cluster->y) +
        (cluster->l - prev_cluster->l) * (cluster->l - prev_cluster->l) +
        (cluster->a - prev_cluster->a) * (cluster->a - prev_cluster->a) +
        (cluster->b - prev_cluster->b) * (cluster->b - prev_cluster->b));
}


__global__ void compute_cluster_error_kernel(Cluster *cluster, Cluster *prev_cluster, float *result, int num_clusters) {
    extern __shared__ float subResult[];

    int index =  threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Loat elements into shared memory
    if (index < num_clusters) {
        subResult[tid] = cluster_difference(&cluster[tid], &prev_cluster[tid]);
    } else {
        subResult[tid] = 0; // Array length is not a multiple of block size
    }

    __syncthreads(); // Sync Block threads

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            subResult[tid] += subResult[tid + s];
        }
        
        __syncthreads(); // Ensure all threads complete reduction step
    }

    // One thread in block copys this blocks result to output array
    if (tid == 0) {
        result[blockIdx.x] = subResult[0];
    }
}

float compute_cluster_error_cuda(Cluster *h_clusters, Cluster *h_prev_clusters, int num_clusters) {
    Cluster *d_clusters;
    Cluster *d_prev_clusters;
    float *d_result, *h_result;

    int numBlocks = (num_clusters + 255) / 256;

    // Allocate host memory
    h_result = (float *)malloc(numBlocks * sizeof(float));

    // Allocate device memory
    cudaError_t err = cudaMalloc(&d_clusters, num_clusters * sizeof(Cluster));
    CHECK_CUDA_ERROR(err);
    err = cudaMalloc(&d_prev_clusters, num_clusters * sizeof(Cluster));
    CHECK_CUDA_ERROR(err);
    err = cudaMalloc(&d_result, numBlocks * sizeof(float));
    CHECK_CUDA_ERROR(err);

    // Copy cluster data to device
    err = cudaMemcpy(d_clusters, h_clusters, num_clusters * sizeof(Cluster), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);
    err = cudaMemcpy(d_prev_clusters, h_prev_clusters, num_clusters * sizeof(Cluster), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    // Launch Kernel
    compute_cluster_error_kernel<<<numBlocks, 256, 256 * sizeof(float)>>>(d_clusters, d_prev_clusters, d_result, num_clusters);
    cudaDeviceSynchronize();

    // Copy result back to host
    err = cudaMemcpy(h_result, d_result, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on host
    float finalResult = 0;
    for (int i = 0; i < numBlocks; i++) {
        finalResult += h_result[i];
    }

    // Clean up memory
    cudaFree(d_clusters);
    cudaFree(d_prev_clusters);
    cudaFree(d_result);
    free(h_result);
    d_clusters = NULL;
    d_prev_clusters = NULL;
    d_result = NULL;
    h_result = NULL;

    return finalResult;
}