#include <cuda_runtime.h>
#include <stdio.h>

#include "types.h"

// CUDA kernel to reset clusters
__global__ void resetClusters(Cluster *clusters, int num_clusters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_clusters) {
        clusters[i].x = 0;
        clusters[i].y = 0;
        clusters[i].l = 0;
        clusters[i].a = 0;
        clusters[i].b = 0;
        clusters[i].n = 0;
    }
}

// CUDA kernel to update clusters
__global__ void updateClusters(Cluster *clusters, int *segmentation_matrix, unsigned char *image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel_index = y * width + x;
        int cluster_idx = segmentation_matrix[pixel_index];

        atomicAdd((float *)&clusters[cluster_idx].x, (float)x);
        atomicAdd((float *)&clusters[cluster_idx].y, (float)y);
        atomicAdd((float *)&clusters[cluster_idx].l, (float)image[3 * pixel_index]);
        atomicAdd((float *)&clusters[cluster_idx].a, (float)image[3 * pixel_index + 1]);
        atomicAdd((float *)&clusters[cluster_idx].b, (float)image[3 * pixel_index + 2]);
        atomicAdd(&clusters[cluster_idx].n, 1);
    }
}

// CUDA kernel to normalize cluster values
__global__ void normalizeClusters(Cluster *clusters, int num_clusters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_clusters) {
        int n = clusters[i].n;
        if (n > 0) {
            clusters[i].x /= n;
            clusters[i].y /= n;
            clusters[i].l /= n;
            clusters[i].a /= n;
            clusters[i].b /= n;
        }
    }
}

// Host function to launch kernels
void compute_cluster_centers(Cluster *h_clusters, int *h_segmentation_matrix, unsigned char *h_image, int width, int height, int num_clusters, int m) {
    Cluster *d_clusters;
    int *d_segmentation_matrix;
    unsigned char *d_image;

    // Allocate device memory
    cudaMalloc(&d_clusters, num_clusters * sizeof(Cluster));
    cudaMalloc(&d_segmentation_matrix, width * height * sizeof(int));
    cudaMalloc(&d_image, 3 * width * height * sizeof(unsigned char));

    // Copy data to device
    cudaMemcpy(d_clusters, h_clusters, num_clusters * sizeof(Cluster), cudaMemcpyHostToDevice);
    cudaMemcpy(d_segmentation_matrix, h_segmentation_matrix, width * height * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, h_image, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Reset clusters
    resetClusters<<<(num_clusters + 255) / 256, 256>>>(d_clusters, num_clusters);
    cudaDeviceSynchronize();

    // Update clusters
    updateClusters<<<gridDim, blockDim>>>(d_clusters, d_segmentation_matrix, d_image, width, height);
    cudaDeviceSynchronize();

    // Normalize clusters
    normalizeClusters<<<(num_clusters + 255) / 256, 256>>>(d_clusters, num_clusters);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_clusters, d_clusters, num_clusters * sizeof(Cluster), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_clusters);
    cudaFree(d_segmentation_matrix);
    cudaFree(d_image);
}
