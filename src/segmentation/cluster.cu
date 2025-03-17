#include "cluster.h"
#include <cuda_runtime.h>
#include <stdio.h>

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

__global__ void updateClusters(Cluster *clusters, int num_clusters, int *segmentation_matrix, unsigned char *image, int width, int height) {
    __shared__ Cluster localClusters[MAX_SUPERPIXELS];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_index = y * width + x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (tid < num_clusters) {
        localClusters[tid].x = 0;
        localClusters[tid].y = 0;
        localClusters[tid].l = 0;
        localClusters[tid].a = 0;
        localClusters[tid].b = 0;
        localClusters[tid].n = 0;
    }
    __syncthreads();


    if (x < width && y < height) {
        int cluster_idx = segmentation_matrix[pixel_index];

        if (cluster_idx >= 0 && cluster_idx < num_clusters) {
            atomicAdd(&localClusters[cluster_idx].x, (float)x);
            atomicAdd(&localClusters[cluster_idx].y, (float)y);
            atomicAdd(&localClusters[cluster_idx].l, (float)image[3 * pixel_index]);
            atomicAdd(&localClusters[cluster_idx].a, (float)image[3 * pixel_index + 1]);
            atomicAdd(&localClusters[cluster_idx].b, (float)image[3 * pixel_index + 2]);
            atomicAdd(&localClusters[cluster_idx].n, 1);
        }
    }

    __syncthreads();

    // Merge local clusters into global memory
    if (tid < num_clusters) {
        atomicAdd(&clusters[tid].x, localClusters[tid].x);
        atomicAdd(&clusters[tid].y, localClusters[tid].y);
        atomicAdd(&clusters[tid].l, localClusters[tid].l);
        atomicAdd(&clusters[tid].a, localClusters[tid].a);
        atomicAdd(&clusters[tid].b, localClusters[tid].b);
        atomicAdd(&clusters[tid].n, localClusters[tid].n);
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
void compute_cluster_centers_cuda(Cluster *h_clusters, int *h_segmentation_matrix, unsigned char *h_image, int width, int height, int num_clusters, int m) {
    Cluster *d_clusters;
    int *d_segmentation_matrix;
    unsigned char *d_image;

    // Allocate device memory
    cudaError_t err = cudaMalloc(&d_clusters, num_clusters * sizeof(Cluster));
    CHECK_CUDA_ERROR(err);
    err = cudaMalloc(&d_segmentation_matrix, width * height * sizeof(int));
    CHECK_CUDA_ERROR(err);

    err = cudaMalloc(&d_image, 3 * width * height * sizeof(unsigned char));
    CHECK_CUDA_ERROR(err);


    // Copy data to device
    err = cudaMemcpy(d_clusters, h_clusters, num_clusters * sizeof(Cluster), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    err = cudaMemcpy(d_segmentation_matrix, h_segmentation_matrix, width * height * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    err = cudaMemcpy(d_image, h_image, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);


    // Kernel launch parameters
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Reset clusters
    resetClusters<<<(num_clusters + 255) / 256, 256>>>(d_clusters, num_clusters);
    cudaDeviceSynchronize();

    // Update clusters
    updateClusters<<<gridDim, blockDim>>>(d_clusters, num_clusters, d_segmentation_matrix, d_image, width, height);
    cudaDeviceSynchronize();

    // Normalize clusters
    normalizeClusters<<<(num_clusters + 255) / 256, 256>>>(d_clusters, num_clusters);
    cudaDeviceSynchronize();

    // Copy results back to host
    err = cudaMemcpy(h_clusters, d_clusters, num_clusters * sizeof(Cluster), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err);

    // Free device memory
    cudaFree(d_clusters);
    cudaFree(d_segmentation_matrix);
    cudaFree(d_image);
    d_clusters = NULL;
    d_segmentation_matrix = NULL;
    d_image = NULL;
}

void initialize_cluster_centers(unsigned char *image, int width, int height, Cluster *clusters, int num_superpixels) {
    int index = 0, pixel_index;
    int grid_spacing = (int)sqrt((height * width)/ num_superpixels);

    for (int y = grid_spacing / 2; y < height; y += grid_spacing) {
        for (int x = grid_spacing / 2; x < width; x += grid_spacing) {

            pixel_index = y * width + x;

            clusters[index].l = image[3 * pixel_index];
            clusters[index].a = image[3 * pixel_index + 1];
            clusters[index].b = image[3 * pixel_index + 2];
            clusters[index].x = x;
            clusters[index].y = y;

            index++;
        }
    }
}

void copy_cluster(Cluster *clusters, Cluster *prev_clusters, int num_clusters) {
    memcpy(prev_clusters, clusters, num_clusters * sizeof(Cluster));
}