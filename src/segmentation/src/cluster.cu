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

__global__ void updateClusters(Cluster *clusters, int num_clusters, int *segmented_matrix, unsigned char *image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_index = y * width + x;

    if (x < width && y < height) {
        int cluster_idx = segmented_matrix[pixel_index];

        if (cluster_idx >= 0 && cluster_idx < num_clusters) {
            atomicAdd(&clusters[cluster_idx].x, (float)x);
            atomicAdd(&clusters[cluster_idx].y, (float)y);
            atomicAdd(&clusters[cluster_idx].l, (float)image[3 * pixel_index]);
            atomicAdd(&clusters[cluster_idx].a, (float)image[3 * pixel_index + 1]);
            atomicAdd(&clusters[cluster_idx].b, (float)image[3 * pixel_index + 2]);
            atomicAdd(&clusters[cluster_idx].n, 1);
        }
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

/**
 * Computes a cluster's average spatial and color information based on it's pixels. 
 * 
 * @param h_clusters: A pointer to an array of clusters
 * @param h_segmented_matrix: A pointer to each pixel's cluster id. 
 * @param h_image: A pointer to the input image. This data should be stored in a 1D array in row-major order. The image is assumed to be in LAB color space.
 * @param width: The width of the image (number of pixels)
 * @param height: The height of the image (number of pixels)
 * @param num_clusters: The number of clusters in the h_clusters array
 * @param m: The compactness factor. Used to balance spatial and color proximity. Higher values enforce spatial uniformity.
 */
void compute_cluster_centers_cuda(Cluster *d_clusters, int *d_segmented_matrix, unsigned char *d_image, int width, int height, int num_clusters, int m) {

    // Kernel launch parameters
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Reset clusters
    resetClusters<<<(num_clusters + 255) / 256, 256>>>(d_clusters, num_clusters);
    cudaDeviceSynchronize();

    // Update clusters
    updateClusters<<<gridDim, blockDim, num_clusters * sizeof(Cluster)>>>(d_clusters, num_clusters, d_segmented_matrix, d_image, width, height);
    cudaDeviceSynchronize();

    // Normalize clusters
    normalizeClusters<<<(num_clusters + 255) / 256, 256>>>(d_clusters, num_clusters);
    cudaDeviceSynchronize();

}

/**
 * Computes a cluster's average spatial and color information based on it's pixels. 
 * 
 * @param h_clusters: A pointer to an array of clusters
 * @param h_segmented_matrix: A pointer to each pixel's cluster id. 
 * @param h_image: A pointer to the input image. This data should be stored in a 1D array in row-major order. The image is assumed to be in LAB color space.
 * @param width: The width of the image (number of pixels)
 * @param height: The height of the image (number of pixels)
 * @param num_clusters: The number of clusters in the h_clusters array
 * @param m: The compactness factor. Used to balance spatial and color proximity. Higher values enforce spatial uniformity.
 */
void compute_cluster_centers_cpu(Cluster *clusters, int *segmented_matrix, unsigned char *image, int width, int height, int num_clusters, int m) {

    // Reset cluster
    for (int i = 0; i < num_clusters; i++) {
        clusters[i].x = 0;
        clusters[i].y = 0;
        clusters[i].l = 0;
        clusters[i].a = 0;
        clusters[i].b = 0;
        clusters[i].n = 0;
    }

    // Update clusters with pixel values
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel_index = y * width + x;
            int cluster_idx = segmented_matrix[pixel_index];

            clusters[cluster_idx].x += x;
            clusters[cluster_idx].y += y;
            clusters[cluster_idx].l += image[3 * pixel_index];
            clusters[cluster_idx].a += image[3 * pixel_index + 1];
            clusters[cluster_idx].b += image[3 * pixel_index + 2];
            clusters[cluster_idx].n++;
        }
    }

    // Normalize Clusters
    for (int i = 0; i < num_clusters; i++) {
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

__global__ void initialize_cluster_centers(unsigned char *image, int width, int height, Cluster *clusters, int num_clusters) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_clusters) return;

    int grid_spacing = (int)sqrtf((float)(height * width) / num_clusters);
    int grid_x = index % (width / grid_spacing);
    int grid_y = index / (width / grid_spacing);

    int x = grid_spacing / 2 + grid_x * grid_spacing;
    int y = grid_spacing / 2 + grid_y * grid_spacing;
    
    if (x >= width || y >= height) return;
    
    int pixel_index = (y * width + x) * 3; // LAB is stored as 3 bytes per pixel

    clusters[index].l = image[pixel_index];
    clusters[index].a = image[pixel_index + 1];
    clusters[index].b = image[pixel_index + 2];
    clusters[index].x = x;
    clusters[index].y = y;
}

/**
 * Initializes a grid of clusters over the width of the image. The clusters are spaced evenly based on the image's size and the number of clusters.
 * 
 * @param image: A pointer to the input image. This data should be stored in a 1D array in row-major order. The image is assumed to be in LAB color space.
 * @param width: The width of the image (number of pixels)
 * @param height: The height of the image (number of pixels)
 * @param clusters: A pointer to an array of Clusters. This shoud be preallocated memory the size of num_clusters.
 * @param num_clusters: The size of the clusters array. 
 */
void initialize_cluster_centers_cuda(unsigned char *d_image, int width, int height, Cluster *d_clusters, int num_clusters) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_clusters + threadsPerBlock - 1) / threadsPerBlock;
    initialize_cluster_centers<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height, d_clusters, num_clusters);
    cudaDeviceSynchronize(); // Ensure the kernel completes before returning
}

/**
 * Initializes a grid of clusters over the width of the image. The clusters are spaced evenly based on the image's size and the number of clusters.
 * 
 * @param image: A pointer to the input image. This data should be stored in a 1D array in row-major order. The image is assumed to be in LAB color space.
 * @param width: The width of the image (number of pixels)
 * @param height: The height of the image (number of pixels)
 * @param clusters: A pointer to an array of Clusters. This shoud be preallocated memory the size of num_clusters.
 * @param num_clusters: The size of the clusters array. 
 */
void initialize_cluster_centers_cpu(unsigned char *image, int width, int height, Cluster *clusters, int num_clusters) {
    if (!image || !clusters) return; // Prevent null pointer access

    int grid_spacing = (int)sqrtf((float)(height * width) / num_clusters);
    if (grid_spacing == 0) grid_spacing = 1; // Prevent division by zero

    int grid_size_x = width / grid_spacing;
    if (grid_size_x == 0) grid_size_x = 1; // Avoid division by zero

    for (int index = 0; index < num_clusters; index++) {
        int grid_x = index % grid_size_x;
        int grid_y = index / grid_size_x;

        int x = grid_spacing / 2 + grid_x * grid_spacing;
        int y = grid_spacing / 2 + grid_y * grid_spacing;
        
        if (x >= width || y >= height) continue;
        
        int pixel_index = (y * width + x) * 3; // LAB is stored as 3 bytes per pixel
    
        clusters[index].l = image[pixel_index];
        clusters[index].a = image[pixel_index + 1];
        clusters[index].b = image[pixel_index + 2];
        clusters[index].x = x;
        clusters[index].y = y;
    }
}

/**
 * Copys the cluster data from one array to another array. 
 * 
 * @param clusters: A pointer to an array of clusters
 * @param prev_clusters: A pointer to where the current cluster will be copied to.
 * @param num_clusters: The size of clusters array. clusters and prev_clusters should be the same size.
 */
void copy_cluster_cuda(Cluster *d_clusters, Cluster *d_prev_clusters, int num_clusters) {
    cudaMemcpy(d_prev_clusters, d_clusters, num_clusters * sizeof(Cluster), cudaMemcpyDeviceToDevice);
}

/**
 * Copys the cluster data from one array to another array. 
 * 
 * @param clusters: A pointer to an array of clusters
 * @param prev_clusters: A pointer to where the current cluster will be copied to.
 * @param num_clusters: The size of clusters array. clusters and prev_clusters should be the same size.
 */
void copy_cluster_cpu(Cluster *clusters, Cluster *prev_clusters, int num_clusters) {
    memcpy(prev_clusters, clusters, num_clusters * sizeof(Cluster));
}