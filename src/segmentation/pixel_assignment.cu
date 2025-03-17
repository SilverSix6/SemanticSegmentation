#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "pixel_assignment.h"

// CUDA kernel for computing distances and assigning pixels to clusters
__global__ void assign_pixels_kernel(unsigned char *image, Cluster *clusters, int *segmented_matrix,
                                     int width, int height, int num_clusters, float m, int grid_spacing)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int pixel_index = y * width + x;
    float min_distance = INFINITY;
    int best_cluster = -1;

    // Get pixel value
    float l = image[3 * pixel_index];
    float a = image[3 * pixel_index + 1];
    float b = image[3 * pixel_index + 2];

    // Search within a 2S x 2S region around the pixel
    int search_radius = 2 * grid_spacing;
    int min_x = max(0, x - search_radius);
    int max_x = min(width - 1, x + search_radius);
    int min_y = max(0, y - search_radius);
    int max_y = min(height - 1, y + search_radius);

    // Check each cluster
    for (int i = 0; i < num_clusters; i++)
    {
        // Check if the cluster center is within the search region
        if (clusters[i].x >= min_x && clusters[i].x <= max_x && clusters[i].y >= min_y && clusters[i].y <= max_y)
        {   
            // Find and update closest cluster
            float dx = x - clusters[i].x;
            float dy = y - clusters[i].y;
            float spatial_distance = dx * dx + dy * dy;
            
            float dl = l - clusters[i].l;
            float da = a - clusters[i].a;
            float db = b - clusters[i].b;
            float color_distance = dl * dl + da * da + db * db;

            float distance = sqrt((color_distance / m) * (color_distance / m) +
                                  (spatial_distance / grid_spacing) * (spatial_distance / grid_spacing));


            if (distance < min_distance)
            {
                min_distance = distance;
                best_cluster = i;
            }
        }
    }

    segmented_matrix[pixel_index] = best_cluster;
}

/**
 * Assigns pixels to the closest cluster based on both position and color information. The choosen cluster's id is stored in the pixels location in the segmented_matrix 
 * 
 * @param h_image: A pointer the the image that is being converted. The image should be a 1D array with pixel data in row major format.
 * @param cluster_centers: A pointer to an array of clusters. 
 * @param segmented_matrix: Pointer to the output matrix of the same dimentions as the input image. Each pixel is assigned the id of the closest cluster.
 * @param width: The width of the image (number of pixels)
 * @param height: The height of the image (number of pixels)
 * @param num_clusters: The number of clusters 
 * @param m: The compactness factor. Used to balance spatial and color proximity. Higher values enforce spatial uniformity.
 */
void assign_pixels_to_clusters_cuda(unsigned char *image, Cluster *cluster_centers, int *segmented_matrix,
                               int width, int height, int num_clusters, float m)
{
    unsigned char *d_image;
    Cluster *d_clusters;
    int *d_segmentation_matrix;

    int grid_spacing = (int)sqrt((width * height) / num_clusters);
    if (grid_spacing == 0 || m == 0 || num_clusters == 0) {
        printf("Invalid inputs in assign_pixel_to_cluster");
        exit(-1);
    }

    size_t image_size = width * height * 3 * sizeof(unsigned char);
    size_t cluster_size = num_clusters * sizeof(Cluster);
    size_t matrix_size = width * height * sizeof(int);

    // Allocate memory on device
    cudaError_t err = cudaMalloc(&d_image, image_size);
    CHECK_CUDA_ERROR(err);
    err = cudaMalloc(&d_clusters, cluster_size);
    CHECK_CUDA_ERROR(err);
    err = cudaMalloc(&d_segmentation_matrix, matrix_size);
    CHECK_CUDA_ERROR(err);

    // Copy image and current cluster to the device
    err = cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);
    err = cudaMemcpy(d_clusters, cluster_centers, cluster_size, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Assign pixels to the nearest cluster
    assign_pixels_kernel<<<gridSize, blockSize>>>(d_image, d_clusters, d_segmentation_matrix,
                                                  width, height, num_clusters, m, grid_spacing);
    cudaDeviceSynchronize();

    // Copy result to ouput buffer
    err = cudaMemcpy(segmented_matrix, d_segmentation_matrix, matrix_size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err);
    
    // Clean up
    cudaFree(d_image);
    cudaFree(d_clusters);
    cudaFree(d_segmentation_matrix);
    d_image = NULL;
    d_clusters = NULL;
    d_segmentation_matrix = NULL;
}