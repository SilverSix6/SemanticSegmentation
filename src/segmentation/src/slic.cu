#include <stdio.h>
#include <math.h>

#include "slic.h"
#include "cluster.h"
#include "cluster_error.h"
#include "pixel_assignment.h"
#include "connectivity.h"
#include "rgb_to_lab.h"

/**
 * Performs Superpixel segmentation using the Simple Linear Iterative Clustering (SLIC) algorithm on a
 * given image. The function groups pixel into superpixels based on color and spatial proximity.
 * 
 * @param image: A pointer to the input image. This data should be stored in a 1D array in row-major order. The image is assumed to be RGB.
 * @param width: The images width (number of pixels)
 * @param height: The images height (number of pixels)
 * @param num_superpixels: The total number of superpixels the system will produce. 
 * @param max_iterations: The number of max number iteration the system will process if the error threshold is not first reached. 
 * @param m: The compactness factor. Used to balance spatial and color proximity. Higher values enforce spatial uniformity.
 * @param threshold: The stopping threshold for convergence.
 * @param clusters: Pointer to array of Cluster structs storing cluster centers. Memory should be pre-allocated.
 * @param segmented_matrix: Pointer to the output matrix of the same dimentions as the input image. Each pixel is assigned the id of it's superpixel label.
 */
extern "C" void slic(unsigned char* image, int width, int height, int num_superpixels, int max_iterations, float m, float threshold, Cluster *clusters, int *segmented_matrix) {
    #ifdef DEBUG
        printf("Debugging enabled. Timing CUDA operations...\n");
    #endif
    
    int deviceCount;
    cudaEvent_t event;

    if (image == NULL || clusters == NULL || segmented_matrix == NULL || max_iterations <= 0 || m <= 0) {
        printf("Invalid Arguments");
        return;
    }
    
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CHECK_CUDA_ERROR(err);

    // Convert RGB to LAB
    START_TIMER(event);
    convert_rgb_to_lab_cuda(image, width, height);
    STOP_TIMER(event, "Time for convert_rgb_to_lab_cuda", -1);

    // Initialize clusters
    START_TIMER(event);
    initialize_cluster_centers(image, width, height, clusters, num_superpixels);
    STOP_TIMER(event, "Time for initialize_cluster_centers", -1);

    Cluster *prev_clusters = (Cluster *)malloc(num_superpixels * sizeof(Cluster));

    // Copy clusters
    START_TIMER(event);
    copy_cluster(clusters, prev_clusters, num_superpixels);
    STOP_TIMER(event, "Time for copy_cluster (initial)", -1);

    // Assign pixels to clusters
    START_TIMER(event);
    assign_pixels_to_clusters_cuda(image, clusters, segmented_matrix, width, height, num_superpixels, m);
    STOP_TIMER(event, "Time for assign_pixels_to_clusters_cuda (initial)", -1);

    int iterations = 0;
    float error = threshold + 1;  // Ensure at least one iteration

    while (iterations < max_iterations && error >= threshold) {
        // Compute new cluster centers
        START_TIMER(event);
        compute_cluster_centers_cuda(clusters, segmented_matrix, image, width, height, num_superpixels, m);
        STOP_TIMER(event, "Time for compute_cluster_centers_cuda", iterations);

        // Assign pixels to clusters
        START_TIMER(event);
        assign_pixels_to_clusters_cuda(image, clusters, segmented_matrix, width, height, num_superpixels, m);
        STOP_TIMER(event, "Time for assign_pixels_to_clusters_cuda", iterations);

        // Compute error
        START_TIMER(event);
        error = compute_cluster_error_cuda(clusters, prev_clusters, num_superpixels);
        STOP_TIMER(event, "Time for compute_cluster_error_cuda", iterations);

        // Copy clusters
        START_TIMER(event);
        copy_cluster(clusters, prev_clusters, num_superpixels);
        STOP_TIMER(event, "Time for copy_cluster", iterations);

        iterations++;
    }

    // Enforce connectivity
    START_TIMER(event);
    enforce_connectivity(segmented_matrix, width, height);
    STOP_TIMER(event, "Time for enforce_connectivity", -1);

    free(prev_clusters);
    prev_clusters = NULL;
}