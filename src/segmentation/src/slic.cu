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
 * given image using CUDA functions. The function groups pixel into superpixels based on color and spatial proximity.
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
extern "C" void slic_gpu(unsigned char* image, int width, int height, int num_superpixels, int max_iterations, float m, float threshold, Cluster *clusters, int *segmented_matrix) {
    unsigned char *d_image;
    Cluster *d_clusters, *d_prev_clusters;
    int *d_segmented_matrix;
    
    #ifdef DEBUG
        printf("Debugging enabled. Timing CUDA operations...\n");
    #endif

    
    int deviceCount;
    cudaEvent_t event;
    cudaEvent_t totalEvent;
    cudaEvent_t iterationEvent;
    
    START_TIMER_SIMPLE(totalEvent);

    // Validate Inputs
    if (image == NULL || clusters == NULL || segmented_matrix == NULL || max_iterations <= 0 || m <= 0) {
        printf("Invalid Arguments");
        return;
    }

    // Check for GPU
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CHECK_CUDA_ERROR(err);

    if (deviceCount < 1) {
        printf("No GPU detected");
        return;
    }

    size_t image_size = width * height * 3 * sizeof(unsigned char);
    size_t cluster_size = num_superpixels * sizeof(Cluster);
    size_t matrix_size = width * height * sizeof(int);
    
    // Initialize device memory
    err = cudaMalloc(&d_image, image_size);
    CHECK_CUDA_ERROR(err);
    err = cudaMalloc(&d_clusters, cluster_size);
    CHECK_CUDA_ERROR(err);
    err = cudaMalloc(&d_prev_clusters, cluster_size);
    CHECK_CUDA_ERROR(err);
    err = cudaMalloc(&d_segmented_matrix, matrix_size);
    CHECK_CUDA_ERROR(err);

    // Copy host memory to device
    err = cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);

    // Convert image from RGB to LAB
    START_TIMER(event);
    convert_rgb_to_lab_cuda(d_image, width, height);
    STOP_TIMER(event, "Time for convert_rgb_to_lab_cuda", -1);

    // Initialize clusters
    START_TIMER(event);
    initialize_cluster_centers_cuda(d_image, width, height, d_clusters, num_superpixels);
    STOP_TIMER(event, "Time for initialize_cluster_centers", -1);

    // Copy clusters
    START_TIMER(event);
    copy_cluster_cuda(d_clusters, d_prev_clusters, num_superpixels);
    STOP_TIMER(event, "Time for copy_cluster_cuda (initial)", -1);

    // Assign pixels to clusters
    START_TIMER(event);
    assign_pixels_to_clusters_cuda(d_image, d_clusters, d_segmented_matrix, width, height, num_superpixels, m);
    STOP_TIMER(event, "Time for assign_pixels_to_clusters_cuda (initial)", -1);

    int iterations = 0;
    float error = threshold + 1;  // Ensure at least one iteration

    
    while (iterations < max_iterations && error >= threshold) {
        START_TIMER_SIMPLE(iterationEvent);

        // Compute new cluster centers
        START_TIMER(event);
        compute_cluster_centers_cuda(d_clusters, d_segmented_matrix, d_image, width, height, num_superpixels, m);
        STOP_TIMER(event, "Time for compute_cluster_centers_cuda", iterations);

        // Assign pixels to clusters
        START_TIMER(event);
        assign_pixels_to_clusters_cuda(d_image, d_clusters, d_segmented_matrix, width, height, num_superpixels, m);
        STOP_TIMER(event, "Time for assign_pixels_to_clusters_cuda", iterations);

        // Compute error
        START_TIMER(event);
        error = compute_cluster_error_cuda(d_clusters, d_prev_clusters, num_superpixels);
        STOP_TIMER(event, "Time for compute_cluster_error_cuda", iterations);

        // Copy clusters
        START_TIMER(event);
        copy_cluster_cuda(d_clusters, d_prev_clusters, num_superpixels);
        STOP_TIMER(event, "Time for copy_cluster", iterations);


        STOP_TIMER_SIMPLE(iterationEvent, "Iteration Processing Time");
        iterations++;
    }

    err = cudaMemcpy(segmented_matrix, d_segmented_matrix, matrix_size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err);
    err = cudaMemcpy(clusters, d_clusters, cluster_size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err);

    // Enforce connectivity
    START_TIMER(event);
    enforce_connectivity(segmented_matrix, width, height);
    STOP_TIMER(event, "Time for enforce_connectivity", -1);


    STOP_TIMER_SIMPLE(totalEvent, "Total Processing Time");

    cudaFree(d_image);
    cudaFree(d_clusters);
    cudaFree(d_prev_clusters);
    cudaFree(d_segmented_matrix);
    d_image = NULL;
    d_clusters = NULL;
    d_prev_clusters = NULL;
    d_segmented_matrix = NULL;
}

/**
 * Performs Superpixel segmentation using the Simple Linear Iterative Clustering (SLIC) algorithm on a
 * given image using the CPU. The function groups pixel into superpixels based on color and spatial proximity.
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
extern "C" void slic_cpu(unsigned char* image, int width, int height, int num_superpixels, int max_iterations, float m, float threshold, Cluster *clusters, int *segmented_matrix) {    
    #ifdef DEBUG
        printf("Debugging enabled. Timing CUDA operations...\n");
    #endif
    
    cudaEvent_t event;
    cudaEvent_t totalEvent;
    cudaEvent_t iterationEvent;

    START_TIMER_SIMPLE(totalEvent);

    // Validate Inputs
    if (image == NULL || clusters == NULL || segmented_matrix == NULL || max_iterations <= 0 || m <= 0) {
        printf("Invalid Arguments");
        return;
    }

    size_t cluster_size = num_superpixels * sizeof(Cluster);    
    Cluster *prev_clusters = (Cluster *) malloc(cluster_size);

    // Convert image from RGB to LAB
    START_TIMER(event);
    convert_rgb_to_lab_cpu(image, width, height);
    STOP_TIMER(event, "Time for convert_rgb_to_lab_cuda", -1);

    // Initialize clusters
    START_TIMER(event);
    initialize_cluster_centers_cpu(image, width, height, clusters, num_superpixels);
    STOP_TIMER(event, "Time for initialize_cluster_centers", -1);

    // Copy clusters
    START_TIMER(event);
    copy_cluster_cpu(clusters, prev_clusters, num_superpixels);
    STOP_TIMER(event, "Time for copy_cluster_cuda (initial)", -1);

    // Assign pixels to clusters
    START_TIMER(event);
    assign_pixels_to_clusters_cpu(image, clusters, segmented_matrix, width, height, num_superpixels, m);
    STOP_TIMER(event, "Time for assign_pixels_to_clusters_cuda (initial)", -1);

    int iterations = 0;
    float error = threshold + 1;  // Ensure at least one iteration

    while (iterations < max_iterations && error >= threshold) {
        START_TIMER_SIMPLE(iterationEvent);
        
        // Compute new cluster centers
        START_TIMER(event);
        compute_cluster_centers_cpu(clusters, segmented_matrix, image, width, height, num_superpixels, m);
        STOP_TIMER(event, "Time for compute_cluster_centers_cuda", iterations);

        // Assign pixels to clusters
        START_TIMER(event);
        assign_pixels_to_clusters_cpu(image, clusters, segmented_matrix, width, height, num_superpixels, m);
        STOP_TIMER(event, "Time for assign_pixels_to_clusters_cuda", iterations);

        // Compute error
        START_TIMER(event);
        error = compute_cluster_error_cpu(clusters, prev_clusters, num_superpixels);
        STOP_TIMER(event, "Time for compute_cluster_error_cuda", iterations);

        // Copy clusters
        START_TIMER(event);
        copy_cluster_cpu(clusters, prev_clusters, num_superpixels);
        STOP_TIMER(event, "Time for copy_cluster", iterations);

        STOP_TIMER_SIMPLE(iterationEvent, "Iteration Processing Time");

        iterations++;
    }

    // Enforce connectivity
    START_TIMER(event);
    enforce_connectivity(segmented_matrix, width, height);
    STOP_TIMER(event, "Time for enforce_connectivity", -1);

    STOP_TIMER_SIMPLE(totalEvent, "Total Processing Time");

    free(prev_clusters);
    prev_clusters = NULL;
}