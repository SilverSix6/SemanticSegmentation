#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "types.h"
#include "cluster_centers.h"
#include "pixel_assignment.h"
#include "connectivity.h"
#include "rgb_to_lab.h"
#include "cluster_error.h"

// Define DEBUG to enable debugging, or comment it out to disable
// #define DEBUG

#ifdef DEBUG
#define START_TIMER(event) cudaEventCreate(&event); cudaEventRecord(event)
#define STOP_TIMER(event, label, iteration)                           \
    {                                                                 \
        cudaEvent_t stopEvent;                                        \
        cudaEventCreate(&stopEvent);                                  \
        cudaEventRecord(stopEvent);                                   \
        cudaEventSynchronize(stopEvent);                              \
        float milliseconds = 0;                                       \
        cudaEventElapsedTime(&milliseconds, event, stopEvent);        \
        if (iteration >= 0)                                           \
            printf("%s (Iteration %d): %.3f ms\n", label, iteration, milliseconds); \
        else                                                          \
            printf("%s: %.3f ms\n", label, milliseconds);             \
        cudaEventDestroy(stopEvent);                                  \
    }
#else
#define START_TIMER(event)
#define STOP_TIMER(event, label, iteration)
#endif

void copy_cluster(Cluster *clusters, Cluster *prev_clusters, int num_clusters);
float compute_cluster_distance(Cluster *cluster, Cluster *prev_cluster);
float compute_cluster_error(Cluster *cluster, Cluster *prev_cluster, int num_clusters);
void initialize_cluster_centers(unsigned char *image, int width, int height, Cluster *clusters, int num_superpixels);

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