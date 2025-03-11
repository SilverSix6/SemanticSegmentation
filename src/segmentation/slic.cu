#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "types.h"
#include "clusterCenters.h"
#include "pixelClusterAssignment.h"
#include "connectivity.h"

void copy_cluster(Cluster *clusters, Cluster *prev_clusters, int num_clusters);
float compute_cluster_distance(Cluster *cluster, Cluster *prev_cluster);
float compute_cluster_error(Cluster *cluster, Cluster *prev_cluster, int num_clusters);
void initialize_cluster_centers(unsigned char *image, int width, int height, Cluster *clusters, int num_superpixels);

/**
*
*
*
 */
extern "C" void slic(unsigned char* image, int width, int height, int num_superpixels, int max_iterations, float m, float threshold, Cluster *clusters, int *segmented_matrix) {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("CUDA not available: %s",cudaGetErrorString(err));
        return;
    }

    int iterations = 0;
    float error = INFINITY;

    initialize_cluster_centers(image,width, height, clusters, num_superpixels);
    Cluster *prev_clusters = (Cluster *) malloc(num_superpixels * sizeof(Cluster));
    copy_cluster(clusters, prev_clusters, num_superpixels);

    assign_pixels_to_clusters(image, clusters, segmented_matrix, width, height, num_superpixels, m);

    // Iterate until convergence
    while (iterations < max_iterations && error >= threshold) {
        // Compute new cluster centers and assign pixels to each cluster
        compute_cluster_centers(clusters, segmented_matrix, image, width, height, num_superpixels, m);
        assign_pixels_to_clusters(image, clusters, segmented_matrix, width, height, num_superpixels, m);

        error = compute_cluster_error(clusters, prev_clusters, num_superpixels);
        copy_cluster(clusters, prev_clusters, num_superpixels);

        iterations++;
    }

    enforce_connectivity(segmented_matrix, width, height);

    free(prev_clusters);
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

float compute_cluster_error(Cluster *cluster, Cluster *prev_cluster, int num_clusters) {
    float sum = 0;

    for (int i = 0; i < num_clusters; i++) {
        sum += compute_cluster_distance(&cluster[i], &prev_cluster[i]);
    }

    return sum;
}

float compute_cluster_distance(Cluster *cluster, Cluster *prev_cluster) {
    return sqrt((cluster->x - prev_cluster->x) * (cluster->x - prev_cluster->x) +
                (cluster->y - prev_cluster->y) * (cluster->y - prev_cluster->y) +
                (cluster->l - prev_cluster->l) * (cluster->l - prev_cluster->l) +
                (cluster->a - prev_cluster->a) * (cluster->a - prev_cluster->a) +
                (cluster->b - prev_cluster->b) * (cluster->b - prev_cluster->b));
}

void copy_cluster(Cluster *clusters, Cluster *prev_clusters, int num_clusters) {
    for (int i = 0; i < num_clusters; i++){
        prev_clusters[i].l = clusters[i].l;
        prev_clusters[i].a = clusters[i].a;
        prev_clusters[i].b = clusters[i].b;
        prev_clusters[i].x = clusters[i].x;
        prev_clusters[i].y = clusters[i].y;
        prev_clusters[i].n = clusters[i].n;
    }
}