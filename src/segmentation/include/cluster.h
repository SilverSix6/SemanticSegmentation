#ifndef CLUSTER_CENTERS_H
#define CLUSTER_CENTERS_H

#include "slic.h"

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
void compute_cluster_centers_cuda(Cluster *h_clusters, int *h_segmentation_matrix, unsigned char *h_image, int width, int height, int num_clusters, int m);

/**
 * Initializes a grid of clusters over the width of the image. The clusters are spaced evenly based on the image's size and the number of clusters.
 * 
 * @param image: A pointer to the input image. This data should be stored in a 1D array in row-major order. The image is assumed to be in LAB color space.
 * @param width: The width of the image (number of pixels)
 * @param height: The height of the image (number of pixels)
 * @param clusters: A pointer to an array of Clusters. This shoud be preallocated memory the size of num_clusters.
 * @param num_clusters: The size of the clusters array. 
 */
void initialize_cluster_centers(unsigned char *image, int width, int height, Cluster *clusters, int num_superpixels);

/**
 * Copys the cluster data from the current iteration to previous iteration. 
 * 
 * @param clusters: A pointer to an array of clusters
 * @param prev_clusters: A pointer to where the current cluster will be copied to.
 * @param num_clusters: The size of clusters array. clusters and prev_clusters should be the same size.
 */
void copy_cluster(Cluster *clusters, Cluster *prev_clusters, int num_clusters);

#endif