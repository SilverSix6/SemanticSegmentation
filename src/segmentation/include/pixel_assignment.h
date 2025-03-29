#ifndef PIXEL_ASSIGNMENT_H
#define PIXEL_ASSIGNMENT_H

#include "slic.h"

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
void assign_pixels_to_clusters_cuda(unsigned char *image, Cluster *cluster_centers, int *segmentation_matrix,
    int width, int height, int num_clusters, float m);

/**
 * Assigns pixels to the closest cluster based on both position and color information. The choosen cluster's id is stored in the pixels location in the segmented_matrix.
 * This function is run entirely on the cpu.
 * 
 * @param image: A pointer the the image that is being converted. The image should be a 1D array with pixel data in row major format.
 * @param cluster_centers: A pointer to an array of clusters. 
 * @param segmented_matrix: Pointer to the output matrix of the same dimentions as the input image. Each pixel is assigned the id of the closest cluster.
 * @param width: The width of the image (number of pixels)
 * @param height: The height of the image (number of pixels)
 * @param num_clusters: The number of clusters 
 * @param m: The compactness factor. Used to balance spatial and color proximity. Higher values enforce spatial uniformity.
 */
void assign_pixels_to_clusters_cpu(unsigned char *image, Cluster *cluster_centers, int *segmented_matrix,
    int width, int height, int num_clusters, float m);

#endif