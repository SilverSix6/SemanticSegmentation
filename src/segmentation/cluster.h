#ifndef CLUSTER_CENTERS_H
#define CLUSTER_CENTERS_H

#include "slic.h"

void compute_cluster_centers_cuda(Cluster *h_clusters, int *h_segmentation_matrix, unsigned char *h_image, int width, int height, int num_clusters, int m);
void initialize_cluster_centers(unsigned char *image, int width, int height, Cluster *clusters, int num_superpixels);
void copy_cluster(Cluster *clusters, Cluster *prev_clusters, int num_clusters);

#endif