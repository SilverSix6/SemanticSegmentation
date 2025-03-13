#ifndef CLUSTER_CENTERS_H
#define CLUSTER_CENTERS_H

#include "types.h"

void compute_cluster_centers_cuda(Cluster *h_clusters, int *h_segmentation_matrix, unsigned char *h_image, int width, int height, int num_clusters, int m);

#endif