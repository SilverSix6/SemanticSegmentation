#ifndef PIXEL_ASSIGNMENT_H
#define PIXEL_ASSIGNMENT_H

#include "slic.h"

void assign_pixels_to_clusters_cuda(unsigned char *image, Cluster *cluster_centers, int *segmentation_matrix,
    int width, int height, int num_clusters, float m);

#endif