#ifndef PIXEL_CLUSTER_ASSIGNMENT_H
#define PIXEL_CLUSTER_ASSIGNMENT_H

#include "types.h"

void assign_pixels_to_clusters(unsigned char *image, Cluster *cluster_centers, int *segmentation_matrix,
    int width, int height, int num_clusters, float m);

#endif