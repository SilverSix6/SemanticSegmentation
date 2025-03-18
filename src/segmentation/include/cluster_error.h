#ifndef CLUSTER_ERROR_H
#define CLUSTER_ERROR_H

#include "slic.h"

/**
 * Computes the difference between the current and previous cluster spatial and color information.
 * 
 * @param h_clusters: A pointer to the current cluster array
 * @param h_prev_clusters: A pointer to the cluster array from the previous iteration
 * @param num_clusters: The number of clusters in both arrays. 
 */
float compute_cluster_error_cuda(Cluster *cluster, Cluster *prev_cluster, int num_clusters);

#endif