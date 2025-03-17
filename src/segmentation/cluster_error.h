#ifndef CLUSTER_ERROR_H
#define CLUSTER_ERROR_H

#include "slic.h"

float compute_cluster_error_cuda(Cluster *cluster, Cluster *prev_cluster, int num_clusters);

#endif