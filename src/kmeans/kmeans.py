# Takes in the clusters of superpixels from the slic algorithm 
# performs k-means clustering based on the cluster's average lab color
# returns the mega clusters
from xmlrpc.client import MAXINT


def kmeans(slic_clusters, threshold, target_clusters):
    """
    Performs k-means clustering on the slic clusters
    :param target_clusters:
    :param threshold:
    :param slic_clusters: matrix of slic clusters and their information
    :return: matrix of mega clusters that contain the cluster ids of the slic clusters
    """

    cluster_groups = [[] for _ in range(target_clusters)]

    # Randomly assign cluster to cluster groups
    slic_cluster_size = len(slic_clusters)
    for i in range(slic_cluster_size):
        cluster_groups[i % target_clusters].append(slic_clusters[i])

    old_centers_lab = []

    # assign values outside of the range for the first iteration of old centers to ensure a second loop
    for i in range(target_clusters):
        old_centers_lab.append([-1000, -1000, -1000])

    # continue until convergence
    while True:

        centers_lab = []
        # calculate the new average lab color of each cluster group
        for i in range(target_clusters):
            centers_lab.append(center(cluster_groups[i], slic_clusters))
            cluster_groups[i].clear()

        # loop over each slic cluster find the closest cluster group
        for j in range(slic_cluster_size):
            # If closest assign slic cluster to the cluster group
            id = closestClusterID(slic_clusters[j], centers_lab)
            cluster_groups[id].append(slic_clusters[j])

        # Check for convergence
        if converged(old_centers_lab, centers_lab, threshold):
            return cluster_groups, centers_lab

        # Update old centers
        old_centers_lab = centers_lab.copy()

# calculates center of the cluster's average lab value
def center(cluster,slic_clusters):
    """
    Calculates the center of the cluster's average lab value
    :param cluster: list of cluster ids
    :param slic_clusters: matrix of slic clusters and their information
    :return: list of average lab values
    """
    sum_l = 0
    sum_a = 0
    sum_b = 0
    for i in range(len(cluster)):
        sum_l += slic_clusters[cluster[i].cid].l
        sum_a += slic_clusters[cluster[i].cid].a
        sum_b += slic_clusters[cluster[i].cid].b
    if len(cluster) == 0:
        return [0,0,0]
    else:
        return [sum_l/len(cluster),sum_a/len(cluster),sum_b/len(cluster)]


def closestClusterID(slic_cluster, centers_lab):

    min_diff_idx = -1
    min_diff = float("inf")

    for i in range(len(centers_lab)):
        diff = (slic_cluster.l - centers_lab[i][0])**2 + (slic_cluster.a - centers_lab[i][1])**2 + (slic_cluster.b - centers_lab[i][2])**2

        if diff < min_diff:
            min_diff = diff
            min_diff_idx = i

    return min_diff_idx


def converged(old_centers, centers_lab, threshold):

    sum = 0

    for i in range(len(centers_lab)):
        sum += (old_centers[i][0] - centers_lab[i][0])**2 + (old_centers[i][1] - centers_lab[i][1])**2 + (old_centers[i][2] - centers_lab[i][2])**2

    return sum < threshold