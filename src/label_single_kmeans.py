from random import randint

from src.kmeans.kmeans import kmeans
from utils.serialize import read_clusters, read_matrix
import numpy as np
import matplotlib.pyplot as plt
from src.labeling.image_load_util import load_single_image
from collections import Counter

def get_error(mega_clusters,slic_clusters,gt,slic_matrix):
    for idx, mega_cluster in enumerate(mega_clusters):
        ground_truth_labels = []

        # Collect ground truth values for all slic clusters in the mega cluster
        for cid in mega_cluster:
            mask = (slic_matrix == (cid + 1))
            labels, counts = np.unique(gt[mask], return_counts=True)
        #     cluster = slic_clusters[cid]
        #     x, y = cluster.x, cluster.y
        #
        #     ground_truth_labels.append(gt[x,y])
        # label_counts = Counter(ground_truth_labels)
    return

def get_random_colors(n):
    """Generate a list of n random colors in hexadecimal format."""
    return [f"#{randint(0, 0xFFFFFF):06x}" for _ in range(n)]

def run_single_kmeans(threshold, target_clusters):

    try:

        matrix_file = open("data/slic/full_slic_matrix.bin", "rb")
        cluster_file = open("data/slic/full_slic_cluster.bin", "rb")
        print("Files loaded")

        # Load the matrix and cluster


        matrix = read_matrix(matrix_file)
        cluster = read_clusters(cluster_file)

        # Do kmeans on super pixels
        mega_clusters, mega_clusters_lab = kmeans(cluster, threshold, target_clusters)

        # Display results
        
        # Original image
        image = load_single_image('data/raw/aachen_000000_000019_leftImg8bit.png')
        # Plot the image
        plt.figure(figsize=(10, 6))
        plt.imshow(image)

        # Overlay clusters
        colors = get_random_colors(len(mega_clusters))  # One color per cluster

        for idx, mega_cluster in enumerate(mega_clusters):
            x_cords = [cluster.x for cluster in mega_cluster]
            y_cords = [cluster.y for cluster in mega_cluster]
            plt.scatter(x_cords, y_cords, color=colors[idx], alpha=0.6, label=f'Cluster {idx + 1}', s=10)


        # Add legend and display
        plt.legend()
        plt.title("K-Means Clustering on Superpixels")
        plt.show(block=True)
        gt = image
        get_error(mega_clusters,cluster,gt,matrix)

    finally:
        matrix_file.close()
        cluster_file.close()
        print("Files closed")