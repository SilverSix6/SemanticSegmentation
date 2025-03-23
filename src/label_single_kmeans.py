from utils.serialize import read_clusters, read_matrix
from kmeans.kmeans import kmeans_cluster
import numpy as np
import matplotlib.pyplot as plt
from src.labeling.image_load_util import load_single_image

def run_single_kmeans():
    
    try:
        matrix_file = open("data/slic/full_slic_matrix.bin", "rb")
        cluster_file = open("data/slic/full_slic_cluster.bin", "rb")
        print("Files loaded")
        
        # Load the matrix and cluster
        matrix = read_matrix(matrix_file)
        cluster = read_clusters(cluster_file)

        # Do kmeans on super pixels
        mega_clusters = kmeans_cluster(matrix,cluster)

        # Display results
        # Original image
        image = load_single_image('data/raw/aachen_000000_000019_leftImg8bit.png')
        # Plot the image
        plt.figure(figsize=(10, 6))
        plt.imshow(image)

        # Overlay clusters
        colors = ['red', 'green', 'blue']  # One color per cluster
        for i, kcluster in enumerate(mega_clusters):
            x_coords = [cluster[cid].x for cid in kcluster]
            y_coords = [cluster[cid].y for cid in kcluster]
            plt.scatter(x_coords, y_coords, color=colors[i], alpha=0.6, label=f'Cluster {i + 1}', s=10)

        # Add legend and display
        plt.legend()
        plt.title("K-Means Clustering on Superpixels")
        plt.show(block=True)

    finally:
        matrix_file.close()
        cluster_file.close()
        print("Files closed")