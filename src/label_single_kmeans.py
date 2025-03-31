from random import randint
import os

from networkx.algorithms.reciprocity import overall_reciprocity

from src.kmeans.kmeans import kmeans
from utils.serialize import read_clusters, read_matrix
import numpy as np
import matplotlib.pyplot as plt
from src.labeling.image_load_util import load_single_image
from collections import Counter

def get_error(mega_clusters,gt,slic_matrix):
    all_cluster_labels = []
    mega_clusters_percentage = []
    for idx, mega_cluster in enumerate(mega_clusters):
        ground_truth_labels = []

        # Collect ground truth values for all slic clusters in the mega cluster
        for cluster in mega_cluster:
            mask = (slic_matrix == (cluster.cid + 1))
            labels, counts = np.unique(gt[mask], return_counts=True)
            ground_truth_labels.extend(labels)
        # Histogram of labels within this mega cluster
        label_counts = Counter(ground_truth_labels)
        all_cluster_labels.append(label_counts)
        if not label_counts:
            print(f"Warning: Mega Cluster {idx + 1} has no labels associated with it.")
            mega_clusters_percentage.append(0)  # Default to 0 if no labels found
            continue
        dominant = max(label_counts.values())
        count_sum = sum(label_counts.values())
        percentage = dominant/count_sum
        mega_clusters_percentage.append(percentage)

        # Display the histogram for this mega cluster
        plt.figure()
        plt.bar(label_counts.keys(), label_counts.values())
        plt.title(f"Histogram for Mega Cluster {idx + 1}")
        plt.xlabel("Label")
        plt.ylabel("Frequency")
        plt.show(block=True)
    return mega_clusters_percentage

def get_random_colors(n):
    """Generate a list of n random colors in hexadecimal format."""
    return [f"#{randint(0, 0xFFFFFF):06x}" for _ in range(n)]

def run_single_kmeans(threshold, target_clusters):

    try:

        matrix_file = open("data/slic/full_slic_matrix.bin", "rb")
        cluster_file = open("data/slic/full_slic_cluster.bin", "rb")
        print("Files loaded")
        # Get 10 images
        image_dir = "/Users/bethralston/Desktop/School/School 2025/COSC444/Project.nosync/SemanticSegmentation/src/data/raw/test-images/leftImg8bit/train/aachen/"
        gt_dir = "/Users/bethralston/Desktop/School/School 2025/COSC444/Project.nosync/SemanticSegmentation/src/data/raw/test-images/gtFine/train/aachen/"
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])[:10]  # Pick first 10 images
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith("gtFine_labelIds.png")])[:10]
        # run through all 10 images
        for image_file,gt_file in zip(image_files, gt_files):

            # Load the matrix and cluster
            matrix = read_matrix(matrix_file)
            cluster = read_clusters(cluster_file)
            overall_percentages = []
            # Load image and ground truth
            image_path = os.path.join(image_dir, image_file)
            gt_path = os.path.join(gt_dir, gt_file)
            image = load_single_image(image_path)
            gt = load_single_image(gt_path)

            # Do kmeans on super pixels
            mega_clusters, mega_clusters_lab = kmeans(cluster, threshold, target_clusters)

            # Display results

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
            percentages = get_error(mega_clusters,gt,matrix)
            overall_percentages.append(percentages)

        # Box plot after processing all images
        plt.figure()
        plt.boxplot(overall_percentages)
        plt.title("Dominant Label Percentage Across 10 Images")
        plt.xlabel("Image Index")
        plt.ylabel("Dominant Label Percentage")
        plt.show()
    finally:
        matrix_file.close()
        cluster_file.close()
        print("Files closed")