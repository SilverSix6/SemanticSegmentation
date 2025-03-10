import copy
import math
import cv2
import numpy as np

class Cluster:
    def __init__(self, cid, x, y):
        self.cid = cid
        self.x = x
        self.y = y
        self.l = 0
        self.a = 0
        self.b = 0
        self.n = 0

    def reset(self):
        self.x = self.y = self.l = self.a = self.b = self.n = 0

def slic(image, num_superpixels, m, max_iterations, threshold):
    """
    Performs SLIC superpixel segmentation.

    :param image: Input image (numpy array).
    :param num_superpixels: Desired number of superpixels.
    :param m: Compactness parameter balancing color and spatial distance.
    :param max_iterations: Maximum number of iterations.
    :param threshold: Convergence threshold.
    :return: segmentation_matrix (NxM cluster assignments), cluster_centers (list of Cluster objects).
    """
    iterations = 0
    error = float('inf')
    height, width, _ = image.shape

    # Use NumPy for efficient array operations
    segmentation_matrix = np.zeros((height, width), dtype=np.int32)

    # Convert to Lab color space for better perceptual uniformity
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    # Initialize clusters
    print('Started Initialization')
    cluster_centers = initialize_cluster_centers(lab_image, num_superpixels)
    prev_cluster_centers = copy.deepcopy(cluster_centers)

    assign_pixels_to_clusters(lab_image, cluster_centers, segmentation_matrix, m)

    # Iterate until convergence
    while iterations < max_iterations and error >= threshold:
        print(f'Started iteration: {iterations}')
        compute_cluster_centers(lab_image, segmentation_matrix, cluster_centers)
        assign_pixels_to_clusters(lab_image, cluster_centers, segmentation_matrix, m)

        # Compute error and update previous cluster centers
        error = compute_cluster_error(cluster_centers, prev_cluster_centers)
        prev_cluster_centers = copy.deepcopy(cluster_centers)

        iterations += 1

    return segmentation_matrix, cluster_centers


def initialize_cluster_centers(image, num_superpixels):
    """
    Initializes cluster centers using a regular grid pattern.

    :param image: The input image in Lab color space.
    :param num_superpixels: Desired number of superpixels.
    :return: List of initialized Cluster objects.
    """
    height, width, _ = image.shape
    grid_spacing = int(np.sqrt((height * width) / num_superpixels))

    cluster_centers = []
    cid = 0
    for y in range(grid_spacing // 2, height, grid_spacing):
        for x in range(grid_spacing // 2, width, grid_spacing):
            if x < width and y < height:
                cluster = Cluster(cid, x, y)
                cluster.l, cluster.a, cluster.b = image[y, x]
                cluster_centers.append(cluster)
                cid += 1

    return cluster_centers


def compute_cluster_centers(image, segmentation_matrix, cluster_centers):
    """
    Updates cluster centers based on assigned pixels.

    :param image: The Lab color space image.
    :param segmentation_matrix: Cluster assignments for each pixel.
    :param cluster_centers: List of Cluster objects.
    """
    # Reset clusters before updating
    for cluster in cluster_centers:
        cluster.reset()

    height, width = segmentation_matrix.shape

    # Update clusters using pixel values
    for y in range(height):
        for x in range(width):
            cluster = cluster_centers[segmentation_matrix[y, x]]
            l, a, b = image[y, x]
            cluster.x += x
            cluster.y += y
            cluster.l += float(l)
            cluster.a += float(a)
            cluster.b += float(b)
            cluster.n += 1

    # Normalize cluster values to compute new centers
    for cluster in cluster_centers:
        if cluster.n > 0:  # Avoid division by zero
            cluster.x /= cluster.n
            cluster.y /= cluster.n
            cluster.l /= cluster.n
            cluster.a /= cluster.n
            cluster.b /= cluster.n


def compute_cluster_error(cluster_centers, prev_cluster_centers):
    """
    Computes the total movement of cluster centers between iterations.

    :param cluster_centers: Current cluster centers.
    :param prev_cluster_centers: Previous iteration's cluster centers.
    :return: Total error value.
    """
    return sum(distance_between_clusters(c1, c2) for c1, c2 in zip(cluster_centers, prev_cluster_centers))


def assign_pixels_to_clusters(image, cluster_centers, segmentation_matrix, m):
    """
    Assigns each pixel to the nearest cluster.

    :param image: Lab color space image.
    :param cluster_centers: List of Cluster objects.
    :param segmentation_matrix: NxM matrix storing cluster IDs for pixels.
    :param m: Compactness parameter.
    """
    height, width, _ = image.shape
    cluster_positions = np.array([(cluster.x, cluster.y) for cluster in cluster_centers])
    cluster_colors = np.array([(cluster.l, cluster.a, cluster.b) for cluster in cluster_centers])

    # Create meshgrid for x, y coordinates
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the grids for vectorized distance calculation
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # Flatten image for faster processing
    image_flat = image.reshape((-1, 3))  # Reshape to (num_pixels, 3)

    # Calculate distances in vectorized form
    distances = np.zeros((len(cluster_centers), len(x_flat)))  # (num_clusters, num_pixels)

    # Compute color and spatial distances
    for i, (l, a, b) in enumerate(cluster_colors):
        color_dist = np.sum((image_flat - np.array([l, a, b])) ** 2, axis=1)
        spatial_dist = (cluster_positions[i, 0] - x_flat) ** 2 + (cluster_positions[i, 1] - y_flat) ** 2
        distances[i, :] = np.sqrt(color_dist + m * spatial_dist)

    # Find the cluster with the minimum distance for each pixel
    segmentation_matrix.flat[:] = np.argmin(distances, axis=0)  # Assign the cluster with the minimum distance


def distance_between_clusters(cluster1, cluster2):
    """
    Computes Euclidean distance between two cluster centers.

    :param cluster1: First cluster.
    :param cluster2: Second cluster.
    :return: Distance value.
    """
    return math.sqrt((cluster1.l - cluster2.l) ** 2 +
                     (cluster1.a - cluster2.a) ** 2 +
                     (cluster1.b - cluster2.b) ** 2 +
                     (cluster1.x - cluster2.x) ** 2 +
                     (cluster1.y - cluster2.y) ** 2)


def distance_function(image, cluster, px, py, m):
    """
    Computes the distance between a pixel and a cluster center.

    :param image: The image in Lab color space.
    :param cluster: The cluster object.
    :param px: Pixel x-coordinate.
    :param py: Pixel y-coordinate.
    :param m: Compactness parameter.
    :return: Computed distance.
    """
    # Color distance in Lab space
    l, a, b = image[px, py]
    color_dist = (l - cluster.l) ** 2 + (a - cluster.a) ** 2 + (b - cluster.b) ** 2

    # Spatial distance (Euclidean)
    spatial_dist = (cluster.x - px) ** 2 + (cluster.y - py) ** 2

    # Total distance with compactness factor
    return np.sqrt(color_dist + m * spatial_dist)

