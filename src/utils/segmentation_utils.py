import numpy as np
import cv2

def segmented_to_color(segmented_matrix, cluster_centers):
    """
    Convert a segmented matrix into a color image by assigning unique colors to each cluster ID.

    :param cluster_centers:
    :param segmented_matrix: np.ndarray of shape (m, n) containing cluster IDs
    :return: np.ndarray of shape (m, n, 3) representing a color image
    """
    num_labels = len(cluster_centers)

    # Generate random colors for each cluster
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)

    # Create a color image
    color_image = np.zeros((*segmented_matrix.shape, 3), dtype=np.uint8)

    # Map each cluster ID to its corresponding color
    for cluster in cluster_centers:
        color_image[segmented_matrix == cluster.cid] = colors[cluster.cid]

    return color_image


def segmented_to_average_color(segmented_matrix, cluster_centers):
    """
    Convert a segmented matrix into a color image by assigning the average Lab color to each cluster ID.

    :param segmented_matrix: np.ndarray of shape (m, n) containing cluster IDs
    :param cluster_centers: List of clusters, each having attributes l, a, and b for Lab color values
    :return: np.ndarray of shape (m, n, 3) representing a color image in BGR color space
    """
    # Create a color image in Lab space
    color_image_lab = np.zeros((*segmented_matrix.shape, 3), dtype=np.uint8)

    # Assign the average Lab color to each cluster ID
    for cluster in cluster_centers:
        color_image_lab[segmented_matrix == cluster.cid] = [cluster.l, cluster.a, cluster.b]

    # Convert from Lab to BGR for display
    color_image_lab = np.clip(color_image_lab, 0, 255).astype(np.uint8)

    return color_image_lab


def average_images(image1, image2):
    """
    Averages two images.

    :param image1: First image (NumPy array).
    :param image2: Second image (NumPy array).
    :return: Averaged image (NumPy array).
    """
    # Ensure both images have the same size and type
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")

    # Convert the images to float32 to avoid overflow/underflow
    image1_float = image1.astype(np.float32)
    image2_float = image2.astype(np.float32)

    # Compute the average pixel values
    averaged_image = (image1_float + image2_float) / 2.0

    # Clip values to ensure they are within the valid range [0, 255] and convert back to uint8
    averaged_image = np.clip(averaged_image, 0, 255).astype(np.uint8)

    return averaged_image
