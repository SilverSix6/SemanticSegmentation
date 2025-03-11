import ctypes
from dataclasses import dataclass

from skimage import color
import numpy as np


# Define the Cluster structure (with x, y, l, a, b as floats and n as int)
class Cluster(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("l", ctypes.c_float),
        ("a", ctypes.c_float),
        ("b", ctypes.c_float),
        ("n", ctypes.c_int)
    ]

@dataclass
class PyCluster:
    x: float
    y: float
    l: float
    a: float
    b: float
    n: int
    cid: int

def convert_cluster(cluster, cid) -> PyCluster:
    return PyCluster(
        x=cluster.x,
        y=cluster.y,
        l=cluster.l,
        a=cluster.a,
        b=cluster.b,
        n=cluster.n,
        cid = cid
    )

def slic(image, num_superpixels, m, max_iterations, threshold):
    """
    Performs SLIC superpixel segmentation.

    :param image: Input image (numpy array, shape: [height, width, 3], dtype: uint8).
    :param num_superpixels: Desired number of superpixels.
    :param m: Compactness parameter balancing color and spatial distance.
    :param max_iterations: Maximum number of iterations.
    :param threshold: Convergence threshold.
    :return: segmentation_matrix (height x width array of cluster assignments),
             cluster_centers (list of Cluster objects).
    """
    # Ensure the image is contiguous and of the right type.

    _, _, channels = image.shape
    if channels != 3:
        raise ValueError("Image must have 3 channels (RGB).")

    lab_image = color.rgb2lab(image)
    lab_image = np.ascontiguousarray(lab_image, dtype=np.uint8)
    height, width, channels = lab_image.shape

    # Flatten the image so it can be passed as a 1D array.
    image_flat = lab_image.flatten()
    image_ptr = image_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    # Load the shared library (adjust the path if necessary)
    libslic = ctypes.CDLL('./segmentation/libslic.so')

    # Define the function signature for slic:
    # void slic(unsigned char* image, int width, int height, int num_superpixels,
    #           int max_iterations, float m, float threshold, Cluster *clusters, int *segmented_matrix)
    libslic.slic.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),  # image pointer
        ctypes.c_int,  # width
        ctypes.c_int,  # height
        ctypes.c_int,  # num_superpixels
        ctypes.c_int,  # max_iterations
        ctypes.c_float,  # m
        ctypes.c_float,  # threshold
        ctypes.POINTER(Cluster),  # clusters pointer
        ctypes.POINTER(ctypes.c_int)  # segmented_matrix pointer
    ]
    libslic.slic.restype = None

    # Allocate the clusters array (size: num_superpixels)
    clusters_array = (Cluster * num_superpixels)()

    # Allocate the segmented matrix (size: width * height)
    segmented_matrix_array = (ctypes.c_int * (width * height))()

    # Call the slic function from the shared library
    libslic.slic(
        image_ptr,
        width,
        height,
        num_superpixels,
        max_iterations,
        ctypes.c_float(m),
        ctypes.c_float(threshold),
        clusters_array,
        segmented_matrix_array
    )

    # Convert the segmented matrix back to a NumPy array and reshape it to (height, width)
    segmentation_matrix = np.ctypeslib.as_array(segmented_matrix_array)
    segmentation_matrix = segmentation_matrix.reshape((height, width))

    # Create a list of Cluster objects from the clusters_array.
    cluster_centers = [convert_cluster(clusters_array[i], i) for i in range(num_superpixels)]

    return segmentation_matrix, cluster_centers