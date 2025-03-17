import time

from matplotlib import pyplot as plt
from segmentation.slic import slic
from src.labeling.image_load_util import load_single_image
from utils.segmentation_utils import segmented_to_color, average_images



def run_single_slic(image_path, num_superpixels, m, max_iterations, threshold):

    image = load_single_image(image_path)

    # Perform slic on image
    start_time = time.perf_counter()
    segmented_matrix, cluster_centers = slic(image, num_superpixels,  m, max_iterations, threshold)
    end_time = time.perf_counter()

    print(f'Test Results: {end_time - start_time} seconds')

    # Display Input Image
    plt.imshow(image)
    plt.show()

    # Display Segmented image with color
    color_segmented_matrix = segmented_to_color(segmented_matrix, cluster_centers)
    plt.imshow(color_segmented_matrix)
    plt.show()

    # Display combination of input image and color image
    combined_image = average_images(image, color_segmented_matrix)
    plt.imshow(combined_image)
    plt.show()

    image.close()
    return
