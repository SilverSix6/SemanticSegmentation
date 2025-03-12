#
# File to test SLIC. Runs segmentation on files in the test image array.
#

import time

import cv2
import numpy as np

from labeling.image_load_util import load_list_image
from segmentation.slic import slic
from utils.segmentation_utils import segmented_to_color, average_images


def run_slic_segment():
    print("Running Segmentation Using SLIC:")

    test_images = ['data/raw/test-images/standard_test_images/fruits.png']

    images = load_list_image(test_images, False)

    # Display each image
    for image in images:

        slic_operation(image)


    # Clean up
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


def slic_operation(image):
    # Show base image

    num_superpixels = 50
    m = 2
    max_iterations = 20
    threshold = 10

    # Perform slic on image
    start_time = time.perf_counter()
    segmented_matrix, cluster_centers = slic(image, num_superpixels,  m, max_iterations, threshold)
    end_time = time.perf_counter()

    cv2.imshow('Image', image)
    cv2.waitKey()

    color_segmented_matrix = segmented_to_color(segmented_matrix, cluster_centers)
    cv2.imshow("Segmentation Matrix", color_segmented_matrix)
    cv2.waitKey()

    combined_image = average_images(image, color_segmented_matrix)
    cv2.imshow("Combined Image: ", combined_image)
    cv2.waitKey()

    print(f'Test Results: {end_time - start_time} seconds')
    return
