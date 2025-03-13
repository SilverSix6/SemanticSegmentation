#
# File to test SLIC. Runs segmentation on files in the test image array.
#

import time

import cv2
import numpy as np

from labeling.image_load_util import load_list_image
from segmentation.slic import slic
from utils.segmentation_utils import segmented_to_color, average_images


def run_single_slic():
    print("Running Segmentation Using SLIC:")

    test_images = ['src/data/raw/test-images/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png']

    images = load_list_image(test_images, False)

    for image in images:
        slic_operation(image)

    # Clean up
    cv2.destroyAllWindows()

    return


def slic_operation(image):
    # Show base image
    num_superpixels = 512
    m = 1
    max_iterations = 10
    threshold = 20

    # Perform slic on image
    start_time = time.perf_counter()
    segmented_matrix, cluster_centers = slic(image, num_superpixels,  m, max_iterations, threshold)
    end_time = time.perf_counter()

    print(f'Test Results: {end_time - start_time} seconds')

    # Display Input Image
    cv2.imshow('Image', image)
    cv2.waitKey()

    # Display Segmented image with color
    color_segmented_matrix = segmented_to_color(segmented_matrix, cluster_centers)
    cv2.imshow("Segmentation Matrix", color_segmented_matrix)
    cv2.waitKey()

    # Display combination of input image and color image
    combined_image = average_images(image, color_segmented_matrix)
    cv2.imshow("Combined Image: ", combined_image)
    cv2.waitKey()


    return
