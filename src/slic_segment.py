import time

import cv2

from labeling.image_load_util import load_list_image


def run_slic_segment():
    print("Running Segmentation Using SLIC:")

    test_images = ['data/raw/test-images/standard_test_images/baboon.png',
                   'data/raw/test-images/standard_test_images/lena_gray_512.tif',
                   'data/raw/test-images/standard_test_images/fruits.png']

    images = load_list_image(test_images, False)

    # Display each image
    for image in images:
        slic_operation(image)
        cv2.waitKey(0)


    # Clean up
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


def slic_operation(image):
    # Show base image
    cv2.imshow('Input Image', image)

    # Perform slic on image
    start_time = time.perf_counter()
    result_image = []
    end_time = time.perf_counter()

    # Display resulting image
    cv2.imread('Segmented Image', result_image)

    print(f'Test Results: {end_time - start_time} seconds')
    return
