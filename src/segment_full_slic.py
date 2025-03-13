import math
import os
import time

import cv2

from src.labeling.image_load_util import load_folder_image
from src.segmentation.slic import slic


def run_full_slic():
    max_number_of_images = 100

    # SLIC settings
    num_superpixels = 512
    m = 1
    max_iterations = 10
    threshold = 20

    print("Processing dataset using...")

    print("Step 1: Loading images locations")
    image_directory = "src/data/raw/test-images/leftImg8bit/train"

    image_filenames =  load_folder_image(image_directory)

    print(f"Loaded {len(image_filenames)} images")

    print("Step 2: Segmenting images using SLIC")
    image_number = 0
    for image_filename in image_filenames:
        print(f"\tProcessing: id: {image_number} path: {image_filename}")

        # Read Image
        image = cv2.imread(image_filename, cv2.IMREAD_COLOR)

        # Segment Image
        start_time = time.perf_counter()
        segmented_matrix, cluster_center = slic(image, num_superpixels, m, max_iterations, threshold)
        end_time = time.perf_counter()

        # Save Results
        print(f"\tSaving Results: id: {image_number} path: {image_filename}")


        image_number += 1

        print(f'\tComplete: {math.floor((end_time - start_time) * 1000)} ms')

        if image_number == max_number_of_images:
            break


    print("DONE")

