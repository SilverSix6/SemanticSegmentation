import math
import os
import time

import cv2

from src.labeling.image_load_util import load_folder_image
from src.segmentation.slic import slic
from src.utils.serialize import write_matrix


def run_full_slic(image_directory, num_superpixels, m, max_iterations, threshold):
    os.makedirs('data/slic/', exist_ok=True)
    matrix_bin_file = "data/slic/full_slic_matrix.bin"
    cluster_bin_file = "data/slic/full_slic_cluster.bin"

    print("Processing dataset using SLIC")
    print("Step 1: Loading images locations")
    image_filenames =  load_folder_image(image_directory)
    print(f"Loaded {len(image_filenames)} images")

    print(f"Step 2: Results will be saved to: {matrix_bin_file} and {cluster_bin_file}")
    matrix_file = open(matrix_bin_file, "wb")
    cluster_file = open(cluster_bin_file, "wb")

    print("Step 3: Segmenting images using SLIC")
    image_number = 0
    try:
        for image_filename in image_filenames:
            print(f"\tProcessing: id: {image_number} path: {image_filename}")

            # Read Image
            image = cv2.imread(image_filename, cv2.IMREAD_COLOR)

            # Segment Image
            start_time = time.perf_counter()
            segmented_matrix, cluster_center = slic(image, num_superpixels, m, max_iterations, threshold)
            end_time = time.perf_counter()

            # Save Results
            print(f"\tSaving Results: id: {image_number}")
            write_matrix(segmented_matrix, matrix_file)
            write_matrix(cluster_center, cluster_file)

            print(f'\tComplete: {math.floor((end_time - start_time) * 1000)} ms')
            image_number += 1

    finally:
        matrix_file.close()
        cluster_file.close()

    print("DONE")

