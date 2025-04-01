import math
import os
import time

import cv2

from src.labeling.image_load_util import load_folder_image, get_full_path_from_root
from src.segmentation.slic import slic
from src.utils.serialize import write_matrix


def run_slic_cpu_vs_gpu(image_filenames, num_superpixels, m, max_iterations, threshold):
    os.makedirs('data/slic/', exist_ok=True)
    matrix_bin_file = "data/slic/full_slic_matrix.bin"
    cluster_bin_file = "data/slic/full_slic_cluster.bin"

    print("Processing image set using SLIC")
    print(f"Running SLIC on {len(image_filenames)} images")

    print(f"Step 1: Results will be saved to: {matrix_bin_file} and {cluster_bin_file}")
    matrix_file = open(matrix_bin_file, "wb")
    cluster_file = open(cluster_bin_file, "wb")

    print("Step 2: Segmenting images using SLIC")
    image_number = 0
    try:
        print("\nStep 2: Segmenting images using SLIC with CPU")
        for image_filename in image_filenames:
            print(f"\tProcessing: id: {image_number} path: {image_filename}")

            # Read Image
            image = cv2.imread(get_full_path_from_root(image_filename), cv2.IMREAD_COLOR)

            # Segment Image
            start_time = time.perf_counter()
            segmented_matrix, cluster_center = slic(image, num_superpixels, m, max_iterations, threshold, False)
            end_time = time.perf_counter()

            # Save Results
            write_matrix(segmented_matrix, matrix_file)
            write_matrix(cluster_center, cluster_file)

            # print(f'\tImage Complete: {math.floor((end_time - start_time) * 1000)} ms\n')
            image_number += 1

        print("\nStep 3: Segmenting images using SLIC with GPU")
        for image_filename in image_filenames:
            print(f"\tProcessing: id: {image_number} path: {image_filename}")

            # Read Image
            image = cv2.imread(get_full_path_from_root(image_filename), cv2.IMREAD_COLOR)

            # Segment Image
            start_time = time.perf_counter()
            segmented_matrix, cluster_center = slic(image, num_superpixels, m, max_iterations, threshold, True)
            end_time = time.perf_counter()

            # Save Results
            write_matrix(segmented_matrix, matrix_file)
            write_matrix(cluster_center, cluster_file)

            # print(f'\tImage Complete: {math.floor((end_time - start_time) * 1000)} ms\n')
            image_number += 1

    finally:
        matrix_file.close()
        cluster_file.close()

    print("DONE")


def run_slic_image_size(image_filename, num_sub_divisions, num_superpixels, m, max_iterations, threshold):
    print("Processing image using SLIC")

    print(f"Running SLIC on 1 image: {image_filename}")

    image = cv2.imread(get_full_path_from_root(image_filename), cv2.IMREAD_COLOR)

    sub_divisions = 0
    while sub_divisions < num_sub_divisions:

        print(f"\tProcessing image with {sub_divisions} sub divisions. path: {image_filename}")

        # Segment Image
        segmented_matrix, cluster_center = slic(image, num_superpixels, m, max_iterations, threshold, True)

        # Half the image
        height, width = image.shape[:2]
        image = cv2.resize(image, (width // 2, height // 2))
        sub_divisions += 1


    print("DONE")

