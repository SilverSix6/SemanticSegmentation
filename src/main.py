from label_single_kmeans import run_single_kmeans
from segment_full_slic import run_full_slic
from segment_single_slic import run_single_slic


def main():
    print("Semantic Segmentation:")
    num_superpixels = 512
    m = 1
    max_iterations = 10
    threshold = 20

    # run_single_slic("data/raw/test-images/standard_test_images/fruits.png", num_superpixels, m, max_iterations, threshold)
    # run_full_slic("src/data/raw/test-images/leftImg8bit/train", num_superpixels, m, max_iterations, threshold)
    run_single_kmeans(threshold, 128)


if __name__ == "__main__":
    main()
