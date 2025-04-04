from label_single_kmeans import run_single_kmeans
from segment_full_slic import run_full_slic
from src.slic_benchmark import run_slic_cpu_vs_gpu, run_slic_image_size
from segment_single_slic import run_single_slic


def main():
    print("Semantic Segmentation:")

    test_images = [
        "src/data/raw/test-images/leftImg8bit/test/berlin/berlin_000095_000019_leftImg8bit.png",
        "src/data/raw/test-images/leftImg8bit/test/bielefeld/bielefeld_000000_006603_leftImg8bit.png",
        "src/data/raw/test-images/leftImg8bit/test/bonn/bonn_000012_000019_leftImg8bit.png",
        "src/data/raw/test-images/leftImg8bit/test/leverkusen/leverkusen_000053_000019_leftImg8bit.png",
        "src/data/raw/test-images/leftImg8bit/test/mainz/mainz_000003_016360_leftImg8bit.png",
        "src/data/raw/test-images/leftImg8bit/test/munich/munich_000385_000019_leftImg8bit.png",
    ]

    num_superpixels = 1024
    m = 4
    max_iterations = 15
    threshold = 200
    gpu = True
    cpu = False

    run_single_slic("src/data/raw/test-images/leftImg8bit/test/berlin/berlin_000095_000019_leftImg8bit.png", num_superpixels, m, max_iterations, threshold, gpu)
    # run_full_slic("src/data/raw/test-images/leftImg8bit/train", num_superpixels, m, max_iterations, threshold)
    # run_slic_cpu_vs_gpu(test_images, num_superpixels, m, max_iterations, threshold)
    # run_slic_image_size(test_images[0], 4, num_superpixels, m, max_iterations, threshold)




if __name__ == "__main__":
    main()

#%%
