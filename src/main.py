from src.segment_full_slic import run_full_slic
from src.segment_single_slic import run_single_slic


def main():
    print("Semantic Segmentation:")
    # run_single_slic()
    run_full_slic()


if __name__ == "__main__":
    main()
