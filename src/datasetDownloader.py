import os
import gdown
import zipfile

LEFTIMG_ID = "1jAP0DSfXlkoiRilHhCBaibNHZAuLZARQ"
GTFINE_ID = "1GOz1F_DEWJuiVF7HoG7NohGhHDismDf6"
DATA_DIR = "data/raw/test-images/"
LEFTIMG_ZIP = os.path.join(DATA_DIR, "leftImg8bit_trainvaltest.zip")
GTFINE_ZIP = os.path.join(DATA_DIR, "gtFine_trainvaltest.zip")

def download_file(file_id, save_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, save_path, quiet=False)

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(LEFTIMG_ZIP):
        download_file(LEFTIMG_ID, LEFTIMG_ZIP)
        extract_zip(LEFTIMG_ZIP, DATA_DIR)

    if not os.path.exists(GTFINE_ZIP):
        download_file(GTFINE_ID, GTFINE_ZIP)
        extract_zip(GTFINE_ZIP, DATA_DIR)

    print("dataset is ready.")
