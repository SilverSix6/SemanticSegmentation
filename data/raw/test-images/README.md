# standard-test-images-for-Image-Processing
Collection of standard test images for image processing

## Run **datasetDownloader.py** in your terminal to download, and extract the dataset to the test-images directory.

The **Cityscapes dataset** consists of high-resolution images captured from 50 cities with **pixel-wise semantic annotations**.  
The dataset includes:
- **Original RGB images** (`leftImg8bit/`)
    - left 8-bit images - train, val, and test sets (5000 images)
- **Segmentation masks** (`gtFine/`)
    - fine annotations for train and val sets (3475 annotated images) and dummy annotations (ignore regions) for the test set (1525 images)
- **Train, validation, and test splits**

## Dataset File Naming Convention

Each image and its corresponding mask share the **same filename prefix**.

### **Example:**
| File Type          | Example File Path |
|--------------------|----------------------------------------------------------------|
| **RGB Image**      | `data/raw/test-images/leftImg8bit/train/berlin/berlin_000000_000019_leftImg8bit.png` |
| **Segmentation Mask** | `data/raw/test-images/gtFine/train/berlin/berlin_000000_000019_gtFine_labelIds.png` |

To match an **RGB image** to its **ground truth mask**, replace **`_leftImg8bit.png`** with **`_gtFine_labelIds.png`**.

## Class Definitions

| **Group**         | **Classes** |
|------------------|------------|
| **flat**        | road · sidewalk · parking+ · rail track+ |
| **human**       | person* · rider* |
| **vehicle**     | car* · truck* · bus* · on rails* · motorcycle* · bicycle* · caravan** · trailer*+ |
| **construction** | building · wall · fence · guard rail* · bridge* · tunnel+ |
| **object**      | pole · pole group+ · traffic sign · traffic light |
| **nature**      | vegetation · terrain |
| **sky**        | sky |
| **void**       | ground+ · dynamic+ · static+ |

## Image size
Most images are 1024×2048


