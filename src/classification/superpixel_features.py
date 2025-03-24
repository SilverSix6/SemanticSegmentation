import numpy as np
from skimage.color import rgb2lab
from skimage.measure import regionprops


def extract_superpixel_features(image, segments):  # This will need the most experimentation
    """
    Extracts features from superpixels based on the input image and the superpixel segmentation. Currently, uses mean
    color and centroid of the superpixel.
    :param image: Input image (numpy array, shape: [height, width, 3], dtype: uint8).
    :param segments: Array of superpixel segment IDs from SLIC (shape: [height, width]).
    :return: features (numpy array, shape: [n_segments, n_features]), segment_labels (numpy array, shape: [n_segments]).
    """
    lab_image = rgb2lab(image)
    regions = regionprops(segments, intensity_image=None)

    features = []
    segment_labels = []

    for region in regions:
        mask = segments == region.label # Mask to extract pixels of the region so we can focus just on a superpixel

        # Mean color features (LAB)
        mean_color = lab_image[mask].mean(axis=0)

        # Position features (centroid normalized by image dimensions)
        centroid = np.array(region.centroid) / np.array(image.shape[:2])

        # Texture or shape features can also be added here

        feature_vector = np.concatenate([mean_color, centroid])
        features.append(feature_vector)
        segment_labels.append(region.label)

    return np.array(features), np.array(segment_labels)


def prepare_labels(label_matrix, segments, segment_labels):
    """
    Prepares the labels for the superpixels based on the most frequent label in the segment. This is important incase
    the superpixel has multiple labels.
    :param label_matrix: Labelled superpixel matrix (shape: [height, width]).
    :param segments: Array of superpixel segment IDs from SLIC (shape: [height, width]).
    :param segment_labels: List of segment labels used for training.
    :return: labels (numpy array, shape: [n_segments]).
    """
    labels = []
    for seg_label in segment_labels:
        mask = segments == seg_label
        # Most frequent label in the segment
        label, counts = np.unique(label_matrix[mask], return_counts=True)
        dominant_label = label[np.argmax(counts)]
        labels.append(dominant_label)
    return np.array(labels)
