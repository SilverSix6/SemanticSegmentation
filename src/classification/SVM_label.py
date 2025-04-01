import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from classification.superpixel_features import extract_superpixel_features, prepare_labels
import matplotlib.pyplot as plt


def train_evaluate_svm(image, segments, label_matrix, test_size=0.3, random_state=42):
    """
    Trains SVM model to classify superpixels based on features extracted from the input image.

    :param image: Input image (numpy array, shape: [height, width, 3], dtype: uint8).
    :param segments: Array of superpixel segment IDs from SLIC (shape: [height, width]).
    :param label_matrix: Matrix of corresponding segment labels (shape: [height, width]).
    :param test_size: Percentage of dataset to use for testing.
    :param random_state: Random seed for reproducibility.
    :return: svm_classifier (trained classifier based on passed data), X_test (test images), y_test (ground truth
    segment labels), y_pred (predicted segmented labels), accuracy (model accuracy).
    """
    # Extract features from superpixels
    features, segment_ids = extract_superpixel_features(image, segments)

    # Prepare corresponding labels
    labels = prepare_labels(label_matrix, segments, segment_ids)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Train SVM Classifier
    svm_classifier = SVC(kernel='linear', C=1.0, random_state=random_state)
    svm_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = svm_classifier.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)

    # Return classifier and predictions
    return svm_classifier, X_test, y_test, y_pred, accuracy


def visualize_predictions(image, segments, segment_labels, predictions):
    """
    Visualizes the predicted segmentation on the input image.
    :param image: Image to visualize. (numpy array, shape: [height, width, 3], dtype: uint8)
    :param segments: Segments array from SLIC for image, same as were used for training. (numpy array, shape: [height, width])
    :param segment_labels: Segment ground truth labels used for training. (numpy array, shape: [n_segments])
    :param predictions: Segment predicted labels from SVM. (numpy array, shape: [n_segments])
    :return: None
    """
    prediction_map = np.zeros(segments.shape, dtype=predictions.dtype)

    label_mapping = dict(zip(segment_labels, predictions))

    for seg_label, pred_label in label_mapping.items():
        prediction_map[segments == seg_label] = pred_label

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Predicted Segmentation")
    plt.imshow(prediction_map, cmap='jet')
    plt.axis('off')

    plt.show()
