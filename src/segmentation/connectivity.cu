#include "slic.h"

// Function to check if a pixel has 4-connectivity
int is_4_connected(int *image, int height, int width, int x, int y) {
    int label = image[y * width + x];
    return (y > 0 && image[(y - 1) * width + x] == label) ||
           (y < height - 1 && image[(y + 1) * width + x] == label) ||
           (x > 0 && image[y * width + x - 1] == label) ||
           (x < width - 1 && image[y * width + x + 1] == label);
}

/**
 * Enforces connectivity of pixels. Assigns pixels to be its most common neighboor. 
 * If there is a tie, the resulting label is randomly choosen between the tieing groups.
 * 
 * @param image: A segmented image.
 * @param width: The image's width (number of pixels)
 * @param height: The image's height (number of pixels) 
 */
void enforce_connectivity(int *image, int width, int height) {
    int dx[] = {-1, 1, 0, 0}; // Left, Right, Up, Down
    int dy[] = {0, 0, -1, 1};

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (!is_4_connected(image, height, width, x, y)) {
                int label_counts[MAX_SUPERPIXELS] = {0}; // Assuming labels range from 0-255
                int max_label = image[y * width + x];
                int max_count = 0;

                // Check neighbors
                for (int i = 0; i < 4; i++) {
                    int nx = x + dx[i];
                    int ny = y + dy[i];
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int label = image[ny * width + nx];
                        label_counts[label]++;
                        if (label_counts[label] > max_count) {
                            max_count = label_counts[label];
                            max_label = label;
                        }
                    }
                }

                // Assign the most common neighboring label
                image[y * width + x] = max_label;
            }
        }
    }
}