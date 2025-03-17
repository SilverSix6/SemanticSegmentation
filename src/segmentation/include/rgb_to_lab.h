#ifndef RGB_TO_LAB_H
#define RGB_TO_LAB_H

#define X_n 0.95047
#define Y_n 1.00000
#define Z_n 1.08883

/**
 * Converts an rgb image to the lab color space. 
 * 
 * @param h_image: A pointer the the image that is being converted. The image should be a 1D array with pixel data in row major format. 
 * @param width: The width of the image (number of pixels)
 * @param height: The height of the image (number of pixels)
 */
void convert_rgb_to_lab_cuda(unsigned char *h_image, int width, int height);

#endif