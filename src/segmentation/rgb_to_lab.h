#ifndef RGB_TO_LAB_H
#define RGB_TO_LAB_H

#define X_n 0.95047
#define Y_n 1.00000
#define Z_n 1.08883

void convert_rgb_to_lab_cuda(unsigned char *h_image, int width, int height);

#endif