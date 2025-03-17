#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "rgb_to_lab.h"
#include "slic.h"

__device__ float xyz_to_lab_fn(float val) {
    if (val > 0.008856) // val > (6 / 29) ** 3
        return cbrt(val);
    // (6 / 29) ** -2 / 3 * val + 4 / 29
    return 7.78703 * val + 0.137931;
}

__global__ void convert_rgb_to_lab_kernel(unsigned char *image, int width, int height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height)
        return;

    int pixel_index = py * width + px;

    // Get RGB and normalize 0->255 to 0->1
    float r = (float)image[3 * pixel_index] / 255;
    float g = (float)image[3 * pixel_index + 1] / 255;
    float b = (float)image[3 * pixel_index + 2] / 255;

    // Convert to XYZ color space
    float x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    float y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    float z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;

    // Normalize XYZ values
    x /= X_n;
    y /= Y_n;
    z /= Z_n;

    // Convert XYZ to LAB and update image
    image[3 * pixel_index] = 116 * xyz_to_lab_fn(y) - 16;
    image[3 * pixel_index + 1] = 500 * (xyz_to_lab_fn(x) - xyz_to_lab_fn(y));
    image[3 * pixel_index + 2] = 200 * (xyz_to_lab_fn(y) - xyz_to_lab_fn(z));

    return;
}


void convert_rgb_to_lab_cuda(unsigned char *h_image, int width, int height) {

    unsigned char *d_image;

    size_t image_size = width * height * 3 * sizeof(unsigned char);

    // Allocate memory on device
    cudaError_t err = cudaMalloc(&d_image, image_size);
    CHECK_CUDA_ERROR(err);

    // Copy image to the device
    err = cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Convert image to lab
    convert_rgb_to_lab_kernel<<<gridSize, blockSize>>>(d_image, width, height);
    cudaDeviceSynchronize();

    // Copy resulting image to host
    err = cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err);

    cudaFree(d_image);
    d_image = NULL;
}


