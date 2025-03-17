#ifndef SLIC_H
#define SLIC_H

struct Cluster {
    float x, y, l, a, b;
    int n;
};

#define BLOCK_SIZE 32 // Define block size for CUDA kernels
#define MAX_SUPERPIXELS 1024 // Set hard limit for number 

#define CHECK_CUDA_ERROR(err) { \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(-1); \
    } \
}

// Define DEBUG to enable debugging, or comment it out to disable
#define DEBUG

#ifdef DEBUG
#define START_TIMER(event) cudaEventCreate(&event); cudaEventRecord(event)
#define STOP_TIMER(event, label, iteration)                           \
    {                                                                 \
        cudaEvent_t stopEvent;                                        \
        cudaEventCreate(&stopEvent);                                  \
        cudaEventRecord(stopEvent);                                   \
        cudaEventSynchronize(stopEvent);                              \
        float milliseconds = 0;                                       \
        cudaEventElapsedTime(&milliseconds, event, stopEvent);        \
        if (iteration >= 0)                                           \
            printf("%s (Iteration %d): %.3f ms\n", label, iteration, milliseconds); \
        else                                                          \
            printf("%s: %.3f ms\n", label, milliseconds);             \
        cudaEventDestroy(stopEvent);                                  \
    }
#else
#define START_TIMER(event)
#define STOP_TIMER(event, label, iteration)
#endif

/**
 * Performs Superpixel segmentation using the Simple Linear Iterative Clustering (SLIC) algorithm on
 * given image. The function groups pixel into superpixels based on color and spatial proximity.
 * 
 * @param image: A pointer to the input image. This data should be stored in a 1D array in row-major order. The image is assumed to be RGB.
 * @param width: The images width (number of pixels)
 * @param height: The images height (number of pixels)
 * @param num_superpixels: The total number of superpixels the system will produce. 
 * @param max_iterations: The number of max number iteration the system will process if the error threshold is not first reached. 
 * @param m: The compactness factor. Used to balance spatial and color proximity. Higher values enforce spatial uniformity.
 * @param threshold: The stopping threshold for convergence.
 * @param clusters: Pointer to array of Cluster structs storing cluster centers. Memory should be pre-allocated.
 * @param segmented_matrix: Pointer to the output matrix of the same dimentions as the input image. Each pixel is assigned the id of it's superpixel label.
 */
extern "C" void slic(unsigned char* image, int width, int height, int num_superpixels, int max_iterations, float m, float threshold, Cluster *clusters, int *segmented_matrix);

#endif