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
// #define DEBUG

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



#endif