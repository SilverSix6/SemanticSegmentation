#ifndef TYPES_H
#define TYPES_H

struct Cluster {
    float x, y, l, a, b;
    int n;
};

#define BLOCK_SIZE 32 // Define block size for CUDA kernels
#define MAX_LABELS 1024

#define CHECK_CUDA_ERROR(err) { \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(-1); \
    } \
}

#endif