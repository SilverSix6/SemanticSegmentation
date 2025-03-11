#ifndef TYPES_H
#define TYPES_H

struct Cluster {
    float x, y, l, a, b;
    int n;
};

#define BLOCK_SIZE 32 // Define block size for CUDA kernels

#endif