nvcc -o libslic.so -shared -Xcompiler -fPIC  -arch=sm_75 clusterCenters.cu pixelClusterAssignment.cu slic.cu
