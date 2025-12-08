// hello.cu

#include <stdio.h>

__global__ void helloFromGPU(){
    int block_number = blockIdx.x;
    int thread_number = threadIdx.x;
    printf("Hello from block %d, thread %d!\n", block_number, blockDim.x*block_number + thread_number);
}

int main() {
    helloFromGPU<<<2, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
