// vector addition

#include <stdio.h>

__global__ void add_vectors(int *d_A, int *d_B, int *d_C) {
        d_C[blockIdx.x * blockDim.x + threadIdx.x] = 
        d_A[blockIdx.x * blockDim.x + threadIdx.x] + 
        d_B[blockIdx.x * blockDim.x + threadIdx.x]; 
}

int main() {
    int N = 1024;
    int bytes = N * sizeof(int);

    int *d_A;
    int *d_B;
    int *d_C;

    int *h_A;
    int *h_B;
    int *h_C;

    h_A = (int*)malloc(bytes);
    h_B = (int*)malloc(bytes);
    h_C = (int*)malloc(bytes);

    for (int i = 0; i < N; i++){
        h_A[i] = i+1;
        h_B[i] = 2*(i+1);
    }

    cudaMalloc((void**)&d_A, bytes); 
    cudaMalloc((void**)&d_B, bytes); 
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    add_vectors<<<4, 256>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("C[%d]: %d\n", i, h_C[i]);
    }

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
