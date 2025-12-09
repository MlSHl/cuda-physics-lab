// matrix addition

#include <stdio.h>

__global__ void add_matricies(int *d_A, int *d_B, int *d_C, int width, int height) {
    int i = (blockIdx.y * blockDim.y) + threadIdx.y;
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;

    int index = i * width + j;

    if (index < width*height) {
        d_C[index] = d_A[index] + d_B[index];
    }
} 

int main() {
    int M = 65;
    int N = 65;
    int total_bytes = N * M*sizeof(int);

    int *h_A;
    int *h_B;
    int *h_C;

    int *d_A;
    int *d_B;
    int *d_C;

    h_A = (int*)malloc(total_bytes);
    h_B = (int*)malloc(total_bytes);
    h_C = (int*)malloc(total_bytes);

    // initialization of host
    for (int i = 0; i < M*N; i++) {
        h_A[i] = i + 1;
        h_B[i] = i + 1;
    }

    // initialization of device
    cudaMalloc((void**)&d_A, total_bytes);
    cudaMalloc((void**)&d_B, total_bytes);
    cudaMalloc((void**)&d_C, total_bytes);

    cudaMemcpy(d_A, h_A, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total_bytes, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid(5, 5);

    add_matricies<<<blockPerGrid, threadPerBlock>>>(d_A, d_B, d_C, N, M);

    cudaMemcpy(h_C, d_C, total_bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i*M + j]); 
            if(j == N-1) printf("\n");
        }
    }

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}
