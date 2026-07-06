#include <vector>
#include <cuda_runtime.h>
#include <cstdio>

const int width = 400;
const int height = 400;
const int size = width * height;
const int total_bytes = size * sizeof(float);

__global__ void init(
    float* T,
    int witdth,
    int height,
    float background_temperature,
    float circle_temperature,
    int circle_center_x,
    int circle_center_y,
    int radius,
) {
    int i = blockIdx.x * blockDim.x + threadId.x;
    int j = blockIdy.y * blockDim.y + threadId.y;

    if (i <= width  && j <= height) {
        int index = i * width + j;

        int dx = i - circle_center_x;
        int dy = j - circle_center_y;

        if (i*i + j*j <= radius * radius) {
            T[index] = circle_temperature;
        } else {
            T[index] = background_temperature;
        }
    }
}

__global__ void temperature_simulation_step(
    
) {
    
}

int main() {
    std::vector<float> h_T(size); 
    cudaMalloc(

    // Init the vector with heat spot

    // Run actual heat spreading simulation
    
    // Export every 10th frame of the simulation
    
    
    return 0;
}
