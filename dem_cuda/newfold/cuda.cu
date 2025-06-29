#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include "cuda.cuh"

__global__ void launchCubicInitialization(float start_x, float start_y, float start_z,
                                          int size_x, int size_y, int size_z,
                                          float particle_radius, float *particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size_x * size_y * size_z)
    {
        int z = idx / (size_x * size_y);
        int y = (idx % (size_x * size_y)) / size_x;
        int x = idx % size_x;
        int particle_index = idx * 3;                                         // Each particle has 3 float values (x, y, z)
        particles[particle_index] = start_x + x * particle_radius * 2.0f;     // x position
        particles[particle_index + 1] = start_y + y * particle_radius * 2.0f; // y position
        particles[particle_index + 2] = start_z + z * particle_radius * 2.0f; // z position
    }
}



void initializeParticles(const std::vector<float>& params, std::vector<Particle>& particles)
{
    // Extract parameters
    float start_x = params[0];
    float start_y = params[1];
    float start_z = params[2];
    int size_x = static_cast<int>(params[3]);
    int size_y = static_cast<int>(params[4]);
    int size_z = static_cast<int>(params[5]);
    float particle_radius = params[6];
    int total_particles = size_x * size_y * size_z;

    float *h_particles = new float[total_particles * 3]; // Each particle has 3 float values (x, y, z)
    // Allocate memory for particles on the device
    float *d_particles;
    cudaMalloc(&d_particles, total_particles * 3 * sizeof(float)); // Each particle has 3 float values (x, y, z)

    // Launch the kernel
    int block_size = 256; // Example block size
    int num_blocks = (total_particles + block_size - 1) / block_size;
    launchCubicInitialization<<<num_blocks, block_size>>>(start_x, start_y, start_z,
                                                          size_x, size_y, size_z,
                                                          particle_radius, d_particles);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy results back: host ← device
    cudaMemcpy(h_particles, d_particles, total_particles * 3 * sizeof(float), cudaMemcpyDeviceToHost);
     err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Free device memory
    cudaFree(d_particles);
    // Convert flat array to vector of Particle structs
    particles.clear();
    particles.reserve(total_particles);
    for (int i = 0; i < total_particles; ++i) {
        Particle p;
        p.position.x = h_particles[i * 3];
        p.position.y = h_particles[i * 3 + 1];
        p.position.z = h_particles[i * 3 + 2];
        particles.push_back(p);
    }
    // Free host buffer
    delete[] h_particles;
}