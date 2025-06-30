#ifndef CUDA_CUH
#define CUDA_CUH
#include <vector>
#include <cuda_runtime.h>
#include "structs.h"

__global__ void launchCubicInitialization(float start_x, float start_y, float start_z,
                                          int size_x, int size_y, int size_z,
                                          float particle_radius, float *particles);



                                        
void initializeParticles(const std::vector<float>& params, std::vector<Particle>& particles);

#endif // CUDA_CUH