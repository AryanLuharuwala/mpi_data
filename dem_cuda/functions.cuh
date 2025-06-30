#ifndef CUBIC_INITIALIZATION_CUH
#define CUBIC_INITIALIZATION_CUH
#include <vector>
#include <cuda_runtime.h>
#include "structs.h"

// Forward declaration of the kernel with correct __global__ qualifier
__global__ void launchCubicInitialization(float *particles, int num_x, int num_y, int num_z, float particle_radius, int id_global);

std::vector<std::vector<float>> initializeParticles(int domain_height, int domain_width, int domain_depth, float particle_radius, int id_global);

// Simulate particle physics including collisions with walls and other particles
void simulateParticles(std::vector<Particle> &particles, const std::vector<Wall> &walls);

// Enhanced simulation with neighbor search info (using precomputed collision pairs from FAISS)
void simulateParticlesWithNeighborInfo(std::vector<Particle> &particles, const std::vector<Wall> &walls, 
                                       const std::vector<std::pair<int, int>> &collision_pairs);

// Apply force fields to particles within specified regions
void applyForceFields(std::vector<Particle> &particles, const std::vector<ForceField> &force_fields);

// Enhanced simulation with both neighbor information and force fields
void simulateParticlesWithForcesAndNeighbors(std::vector<Particle> &particles, 
                                            const std::vector<Wall> &walls,
                                            const std::vector<std::pair<int, int>> &collision_pairs,
                                            const std::vector<ForceField> &force_fields);

#endif // CUBIC_INITIALIZATION_CUH