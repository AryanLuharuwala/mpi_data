#include <vector>
#include <utility> // for std::pair
#include <iostream>
#include <cuda_runtime.h>

// Forward declaration of the kernel with correct __global__ qualifier
__global__ void launchCubicInitialization(float *particles, int num_x, int num_y, int num_z, float particle_radius, int id_global);

std::vector<std::vector<float>> initializeParticles(int domain_height, int domain_width, int domain_depth, float particle_radius, int id_global) {
    // Initialize particles with the given dimensions and global ID
    // BCC lattice is 2r(a*domain_height, b*domain_width, c*domain_depth)
    int num_x = domain_height / (2 * particle_radius);
    int num_y = domain_width / (2 * particle_radius);
    int num_z = domain_depth / (2 * particle_radius);
    
    // Calculate total number of particles
    int total_particles = num_x * num_y * num_z;
    
    // Create device array to store particle data (4 floats per particle: x, y, z, id)
    float *d_particles;
    cudaMalloc(&d_particles, total_particles * 4 * sizeof(float));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (total_particles + blockSize - 1) / blockSize;
    launchCubicInitialization<<<numBlocks, blockSize>>>(d_particles, num_x, num_y, num_z, particle_radius, id_global);
    
    // Create host array to receive results
    float *h_particles = new float[total_particles * 4];
    cudaMemcpy(h_particles, d_particles, total_particles * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Convert to vector of vectors format
    std::vector<std::vector<float>> particles;
    for (int i = 0; i < total_particles; i++) {
        std::vector<float> particle = {
            h_particles[i * 4],     // x
            h_particles[i * 4 + 1], // y
            h_particles[i * 4 + 2], // z
            h_particles[i * 4 + 3]  // id
        };
        particles.push_back(particle);
    }
    
    // Clean up
    delete[] h_particles;
    cudaFree(d_particles);
    
    return particles;
}

// Correct kernel implementation
__global__ void launchCubicInitialization(float *particles, int num_x, int num_y, int num_z, float particle_radius, int id_global) {
    // Kernel to initialize particles in a cubic lattice
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_x * num_y * num_z) {
        int x = idx % num_x;
        int y = (idx / num_x) % num_y;
        int z = idx / (num_x * num_y);
        
        // Calculate the position of the particle and store components separately
        particles[idx * 4] = (x * 2 * particle_radius) + float((id_global) % num_x * (2*particle_radius * num_x));      // x position
        particles[idx * 4 + 1] = y * 2 * particle_radius + float((int(id_global) / num_x) * (2*particle_radius * num_y));  // y position
        particles[idx * 4 + 2] = z * 2 * particle_radius;  // z position
        particles[idx * 4 + 3] = float(id_global);                // particle id
    }
}

#include "structs.h"
#include <cuda_runtime.h>

// Helper device function for particle-particle collision response
__device__ void handleParticleCollision(Particle &p, Particle &other) {
    // Calculate distance between particles
    float dx = p.position.x - other.position.x;
    float dy = p.position.y - other.position.y;
    float dz = p.position.z - other.position.z;
    
    // Calculate squared distance
    float distSq = dx*dx + dy*dy + dz*dz;
    
    // Sum of radii
    float sumRadii = p.properties.radius + other.properties.radius;
    float sumRadiiSq = sumRadii * sumRadii;
    
    // Check for collision
    if (distSq < sumRadiiSq && distSq > 0.0001f) { // Small epsilon to avoid division by zero
        // Calculate actual distance
        float dist = sqrtf(distSq);
        
        // Calculate overlap
        float overlap = sumRadii - dist;
        
        // Calculate unit normal vector from other to this particle
        float nx = dx / dist;
        float ny = dy / dist;
        float nz = dz / dist;
        
        // Calculate relative velocity
        float rvx = p.state.velocity_x - other.state.velocity_x;
        float rvy = p.state.velocity_y - other.state.velocity_y;
        float rvz = p.state.velocity_z - other.state.velocity_z;
        
        // Calculate relative velocity along the normal
        float normalVelocity = rvx * nx + rvy * ny + rvz * nz;
        
        // Only apply force if particles are approaching each other
        if (normalVelocity < 0) {
            // Compute average properties for collision response
            float avg_stiffness = (p.properties.stiffness + other.properties.stiffness) * 0.5f;
            float avg_damping = (p.properties.damping + other.properties.damping) * 0.5f;
            float avg_restitution = (p.properties.restitution + other.properties.restitution) * 0.5f;
            
            // Calculate spring force (proportional to overlap)
            float springForce = avg_stiffness * overlap;
            
            // Calculate damping force (proportional to normal velocity)
            float dampingForce = avg_damping * normalVelocity;
            
            // Total force magnitude (spring - damping)
            float totalForce = springForce - dampingForce;
            
            // Apply forces along normal direction
            p.state.force_x += totalForce * nx;
            p.state.force_y += totalForce * ny;
            p.state.force_z += totalForce * nz;
            
            // Apply friction forces if sliding occurs
            // Calculate tangential component of relative velocity
            float tangVelX = rvx - normalVelocity * nx;
            float tangVelY = rvy - normalVelocity * ny;
            float tangVelZ = rvz - normalVelocity * nz;
            
            float tangVelMagSq = tangVelX*tangVelX + tangVelY*tangVelY + tangVelZ*tangVelZ;
            
            if (tangVelMagSq > 0.0001f) { // Only apply friction if there's significant tangential velocity
                float tangVelMag = sqrtf(tangVelMagSq);
                float friction = (p.properties.friction + other.properties.friction) * 0.5f;
                
                // Friction force is proportional to normal force and opposite to tangential velocity
                float frictionMag = friction * totalForce;
                
                // Apply friction force
                p.state.force_x -= frictionMag * tangVelX / tangVelMag;
                p.state.force_y -= frictionMag * tangVelY / tangVelMag;
                p.state.force_z -= frictionMag * tangVelZ / tangVelMag;
            }
        }
    }
}

// CUDA kernel for particle simulation with wall collisions
extern "C" __global__ void simulateParticlesKernel(Particle *particles, int num_particles, Wall *walls, int num_walls, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        Particle &p = particles[idx];
        
        // Reset forces
        p.state.force_x = 0.0f;
        p.state.force_y = 0.0f;
        p.state.force_z = -9.81f * p.properties.mass; // Gravity
        
        // Wall collisions
        for (int w = 0; w < num_walls; w++) {
            Wall wall = walls[w];
            
            // Calculate distance from particle to wall
            float dx = p.position.x - wall.centre_x;
            float dy = p.position.y - wall.centre_y;
            float dz = p.position.z - wall.centre_z;
            
            // Project onto wall normal
            float dist = dx * wall.normal_x + dy * wall.normal_y + dz * wall.normal_z;
            
            // Check for collision
            if (dist < p.properties.radius) {
                // Calculate overlap
                float overlap = p.properties.radius - dist;
                
                // Apply spring force
                float springForce = p.properties.stiffness * overlap;
                
                // Apply damping based on normal velocity
                float nVel = p.state.velocity_x * wall.normal_x + 
                             p.state.velocity_y * wall.normal_y + 
                             p.state.velocity_z * wall.normal_z;
                float dampingForce = p.properties.damping * nVel;
                
                // Total force
                float totalForce = springForce - dampingForce;
                
                // Apply forces along normal
                p.state.force_x += totalForce * wall.normal_x;
                p.state.force_y += totalForce * wall.normal_y;
                p.state.force_z += totalForce * wall.normal_z;
            }
        }
        
        // Particle-particle collisions (O(N²) version - inefficient for large particle counts)
        // We don't use shared memory for complete particles due to their large size
        // Instead, we directly access particles from global memory
        // which is slower but doesn't hit shared memory limits
        
        // Number of particles to process in this block
        int block_start = blockIdx.x * blockDim.x;
        int block_end = min(block_start + blockDim.x, num_particles);
        
        // Check collisions with particles in the same block
        for (int j = block_start; j < block_end; j++) {
            // Skip self-collision
            if (j == idx) continue;
            
            // Get the other particle directly from global memory
            Particle other_particle = particles[j];
            handleParticleCollision(p, other_particle);
        }
        
        // Update acceleration
        p.state.acceleration_x = p.state.force_x / p.properties.mass;
        p.state.acceleration_y = p.state.force_y / p.properties.mass;
        p.state.acceleration_z = p.state.force_z / p.properties.mass;
        
        // Update velocity (Semi-implicit Euler)
        p.state.velocity_x += p.state.acceleration_x * dt;
        p.state.velocity_y += p.state.acceleration_y * dt;
        p.state.velocity_z += p.state.acceleration_z * dt;
        
        // Update position
        p.position.x += p.state.velocity_x * dt;
        p.position.y += p.state.velocity_y * dt;
        p.position.z += p.state.velocity_z * dt;
    }
}

// CUDA kernel for particle simulation with precomputed neighbor list
extern "C" __global__ void simulateParticlesWithNeighborsKernel(
    Particle *particles, 
    int num_particles, 
    Wall *walls, 
    int num_walls, 
    int2 *collision_pairs, 
    int num_pairs, 
    float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        Particle &p = particles[idx];
        
        // Reset forces
        p.state.force_x = 0.0f;
        p.state.force_y = 0.0f;
        p.state.force_z = -9.81f * p.properties.mass; // Gravity
        
        // Wall collisions
        for (int w = 0; w < num_walls; w++) {
            Wall wall = walls[w];
            
            // Calculate distance from particle to wall
            float dx = p.position.x - wall.centre_x;
            float dy = p.position.y - wall.centre_y;
            float dz = p.position.z - wall.centre_z;
            
            // Project onto wall normal
            float dist = dx * wall.normal_x + dy * wall.normal_y + dz * wall.normal_z;
            
            // Check for collision
            if (dist < p.properties.radius) {
                // Calculate overlap
                float overlap = p.properties.radius - dist;
                
                // Apply spring force
                float springForce = p.properties.stiffness * overlap;
                
                // Apply damping based on normal velocity
                float nVel = p.state.velocity_x * wall.normal_x + 
                             p.state.velocity_y * wall.normal_y + 
                             p.state.velocity_z * wall.normal_z;
                float dampingForce = p.properties.damping * nVel;
                
                // Total force
                float totalForce = springForce - dampingForce;
                
                // Apply forces along normal
                p.state.force_x += totalForce * wall.normal_x;
                p.state.force_y += totalForce * wall.normal_y;
                p.state.force_z += totalForce * wall.normal_z;
            }
        }
        
        // Update acceleration
        p.state.acceleration_x = p.state.force_x / p.properties.mass;
        p.state.acceleration_y = p.state.force_y / p.properties.mass;
        p.state.acceleration_z = p.state.force_z / p.properties.mass;
        
        // Update velocity (Semi-implicit Euler)
        p.state.velocity_x += p.state.acceleration_x * dt;
        p.state.velocity_y += p.state.acceleration_y * dt;
        p.state.velocity_z += p.state.acceleration_z * dt;
        
        // Update position
        p.position.x += p.state.velocity_x * dt;
        p.position.y += p.state.velocity_y * dt;
        p.position.z += p.state.velocity_z * dt;
    }
}

// Process collision pairs in a separate kernel for better parallelism
extern "C" __global__ void processCollisionPairsKernel(
    Particle *particles, 
    int2 *collision_pairs, 
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pairs) {
        int2 pair = collision_pairs[idx];
        int i = pair.x;
        int j = pair.y;
        
        // Avoid out-of-bounds access
        if (i < 0 || j < 0) return;
        
        // Process collision between particles[i] and particles[j]
        Particle &p1 = particles[i];
        Particle &p2 = particles[j];
        
        // Calculate distance between particles
        float dx = p1.position.x - p2.position.x;
        float dy = p1.position.y - p2.position.y;
        float dz = p1.position.z - p2.position.z;
        
        // Calculate squared distance
        float distSq = dx*dx + dy*dy + dz*dz;
        
        // Sum of radii
        float sumRadii = p1.properties.radius + p2.properties.radius;
        float sumRadiiSq = sumRadii * sumRadii;
        
        // Check for collision
        if (distSq < sumRadiiSq && distSq > 0.0001f) { // Small epsilon to avoid division by zero
            // Calculate actual distance
            float dist = sqrtf(distSq);
            
            // Calculate overlap
            float overlap = sumRadii - dist;
            
            // Calculate unit normal vector from p2 to p1
            float nx = dx / dist;
            float ny = dy / dist;
            float nz = dz / dist;
            
            // Calculate relative velocity
            float rvx = p1.state.velocity_x - p2.state.velocity_x;
            float rvy = p1.state.velocity_y - p2.state.velocity_y;
            float rvz = p1.state.velocity_z - p2.state.velocity_z;
            
            // Calculate relative velocity along the normal
            float normalVelocity = rvx * nx + rvy * ny + rvz * nz;
            
            // Only apply force if particles are approaching each other
            if (normalVelocity < 0) {
                // Compute average properties for collision response
                float avg_stiffness = (p1.properties.stiffness + p2.properties.stiffness) * 0.5f;
                float avg_damping = (p1.properties.damping + p2.properties.damping) * 0.5f;
                float avg_restitution = (p1.properties.restitution + p2.properties.restitution) * 0.5f;
                
                // Calculate spring force (proportional to overlap)
                float springForce = avg_stiffness * overlap;
                
                // Calculate damping force (proportional to normal velocity)
                float dampingForce = avg_damping * normalVelocity;
                
                // Total force magnitude (spring - damping)
                float totalForce = springForce - dampingForce;
                
                // Apply forces along normal direction using atomic operations to avoid race conditions
                atomicAdd(&p1.state.force_x, totalForce * nx);
                atomicAdd(&p1.state.force_y, totalForce * ny);
                atomicAdd(&p1.state.force_z, totalForce * nz);
                
                atomicAdd(&p2.state.force_x, -totalForce * nx);
                atomicAdd(&p2.state.force_y, -totalForce * ny);
                atomicAdd(&p2.state.force_z, -totalForce * nz);
                
                // Apply friction forces if sliding occurs
                // Calculate tangential component of relative velocity
                float tangVelX = rvx - normalVelocity * nx;
                float tangVelY = rvy - normalVelocity * ny;
                float tangVelZ = rvz - normalVelocity * nz;
                
                float tangVelMagSq = tangVelX*tangVelX + tangVelY*tangVelY + tangVelZ*tangVelZ;
                
                if (tangVelMagSq > 0.0001f) { // Only apply friction if there's significant tangential velocity
                    float tangVelMag = sqrtf(tangVelMagSq);
                    float friction = (p1.properties.friction + p2.properties.friction) * 0.5f;
                    
                    // Friction force is proportional to normal force and opposite to tangential velocity
                    float frictionMag = friction * totalForce;
                    
                    // Unit tangent vector
                    float tx = tangVelX / tangVelMag;
                    float ty = tangVelY / tangVelMag;
                    float tz = tangVelZ / tangVelMag;
                    
                    // Apply friction force
                    atomicAdd(&p1.state.force_x, -frictionMag * tx);
                    atomicAdd(&p1.state.force_y, -frictionMag * ty);
                    atomicAdd(&p1.state.force_z, -frictionMag * tz);
                    
                    atomicAdd(&p2.state.force_x, frictionMag * tx);
                    atomicAdd(&p2.state.force_y, frictionMag * ty);
                    atomicAdd(&p2.state.force_z, frictionMag * tz);
                }
            }
        }
    }
}

void simulateParticles(std::vector<Particle> &particles, const std::vector<Wall> &walls) {
    if (particles.empty()) return;
    
    float dt = 0.001f; // Time step in seconds
    
    // Allocate device memory
    Particle *d_particles;
    Wall *d_walls;
    cudaMalloc(&d_particles, particles.size() * sizeof(Particle));
    cudaMalloc(&d_walls, walls.size() * sizeof(Wall));
    
    // Copy data to device
    cudaMemcpy(d_particles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_walls, walls.data(), walls.size() * sizeof(Wall), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (particles.size() + blockSize - 1) / blockSize;
    dim3 grid(numBlocks), block(blockSize);
    simulateParticlesKernel<<<grid, block>>>(d_particles, particles.size(), d_walls, walls.size(), dt);
    
    // Copy results back
    cudaMemcpy(particles.data(), d_particles, particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);
    
    // Update the CPU vector neighbors from the GPU array neighbors
#ifndef __CUDA_ARCH__
    for (auto &p : particles) {
        p.neighbor_vector.clear();
        for (int i = 0; i < p.num_neighbors; i++) {
            if (p.neighbors[i] >= 0) { // Valid neighbor ID
                p.neighbor_vector.push_back(p.neighbors[i]);
            }
        }
    }
#endif
    
    // Clean up
    cudaFree(d_particles);
    cudaFree(d_walls);
}

void simulateParticlesWithNeighborInfo(std::vector<Particle> &particles, const std::vector<Wall> &walls, 
                                       const std::vector<std::pair<int, int>> &collision_pairs) {
    if (particles.empty()) return;
    
    float dt = 0.001f; // Time step in seconds
    
    // Prepare particles for device - copy neighbors to fixed-size arrays
    for (auto &p : particles) {
        // Initialize fixed array with -1 (no neighbor)
        for (int i = 0; i < MAX_NEIGHBORS; i++) {
            p.neighbors[i] = -1;
        }
        p.num_neighbors = 0;
        
        // Copy from vector to fixed array (if vector exists)
#ifndef __CUDA_ARCH__
        int count = 0;
        for (const auto &neighbor_id : p.neighbor_vector) {
            if (count < MAX_NEIGHBORS) {
                p.neighbors[count++] = neighbor_id;
            } else {
                break;
            }
        }
        p.num_neighbors = count;
#endif
    }
    
    // Allocate device memory
    Particle *d_particles;
    Wall *d_walls;
    int2 *d_collision_pairs;
    
    cudaMalloc(&d_particles, particles.size() * sizeof(Particle));
    cudaMalloc(&d_walls, walls.size() * sizeof(Wall));
    cudaMalloc(&d_collision_pairs, collision_pairs.size() * sizeof(int2));
    
    // Copy particle and wall data to device
    cudaMemcpy(d_particles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_walls, walls.data(), walls.size() * sizeof(Wall), cudaMemcpyHostToDevice);
    
    // Convert collision pairs to CUDA int2 format and copy to device
    std::vector<int2> formatted_pairs(collision_pairs.size());
    for (size_t i = 0; i < collision_pairs.size(); i++) {
        formatted_pairs[i].x = collision_pairs[i].first;
        formatted_pairs[i].y = collision_pairs[i].second;
    }
    cudaMemcpy(d_collision_pairs, formatted_pairs.data(), collision_pairs.size() * sizeof(int2), cudaMemcpyHostToDevice);
    
    // Process particle movement and wall collisions
    int blockSize = 256;
    int particleBlocks = (particles.size() + blockSize - 1) / blockSize;
    dim3 particleGrid(particleBlocks), particleBlock(blockSize);
    simulateParticlesWithNeighborsKernel<<<particleGrid, particleBlock>>>(
        d_particles, particles.size(), d_walls, walls.size(), 
        d_collision_pairs, collision_pairs.size(), dt);
    
    // Process particle-particle collisions
    int collisionBlocks = (collision_pairs.size() + blockSize - 1) / blockSize;
    dim3 collisionGrid(collisionBlocks), collisionBlock(blockSize);
    processCollisionPairsKernel<<<collisionGrid, collisionBlock>>>(d_particles, d_collision_pairs, collision_pairs.size());
    
    // Copy results back
    cudaMemcpy(particles.data(), d_particles, particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_particles);
    cudaFree(d_walls);
    cudaFree(d_collision_pairs);
}

// Apply external force fields to particles within defined regions
void applyForceFields(std::vector<Particle> &particles, const std::vector<ForceField> &force_fields) {
    for (auto &particle : particles) {
        // Check each force field
        for (const auto &field : force_fields) {
            // Skip inactive force fields
            if (!field.active) {
                continue;
            }
            
            // Check if particle is within force field bounds
            bool in_x_range = (particle.position.x >= field.min_x && particle.position.x <= field.max_x);
            bool in_y_range = (particle.position.y >= field.min_y && particle.position.y <= field.max_y);
            bool in_z_range = (particle.position.z >= field.min_z && particle.position.z <= field.max_z);
            
            if (in_x_range && in_y_range && in_z_range) {
                // Particle is in the force field region, apply forces
                particle.state.force_x += field.force_x;
                particle.state.force_y += field.force_y;
                particle.state.force_z += field.force_z;
            }
        }
    }
}

// Combined simulation function with both force fields and neighbors
void simulateParticlesWithForcesAndNeighbors(
    std::vector<Particle> &particles, 
    const std::vector<Wall> &walls,
    const std::vector<std::pair<int, int>> &collision_pairs,
    const std::vector<ForceField> &force_fields) {
    
    // First apply all force fields
    applyForceFields(particles, force_fields);
    
    // Then proceed with normal simulation using neighbor info
    simulateParticlesWithNeighborInfo(particles, walls, collision_pairs);
}

