#ifndef STRUCTS_H
#define STRUCTS_H

struct Particle_position {
    float x;  // x position
    float y;  // y position
    float z;  // z position
    int id;   // particle id
};

struct Particle_state {
    float velocity_x;  // x component of velocity
    float velocity_y;  // y component of velocity
    float velocity_z;  // z component of velocity
    float acceleration_x;  // x component of acceleration
    float acceleration_y;  // y component of acceleration   
    float acceleration_z;  // z component of acceleration
    float force_x;  // x component of force
    float force_y;  // y component of force
    float force_z;  // z component of force
};

struct Particle_properties {
    float mass;  // mass of the particle
    float radius;  // radius of the particle
    float restitution;  // coefficient of restitution
    float friction;  // coefficient of friction
    float young_modulus;  // Young's modulus
    float shear_modulus;  // shear modulus
    float damping;  // damping factor
    float stiffness;  // stiffness of the particle
};

// Maximum number of neighbors to track per particle
#define MAX_NEIGHBORS 24  // Reduced to avoid using too much shared memory in CUDA kernels

struct Particle {
    Particle_position position;  // position of the particle
    Particle_state state;  // state of the particle
    Particle_properties properties;  // properties of the particle
    
    // Device-compatible neighbor storage (fixed size array)
    int neighbors[MAX_NEIGHBORS];  // list of neighboring particle ids
    int num_neighbors;            // actual number of neighbors
    
    // Host-only neighbor storage (used only on CPU)
#ifndef __CUDA_ARCH__
    std::vector<int> neighbor_vector;  // dynamic neighbor storage for CPU code
#endif

    // Comparison operator for using Particle in ordered containers
    bool operator<(const Particle& other) const {
        // Compare based on ID first
        if (position.id != other.position.id) {
            return position.id < other.position.id;
        }
        
        // If IDs are the same, compare positions
        if (position.x != other.position.x) {
            return position.x < other.position.x;
        }
        if (position.y != other.position.y) {
            return position.y < other.position.y;
        }
        return position.z < other.position.z;
    }
    
    // Equality operator for comparison
    bool operator==(const Particle& other) const {
        return position.id == other.position.id &&
               position.x == other.position.x &&
               position.y == other.position.y &&
               position.z == other.position.z;
    }
};

struct Wall {
    float normal_x;  // x component of the normal vector
    float normal_y;  // y component of the normal vector
    float normal_z;  // z component of the normal vector
    float centre_x;  // x coordinate of the center
    float centre_y;  // y coordinate of the center
    float centre_z;  // z coordinate of the center
    int id;
};

// Structure to define a force field that applies to particles within a region
struct ForceField {
    // Region boundaries (min/max for each dimension)
    float min_x, max_x;
    float min_y, max_y;
    float min_z, max_z;
    
    // Force components
    float force_x, force_y, force_z;
    
    // Optional: Duration or activation time
    float start_time;
    float duration;
    bool active;
};

// MPI data transfer structures for more efficient communication
struct ParticleTransfer {
    int source_domain;
    int target_domain;
    std::vector<Particle> particles;
};

// Serialization helper for MPI transfers (makes particles vector serializable)
struct SerializableParticleBuffer {
    int count;
    std::vector<Particle> particles;
};

#endif // STRUCTS_H