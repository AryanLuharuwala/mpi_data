#ifndef MASTER_H
#define MASTER_H

#include <vector>
#include <map>
#include "structs.h"

/**
 * Main function for the master process to coordinate the simulation
 * @param rank The MPI rank of the process
 * @param size The total number of MPI processes
 */
void masterProcess(int rank, int size);

/**
 * Helper function to facilitate particle migration between domains
 * @param gpu_map Map of GPU IDs to their rank and thread
 */
void migrationHelper(const std::map<int, std::pair<int, int>> &gpu_map);

/**
 * Sends domain parameters and assigns unique particle IDs
 * @param domain_params Map of domain parameters for each GPU
 * @param gpu_map Map of GPU IDs to their rank and thread
 */
void sendDomainParametersAndUniqueParticleId(const std::map<int, std::vector<float>> &domain_params,
                                             const std::map<int, std::pair<int, int>> &gpu_map);

/**
 * Receives search results from GPUs, aggregates them, and sends back to respective domains
 * @param gpu_map Map of GPU IDs to their rank and thread
 */
void recieveSearchedResultsAndSendToOrigin(const std::map<int, std::pair<int, int>> &gpu_map);

/**
 * Finds a particle in a vector by its ID
 * @param begin Iterator to the beginning of the vector
 * @param end Iterator to the end of the vector
 * @param particle The particle to find
 * @return Iterator to the found particle or end if not found
 */
std::vector<Particle>::iterator findParticle(std::vector<Particle>::iterator begin,
                                             std::vector<Particle>::iterator end,
                                             const Particle &particle);

/**
 * Receives particle data from each GPU and distributes to other domains
 * @param gpu_map Map of GPU IDs to their rank and thread
 */
void receiveParticleDataAndSendToDomains(const std::map<int, std::pair<int, int>> &gpu_map);

/**
 * Finds the optimal grid dimensions for domain decomposition
 * @param num_domains The number of domains (GPUs)
 * @return A pair containing the number of rows and columns
 */
std::pair<int, int> findGridDimensions(int num_domains);

/**
 * Decomposes the simulation domain across available GPUs
 * @param domain_params Output map of domain parameters for each GPU
 * @param gpu_map Map of GPU IDs to their rank and thread
 * @param domain_height Height of the overall domain
 * @param domain_width Width of the overall domain
 * @param domain_depth Depth of the overall domain
 * @param particle_radius Radius of particles
 */
void domainDecomposition(std::map<int, std::vector<float>> &domain_params, 
                         std::map<int, std::pair<int, int>> &gpu_map, 
                         int domain_height, int domain_width, int domain_depth, 
                         float particle_radius);

/**
 * Validates the domain dimensions and particle radius
 * @param domain_height Height of the domain
 * @param domain_width Width of the domain
 * @param domain_depth Depth of the domain
 * @param particle_radius Radius of particles
 */
void checkDomainValidity(int domain_height, int domain_width, int domain_depth, 
                         float particle_radius);

/**
 * Collects information about available GPUs across MPI processes
 * @param gpu_map Output map of GPU IDs to their rank and thread
 * @param size The total number of MPI processes
 */
void howManyGpus(std::map<int, std::pair<int, int>> &gpu_map, int size);

#endif // MASTER_H
