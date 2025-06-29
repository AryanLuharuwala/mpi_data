#include <iostream>
#include <mpi.h>
#include <vector>
#include "structs.h"
#include "dataUtils.h"
#include <omp.h>
#include <cuda_runtime.h>
#include "cuda.cuh"
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include "worker.h"

namespace faiss
{
    namespace gpu
    {
        // GPU accelerate two-stage search using IVFPQ index
        void rangeSearchFaiss(GpuIndexIVFPQ *index, const std::vector<Particle> &all_particles, const std::vector<Particle> &query_particles,
                              float radius_sq, int k_nearest, // Number of nearest neighbours to search radius from
                              std::vector<std::pair<Particle, std::vector<Particle>>> &particle_neighbors, 
                              int nprobe = 8) // Number of closest centroids to search
        {
            if (all_particles.empty() || query_particles.empty() || index->ntotal == 0)
            {
                return;
            }

            // Extract query positions
            std::vector<float> query_positions;
            query_positions.reserve(query_particles.size() * 3);

            for (const auto &p : query_particles)
            {
                query_positions.push_back(p.position.x);
                query_positions.push_back(p.position.y);
                query_positions.push_back(p.position.z);
            }

            // First stage: Find k nearest neighbours for each query particle
            faiss::gpu::GpuIndexIVFPQ *gpu_index = index;
            
            // Set nprobe for search (number of clusters to explore)
            int old_nprobe = gpu_index->nprobe;
            gpu_index->nprobe = nprobe;

            // Prepare output distances and indices
            int num_queries = query_particles.size();
            std::vector<float> distances(num_queries * k_nearest);
            std::vector<faiss::idx_t> indices(num_queries * k_nearest);
            
            std::cout << "debug: Performing k_nearest neighbors search with k = " << k_nearest 
                      << " and nprobe = " << nprobe << std::endl;
                      
            // Perform k_nearest neighbors search
            gpu_index->search(num_queries, query_positions.data(), k_nearest, distances.data(), indices.data());
            
            // Restore original nprobe value
            gpu_index->nprobe = old_nprobe;
            
            // Second stage: Filter results based on radius
            std::cout << "Filtering neighbors within radius squared: " << radius_sq << std::endl;
            particle_neighbors.reserve(num_queries);
            for (int i = 0; i < num_queries; ++i)
            {
                Particle query_particle = query_particles[i];
                std::vector<Particle> neighbors;
                for (int j = 0; j < k_nearest; ++j)
                {
                    int idx = indices[i * k_nearest + j];
                    if (idx < 0 || idx >= all_particles.size())
                    {
                        continue; // Skip invalid indices
                    }
                    if (all_particles[idx].position.id == query_particle.position.id)
                    {
                        continue; // Skip self-collisions
                    }
                    Particle neighbor_particle = all_particles[idx];
                    if (distances[i * k_nearest + j] <= radius_sq)
                    {
                        neighbors.push_back(neighbor_particle);
                    }
                }
                // Store results
                particle_neighbors.emplace_back(query_particle, std::move(neighbors));
            }
        }
    }
}

void workerProcess(int rank, int size)
{

    Particle config_particle;

    std::cout << "Worker " << rank << " started processing" << std::endl;
    int cuda_size;

    cudaGetDeviceCount(&cuda_size);
    int thr_size = omp_get_num_threads();
    std::cout<< thr_size << " threads available for worker " << rank << std::endl;
    omp_set_num_threads(cuda_size);
    MPI_Barrier(MPI_COMM_WORLD);
    sendData<int>(cuda_size, 0, 0);

#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        

        std::cout << "Thread " << thread_num << " processing data" << std::endl;

        cudaSetDevice(thread_num % cuda_size);

        sendData<std::pair<int, int>>({rank, thread_num}, 0, 1);

        std::vector<float> params;
        params.resize(8);

        // Process the data for this thread
        params = receiveData<std::vector<float>>(0, thread_num);

        std::vector<Particle> particles;
        initializeParticles(params, particles);

        initializeParticlesToConfig(particles, config_particle);

        notInit(particles, config_particle, params, thread_num);

        // Next step
    }
}

void notInit(std::vector<Particle> &particles, const Particle &config_particle, const std::vector<float> &params, int thread_num)
{

    // Build Faiss Index
    auto faiss_index = buildFaissIndex(particles, thread_num);

    // particles initialized with config_particle

    std::vector<Particle> inzone, outzone;

    classifyInZone(particles, inzone, outzone, params, 4);
    std::cout<< inzone.size() << " inzone particles and " << outzone.size() << " outzone particles for thread " << thread_num << std::endl;

    std::vector<std::pair<Particle, std::vector<Particle>>> inzone_particle_neighbors;
    // Perform range search for each domain
    inzone_particle_neighbors = searchNearestWithinRadius(
        faiss_index, inzone, particles, 0.5f * 2.0f);
    std::cout<< inzone_particle_neighbors.size() << " inzone particle neighbors found for thread " << thread_num << std::endl;
    // Here we process the outzone particles
    // First sent to master, where it is distributed to other domains
    // Each domain accumulates all the searches
    // compiles the results and sends back to the master
    // Master then sends the results to the respective domains
    // domain accumulate the results as there may be multiple domains returning results for the particle

    sendData<std::vector<Particle>>(outzone, 0, thread_num + 2);

    // Receive the outzone particles from master
    receiveAndProcessSearchQueryFromMaster(
        faiss_index, particles, params, thread_num + 3, thread_num + 4);

    // Receive the aggregated search results from the master
    std::map<Particle, std::vector<Particle>> domain_wise_search_results_for_this_domain =
        receiveData<std::map<Particle, std::vector<Particle>>>(0, thread_num + 5);

    std::vector<std::pair<Particle, std::vector<Particle>>> outzone_particle_neighbors;
    outzone_particle_neighbors = searchNearestWithinRadius(
        faiss_index, outzone, particles, config_particle.properties.radius * 2.0f);

    // Now we have the inzone and outzone particle neighbors
    // We can combine them into a single result set
    // outzone_particle_neighbors contains the neighbors for the outzone particles of this domain
    // inzone_particle_neighbors contains the neighbors for the inzone particles of this domain
    // domain_wise_search_results contains the neighbors for the inzone particles of other domains
    // We need to combine these results

    std::vector<std::pair<Particle, std::vector<Particle>>> combined_particle_neighbors;

    combineParticlesAcrossDomains(
        inzone_particle_neighbors, outzone_particle_neighbors, domain_wise_search_results_for_this_domain,
        combined_particle_neighbors);

    // domain_wise_search_results_for_this_domain
    // Now we have the combined results for this domain
    // Combined_particle_neighbours
    // Run any simulation or processing on the combined results

    //  simulate(combined_particle_neighbors, particles, params);
    std::cout<<"hii"<<std::endl;
    postProcessing(particles, params);
    delete faiss_index;    // Clean up the Faiss index
    faiss_index = nullptr; // Set to nullptr to avoid dangling pointer
                           // particles are now updated with the results of the simulation
}

void receiveAndProcessSearchQueryFromMaster(
    faiss::gpu::GpuIndexIVFPQ *faiss_index,
    const std::vector<Particle> &particles,
    const std::vector<float> &params,
    int receive_tag,
    int send_tag)
{
    std::vector<std::pair<int, std::vector<Particle>>> search_queries = receiveData<std::vector<std::pair<int, std::vector<Particle>>>>(
        0, receive_tag); // Receive the outzone particles from master

    int num_queries = search_queries.size();
    std::vector<std::pair<int, std::vector<std::pair<Particle, std::vector<Particle>>>>> search_results(num_queries);
    // Prepare the search queries
    for (int i = 0; i < num_queries; ++i)
    {
        std::vector<Particle> search = search_queries[i].second;
        std::vector<std::pair<Particle, std::vector<Particle>>> result = searchNearestWithinRadius(
            faiss_index, particles, search, particles[0].properties.radius * 2.0f);
        search_results[i].first = search_queries[i].first; // Store the domain id
        search_results[i].second = result;                 // Store the results for this domain
    }
    // Send the search results back to the master
    sendData<std::vector<std::pair<int, std::vector<std::pair<Particle, std::vector<Particle>>>>>>(
        search_results, 0, send_tag);
}

void combineParticlesAcrossDomains(
    const std::vector<std::pair<Particle, std::vector<Particle>>> &inzone_particle_neighbors,
    const std::vector<std::pair<Particle, std::vector<Particle>>> &outzone_particle_neighbors,
    const std::map<Particle, std::vector<Particle>> &domain_wise_search_results_for_this_domain,
    std::vector<std::pair<Particle, std::vector<Particle>>> &combined_particle_neighbors)
{
    //
    // Concatenate the outzone results with the inzone results
    for (const auto &result : inzone_particle_neighbors)
    {
        Particle query_particle = result.first;
        std::vector<Particle> neighbors = result.second;
        // Inzone and outzone are independent, so we can just add the neighbors
        combined_particle_neighbors.emplace_back(query_particle, std::move(neighbors));
    }
    for (const auto &result : outzone_particle_neighbors)
    {
        Particle query_particle = result.first;
        std::vector<Particle> neighbors = result.second;
        // domain_wise_search_results_for_this_domain
        // find the particle in domain_wise_search_results_for_this_domain

        auto it = domain_wise_search_results_for_this_domain.find(query_particle);
        if (it != domain_wise_search_results_for_this_domain.end())
        {
            // If the particle is found in the domain-wise search results, we can add its neighbors
            const auto &domain_neighbors = it->second;
            // Add each neighbor to the neighbors vector
            neighbors.insert(neighbors.end(), domain_neighbors.begin(), domain_neighbors.end());
        }
        // Inzone and outzone are independent, so we can just add the neighbors
        combined_particle_neighbors.emplace_back(query_particle, std::move(neighbors));
    }
}

void postProcessing(
    std::vector<Particle> &particles,
    const std::vector<float> &params)
{
    std::vector<Particle> inzone, outzone;
    int thread_num = omp_get_thread_num();
    classifyInZone(particles, inzone, outzone, params, 0);
    // Send the outzone particles back to the master
    sendData<std::vector<Particle>>(outzone, 0, thread_num + 6);

    std::vector<Particle> particles_migrated;
    // Receive the particles to migrate from the master
    particles_migrated = receiveData<std::vector<Particle>>(0, thread_num + 7);

    std::vector<Particle> migrated_inzone, migrated_outzone;

    classifyInZone(particles_migrated, migrated_inzone, migrated_outzone, params, 0);
    particles.clear();
    particles.reserve(inzone.size() + migrated_inzone.size());
    // Combine the inzone particles with the migrated particles
    particles.insert(particles.end(), inzone.begin(), inzone.end());
    particles.insert(particles.end(), migrated_inzone.begin(), migrated_inzone.end());
}

// We can use this for zoning as well as domain search
void classifyInZone(
    const std::vector<Particle> &particles,
    std::vector<Particle> &inzone,
    std::vector<Particle> &outzone,
    const std::vector<float> &params,
    int threshold_multiplier )
{
    // Params has the start x and x and start y and y and start z and z and particle radius
    // Using a threshold of 4 particle radius
    // if position is within the zone defined by pos_x <= start_x + x - radius * threshold_multiplier
    // and pos_y <= start_y + y - radius * threshold_multiplier
    // and pos_z <= start_z +z - radius * threshold_multiplier
    // pos_x >= start_x + radius * threshold_multiplier
    // and pos_y >= start_y + radius * threshold_multiplier
    // and pos_z >= start_z + radius * threshold_multiplier
    float start_x = params[0];
    float start_y = params[1];
    float start_z = params[2];
    float x = params[3];
    float y = params[4];
    float z = params[5];
    float radius = params[6];
    float threshold = radius * threshold_multiplier;
    float end_x = start_x + x - threshold;
    float end_y = start_y + y - threshold;
    float end_z = start_z + z - threshold;
    inzone.clear();
    outzone.clear();
    if (particles.empty())
    {
        return; // No particles to classify
    }
    for (const auto &particle : particles)
    {
        if (particle.position.x >= start_x + threshold &&
            particle.position.x <= end_x &&
            particle.position.y >= start_y + threshold &&
            particle.position.y <= end_y &&
            particle.position.z >= start_z + threshold &&
            particle.position.z <= end_z)
        {
            inzone.push_back(particle);
        }
        else
        {
            outzone.push_back(particle);
        }
    }
}

std::vector<std::pair<Particle, std::vector<Particle>>> searchNearestWithinRadius(
    faiss::gpu::GpuIndexIVFPQ *index,
    const std::vector<Particle> &all_particles,
    const std::vector<Particle> &query_particles,
    float search_radius,
    int k) // Default to 50 nearest neighbors
{
    std::vector<std::pair<Particle, std::vector<Particle>>> results;

    float radius_sq = search_radius * search_radius;
    std::cout << "Searching within radius: " << search_radius << " (squared: " << radius_sq << ")" << std::endl;
    // Perform two-stage search
    faiss::gpu::rangeSearchFaiss(
        index, all_particles, query_particles,
        radius_sq, k, results, 8);

    return results;
}

faiss::gpu::GpuIndexIVFPQ *buildFaissIndex(const std::vector<Particle> &particles, int device)
{
    // Extract positions to a flat array for FAISS
    std::vector<float> position_data;
    position_data.reserve(particles.size() * 3);

    for (const auto &particle : particles)
    {
        position_data.push_back(particle.position.x);
        position_data.push_back(particle.position.y);
        position_data.push_back(particle.position.z);
    }

    // Create GPU resources (make static so it persists)
    static faiss::gpu::StandardGpuResources gpu_resources;

    // Configure GPU index
    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = device;
    
    // Define IVFPQ parameters
    int nlist = std::min(256, int(particles.size() / 30)); // Number of clusters/centroids (adjust based on data size)
    int m = 1;      // Number of subquantizers (limited by small dimension)
    int bits = 8;   // Bits per subquantizer (usually 8)

    // For small dimensions (d=3), we need to be careful with parameters
    if (nlist < 10) nlist = 10; // Ensure minimum number of clusters

    // Create the IVFPQ index directly - the quantizer is created internally
    faiss::gpu::GpuIndexIVFPQ *gpu_index = 
        new faiss::gpu::GpuIndexIVFPQ(&gpu_resources, 
                                     3,              // dimension
                                     nlist,          // number of centroids
                                     m,              // number of subquantizers
                                     bits,           // bits per subquantizer
                                     faiss::METRIC_L2,
                                     config);
    
    // Add positions to index
    if (!position_data.empty()) {
        // Train the index first (required for IVFPQ)
        gpu_index->train(particles.size(), position_data.data());
        
        // Add the vectors
        gpu_index->add(particles.size(), position_data.data());
        
        std::cout << "Built IVFPQ index with " << nlist << " clusters, " 
                  << m << " subquantizers, and " << bits << " bits" << std::endl;
    } else {
        std::cout << "Warning: No particles to index" << std::endl;
    }

    return gpu_index;
}

void initializeParticlesToConfig(std::vector<Particle> &particles, const Particle &config_particle)
{
    sendData<int>(particles.size(), 0, omp_get_thread_num());
    int id_start = receiveData<int>(0, omp_get_thread_num());
    std::cout<<particles[1000].position.x << " " << particles[1000].position.y << " " << particles[1000].position.z << std::endl;
    for (int i = 0; i < particles.size(); i++)
    {
        particles[i].position.id = id_start + i;
        particles[i].state.velocity_x = config_particle.state.velocity_x;
        particles[i].state.velocity_y = config_particle.state.velocity_y;
        particles[i].state.velocity_z = config_particle.state.velocity_z;
        particles[i].state.acceleration_x = config_particle.state.acceleration_x;
        particles[i].state.acceleration_y = config_particle.state.acceleration_y;
        particles[i].state.acceleration_z = config_particle.state.acceleration_z;
        particles[i].state.force_x = config_particle.state.force_x;
        particles[i].state.force_y = config_particle.state.force_y;
        particles[i].state.force_z = config_particle.state.force_z;
        particles[i].properties.mass = config_particle.properties.mass;
        particles[i].properties.radius = config_particle.properties.radius;
        particles[i].properties.restitution = config_particle.properties.restitution;
        particles[i].properties.friction = config_particle.properties.friction;
        particles[i].properties.young_modulus = config_particle.properties.young_modulus;
        particles[i].properties.shear_modulus = config_particle.properties.shear_modulus;
        particles[i].properties.damping = config_particle.properties.damping;
        particles[i].properties.stiffness = config_particle.properties.stiffness;
        particles[i].num_neighbors = 0; // Initialize neighbor count
        for (int j = 0; j < MAX_NEIGHBORS; j++)
        {
            particles[i].neighbors[j] = -1; // Initialize neighbors to -1
        }
    }
}