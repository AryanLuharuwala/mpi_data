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

namespace faiss
{
    namespace gpu
    {
        // GPU accelerate two-stage search
        void rangeSearchFaiss(GpuIndexFlatL2 *index, const std::vector<Particle> &all_particles, const std::vector<Particle> &query_particles,
                              float radius_sq, int k_nearest, // Number of nearest neighbours to search radius from
                              std::vector<std::pair<Particle, std::vector<Particle>>> &particle_neighbors)
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
            faiss::gpu::GpuIndexFlatL2 *gpu_index = index;

            // Prepare output distances and indices
            int num_queries = query_particles.size();
            std::vector<float> distances(num_queries * k_nearest);
            std::vector<faiss::idx_t> indices(num_queries * k_nearest);

            // Perform k_nearest neighbors search
            gpu_index->search(num_queries, query_positions.data(), k_nearest, distances.data(), indices.data());
            // Second stage: Filter results based on radius
            results.reserve(num_queries);
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

    // Receive the nested vector of integers
    std::vector<std::vector<std::pair<int, int>>> data;
    data = receiveData<std::vector<std::vector<std::pair<int, int>>>>(0);
    std::cout << "Worker " << rank << " received data:\n ";

    int cuda_size;

    cudaGetDeviceCount(&cuda_size);

    omp_set_num_threads(cuda_size);
    int size = omp_get_num_threads();
    sendData<int>(size, 0, 0);

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

        // Build Faiss Index
        auto faiss_index = buildFaissIndex(particles, thread_num);

        // particles initialized with config_particle

        std::vector<Particle> inzone, outzone;

        classifyInZone(particles, inzone, outzone, params, 4);

        //
        std::vector<std::pair<Particle, std::vector<Particle>>> inzone_particle_neighbors;
        // Perform range search for each domain
        inzone_particle_neighbors = searchNearestWithinRadius(
            faiss_index, inzone, particles, config_particle.properties.radius * 2.0f);

        // Here we process the outzone particles
        // First sent to master, where it is distributed to other domains
        // Each domain accumulates all the searches
        // compiles the results and sends back to the master
        // Master then sends the results to the respective domains
        // domain accumulate the results as there may be multiple domains returning results for the particle
        sendData<std::vector<Particle>>(outzone, 0, thread_num + 2);

        std::vector<std::pair<int, std::vector<Particle>>> search_queries = receiveData<std::vector<std::pair<int, std::vector<Particle>>>>(
            0, thread_num + 3); // Receive the outzone particles from master

        int num_queries = search_queries.size();
        std::vector<std::pair<int, std::vector<std::pair<Particle, std::vector<Particle>>>>> search_results(num_queries);
        // Prepare the search queries
        for (int i = 0; i < num_queries; ++i)
        {
            std::vector<Particle> search = search_queries[i].second;
            std::vector<std::pair<Particle, std::vector<Particle>>> result = searchNearestWithinRadius(
                faiss_index, particles, search, config_particle.properties.radius * 2.0f);
            search_results[i].first = search_queries[i].first; // Store the domain id
            search_results[i].second = result;                 // Store the results for this domain
        }
        // Send the search results back to the master
        sendData < std::vector<std::pair<int, std::vector<std::pair<Particle, std::vector<Particle>>>>>(
                       search_results, 0, thread_num + 4);

        // Receive the aggregated search results from the master
        std::map < Particle, std::vector < Particle >>> domain_wise_search_results_for_this_domain =
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
        // domain_wise_search_results_for_this_domain
        // Now we have the combined results for this domain
        // Combined_particle_neighbours
        // Run any simulation or processing on the combined results
        simulate(combined_particle_neighbors, particles, params);


        classifyInZone(particles, inzone, outzone, params, 0);
        // Send the outzone particles back to the master
        sendData<std::vector<Particle>>(outzone, 0, thread_num + 6);

        std::vector<Particle> particles_migrated;
        // Receive the particles to migrate from the master
        particles_migrated = receiveData<std::vector<Particle>>(0, thread_num + 7); 
        std::vector<Particle> migrated_inzone;
        classifyInzone(particles_migrated, migrated_inzone, nullptr, params, 0);
        particles.clear();
        particles.reserve(inzone.size() + migrated_inzone.size());
        // Combine the inzone particles with the migrated particles
        particles.insert(particles.end(), inzone.begin(), inzone.end());
        particles.insert(particles.end(), migrated_inzone.begin(), migrated_inzone.end());

        // next step
    }
}

// We can use this for zoning as well as domain search
void classifyInZone(
    const std::vector<Particle> &particles,
    std::vector<Particle> &inzone,
    std::vector<Particle> &outzone,
    const std::vector<float> &params,
    int threshold_multiplier = 4)
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
    faiss::gpu::GpuIndexFlatL2 *index,
    const std::vector<Particle> &all_particles,
    const std::vector<Particle> &query_particles,
    float search_radius,
    int k = 50) // Default to 50 nearest neighbors
{
    std::vector<std::pair<Particle, std::vector<Particle>>> results;

    float radius_sq = search_radius * search_radius;

    // Perform two-stage search
    faiss::gpu::rangeSearchFaiss(
        index, all_particles, query_particles,
        radius_sq, k, results);

    return results;
}

faiss::gpu::GpuIndexFlatL2 *buildFaissIndex(const std::vector<Particle> &particles, int device)
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
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;

    // Create L2 distance index for 3D points
    faiss::gpu::GpuIndexFlatL2 *gpu_index =
        new faiss::gpu::GpuIndexFlatL2(&gpu_resources, 3, config);

    // Add positions to index
    if (!position_data.empty())
    {
        gpu_index->add(particles.size(), position_data.data());
    }

    return gpu_index;
}

void initializeParticlesToConfig(std::vector<Particle> &particles, const Particle &config_particle)
{
    sendData<int>(particles.size(), 0, omp_get_thread_num());
    int id_start = receiveData<int>(0, omp_get_thread_num());

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