#ifndef WORKER_H
#define WORKER_H

#include <iostream>
#include <mpi.h>
#include <vector>
#include <map>
#include <utility>
#include <omp.h>
#include <cuda_runtime.h>
#include "structs.h"
#include "dataUtils.h"
#include "cuda.cuh"
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

// Forward declarations
void initializeParticles(const std::vector<float>& params, std::vector<Particle>& particles);

namespace faiss {
namespace gpu {
    // GPU accelerate two-stage search
    void rangeSearchFaiss(GpuIndexIVFPQ* index, const std::vector<Particle>& all_particles,
                         const std::vector<Particle>& query_particles, float radius_sq, int k_nearest,
                         std::vector<std::pair<Particle, std::vector<Particle>>>& particle_neighbors);
}
}

// Function declarations
void workerProcess(int rank, int size);

void notInit(std::vector<Particle>& particles, const Particle& config_particle, 
            const std::vector<float>& params, int thread_num);

void receiveAndProcessSearchQueryFromMaster(
    faiss::gpu::GpuIndexIVFPQ* faiss_index,
    const std::vector<Particle>& particles,
    const std::vector<float>& params,
    int receive_tag,
    int send_tag);

void combineParticlesAcrossDomains(
    const std::vector<std::pair<Particle, std::vector<Particle>>>& inzone_particle_neighbors,
    const std::vector<std::pair<Particle, std::vector<Particle>>>& outzone_particle_neighbors,
    const std::map<Particle, std::vector<Particle>>& domain_wise_search_results_for_this_domain,
    std::vector<std::pair<Particle, std::vector<Particle>>>& combined_particle_neighbors);

void postProcessing(
    std::vector<Particle>& particles,
    const std::vector<float>& params);

void classifyInZone(
    const std::vector<Particle>& particles,
    std::vector<Particle>& inzone,
    std::vector<Particle>& outzone,
    const std::vector<float>& params,
    int threshold_multiplier = 4);

std::vector<std::pair<Particle, std::vector<Particle>>> searchNearestWithinRadius(
    faiss::gpu::GpuIndexIVFPQ* index,
    const std::vector<Particle>& all_particles,
    const std::vector<Particle>& query_particles,
    float search_radius,
    int k = 50);

faiss::gpu::GpuIndexIVFPQ *buildFaissIndex(const std::vector<Particle>& particles, int device);

void initializeParticlesToConfig(std::vector<Particle>& particles, const Particle& config_particle);

#endif // WORKER_H
