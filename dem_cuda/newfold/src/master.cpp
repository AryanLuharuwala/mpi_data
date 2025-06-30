#include <mpi.h>
#include <vector>
#include <type_traits>
#include "../include/dataUtils.h"
#include <iostream>
#include <cmath>
#include "../include/structs.h"
#include <algorithm>
#include "../include/master.h"

void masterProcess(int rank, int size)
{

    MPI_Barrier(MPI_COMM_WORLD);
    
    std::map<int, std::pair<int, int>> gpu_map;
    
    howManyGpus(gpu_map, size);
    
    std::cout << "Number of GPUs: " << gpu_map.size() << std::endl;
    int domain_height = 100;
    int domain_width = 100;
    int domain_depth = 100;
    float particle_radius = 0.5f;


        printSpaced(5);

    checkDomainValidity(domain_height, domain_width, domain_depth, particle_radius);
    
        printSpaced(5);
    
    std::map<int, std::vector<float>> domain_params;
    // domain params if of the form {x_start, y_start, z_start, x_size, y_size, z_size, particle_radius}

    domainDecomposition(domain_params, gpu_map, domain_height, domain_width, domain_depth, particle_radius);
    
        printSpaced(5);
    
    // Send the domain parameters to each GPU
    sendDomainParametersAndUniqueParticleId(domain_params, gpu_map);
    
        printSpaced(5);
    
    receiveParticleDataAndSendToDomains(gpu_map);
    
        printSpaced(5);
    
    // Recieve the searched results and accumulate then send to respective GPUs
    recieveSearchedResultsAndSendToOrigin(gpu_map);
    
        printSpaced(5);
    
    migrationHelper(gpu_map);
    
        printSpaced(5);
}

void printSpaced(int loop)
{
    for (int i = 0; i < loop; i++)
    {
        std::cout << std::endl;
    }
}
void migrationHelper(const std::map<int, std::pair<int, int>> &gpu_map)
{

    std::vector<Particle> all_migration_particles;
    // Migration helper
    for (const auto &pair : gpu_map)
    {
        int gpu_id = pair.first;
        int gpu_rank = pair.second.first;
        int gpu_thread = pair.second.second;

        // Receive the outzone particles from each GPU
        std::vector<Particle> outzone_particles = receiveData<std::vector<Particle>>(gpu_rank, gpu_thread + 6);
        // Process the outzone particles as needed
        all_migration_particles.insert(all_migration_particles.end(), outzone_particles.begin(), outzone_particles.end());
        // For example, you can send them to the master or perform further processing
        std::cout << "Received " << outzone_particles.size() << " outzone particles from GPU " << gpu_id << std::endl;
    }
    // Send the outzone particles back to each GPU
    for (const auto &pair : gpu_map)
    {
        int gpu_id = pair.first;
        int gpu_rank = pair.second.first;
        int gpu_thread = pair.second.second;
        // Send the outzone particles back to the master
        sendData<std::vector<Particle>>(all_migration_particles, gpu_rank, gpu_thread + 7);
        std::cout << "Sent outzone particles to GPU " << gpu_id << " with rank " << gpu_rank << " and thread " << gpu_thread << std::endl;
    }
}

void sendDomainParametersAndUniqueParticleId(const std::map<int, std::vector<float>> &domain_params,
                                             const std::map<int, std::pair<int, int>> &gpu_map)
{
    int num_particles = 0;
    for (const auto &pair : gpu_map)
    {
        int gpu_id = pair.first;
        int gpu_rank = pair.second.first;
        int gpu_thread = pair.second.second;
        // Receive the number of particles from each GPU
        int particles = receiveData<int>(gpu_rank, gpu_thread);
        sendData<int>(num_particles, gpu_rank, gpu_thread);
        num_particles += particles;
        std::cout << "GPU " << gpu_id << " has " << num_particles << " particles." << std::endl;
    }
    std::cout << "Total number of particles across all domains: " << num_particles << std::endl;
}

void recieveSearchedResultsAndSendToOrigin(const std::map<int, std::pair<int, int>> &gpu_map)
{
    std::map<int, std::map<Particle, std::vector<Particle>>> domain_wise_search_results;
    // Receive search results from each GPU
    for (const auto &pair : gpu_map)
    {
        int gpu_id = pair.first;
        int gpu_rank = pair.second.first;
        int gpu_thread = pair.second.second;
        std::cout << "Receiving search results from GPU " << gpu_id << " with rank " << gpu_rank << " and thread " << gpu_thread << std::endl;
        // Receive the search results from each GPU
        std::vector<std::pair<int, std::vector<std::pair<Particle, std::vector<Particle>>>>> search_results =
            receiveData<std::vector<std::pair<int, std::vector<std::pair<Particle, std::vector<Particle>>>>>>(gpu_rank, gpu_thread + 4);
        std::cout << "Received search results from GPU " << gpu_id << " with rank " << gpu_rank << " and thread " << gpu_thread << std::endl;
        // Process the search results as needed
        // there are 1 2 , 2 3, 1 3 data in this
        // all of 1 need to be compiled and sent to 1
        // all of 2 need to be compiled and sent to 2
        // all of 3 need to be compiled and sent to 3
        // Find whether the particle neighbour is already present as a neighbour in the domain
        for (const auto &result : search_results)
        {
            int domain_id = result.first;
            const auto &particle_results = result.second;
            if (domain_wise_search_results.find(domain_id) == domain_wise_search_results.end())
            {
                domain_wise_search_results[domain_id] = std::map<Particle, std::vector<Particle>>();
            }
            // Iterate through the particle results
            for (const auto &particle_result : particle_results)
            {
                const Particle &particle = particle_result.first;
                const std::vector<Particle> &neighbours = particle_result.second;
                if (domain_wise_search_results[domain_id].find(particle) == domain_wise_search_results[domain_id].end())
                {
                    // If the particle is not already present, initialize its neighbour vector
                    domain_wise_search_results[domain_id][particle] = std::vector<Particle>();
                }
                for (const auto &neighbour : neighbours)
                {
                    if (findParticle(domain_wise_search_results[domain_id][particle].begin(),
                                     domain_wise_search_results[domain_id][particle].end(), neighbour) ==
                        domain_wise_search_results[domain_id][particle].end())
                    { // If the neighbour is not already present, add it
                      // Add the neighbour to the particle's neighbour vector
                      // This ensures that we do not add duplicate neighbours
                      // and that we only add unique neighbours for each particle
                      // This is important as we are using a map to store the neighbours
                      // and we want to avoid duplicate entries
                        domain_wise_search_results[domain_id][particle].push_back(neighbour);
                    }
                }
            }
        }
    }
    // Send the aggregated search results back to each GPU
    for (const auto &pair : gpu_map)
    {
        int gpu_id = pair.first;
        int gpu_rank = pair.second.first;
        int gpu_thread = pair.second.second;
        // Send the aggregated search results back to the master
        sendData<std::map<Particle, std::vector<Particle>>>(domain_wise_search_results[gpu_id], gpu_rank, gpu_thread + 5);
        std::cout << "Sent aggregated search results from GPU " << gpu_id << " with rank " << gpu_rank << " and thread " << gpu_thread << std::endl;
    }
}

std::vector<Particle>::iterator findParticle(std::vector<Particle>::iterator begin,
                                             std::vector<Particle>::iterator end,
                                             const Particle &particle)
{
    // Find the particle in the vector
    return std::find_if(begin, end, [&particle](const Particle &p)
                        { return p.position.id == particle.position.id; });
    // return 0;
}

void receiveParticleDataAndSendToDomains(const std::map<int, std::pair<int, int>> &gpu_map)
{
    // Receive particle data from each GPU
    for (const auto &pair : gpu_map)
    {
        std::vector<std::pair<int, std::vector<Particle>>> particle_to_send_to_domains;
        int gpu_id = pair.first;
        int gpu_rank = pair.second.first;
        int gpu_thread = pair.second.second;

        // Receive the number of particles from each GPU
        std::vector<Particle> num_particles = receiveData<std::vector<Particle>>(gpu_rank, gpu_thread + 2);

        for (const auto &pair_2 : gpu_map)
        {
            if (pair_2 != pair)
            {
                int other_gpu_id = pair_2.first;
                int other_gpu_rank = pair_2.second.first;
                int other_gpu_thread = pair_2.second.second;

                particle_to_send_to_domains.push_back({other_gpu_id, num_particles});
                //     // Send the number of particles to the other GPU
                //     sendData<std::vector<Particle>>(num_particles, other_gpu_rank, other_gpu_thread + 3);
            }
        }
        // Send the particle data to each GPU
        sendData<std::vector<std::pair<int, std::vector<Particle>>>>(particle_to_send_to_domains, gpu_rank, gpu_thread + 3);
        std::cout << "Sent particle data to GPU " << gpu_id << " with rank " << gpu_rank << " and thread " << gpu_thread << std::endl;
    }
}

std::pair<int, int> findGridDimensions(int num_domains)
{
    // Calculate the number of rows and columns for the grid
    int sqrt = static_cast<int>(std::sqrt(num_domains));
    for (int i = sqrt; i > 0; --i)
    {
        if (num_domains % i == 0)
        {
            int rows = i;
            int cols = num_domains / i;
            std::cout << "Grid dimensions: " << rows << " rows, " << cols << " columns." << std::endl;
            return {rows, cols};
        }
    }
    throw std::runtime_error("Failed to find grid dimensions for the number of domains.");
}

void domainDecomposition(std::map<int, std::vector<float>> &domain_params, std::map<int, std::pair<int, int>> &gpu_map, int domain_height, int domain_width, int domain_depth, float particle_radius)
{
    // Calculate the number of domains
    int num_domains = gpu_map.size();
    if (num_domains == 0)
    {
        throw std::runtime_error("No GPUs detected for domain decomposition.");
    }

    int rows, cols;
    std::pair<int, int> grid_dimensions = findGridDimensions(num_domains);
    rows = grid_dimensions.first;
    cols = grid_dimensions.second;

    float x = static_cast<float>(domain_width) / cols;
    float y = static_cast<float>(domain_height) / rows;
    float z = static_cast<float>(domain_depth) / 1; // Assuming depth is not divided for simplicity
    checkDomainValidity(x, y, z, particle_radius);
    std::cout << "Domain decomposition parameters: x=" << x << ", y=" << y << ", z=" << z << ", particle_radius=" << particle_radius << std::endl;
    std::cout << "Height: " << domain_height << ", Width: " << domain_width << ", Depth: " << domain_depth << std::endl;

    // Domain parameters for each GPU
    domain_params.clear();

    for (const auto &pair : gpu_map)
    {
        int gpu_id = pair.first;
        int gpu_rank = pair.second.first;
        int gpu_thread = pair.second.second;

        // Calculate the domain boundaries for each GPU
        float x_start = (gpu_id % cols) * x;
        float y_start = (gpu_id / cols) * y;
        float z_start = 0.0f; // Assuming z starts from 0

        int num_particles_x = static_cast<int>(std::ceil(x / particle_radius));
        int num_particles_y = static_cast<int>(std::ceil(y / particle_radius));
        int num_particles_z = static_cast<int>(std::ceil(z / particle_radius));
        float total_particles = num_particles_x * num_particles_y * num_particles_z;

        std::vector<float> params = {x_start, y_start, z_start, x, y, z, particle_radius, total_particles};
        domain_params[gpu_id] = params;

        // Send domain parameters to each GPU
        sendData<std::vector<float>>(params, gpu_rank, gpu_thread);
    }
}

void checkDomainValidity(int domain_height, int domain_width, int domain_depth, float particle_radius)
{
    if (domain_height <= 0 || domain_width <= 0 || domain_depth <= 0)
    {
        throw std::invalid_argument("Domain dimensions must be positive.");
    }
    if (particle_radius <= 0)
    {
        throw std::invalid_argument("Particle radius must be positive.");
    }
    // domain should be multiple of particle_radius
    // reduce domain_height, domain_width, domain_depth to valid values
    domain_height = static_cast<int>(domain_height / particle_radius);
    domain_width = static_cast<int>(domain_width / particle_radius);
    domain_depth = static_cast<int>(domain_depth / particle_radius);
    domain_height *= particle_radius;
    domain_width *= particle_radius;
    domain_depth *= particle_radius;
}

void howManyGpus(std::map<int, std::pair<int, int>> &gpu_map, int size)
{
    int id = 0;
    for (int i = 1; i < size; i++)
    {
        int count = 0;
        count = receiveData<int>(i, 0);
        while (count--)
        {
            auto r = receiveData<std::pair<int, int>>(i, 1);
            gpu_map.insert({id, {r.first, r.second}});
            id++;
        }
    }
    std::cout << "Master process detected " << gpu_map.size() << " GPUs." << std::endl;
}