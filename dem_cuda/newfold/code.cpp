#include <mpi.h>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>    
#include <functional>

// Forward declarations
void masterProcess(int rank, int size);
void workerProcess(int rank, int size);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // Initialize MPI environment
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Hello from process " << rank << " of " << size << std::endl;

    if (rank == 0) {
        masterProcess(rank, size);
    } else {
        workerProcess(rank, size);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}