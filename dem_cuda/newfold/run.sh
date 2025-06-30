#!/bin/bash

#SBATCH -J particle_sim            # job name
#SBATCH -p gpu                     # partition name
#SBATCH -N 2                       # number of nodes (3: one master + 3 workers)
#SBATCH --ntasks=2                 # total MPI processes (3 workers + 1 master)
#SBATCH --ntasks-per-node=1        # MPI ranks per node
#SBATCH --gres=gpu:2               # GPUs per node (2 GPUs per worker node)
##SBATCH --cpus-per-task=2          # CPU cores per MPI process (2 threads each)
#SBATCH -t 01:00:00                # walltime (2 hours)
#SBATCH -o particle_sim_%j.out     # output file
#SBATCH -e particle_sim_%j.err     # error file
#SBATCH --reservation hackathon

# Load required modules
module load lib/intel/2022/tbb/oneapi-2021.5.1
module load compiler/cuda
module load compiler/intel-mpi/mpi-2020-v4 
module load lib/intel/2022/mkl/oneapi-2022.0.2
module load compiler/gcc/12.3.0

 #Set environment variables
export PATH=/opt/ohpc/pub/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/home/iitkgp01/faiss:$LD_LIBRARY_PATH
cd /home/iitkgp01/newfold/newfold
export LD_LIBRARY_PATH=/home/iitkgp01/libs/lapack_build/lapack/lib/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/ohpc/pub/cuda-12.2/lib64:$LD_LIBRARY_PATH
source /home/opt_ohpc_pub/apps/intel/oneapi/2022/setvars.sh


# detect MPI C++ compiler
if command -v mpiicpc >/dev/null 2>&1; then
  echo "Using Intel MPI compiler"
  MAKE_OPTS="USE_INTEL=1"
elif command -v mpiicpc >/dev/null 2>&1; then
  echo "Using default MPI compiler"
  MAKE_OPTS=""
else
  echo "Error: no MPI C++ compiler found."
  exit 1
fi

# build
make clean
make $MAKE_OPTS || { echo "Build failed"; exit 1; }


# run with NP ranks (override with NP env var)
echo $SLURM_NTASKS


# srun --mpi=pmix -n $NP $(pwd)/bin/mpi_dem "$@"


export MKLROOT=/home/opt_ohpc_pub/apps/intel/oneapi/2022/mkl/2022.0.2
export LD_LIBRARY_PATH=/home/opt_ohpc_pub/apps/intel/oneapi/2022/mkl/2022.0.2/lib/ia32:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\
$MKLROOT/lib/intel64:\
$FAISS_PATH/faiss:\
$CUDA_PATH/lib64:\
/home/opt_ohpc_pub/apps/intel/oneapi/2022/compiler/2022.0.2/linux/compiler/lib/intel64_lin:\
/home/opt_ohpc_pub/apps/intel/oneapi/2022/mpi/latest/lib/release:\
/home/opt_ohpc_pub/apps/intel/oneapi/2022/mpi/latest/libfabric/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/iitkgp01/libs/lapack_build/lapack/lib:$LD_LIBRARY_PATH

module load compiler/gcc/12.3.0



echo "Starting work"


mpirun -n $SLURM_NTASKS $(pwd)/bin/mpi_dem "$@"
mpirun -bootstrap slurm -n $SLURM_NTASKS $(pwd)/bin/mpi_dem "$@" 

