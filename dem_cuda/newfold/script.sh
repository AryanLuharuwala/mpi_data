
#!/bin/bash

#SBATCH -J particle_sim            # job name
#SBATCH -p gpu                     # partition name
#SBATCH -N 1                       # number of nodes (3: one master + 3 workers)
#SBATCH --ntasks=2                 # total MPI processes (3 workers + 1 master)
#SBATCH --ntasks-per-node=2        # MPI ranks per node
#SBATCH --gres=gpu:2               # GPUs per node (2 GPUs per worker node)
##SBATCH --cpus-per-task=2          # CPU cores per MPI process (2 threads each)
#SBATCH -t 02:00:00                # walltime (2 hours)
#SBATCH -o particle_sim_%j.out     # output file
#SBATCH -e particle_sim_%j.err     # error file


# Load required modules
module load compiler/cuda
module load compiler/intel-mpi/mpi-2020-v4 
module load compiler/gcc

 #Set environment variables
export PATH=/opt/ohpc/pub/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/home/iitkgp01/faiss:$LD_LIBRARY_PATH
cd /home/iitkgp01/folder1
make

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# Application executable
exe=./mpi_dem

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of MPI processes: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs requested: $SLURM_GPUS_ON_NODE per node"

# Run the distributed particle simulator
mpirun -bootstrap slurm -n $SLURM_NTASKS $exe \
   
