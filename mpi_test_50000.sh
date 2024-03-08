#!/bin/bash -l
#SBATCH --qos=default                      # SLURM qos
#SBATCH --nodes=5                         # number of nodes
#SBATCH --ntasks=5                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --time=10:00:00                    # time (HH:MM:SS)
#SBATCH --partition=cpu                    # partition
#SBATCH --account=p200301                  # project account
#SBATCH --cpus-per-task=256                # CORES per task

module load ifpgasdk && module load 520nmx && module load CMake && module load intel && module load deploy/EasyBuild
cd build
make mpi

srun mpi /project/home/p200301/tests/matrix50000.bin /project/home/p200301/tests/rhs50000.bin output_mpi.bin 50000 1e-16