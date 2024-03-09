#!/bin/bash -l
#SBATCH --qos=default                      # SLURM qos
#SBATCH --nodes=3                          # number of nodes
#SBATCH --ntasks=3                        # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --time=00:05:00                    # time (HH:MM:SS)
#SBATCH --partition=fpga                   # partition
#SBATCH --account=p200301                  # project account
#SBATCH --cpus-per-task=8                  # CORES per task

module load ifpgasdk && module load 520nmx && module load CMake && module load intel && module load deploy/EasyBuild
cd build
git pull
make multi_node
srun multi_node /project/home/p200301/tests/matrix1000.bin /project/home/p200301/tests/rhs1000.bin output_mpi.bin 1000 1e-16
