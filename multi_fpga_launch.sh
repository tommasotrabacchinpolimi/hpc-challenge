#!/bin/bash -l
#SBATCH --qos=default                      # SLURM qos
#SBATCH --nodes=5                          # number of nodes
#SBATCH --ntasks=5                        # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --time=00:30:00                    # time (HH:MM:SS)
#SBATCH --partition=fpga                   # partition
#SBATCH --account=p200301                  # project account
#SBATCH --cpus-per-task=200                  # CORES per task

module load ifpgasdk && module load 520nmx && module load CMake && module load intel && module load deploy/EasyBuild
cd build
git pull
make multi_node
srun multi_node /project/home/p200301/tests/matrix10000.bin /project/home/p200301/tests/rhs10000.bin output_mpi.bin 10000 1e-16
srun multi_node /project/home/p200301/tests/matrix20000.bin /project/home/p200301/tests/rhs20000.bin output_mpi.bin 20000 1e-16
srun multi_node /project/home/p200301/tests/matrix30000.bin /project/home/p200301/tests/rhs30000.bin output_mpi.bin 30000 1e-16
srun multi_node /project/home/p200301/tests/matrix40000.bin /project/home/p200301/tests/rhs40000.bin output_mpi.bin 40000 1e-16
srun multi_node /project/home/p200301/tests/matrix50000.bin /project/home/p200301/tests/rhs50000.bin output_mpi.bin 50000 1e-16
