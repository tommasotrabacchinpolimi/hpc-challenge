#!/bin/bash -l
#SBATCH --qos=default                      # SLURM qos
#SBATCH --nodes=50                         # number of nodes
#SBATCH --ntasks=50                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --time=03:00:00                    # time (HH:MM:SS)
#SBATCH --partition=cpu                    # partition
#SBATCH --account=p200301                  # project account
#SBATCH --cpus-per-task=256                # CORES per task

module load ifpgasdk && module load 520nmx && module load CMake && module load intel && module load deploy/EasyBuild
cd build
make mpi



srun mpi /project/home/p200301/tests/matrix10000.bin /project/home/p200301/tests/rhs10000.bin output_mpi.bin 50000 1e-16 50000
srun mpi /project/home/p200301/tests/matrix10000.bin /project/home/p200301/tests/rhs10000.bin output_mpi.bin 70000 1e-16 70000
srun mpi /project/home/p200301/tests/matrix10000.bin /project/home/p200301/tests/rhs10000.bin output_mpi.bin 80000 1e-16 80000
srun mpi /project/home/p200301/tests/matrix10000.bin /project/home/p200301/tests/rhs10000.bin output_mpi.bin 90000 1e-16 90000
srun mpi /project/home/p200301/tests/matrix10000.bin /project/home/p200301/tests/rhs10000.bin output_mpi.bin 100000 1e-16 100000
srun mpi /project/home/p200301/tests/matrix10000.bin /project/home/p200301/tests/rhs10000.bin output_mpi.bin 150000 1e-16 150000
srun mpi /project/home/p200301/tests/matrix20000.bin /project/home/p200301/tests/rhs20000.bin output_mpi.bin 200000 1e-16 200000
srun mpi /project/home/p200301/tests/matrix30000.bin /project/home/p200301/tests/rhs30000.bin output_mpi.bin 300000 1e-16 300000
srun mpi /project/home/p200301/tests/matrix40000.bin /project/home/p200301/tests/rhs40000.bin output_mpi.bin 400000 1e-16 400000
srun mpi /project/home/p200301/tests/matrix50000.bin /project/home/p200301/tests/rhs50000.bin output_mpi.bin 500000 1e-16 500000
srun mpi /project/home/p200301/tests/matrix70000.bin /project/home/p200301/tests/rhs70000.bin output_mpi.bin 600000 1e-16 600000
srun mpi /project/home/p200301/tests/matrix70000.bin /project/home/p200301/tests/rhs70000.bin output_mpi.bin 700000 1e-16 700000
srun mpi /project/home/p200301/tests/matrix70000.bin /project/home/p200301/tests/rhs70000.bin output_mpi.bin 800000 1e-16 800000
srun mpi /project/home/p200301/tests/matrix70000.bin /project/home/p200301/tests/rhs70000.bin output_mpi.bin 900000 1e-16 900000
srun mpi /project/home/p200301/tests/matrix70000.bin /project/home/p200301/tests/rhs70000.bin output_mpi.bin 1000000 1e-16 1000000