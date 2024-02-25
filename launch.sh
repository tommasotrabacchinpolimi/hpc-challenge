#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

#SBATCH --time=03:00:00
#SBATCH --account=p200301
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --cpus-per-task=1

module load ifpgasdk
module load 520nmx
cd hpc-challenge/src/fpga
aoc -board=p520_hpc_m210h_g3x16 -fp-relaxed -DINTEL_CL -o CG_kernel kernels.cl -v -report

