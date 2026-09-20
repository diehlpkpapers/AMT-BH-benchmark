#!/bin/bash
#SBATCH --partition=all
#SBATCH --nodelist=simcl1n[1-4]
#SBATCH --job-name="scenario2_final"
#SBATCH --output=scenario2.out
#SBATCH --error=scenario2.err
#SBATCH --time=06:30:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1       # one MPI rank on each node
#SBATCH --cpus-per-task=48        # allocate all 48 physical cores
#SBATCH --exclusive

set -e
set -x

threads=$SLURM_CPUS_PER_TASK

echo "Running scenario 2... at $(date)"
srun --exclusive -N 4 -c $threads --cpu-bind=cores ./build_hpx/simulate \
      --hpx:threads=$threads \
      --file ./data/state_vectors_csvs/scenario2_300149.csv \
      --dt 1h --tend 1y --vs 7d \
      --outdir "sim_s2_${SLURM_JOB_ID}" \
      --theta 1.05 --fc let
echo "Scenario 2 complete at $(date)"