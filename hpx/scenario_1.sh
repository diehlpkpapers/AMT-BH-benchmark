#!/bin/bash
#SBATCH --partition=all
#SBATCH --nodelist=simcl1n[1-4]
#SBATCH --job-name="scenario1_final"
#SBATCH --output=scenario1.out
#SBATCH --error=scenario1.err
#SBATCH --time=06:30:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1       # one MPI rank on each node
#SBATCH --cpus-per-task=48        # allocate all 48 physical cores
#SBATCH --exclusive

set -e
set -x

threads=$SLURM_CPUS_PER_TASK

echo "Running scenario 1... at $(date)"
srun --exclusive -N 4 -c $threads --cpu-bind=cores ./build_hpx/simulate \
      --hpx:threads=$threads \
      --file ./data/state_vectors_csvs/scenario1_19054.csv \
      --dt 1h --tend 12y --vs 2d \
      --outdir "sim_s1_${SLURM_JOB_ID}" \
      --theta 1.05 --fc let
echo "Scenario 1 complete at $(date)"