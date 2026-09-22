#!/bin/bash

#SBATCH --partition=all
#SBATCH --nodelist=simcl1n[1-4]
#SBATCH --job-name="benchmark_S1"
#SBATCH --output=benchmark_S1_new.out
#SBATCH --error=benchmark_S1_new.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1

# Exit immediately if a command exits with a non-zero status
set -e
set -x

# ==============================================================================
#                             BENCHMARK CONFIGURATION
# ==============================================================================
# --- Simulation Parameters ---
file="../data/state_vectors_csvs/scenario1_19054.csv"
dt="1h"
tend="60d" # Longer duration for rebalancing analysis
vs="2d"    # More frequent visualization for analysis
theta="1.05"

# --- Benchmark Directory Setup ---
sanitized_file=$(basename "$file" .csv | tr -c 'A-Za-z0-9' '_')
benchmark_dir="benchmark_S1_${SLURM_JOB_ID}"
# --- NEW: Real location lives on scratch ---
SCRATCH_ROOT=/data/scratch-simcl1/$USER
REAL_DIR="${SCRATCH_ROOT}/${benchmark_dir}"

mkdir -p "${REAL_DIR}/vs_outputs"   # Create the real folders on scratch
ln -sfn "${REAL_DIR}" "${benchmark_dir}"
# This creates a symlink in the current directory:
# ./benchmark_S1_<jobid>  →  /data/scratch-simcl1/$USER/benchmark_S1_<jobid>
# All script operations on benchmark_dir will now transparently write to scratch.
# ---------------------------------------------------------------------------

# --- Data and Plot Files ---
data_max_depth_file="${benchmark_dir}/data_max_depth.dat"
plot_max_depth_file="${benchmark_dir}/plot_max_depth.png"

data_bucket_bits_file="${benchmark_dir}/data_bucket_bits.dat"
plot_bucket_bits_file="${benchmark_dir}/plot_bucket_bits.png"

data_nodes_file="${benchmark_dir}/data_nodes.dat"
plot_nodes_file="${benchmark_dir}/plot_nodes.png"

data_bodies_file="${benchmark_dir}/data_bodies.dat"
plot_bodies_file="${benchmark_dir}/plot_bodies_combined.png" # Updated plot name

data_rebalance_file="${benchmark_dir}/data_rebalance.dat"
plot_rebalance_file="${benchmark_dir}/plot_rebalance.png"

data_let_direct_bodies_file="${benchmark_dir}/data_let_direct_bodies.dat" # New data file

# --- Parameter Ranges for Tests ---
max_depth_values=(2 4 8 12 16 21)
bucket_bits_values=(3 9 12 15 18 21)
node_counts=(1 2 3 4)
body_counts=(100 1000 5000 10000 15000 19054)
rebalance_values=(1 5 10 24 50 999999)

# --- Fixed Parameters for Control Runs ---
FIXED_BODIES=19054
FIXED_NODES=4
FIXED_THREADS=48
FIXED_MAX_DEPTH=21
FIXED_BUCKET_BITS=18
FIXED_REBALANCE=24
FIXED_FC_POLICY="let"

# ==============================================================================
#                               SETUP & HELPERS
# ==============================================================================

# --- Change to Build Directory ---
BUILD_DIR="./build_hpx"
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory '$BUILD_DIR' not found. Please compile the project first."
    exit 1
fi
cd "$BUILD_DIR"

simulate_binary="./simulate"
if [ ! -x "$simulate_binary" ]; then
    echo "Error: Simulation binary '$simulate_binary' not found or not executable."
    exit 1
fi

# --- Simulation Runner Function ---
# Arguments:
#   $1: bodies        $6: fc_policy
#   $2: nodes         $7: is_reference ("true" or "false")
#   $3: threads       $8: bucket_bits
#   $4: max_depth     $9: test_name (for outdir)
#   $5: rebalance_interval $10: param_value (for outdir)
# Returns: "total_time summed_dist_error"
run_simulation() {
    local bodies=$1
    local nodes=$2
    local threads=$3
    local max_depth=$4
    local rebalance=$5
    local fc_policy=$6
    local is_ref=$7
    local bucket_bits=$8
    local test_name=$9
    local param_value=${10}

    # Create a unique output directory for visualization files
    local outdir="${OLDPWD}/${benchmark_dir}/vs_outputs/${test_name}/${param_value}"
    mkdir -p "$outdir"

    local cmd=(srun --exclusive -N "$nodes" -c "$threads" --cpu-bind=cores "$simulate_binary"
                --hpx:threads="$threads"
                --file "$file"
                --dt "$dt" --tend "$tend" --vs "$vs"
                --outdir "$outdir"
                --theta "$theta"
                --bodies "$bodies"
                --fc "$fc_policy"
                --max-depth "$max_depth"
                --rebalance-interval "$rebalance"
                --bucket-bits "$bucket_bits")

    if [ "$is_ref" = "true" ]; then
        cmd+=("-r")
    fi

    local simulation_output
    simulation_output=$("${cmd[@]}" 2>&1)

    local total_time
    total_time=$(echo "$simulation_output" | grep "TOTAL_TIME" | awk '{print $2}')
    local summed_dist_error
    summed_dist_error=$(echo "$simulation_output" | grep "SUMMED_DIST_ERROR" | awk '{print $2}')

    total_time=${total_time:-"na"}
    summed_dist_error=${summed_dist_error:-"na"}

    echo "$total_time $summed_dist_error"
}

# --- Initialize Data Files with Headers ---
echo "max_depth total_time summed_dist_error" > "${OLDPWD}/${data_max_depth_file}"
echo "bucket_bits total_time summed_dist_error" > "${OLDPWD}/${data_bucket_bits_file}"
echo "nodes total_time summed_dist_error" > "${OLDPWD}/${data_nodes_file}"
echo "bodies total_time summed_dist_error" > "${OLDPWD}/${data_bodies_file}"
echo "rebalance_interval total_time summed_dist_error" > "${OLDPWD}/${data_rebalance_file}"
echo "bodies total_time summed_dist_error" > "${OLDPWD}/${data_let_direct_bodies_file}"

# ==============================================================================
#                               REFERENCE RUN
# ==============================================================================
echo "============================================================"
echo "PHASE 0: Generating Reference Data (Ground Truth)"
echo "============================================================"
run_simulation "$FIXED_BODIES" "1" "$FIXED_THREADS" "21" "1" "tree" "true" "18" "reference" "run"
echo "Reference run completed. reference/final_ref.csv has been generated."

# ==============================================================================
#                               BENCHMARK PHASES
# ==============================================================================

echo "============================================================"
echo "PHASE 1: Runtime & Error vs. LET Max Depth"
echo "============================================================"
for depth in "${max_depth_values[@]}"; do
    metrics=$(run_simulation "$FIXED_BODIES" "$FIXED_NODES" "$FIXED_THREADS" "$depth" "$FIXED_REBALANCE" "$FIXED_FC_POLICY" "false" "$FIXED_BUCKET_BITS" "max_depth" "d_${depth}")
    total_time=$(echo "$metrics" | awk '{print $1}')
    dist_error=$(echo "$metrics" | awk '{print $2}')
    echo "$depth $total_time $dist_error" >> "${OLDPWD}/${data_max_depth_file}"
done

echo "============================================================"
echo "PHASE 2: Runtime & Error vs. Bucket Bits"
echo "============================================================"
for bits in "${bucket_bits_values[@]}"; do
    metrics=$(run_simulation "$FIXED_BODIES" "$FIXED_NODES" "$FIXED_THREADS" "$FIXED_MAX_DEPTH" "$FIXED_REBALANCE" "$FIXED_FC_POLICY" "false" "$bits" "bucket_bits" "b_${bits}")
    total_time=$(echo "$metrics" | awk '{print $1}')
    dist_error=$(echo "$metrics" | awk '{print $2}')
    echo "$bits $total_time $dist_error" >> "${OLDPWD}/${data_bucket_bits_file}"
done

echo "============================================================"
echo "PHASE 3: Runtime & Error vs. MPI Node Count"
echo "============================================================"
for nodes in "${node_counts[@]}"; do
    metrics=$(run_simulation "$FIXED_BODIES" "$nodes" "$FIXED_THREADS" "$FIXED_MAX_DEPTH" "$FIXED_REBALANCE" "$FIXED_FC_POLICY" "false" "$FIXED_BUCKET_BITS" "nodes" "n_${nodes}")
    total_time=$(echo "$metrics" | awk '{print $1}')
    dist_error=$(echo "$metrics" | awk '{print $2}')
    echo "$nodes $total_time $dist_error" >> "${OLDPWD}/${data_nodes_file}"
done

echo "============================================================"
echo "PHASE 4: Runtime & Error vs. Body Count (FC=let)"
echo "============================================================"
for bodies in "${body_counts[@]}"; do
    metrics=$(run_simulation "$bodies" "$FIXED_NODES" "$FIXED_THREADS" "$FIXED_MAX_DEPTH" "$FIXED_REBALANCE" "$FIXED_FC_POLICY" "false" "$FIXED_BUCKET_BITS" "bodies_let" "b_${bodies}")
    total_time=$(echo "$metrics" | awk '{print $1}')
    dist_error=$(echo "$metrics" | awk '{print $2}')
    echo "$bodies $total_time $dist_error" >> "${OLDPWD}/${data_bodies_file}"
done

echo "============================================================"
echo "PHASE 5: Runtime & Error vs. Rebalance Interval"
echo "============================================================"
for interval in "${rebalance_values[@]}"; do
    metrics=$(run_simulation "$FIXED_BODIES" "$FIXED_NODES" "$FIXED_THREADS" "$FIXED_MAX_DEPTH" "$interval" "$FIXED_FC_POLICY" "false" "$FIXED_BUCKET_BITS" "rebalance" "i_${interval}")
    total_time=$(echo "$metrics" | awk '{print $1}')
    dist_error=$(echo "$metrics" | awk '{print $2}')
    echo "$interval $total_time $dist_error" >> "${OLDPWD}/${data_rebalance_file}"
done

echo "============================================================"
echo "PHASE 6: Runtime & Error vs. Body Count (FC=let_direct)"
echo "============================================================"
for bodies in "${body_counts[@]}"; do
    metrics=$(run_simulation "$bodies" "$FIXED_NODES" "$FIXED_THREADS" "$FIXED_MAX_DEPTH" "$FIXED_REBALANCE" "let_direct" "false" "$FIXED_BUCKET_BITS" "bodies_let_direct" "b_${bodies}")
    total_time=$(echo "$metrics" | awk '{print $1}')
    dist_error=$(echo "$metrics" | awk '{print $2}')
    echo "$bodies $total_time $dist_error" >> "${OLDPWD}/${data_let_direct_bodies_file}"
done

# ==============================================================================
#                               PLOTTING
# ==============================================================================
echo "============================================================"
echo "PHASE 7: Generating Plots"
echo "============================================================"
cd "${OLDPWD}"

# --- Generic Plot Function ---
plot_graph() {
    local output_file=$1
    local data_file=$2
    local title=$3
    local xlabel=$4

    gnuplot <<EOF
set terminal png size 1000,800
set output "$output_file"
set title "$title\n(Scenario 1: 19k bodies)"
set xlabel "$xlabel"
set ylabel "Runtime (s)"
set y2label "Summed Distance Error"
set grid
set key outside
set ytics nomirror
set y2tics
set y2range [0:*]

set label "File: $(basename $file)\nDT: $dt, T_END: $tend, Theta: $theta" at graph 0.02, 0.95 left

plot "$data_file" using 1:2 with linespoints title 'Runtime' lc rgb 'blue', \
     '' using 1:3 axes x1y2 with linespoints title 'Error' lc rgb 'red'
EOF
    echo "Plot generated: $output_file"
}

# --- Combined Plot for Body Count ---
gnuplot <<EOF
set terminal png size 1200,800
set output "$plot_bodies_file"
set title "Runtime & Error vs. Body Count (let vs. let_direct)\n(Scenario 1: 19k bodies)"
set xlabel "Number of Bodies"
set ylabel "Runtime (s)"
set y2label "Summed Distance Error"
set grid
set key outside
set ytics nomirror
set y2tics
set y2range [0:*]

set label "File: $(basename $file)\nDT: $dt, T_END: $tend, Theta: $theta" at graph 0.02, 0.95 left

plot "$data_bodies_file" using 1:2 with linespoints title 'Runtime (let)' lc rgb 'blue', \
     '' using 1:3 axes x1y2 with linespoints title 'Error (let)' lc rgb 'red', \
     "$data_let_direct_bodies_file" using 1:2 with linespoints title 'Runtime (let_direct)' lc rgb 'cyan', \
     '' using 1:3 axes x1y2 with linespoints title 'Error (let_direct)' lc rgb 'orange'
EOF
echo "Plot generated: $plot_bodies_file"


# --- Generate other plots ---
plot_graph "$plot_max_depth_file" "$data_max_depth_file" "Runtime & Error vs. LET Max Depth" "Max Depth"
plot_graph "$plot_bucket_bits_file" "$data_bucket_bits_file" "Runtime & Error vs. Bucket Bits" "Bucket Bits"
plot_graph "$plot_nodes_file" "$data_nodes_file" "Runtime & Error vs. MPI Nodes" "Node Count"
plot_graph "$plot_rebalance_file" "$data_rebalance_file" "Runtime & Error vs. Rebalance Interval" "Rebalance Interval (steps)"

echo "All plots generated in '$benchmark_dir'."