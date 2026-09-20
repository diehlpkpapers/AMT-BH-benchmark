#pragma once

#include <fstream>
#include <vector>
#include <string>
#include "body.h"
#include "tinyxml2.h"
#include "cxxopts.hpp"
#include "linear_octree.h"
#include "bounding_box.h"
#include "hpx_collectives.h"

// Reads body data (mass, position, velocity) from a specified CSV file
// Returns true on success, false on failure
bool readCSV(
    const std::string& filename,
    std::vector<uint64_t>& ids,
    std::vector<double>& masses,
    std::vector<Position>& positions,
    std::vector<Velocity>& velocities,
    int body_count = -1 // limit on the number of bodies to read
);

/**
 * @brief Writes the simulation state for one rank to a VTP snapshot file.
 * @param rank The rank of the calling process.
 * @param step_counter The current visualization step number.
 * @param ... (other params)
 * @param output_dir The root directory for visualization files.
 */
void writeSnapshot(int rank, int vs_counter,
                   const std::vector<uint64_t>& local_ids,
                   const std::vector<double>& local_masses,
                   const std::vector<Position>& local_positions,
                   const std::vector<Velocity>& local_velocities,
                   const std::vector<Acceleration>& local_accelerations,
                   const std::string& vs_dir);

/**
 * @brief Creates or updates the main timeline (.pvd) file for particle snapshots.
 *
 * This function adds a new entry to the PVD file, pointing to the snapshot
 * files that were just written for the current timestep.
 *
 * @param cli_args The parsed command-line arguments, used for generating a unique filename.
 * @param num_ranks The total number of MPI ranks.
 * @param step_counter The current visualization step number.
 * @param current_time The current simulation time.
 * @param output_dir The root directory for visualization files.
 */
void updatePVDFile(const cxxopts::ParseResult& args,
                   int size,
                   int vs_counter,
                   double current_time,
                   const std::string& vs_dir);


/**
 * @brief Writes the Locally Essential Tree (LET) nodes received by a rank to a VTU file.
 *
 * This is a visualization function to debug and analyze which pseudo-leaves
 * each rank receives from its peers.
 */
void writeReceivedLETs(int rank, int vis_step,
                       const std::vector<NodeRecord>& received_nodes,
                       const std::vector<int>& recv_counts,
                       const BoundingBox& global_bb,
                       const std::string& vs_dir);

/**
 * @brief Creates or updates the timeline (.pvd) file for the LET visualization.
 */
void updateReceivedLETPVDFile(const cxxopts::ParseResult& args,
                              int size,
                              int vs_counter,
                              double current_time,
                              const std::string& vs_dir);

void writeHistogram(int vis_step,
                    const std::vector<std::pair<long long, int>>& hist,
                    const BoundingBox& global_bb,
                    const std::string& out_dir,
                    int bucket_bits);

void updateHistogramPVDFile(const cxxopts::ParseResult& args,
                            int vs_counter,
                            double current_time,
                            const std::string& out_dir);

void saveReferenceCSV(const std::string& dir, const std::vector<Position>& positions, const std::vector<uint64_t>& ids);
std::vector<Position> loadReferenceCSV(const std::string& dir, int num_bodies);
double computeDistanceSum(const std::vector<Position>& local_pos, const std::vector<uint64_t>& local_ids,
    const std::vector<Position>& ref_pos, hpxc::Ctx& ctx);

// Creates <out_dir>/energy.csv (truncating) and writes the header row.
// Rank 0 only. Returns the open stream; caller keeps it for the whole run.
std::ofstream openEnergyLog(const std::string& out_dir);

// Appends one measurement and flushes, so a killed run still has usable data.
void appendEnergyRow(std::ofstream& os, int step, double t,
    double kinetic, double potential, double total, double rel_drift);