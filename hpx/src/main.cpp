#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/collectives/broadcast.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>


#include "cxxopts.hpp"
#include "parse_time.h"
#include "body.h"
#include "io.h"
#include "morton_keys.h"
#include "linear_octree.h"
#include "load_balancing.h"
#include "exchange.h"
#include "traversal.h"
#include "utility.h"
#include "policy.h"
#include "hpx_collectives.h"
#include "energy.h"
#include "numa_layout.h"

namespace {

// The argc/argv this process was started with. Every locality (rank) is
// started by mpirun with the identical command line, so each locality's
// worker action can just re-parse it locally instead of having it shipped
// as action arguments.
int    g_argc = 0;
char** g_argv = nullptr;

// Pairs a final position with its body id, for the single-shot gather to
// the root rank used by the reference/error-analysis mechanism.
struct RefEntry {
    Position pos;
    uint64_t id;
};

}    // namespace

HPX_IS_BITWISE_SERIALIZABLE(RefEntry);

namespace {

// ---------------------------------------------------------------------------
// Per-phase wall-clock timers, opt-in via --timers.
//
// Phase -> code mapping:
//   Domain        step 3's bbox/Morton-code/histogram computation (excludes
//                 the periodic body migration itself)
//   Migrate       step 3's rebalance_bodies() call (only on rebalance steps)
//   TreeBuild     step 4-5: sort by Morton key + build the local octree
//   TreeExchange  step 6: exchangeFullTrees / exchangeEssentialTrees
//   Interact      step 7 only: computeAccelerations*()
//   Energy        step 8b: computeGlobalEnergy() (only when --energy is on)
//   Output        step 9: snapshot/PVD writes, including the barrier wait
//   StepTotal     the entire loop body for that step
//   (Other, derived = StepTotal - sum of the above: the untimed kick/drift
//    integrator steps, plus any straggler skew between localities)
//
// Every start() begins with a barrier, so each locality's measured duration
// for a phase is that locality's own work, not work inherited from a
// previous phase's queueing/skew; the barrier is the "synchronization point"
// the timer requirement explicitly accepts as the cost of measuring. Once
// per step, a single all_reduce with an elementwise-max functor reduces
// every locality's measured durations into one array - the straggler's time
// per phase, which is what determines wall time. Both the barrier and the
// all_reduce are no-ops (branch out immediately) for any disabled phase, and
// endStep() itself is a no-op unless at least one phase is enabled - the
// existing benchmark scripts, which never pass --timers, pay nothing.
// ---------------------------------------------------------------------------
struct PhaseTimers {
    enum class Phase : std::size_t {
        Domain = 0, Migrate, TreeBuild, TreeExchange,
        Interact, Energy, Output, StepTotal, Count
    };
    static constexpr std::size_t N = static_cast<std::size_t>(Phase::Count);
    using clock = std::chrono::high_resolution_clock;

    hpxc::Ctx* ctx = nullptr;
    bool per_step = false;
    bool any = false;
    std::array<bool, N> enabled{};
    std::array<clock::time_point, N> t0{};
    std::array<double, N> cur{};        // this step, local to this locality
    std::array<bool, N> ran{};          // did this phase run this step
    std::array<double, N> sum{};        // run aggregate (reduced, so identical everywhere)
    std::array<long long, N> n{};       // number of steps each phase ran
    long long steps_timed = 0;
    double input_seconds = 0.0;

    void configure(hpxc::Ctx& c, std::array<bool, N> phase_enabled, bool per_step_) {
        ctx = &c;
        enabled = phase_enabled;
        per_step = per_step_;
        any = false;
        for (std::size_t i = 0; i + 1 < N; ++i)    // exclude StepTotal itself
            any = any || enabled[i];
        enabled[static_cast<std::size_t>(Phase::StepTotal)] = any;
    }

    void beginStep() {
        if (!any) return;
        cur.fill(0.0);
        ran.fill(false);
    }

    void start(Phase p) {
        std::size_t i = static_cast<std::size_t>(p);
        if (!enabled[i]) return;
        ctx->barrier.wait();
        t0[i] = clock::now();
    }

    void stop(Phase p) {
        std::size_t i = static_cast<std::size_t>(p);
        if (!enabled[i]) return;
        cur[i] += std::chrono::duration<double>(clock::now() - t0[i]).count();
        ran[i] = true;
    }

    // Reduces this step's per-locality durations to the cross-locality max
    // (identical result on every locality) and updates the run aggregate.
    // Returns a zero-filled array if no phase is enabled.
    std::array<double, N> endStep() {
        std::array<double, N> g{};
        if (!any) return g;
        g = hpx::collectives::all_reduce(ctx->comm_timers, cur, hpxc::MaxTimers{},
            hpxc::this_site_arg(ctx->rank), hpxc::generation_arg(ctx->gen_timers++)).get();
        for (std::size_t i = 0; i < N; ++i) {
            if (!enabled[i]) continue;
            sum[i] += g[i];
            if (ran[i] || i == static_cast<std::size_t>(Phase::StepTotal)) ++n[i];
        }
        ++steps_timed;
        return g;
    }
};

static_assert(PhaseTimers::N == hpxc::kTimerSlots,
    "PhaseTimers::Phase::Count must match hpxc::kTimerSlots (hpxc::MaxTimers' array size)");

// Parses a comma-separated --timers spec into the set of enabled phases.
// Returns false and sets `bad_token` on an unrecognised token; every
// locality parses the identical string, so this never desynchronises ranks.
bool parseTimerGroups(const std::string& spec,
    std::array<bool, PhaseTimers::N>& enabled, std::string& bad_token)
{
    enabled.fill(false);
    using Phase = PhaseTimers::Phase;

    auto enable_group = [&](const std::string& tok) -> bool {
        if (tok == "none") return true;
        if (tok == "all" || tok == "loadbal") {
            enabled[static_cast<std::size_t>(Phase::Domain)] = true;
            enabled[static_cast<std::size_t>(Phase::Migrate)] = true;
        }
        if (tok == "all" || tok == "tree") {
            enabled[static_cast<std::size_t>(Phase::TreeBuild)] = true;
            enabled[static_cast<std::size_t>(Phase::TreeExchange)] = true;
        }
        if (tok == "all" || tok == "interactions") {
            enabled[static_cast<std::size_t>(Phase::Interact)] = true;
            enabled[static_cast<std::size_t>(Phase::Energy)] = true;
        }
        if (tok == "all" || tok == "output") {
            enabled[static_cast<std::size_t>(Phase::Output)] = true;
        }
        return tok == "all" || tok == "loadbal" || tok == "tree" ||
               tok == "interactions" || tok == "output";
    };

    std::stringstream ss(spec);
    std::string tok;
    bool any_token = false;
    while (std::getline(ss, tok, ',')) {
        // trim whitespace
        std::size_t b = tok.find_first_not_of(" \t");
        std::size_t e = tok.find_last_not_of(" \t");
        tok = (b == std::string::npos) ? "" : tok.substr(b, e - b + 1);
        if (tok.empty()) continue;
        any_token = true;
        if (!enable_group(tok)) {
            bad_token = tok;
            return false;
        }
    }
    (void)any_token;    // an empty/whitespace-only spec behaves like "none"
    return true;
}

double performReferenceAndErrorAnalysis(
    bool is_reference,
    const std::string& ref_dir,
    hpxc::Ctx& ctx,
    int N_total,
    const std::vector<Position>& local_pos,
    const std::vector<uint64_t>& local_ids)
{
    double summed_dist_error = 0.0;

    std::vector<RefEntry> local_entries(local_pos.size());
    for (std::size_t i = 0; i < local_pos.size(); ++i)
        local_entries[i] = {local_pos[i], local_ids[i]};

    if (is_reference) {
        if (ctx.rank == 0) {
            auto per_site = hpx::collectives::gather_here(ctx.comm_gather_ref, std::move(local_entries),
                hpxc::this_site_arg(ctx.rank), hpxc::generation_arg(ctx.gen_gather_ref++)).get();
            auto all_entries = hpxc::flatten(std::move(per_site));

            std::vector<Position> final_positions(N_total);
            std::vector<uint64_t> final_ids(N_total);
            for (std::size_t i = 0; i < all_entries.size(); ++i) {
                final_positions[i] = all_entries[i].pos;
                final_ids[i]       = all_entries[i].id;
            }
            saveReferenceCSV(ref_dir, final_positions, final_ids);
            std::cout << "Reference positions saved to " << ref_dir << "/final_ref.csv" << std::endl;
        } else {
            hpx::collectives::gather_there(ctx.comm_gather_ref, std::move(local_entries),
                hpxc::this_site_arg(ctx.rank), hpxc::generation_arg(ctx.gen_gather_ref++)).get();
        }
    } else {
        if (std::filesystem::exists(std::filesystem::path(ref_dir) / "final_ref.csv")) {
            std::vector<Position> ref_positions = loadReferenceCSV(ref_dir, N_total);
            summed_dist_error = computeDistanceSum(local_pos, local_ids, ref_positions, ctx);
        } else {
            if (ctx.rank == 0)
                std::cout << "Reference file not found at " << ref_dir << "/final_ref.csv, skipping error analysis." << std::endl;
        }
    }
    return summed_dist_error;
}

void worker_main()
{
    std::size_t rank = hpx::get_locality_id();
    std::size_t size = hpx::get_num_localities(hpx::launch::sync);
    hpxc::Ctx ctx(rank, size);

    // CLI parsing (argv is identical across every locality; --hpx:* flags
    // mixed into it are simply ignored here).
    cxxopts::Options opt(g_argv[0], "Phase-2 Barnes-Hut");
    opt.allow_unrecognised_options();
    opt.add_options()
        ("h,help",    "Print usage")
        ("f,file",    "CSV file (mass,x,y,z,vx,vy,vz)",   cxxopts::value<std::string>())
        ("dt",        "Time step",                       cxxopts::value<std::string>()->default_value("1d"))
        ("tend",      "End time",                       cxxopts::value<std::string>()->default_value("1y"))
        ("vs",        "Visualization step interval", cxxopts::value<std::string>()->default_value("10d"))
        ("o,outdir",  "Output directory for visualization", cxxopts::value<std::string>()->default_value("sim_out"))
        ("b,bodies",  "number of bodies to simulate", cxxopts::value<int>()->default_value("-1"))
        ("fc", "Force calculation: tree|let|let_direct", cxxopts::value<std::string>()->default_value("let"))
        ("theta",     "BH opening angle",               cxxopts::value<double>()->default_value("0.5"))
        ("softening", "Gravitational softening length epsilon (same distance unit as "
                     "the input positions, e.g. AU). 0 (default) disables softening - "
                     "fine for well-separated bodies (e.g. solar-system data), but "
                     "close encounters in dense datasets (e.g. galaxy models) need a "
                     "nonzero epsilon to avoid near-singular forces.",
                     cxxopts::value<double>()->default_value("0.0"))
        ("max-depth", "Maximum depth for LET traversal (0-21, 21 means full traversal)", cxxopts::value<int>()->default_value("21"))
        ("bucket-bits", "Number of bits for histogram buckets (2^n buckets)", cxxopts::value<int>()->default_value("18"))
        ("rebalance-interval", "Steps between load rebalancing", cxxopts::value<int>()->default_value("24"))
        ("leaf-capacity", "Maximum number of bodies per octree leaf (>=1). Subdivision "
                         "stops as soon as a cell holds at most this many bodies, and "
                         "near leaves are resolved by exact pairwise summation. 1 "
                         "(default) is the classic one-body-per-leaf rule; larger "
                         "values trade a shallower tree for more direct pairs.",
                         cxxopts::value<int>()->default_value("1"))
        ("r,reference", "Run as reference and save final positions", cxxopts::value<bool>()->default_value("false"))
        ("energy", "Track total energy (KE+PE) for conservation diagnostics", cxxopts::value<bool>()->default_value("false"))
        ("et", "Energy tracking interval (like --vs); 0d = every step", cxxopts::value<std::string>()->default_value("0d"))
        ("energy-exact", "Use exact O(N^2) direct-sum potential energy instead of the "
                         "Barnes-Hut/LET approximation (validation only, small N)", cxxopts::value<bool>()->default_value("false"))
        ("timers", "Comma-separated phases to time: loadbal,tree,interactions,output "
                  "(also 'all' or 'none'). Timed phases add a barrier per phase per "
                  "step plus one max-reduction per step. Input timing is always on.",
                  cxxopts::value<std::string>()->default_value("none"))
        ("timers-per-step", "Print one timer line for every timed step (default: only "
                            "the first step's breakdown plus the run aggregate)",
                  cxxopts::value<bool>()->default_value("false"));

    const auto args = opt.parse(g_argc, g_argv);
    if (args.count("help")) {
        if (rank == 0) std::cout << opt.help() << std::endl;
        return;
    }
    const double dt    = parseTime(args["dt"].as<std::string>());
    const double t_end = parseTime(args["tend"].as<std::string>());
    const double vs_interval = parseTime(args["vs"].as<std::string>());
    const std::string out_dir = args["outdir"].as<std::string>();
    const double theta = args["theta"].as<double>();
    const double softening = args["softening"].as<double>();
    if (softening < 0.0) {
        if (rank == 0)
            std::cerr << "Error: --softening must be >= 0 (got " << softening << ")\n";
        return;
    }
    const int nbodies = args["bodies"].as<int>();
    const int max_let_depth = args["max-depth"].as<int>();
    const int bucket_bits = args["bucket-bits"].as<int>();
    const int rebalance_interval = args["rebalance-interval"].as<int>();
    const int leaf_capacity = args["leaf-capacity"].as<int>();
    if (leaf_capacity < 1) {
        if (rank == 0)
            std::cerr << "Error: --leaf-capacity must be >= 1 (got " << leaf_capacity << ")\n";
        return;    // replicated failure: every locality parses the same argv and
                   // returns before entering any collective
    }
    const bool energy_on = args["energy"].as<bool>();
    const double et_interval = parseTime(args["et"].as<std::string>());
    const bool energy_exact = args["energy-exact"].as<bool>();
    const bool timers_per_step = args["timers-per-step"].as<bool>();

    std::array<bool, PhaseTimers::N> timer_phases{};
    {
        std::string bad_token;
        if (!parseTimerGroups(args["timers"].as<std::string>(), timer_phases, bad_token)) {
            if (rank == 0)
                std::cerr << "Error: unknown --timers phase '" << bad_token
                          << "' (expected loadbal,tree,interactions,output,all,none)\n";
            return;    // replicated failure: every locality returns, no rank left in a collective
        }
    }
    PhaseTimers timers;
    timers.configure(ctx, timer_phases, timers_per_step);

    const bool is_reference = args["reference"].as<bool>();
    const std::string ref_dir = "reference";

    FCPolicy fcPol;
    const std::string fc_str = args["fc"].as<std::string>();

    if (is_reference) {
        fcPol = FCPolicy::Tree;
        if (rank == 0)
            std::cout << "Running in reference mode. Force calculation method will be 'tree'." << std::endl;
    } else if (fc_str == "tree") {
        fcPol = FCPolicy::Tree;
    } else if (fc_str == "let_direct") {
        fcPol = FCPolicy::LET_DirectSum;
    } else {
        fcPol = FCPolicy::LET;
    }

    // Input timing is unconditional (not gated by --timers): the doc requires
    // it always be trackable so it can be excluded from other comparisons,
    // and it costs nothing extra - rank 0's own wall time already is the
    // true input time, since every other locality is blocked inside
    // broadcast_from/scatter_from until rank 0 finishes reading and scattering.
    auto input_t0 = PhaseTimers::clock::now();

    // Rank 0 loads the CSV, then scatters slices to every other locality.
    std::vector<Position>  init_pos;
    std::vector<Velocity>  init_vel;
    std::vector<double>    init_mass;
    std::vector<uint64_t>  init_ids;

    int N_total_or_error = 0;    // -1 signals a fatal error to every locality
    if (rank == 0) {
        if (!args.count("file")) {
            std::cerr << "Error: input file required (-f)\n";
            N_total_or_error = -1;
        } else if (!readCSV(args["file"].as<std::string>(), init_ids, init_mass, init_pos, init_vel, nbodies)) {
            N_total_or_error = -1;
        } else {
            std::cout << "Loaded " << init_mass.size() << " bodies\n";
            std::filesystem::create_directories(out_dir);
            if (is_reference) std::filesystem::create_directories(ref_dir);
            N_total_or_error = static_cast<int>(init_mass.size());
        }
    }

    N_total_or_error = rank == 0
        ? hpx::collectives::broadcast_to(ctx.comm_bcast_ntotal, N_total_or_error,
              hpxc::this_site_arg(rank), hpxc::generation_arg(ctx.gen_bcast_ntotal++)).get()
        : hpx::collectives::broadcast_from<int>(ctx.comm_bcast_ntotal,
              hpxc::this_site_arg(rank), hpxc::generation_arg(ctx.gen_bcast_ntotal++)).get();

    if (N_total_or_error < 0) return;    // fatal error already reported by rank 0
    const int N_total = N_total_or_error;

    std::vector<int> counts(size), displs(size);
    int base = N_total / static_cast<int>(size), rem = N_total % static_cast<int>(size);
    for (std::size_t i = 0; i < size; ++i) {
        counts[i] = base + (static_cast<int>(i) < rem);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }

    std::vector<Position>  local_pos;
    std::vector<Velocity>  local_vel;
    std::vector<double>    local_mass;
    std::vector<uint64_t>  local_ids;

    std::vector<hpxc::TransferBody> my_payload;
    if (rank == 0) {
        std::vector<std::vector<hpxc::TransferBody>> per_dest(size);
        for (std::size_t dst = 0; dst < size; ++dst) {
            per_dest[dst].resize(counts[dst]);
            for (int i = 0; i < counts[dst]; ++i) {
                int src_i = displs[dst] + i;
                per_dest[dst][i] = {init_pos[src_i], init_mass[src_i], init_vel[src_i], init_ids[src_i]};
            }
        }
        my_payload = hpx::collectives::scatter_to(ctx.comm_scatter_init, std::move(per_dest),
            hpxc::this_site_arg(rank), hpxc::generation_arg(ctx.gen_scatter_init++)).get();
    } else {
        my_payload = hpx::collectives::scatter_from<std::vector<hpxc::TransferBody>>(
            ctx.comm_scatter_init, hpxc::this_site_arg(rank), hpxc::generation_arg(ctx.gen_scatter_init++)).get();
    }

    // Grows each array in per-domain slices (numa_layout.h) so value-init -
    // the first write to each page - happens on the domain that reads it.
    const NumaLayout& numa_layout = NumaLayout::instance();
    numa_layout.growWithFirstTouch(local_pos,  my_payload.size());
    numa_layout.growWithFirstTouch(local_vel,  my_payload.size());
    numa_layout.growWithFirstTouch(local_mass, my_payload.size());
    numa_layout.growWithFirstTouch(local_ids,  my_payload.size());
    for (std::size_t i = 0; i < my_payload.size(); ++i) {
        local_pos[i]  = my_payload[i].pos;
        local_mass[i] = my_payload[i].mass;
        local_vel[i]  = my_payload[i].vel;
        local_ids[i]  = my_payload[i].id;
    }
    timers.input_seconds = std::chrono::duration<double>(PhaseTimers::clock::now() - input_t0).count();

    // Main loop
    const double G = 1.48812e-34;
    const double soft2 = softening * softening;    // Plummer eps^2; shared by the force and energy paths
    std::vector<Acceleration> local_acc;
    numa_layout.growWithFirstTouch(local_acc, local_pos.size());
    double next_vis_time = 0.0;
    int vis_step = 0;
    double t = 0.0;

    double next_energy_time  = 0.0;
    double energy_ref_total  = 0.0;   // E at the first sample (baseline)
    bool   have_energy_ref   = false;
    double energy_last_total = 0.0;
    double energy_max_abs_dev = 0.0;  // max |E_k - E_0| / |E_0|
    std::ofstream energy_log;
    if (energy_on && rank == 0) {
        energy_log = openEnergyLog(out_dir);
        if (energy_exact && N_total > 20000)
            std::cout << "Warning: --energy-exact is O(N^2) with N=" << N_total
                      << "; consider -b or a larger --et\n";
    }

    if (rank == 0) {
        std::cout << "Starting simulation (dt=" << dt << ", tend=" << t_end << ")\n";
        std::cout << "LEAF_CAPACITY           " << leaf_capacity << std::endl;
        if (timers.any) {
            std::ostringstream os;
            os << "TIMERS_ENABLED          ";
            bool first = true;
            auto tag = [&](PhaseTimers::Phase p, const char* name) {
                if (!timers.enabled[static_cast<std::size_t>(p)]) return;
                if (!first) os << ',';
                os << name;
                first = false;
            };
            tag(PhaseTimers::Phase::Domain,       "domain");
            tag(PhaseTimers::Phase::Migrate,      "migrate");
            tag(PhaseTimers::Phase::TreeBuild,    "tree_build");
            tag(PhaseTimers::Phase::TreeExchange, "tree_exchange");
            tag(PhaseTimers::Phase::Interact,     "interact");
            tag(PhaseTimers::Phase::Energy,       "energy");
            tag(PhaseTimers::Phase::Output,       "output");
            std::cout << os.str() << std::endl;
            if (timers_per_step) {
                std::cout << "TIMERS_HEADER           step t step_total domain migrate "
                             "tree_build tree_exchange interact energy output other" << std::endl;
            }
        }
    }

    auto sim_start_time = std::chrono::high_resolution_clock::now();
    auto last_vis_time = sim_start_time;

    std::vector<std::vector<uint64_t>> rank_domain_keys;
    std::vector<std::pair<long long, int>> last_global_hist;
    bool printed_step0_timers = false;

    // Step 0 pays every one-time cost the rest of the run doesn't: HPX
    // thread-pool/parcelport warm-up, lazily created communicators, the
    // first (cold) tree build and LET exchange, and first-touch page
    // faults. It is also the only step *guaranteed* to migrate bodies
    // (step % rebalance_interval == 0), write a snapshot, and take an
    // energy sample (all gated by "|| step == 0" below), so it is not
    // representative of steady-state per-step cost. Measured on rank 0
    // with no barrier, on the same clock/basis as TOTAL_TIME below, so
    // STEP0_WALLTIME + STEADY_TIME == TOTAL_TIME to the printed digit.
    long long steps_run = 0;
    auto step0_end_time = sim_start_time;

    for (int step = 0; t < t_end; ++step, t += dt) {
        timers.beginStep();
        timers.start(PhaseTimers::Phase::StepTotal);

        // 1. half kick
        if (t > 0) {
            hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), local_pos.size(), [&](std::size_t i) {
                local_vel[i].x += 0.5 * dt * local_acc[i].x;
                local_vel[i].y += 0.5 * dt * local_acc[i].y;
                local_vel[i].z += 0.5 * dt * local_acc[i].z;
            });
        }

        // 2. drift
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), local_pos.size(), [&](std::size_t i) {
            local_pos[i].x += dt * local_vel[i].x;
            local_pos[i].y += dt * local_vel[i].y;
            local_pos[i].z += dt * local_vel[i].z;
        });

        // 3. Domain decomposition and load balancing
        timers.start(PhaseTimers::Phase::Domain);
        BoundingBox local_bb = compute_local_bbox(local_pos);
        BoundingBox global_bb = compute_global_bbox(local_bb, ctx);
        BoundingBox bbox_for_histogram_vis = global_bb;
        std::vector<uint64_t> codes = generateMortonCodes(local_pos, global_bb);

        std::vector<int> splitters;
        update_rank_domains(static_cast<int>(size), codes, bucket_bits, rank_domain_keys, last_global_hist, splitters, ctx);
        timers.stop(PhaseTimers::Phase::Domain);

        // With a single locality there are no other ranks to migrate bodies
        // to or from - rebalance_bodies would be a no-op that still pays a
        // full all_to_all round-trip plus clearing/rebuilding every local
        // array from scratch, so skip the whole block entirely.
        if (size > 1 && step % rebalance_interval == 0) {
            if (rank == 0) std::cout << "Step " << step << ": Rebalancing particles...\n";

            timers.start(PhaseTimers::Phase::Migrate);
            rebalance_bodies(static_cast<int>(rank), static_cast<int>(size), splitters, codes,
                             local_pos, local_mass, local_vel, local_ids,
                             bucket_bits, ctx);

            // Re-touch local_acc: rebalance is the only point body count
            // (and thus this placement) can go stale.
            numa_layout.growWithFirstTouch(local_acc, local_pos.size());

            local_bb = compute_local_bbox(local_pos);
            global_bb = compute_global_bbox(local_bb, ctx);
            codes = generateMortonCodes(local_pos, global_bb);
            timers.stop(PhaseTimers::Phase::Migrate);
        }

        // 4-5. prepare for and build the local tree
        timers.start(PhaseTimers::Phase::TreeBuild);
        sortBodiesByMortonKey(codes, local_pos, local_mass, local_vel, local_ids);
        OctreeMap my_tree = buildOctreeBottomUp(codes, local_pos, local_mass, leaf_capacity);
        timers.stop(PhaseTimers::Phase::TreeBuild);

        // 6. Exchange trees
        OctreeMap full_tree;
        std::vector<NodeRecord> remote_nodes;
        std::vector<int> remote_node_counts;

        timers.start(PhaseTimers::Phase::TreeExchange);
        switch (fcPol) {
            case FCPolicy::Tree:
                exchangeFullTrees(my_tree, full_tree, static_cast<int>(rank), static_cast<int>(size), ctx);
                break;
            case FCPolicy::LET:
            case FCPolicy::LET_DirectSum:
                exchangeEssentialTrees(my_tree, remote_nodes, remote_node_counts, global_bb, theta,
                    static_cast<int>(rank), static_cast<int>(size), rank_domain_keys, max_let_depth, bucket_bits, leaf_capacity, ctx);
                break;
            default:
                throw std::runtime_error("invalid force-calculation policy");
        }
        timers.stop(PhaseTimers::Phase::TreeExchange);

        // At a single locality, exchangeFullTrees leaves full_tree untouched
        // (nothing to gather/merge - see there), so FCPolicy::Tree must read
        // my_tree directly instead.
        const OctreeMap& tree_for_tree_policy = (size > 1) ? full_tree : my_tree;

        // 7. Compute new accelerations
        timers.start(PhaseTimers::Phase::Interact);
        switch (fcPol) {
            case FCPolicy::Tree:
                computeAccelerations(tree_for_tree_policy, codes, local_pos, local_mass, theta, G, soft2, global_bb, local_acc);
                break;
            case FCPolicy::LET:
                computeAccelerationsWithLET(my_tree, remote_nodes, codes, local_pos, local_mass, theta, G, soft2, global_bb, local_acc, static_cast<int>(rank));
                break;
            case FCPolicy::LET_DirectSum:
                computeAccelerationsWithRemoteDirectSum(my_tree, remote_nodes, codes, local_pos, local_mass, theta, G, soft2, global_bb, local_acc);
                break;
        }
        timers.stop(PhaseTimers::Phase::Interact);

        // 8. Second half-kick
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), local_pos.size(), [&](std::size_t i) {
            local_vel[i].x += 0.5 * dt * local_acc[i].x;
            local_vel[i].y += 0.5 * dt * local_acc[i].y;
            local_vel[i].z += 0.5 * dt * local_acc[i].z;
        });

        // 8b. Energy-conservation diagnostics. Positions and velocities are
        //     time-synchronised at exactly this point of the KDK leapfrog,
        //     which is the only place KE and PE may be combined.
        //     The predicate below depends only on replicated state (t, dt,
        //     flags), so every locality enters the collective on the same
        //     steps - required by the ctx.gen_energy generation gate.
        if (energy_on && (t >= next_energy_time || step == 0)) {
            timers.start(PhaseTimers::Phase::Energy);
            // FCPolicy::LET: computeAccelerationsWithLET() already merged the
            // remote pseudo-leaves into my_tree in place, so my_tree is the
            // same tree the force walk used - do NOT merge again.
            const OctreeMap& energy_tree = (fcPol == FCPolicy::Tree) ? tree_for_tree_policy : my_tree;

            EnergySample e = computeGlobalEnergy(
                fcPol, energy_exact, energy_tree, remote_nodes, codes,
                local_pos, local_vel, local_mass, local_ids,
                theta, G, soft2, global_bb, ctx);
            timers.stop(PhaseTimers::Phase::Energy);

            if (!have_energy_ref) { energy_ref_total = e.total; have_energy_ref = true; }
            const double rel_drift = (energy_ref_total != 0.0)
                ? (e.total - energy_ref_total) / std::abs(energy_ref_total) : 0.0;
            energy_last_total  = e.total;
            energy_max_abs_dev = std::max(energy_max_abs_dev, std::abs(rel_drift));

            if (rank == 0) {
                std::ostringstream os;    // don't touch std::cout's sticky format state
                os << std::scientific << std::setprecision(9)
                   << "ENERGY_STEP  " << step << ' ' << t << ' '
                   << e.kinetic << ' ' << e.potential << ' ' << e.total << ' '
                   << rel_drift;
                std::cout << os.str() << std::endl;
                appendEnergyRow(energy_log, step, t, e.kinetic, e.potential, e.total, rel_drift);
            }
            next_energy_time += et_interval;
        }

        // 9. Visualization and report
        if (t >= next_vis_time || step == 0) {
            timers.start(PhaseTimers::Phase::Output);
            writeSnapshot(static_cast<int>(rank), vis_step, local_ids, local_mass,
                          local_pos, local_vel, local_acc, out_dir);

            if (fcPol == FCPolicy::LET) {
                writeReceivedLETs(static_cast<int>(rank), vis_step, remote_nodes, remote_node_counts, global_bb, out_dir);
            }

            if (rank == 0 && !last_global_hist.empty()) {
                writeHistogram(vis_step, last_global_hist, bbox_for_histogram_vis, out_dir, bucket_bits);
            }

            ctx.barrier.wait();

            if (rank == 0) {
                auto current_vis_time = std::chrono::high_resolution_clock::now();
                auto vis_interval_duration = std::chrono::duration_cast<std::chrono::duration<double>>(current_vis_time - last_vis_time).count();
                last_vis_time = current_vis_time;

                updatePVDFile(args, static_cast<int>(size), vis_step, t, out_dir);
                std::cout << "Saved frame " << vis_step << " at t=" << t
                          << " (Time since last frame: " << std::fixed << std::setprecision(2) << vis_interval_duration << " s)" << std::endl;

                if (fcPol == FCPolicy::LET) {
                    updateReceivedLETPVDFile(args, static_cast<int>(size), vis_step, t, out_dir);
                }
                if (!last_global_hist.empty()) {
                    updateHistogramPVDFile(args, vis_step, t, out_dir);
                }
            }

            next_vis_time += vs_interval;
            vis_step++;
            timers.stop(PhaseTimers::Phase::Output);
        }

        // End of step: reduce this step's timings (max across localities)
        // and print the first-timestep breakdown / optional per-step line.
        timers.stop(PhaseTimers::Phase::StepTotal);
        if (timers.any) {
            auto g = timers.endStep();
            if (rank == 0) {
                using Phase = PhaseTimers::Phase;
                double other = g[static_cast<std::size_t>(Phase::StepTotal)];
                for (std::size_t i = 0; i + 1 < PhaseTimers::N; ++i)
                    if (timers.enabled[i]) other -= g[i];

                if (!printed_step0_timers) {
                    printed_step0_timers = true;
                    std::ostringstream os;
                    os << std::fixed << std::setprecision(6);
                    os << "STEP0_STEP_TOTAL        " << g[static_cast<std::size_t>(Phase::StepTotal)];
                    if (timers.enabled[static_cast<std::size_t>(Phase::Domain)])
                        os << '\n' << "STEP0_DOMAIN            " << g[static_cast<std::size_t>(Phase::Domain)];
                    if (timers.enabled[static_cast<std::size_t>(Phase::Migrate)])
                        os << '\n' << "STEP0_MIGRATE           " << g[static_cast<std::size_t>(Phase::Migrate)];
                    if (timers.enabled[static_cast<std::size_t>(Phase::TreeBuild)])
                        os << '\n' << "STEP0_TREE_BUILD        " << g[static_cast<std::size_t>(Phase::TreeBuild)];
                    if (timers.enabled[static_cast<std::size_t>(Phase::TreeExchange)])
                        os << '\n' << "STEP0_TREE_EXCHANGE     " << g[static_cast<std::size_t>(Phase::TreeExchange)];
                    if (timers.enabled[static_cast<std::size_t>(Phase::Interact)])
                        os << '\n' << "STEP0_INTERACT          " << g[static_cast<std::size_t>(Phase::Interact)];
                    if (timers.enabled[static_cast<std::size_t>(Phase::Energy)])
                        os << '\n' << "STEP0_ENERGY            " << g[static_cast<std::size_t>(Phase::Energy)];
                    if (timers.enabled[static_cast<std::size_t>(Phase::Output)])
                        os << '\n' << "STEP0_OUTPUT            " << g[static_cast<std::size_t>(Phase::Output)];
                    os << '\n' << "STEP0_OTHER             " << other;
                    std::cout << os.str() << std::endl;
                }

                if (timers_per_step) {
                    std::ostringstream os;
                    os << "TIMERS_STEP             " << step << ' '
                       << std::fixed << std::setprecision(4) << t << ' '
                       << std::setprecision(6)
                       << g[static_cast<std::size_t>(Phase::StepTotal)] << ' '
                       << g[static_cast<std::size_t>(Phase::Domain)] << ' '
                       << g[static_cast<std::size_t>(Phase::Migrate)] << ' '
                       << g[static_cast<std::size_t>(Phase::TreeBuild)] << ' '
                       << g[static_cast<std::size_t>(Phase::TreeExchange)] << ' '
                       << g[static_cast<std::size_t>(Phase::Interact)] << ' '
                       << g[static_cast<std::size_t>(Phase::Energy)] << ' '
                       << g[static_cast<std::size_t>(Phase::Output)] << ' '
                       << other;
                    std::cout << os.str() << std::endl;
                }
            }
        }

        if (++steps_run == 1)
            step0_end_time = std::chrono::high_resolution_clock::now();
    }

    auto sim_end_time = std::chrono::high_resolution_clock::now();

    double summed_dist_error = performReferenceAndErrorAnalysis(
        is_reference, ref_dir, ctx, N_total, local_pos, local_ids
    );

    if (rank == 0) {
        auto total_sim_duration_s = std::chrono::duration_cast<std::chrono::duration<double>>(sim_end_time - sim_start_time).count();
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "TOTAL_TIME          " << total_sim_duration_s << std::endl;
        std::cout << "SUMMED_DIST_ERROR   " << summed_dist_error << std::endl;

        // First-timestep isolation: split TOTAL_TIME into step 0 (warm-up,
        // guaranteed migrate/output/energy - see the comment where steps_run
        // is declared) and the steady-state remainder, on the same rank-0,
        // barrier-free clock basis as TOTAL_TIME above, unconditionally (one
        // extra clock read) so this is available on the unperturbed
        // configuration the benchmark scripts use, not only under --timers.
        // Deliberately named to avoid the substring "TOTAL_TIME": the
        // benchmark_S*.sh scripts harvest total time via an unanchored
        // `grep "TOTAL_TIME"`, and a second match would corrupt that parse.
        const double step0_s = std::chrono::duration_cast<std::chrono::duration<double>>(
            step0_end_time - sim_start_time).count();
        const double steady_s = (steps_run > 1) ? (total_sim_duration_s - step0_s) : 0.0;
        {
            std::ostringstream os;
            os << std::fixed << std::setprecision(6);
            os << "STEPS                   " << steps_run << '\n'
               << "STEP0_WALLTIME          " << step0_s << '\n'
               << "STEADY_TIME             " << steady_s << '\n'
               << "STEADY_AVG_STEP         "
               << (steps_run > 1 ? steady_s / static_cast<double>(steps_run - 1) : 0.0);
            std::cout << os.str() << std::endl;
        }

        if (energy_on && have_energy_ref) {
            std::ostringstream os;
            os << std::scientific << std::setprecision(6)
               << "ENERGY_INITIAL      " << energy_ref_total << '\n'
               << "ENERGY_FINAL        " << energy_last_total << '\n'
               << "ENERGY_REL_DRIFT     " << (energy_ref_total != 0.0
                     ? (energy_last_total - energy_ref_total) / std::abs(energy_ref_total) : 0.0) << '\n'
               << "ENERGY_MAX_REL_DEV   " << energy_max_abs_dev;
            std::cout << os.str() << std::endl;
        }

        // Per-phase aggregate over the whole run. PHASE_INPUT is printed
        // unconditionally (see the input-timing comment above); every other
        // PHASE_* line only appears for a phase that was actually enabled via
        // --timers. TOTAL_TIME already excludes input (sim_start_time is
        // taken after the initial scatter), so no TOTAL_TIME_EXCL_INPUT is
        // needed; TOTAL_TIME_EXCL_OUTPUT is the doc's "exclude I/O" figure.
        {
            using Phase = PhaseTimers::Phase;
            auto idx = [](Phase p) { return static_cast<std::size_t>(p); };

            std::ostringstream os;
            os << std::fixed << std::setprecision(6);
            os << "PHASE_INPUT             " << timers.input_seconds;
            if (timers.any) {
                os << '\n' << "PHASE_STEP_TOTAL        " << timers.sum[idx(Phase::StepTotal)]
                   << '\n' << "PHASE_STEPS             " << timers.steps_timed;
                if (timers.enabled[idx(Phase::Domain)])
                    os << '\n' << "PHASE_DOMAIN            " << timers.sum[idx(Phase::Domain)]
                       << '\n' << "PHASE_DOMAIN_N          " << timers.n[idx(Phase::Domain)];
                if (timers.enabled[idx(Phase::Migrate)])
                    os << '\n' << "PHASE_MIGRATE           " << timers.sum[idx(Phase::Migrate)]
                       << '\n' << "PHASE_MIGRATE_N         " << timers.n[idx(Phase::Migrate)];
                if (timers.enabled[idx(Phase::TreeBuild)])
                    os << '\n' << "PHASE_TREE_BUILD        " << timers.sum[idx(Phase::TreeBuild)]
                       << '\n' << "PHASE_TREE_BUILD_N      " << timers.n[idx(Phase::TreeBuild)];
                if (timers.enabled[idx(Phase::TreeExchange)])
                    os << '\n' << "PHASE_TREE_EXCHANGE     " << timers.sum[idx(Phase::TreeExchange)]
                       << '\n' << "PHASE_TREE_EXCHANGE_N   " << timers.n[idx(Phase::TreeExchange)];
                if (timers.enabled[idx(Phase::Interact)])
                    os << '\n' << "PHASE_INTERACT          " << timers.sum[idx(Phase::Interact)]
                       << '\n' << "PHASE_INTERACT_N        " << timers.n[idx(Phase::Interact)];
                if (timers.enabled[idx(Phase::Energy)])
                    os << '\n' << "PHASE_ENERGY            " << timers.sum[idx(Phase::Energy)]
                       << '\n' << "PHASE_ENERGY_N          " << timers.n[idx(Phase::Energy)];
                if (timers.enabled[idx(Phase::Output)])
                    os << '\n' << "PHASE_OUTPUT            " << timers.sum[idx(Phase::Output)]
                       << '\n' << "PHASE_OUTPUT_N          " << timers.n[idx(Phase::Output)];

                double other_total = timers.sum[idx(Phase::StepTotal)];
                for (std::size_t i = 0; i + 1 < PhaseTimers::N; ++i)
                    if (timers.enabled[i]) other_total -= timers.sum[i];
                os << '\n' << "PHASE_OTHER             " << other_total;

                if (timers.enabled[idx(Phase::Output)]) {
                    os << '\n' << "TOTAL_TIME_EXCL_OUTPUT  "
                       << (total_sim_duration_s - timers.sum[idx(Phase::Output)]);
                }
            }
            std::cout << os.str() << std::endl;
        }
    }
}

}    // namespace

HPX_PLAIN_ACTION(worker_main, worker_main_action);

int hpx_main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    std::vector<hpx::future<void>> futs;
    futs.reserve(localities.size());
    for (auto const& loc : localities)
        futs.push_back(hpx::async<worker_main_action>(loc));

    hpx::when_all(futs).get();

    return hpx::finalize();
}

int main(int argc, char** argv)
{
    g_argc = argc;
    g_argv = argv;
    return hpx::init(argc, argv);
}
