"""
Run configuration.
"""

# Gravitational constant in the units of the benchmark data generator:
# astronomical units, days and kilograms.  Override with `--G` for data sets in
# other units.
const G_SI = 6.67430e-11              # m^3 / (kg s^2)
const AU_IN_M = 1.49597870691e11
const DAY_IN_S = 86400.0
const G_AU_DAY_KG = G_SI * DAY_IN_S^2 / AU_IN_M^3

# Time units of the reference C++ implementation: the internal unit is days
# and a duration may carry a suffix h/H, d/D, m/M, y/Y.  DAY_IN_HOUR is the
# reference's rounded 0.0416667 rather than the exact 1/24.
const DAY_IN_HOUR = 0.0416667
const DAY_IN_MONTH = 30.4167
const DAY_IN_YEAR = 365.25

"""
    parse_duration(s) -> Float64

A time in days.  Accepts a bare number (already days, so unit-free data sets
keep working) or a number with one of the reference's suffixes: `1h`, `0.5d`,
`3m`, `1y`.
"""
function parse_duration(s::AbstractString)
    str = strip(s)
    isempty(str) && error("empty duration")
    tail = last(str)
    (isdigit(tail) || tail == '.') && return parse(Float64, str)
    isletter(tail) ||
        error("duration \"$str\": the last character must be a digit or one of h, d, m, y")
    head = str[1:prevind(str, lastindex(str))]
    isempty(head) && error("duration \"$str\": no number before the unit")
    value = parse(Float64, head)
    factor = if tail == 'h' || tail == 'H'
        DAY_IN_HOUR
    elseif tail == 'd' || tail == 'D'
        1.0
    elseif tail == 'm' || tail == 'M'
        DAY_IN_MONTH
    elseif tail == 'y' || tail == 'Y'
        DAY_IN_YEAR
    else
        error("duration \"$str\": unit must be one of h, H, d, D, m, M, y, Y")
    end
    return value * factor
end

"Days as the reference's pretty string, e.g. `1 years` or `9 months 26 days`."
function format_duration(days::Float64)
    days > 0 || return "$(days) days"
    out = String[]
    y = floor(Int, days / DAY_IN_YEAR); days -= y * DAY_IN_YEAR
    mo = floor(Int, days / DAY_IN_MONTH); days -= mo * DAY_IN_MONTH
    d = floor(Int, days); days -= d
    h = days / DAY_IN_HOUR
    y != 0 && push!(out, "$y years")
    mo != 0 && push!(out, "$mo months")
    d != 0 && push!(out, "$d days")
    h >= 0.005 && push!(out, string(round(h; digits=2), " hours"))
    return isempty(out) ? "0 hours" : join(out, " ")
end

@kwdef mutable struct Config
    # input / output
    input::String = ""                      # CSV path; empty means synthetic
    output::String = "output/run"
    generate::Int = 0                        # synthetic particle count
    generate_kind::Symbol = :collision       # :collision | :plummer
    total_mass::Float64 = 1.0e11 * 1.98847e30   # kg, for generated data
    length_scale::Float64 = 2.0e5               # AU, Plummer scale radius
    seed::Int = 20260918
    # time integration
    dt::Float64 = DAY_IN_HOUR                # 1 hour, the reference's default
    steps::Int = 10
    t_end::Float64 = -1.0                    # days; overrides `steps` when > 0
    # physics
    G::Float64 = G_AU_DAY_KG
    softening::Float64 = 1e-11
    soften_potential::Bool = false
    recenter_momentum::Bool = false
    # Barnes-Hut
    algorithm::Symbol = :barneshut            # :barneshut | :bruteforce
    theta::Float64 = 0.5
    leaf_capacity::Int = 16
    max_depth::Int = MAX_LEVEL
    max_branches::Int = 0                     # 0 == automatic (see branch_budget)
    export_groups::Int = 0                    # 0 == automatic (see export_groups)
    import_fanin::Int = 0                     # 0 == automatic (see import_fanin)
    # Dagger execution
    backend::Symbol = :distributed            # :distributed (addprocs) | :mpi
    workers::Int = 0                          # addprocs count; 0 == reuse existing
    partitions::Int = 0                       # 0 == one per process; never more
    task_batch_size::Int = 0                  # 0 == one task per tile
    tree_tasks::Int = 0                       # subtree tasks per partition; 0 == derive
    tiles::Int = 0                            # tiles per partition; 0 == derive
    tiles_per_thread::Int = 4                 # oversubscribe tiles for balancing
    max_tiles::Int = 64                       # cap on tiles per partition
    parallel_sort::Bool = false               # sample sort instead of one sort per partition
    threads::Int = 0                          # Julia threads per worker process
    use_master::Bool = true                   # process 1 owns a partition too
    check_uniformity::Bool = false
    # load balancing
    load_balance::Bool = false
    rebalance_interval::Int = 0               # steps; 0 == never
    # verification and reporting
    energy_interval::Int = 1                  # 0 == off
    vtk_interval::Int = 0                     # snapshot every N steps; 0 == off
    vs::Float64 = -1.0                        # snapshot every N days; sets vtk_interval
    vs_count::Int = 0                         # total snapshots wanted; sets vtk_interval
    vs_dir::String = ""                       # "" == <output>/sim_out
    vtk_format::Symbol = :vtp                 # :vtp (reference) | :vtu (parallel unstructured)
    initial_state_csv::Bool = false           # write initial_state.csv like the reference
    csv_output::Bool = true
    verify_tree::Bool = false
    detailed_timers::Bool = true
    disable_timers::String = ""               # comma-separated phase names
    warmup_steps::Int = 1                     # excluded from the reported timings
    quiet::Bool = false
end

physics(cfg::Config) = PhysicsParams(cfg.G, cfg.softening, cfg.theta, cfg.soften_potential)

const _CONFIG_HELP = """
nbody-dagger -- hybrid Barnes-Hut with Julia + Dagger (processes x threads)

usage: julia --project bin/simulate.jl [--key=value ...]
       julia --project bin/simulate.jl --config=run.toml [--key=value ...]

input / output
  --input=PATH              CSV data set (id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z)
  --generate=N              generate N particles instead of reading a file
  --generate_kind=collision|plummer
  --total_mass=FLOAT        total mass of the generated system
  --length_scale=FLOAT      Plummer scale radius of the generated system
  --seed=INT                RNG seed for generated data
  --file=PATH               alias for --input (the C++ reference's flag name)
  --output=DIR              output directory (default output/run)
  --csv_output=BOOL         write final_state.csv after the last step
  --initial_state_csv=BOOL  also write initial_state.csv before the first step

visualization (all off by default; give at most one of --vs/--vs_count)
  --vs=DURATION             snapshot every DURATION of simulated time (1d, 6h)
  --vs_count=INT            write exactly INT snapshots, evenly spaced
  --vtk_interval=INT        snapshot every INT steps
  --vs_dir=DIR              where snapshots go (default <output>/sim_out)
  --vtk_format=vtp|vtu      vtp: the reference's .vtp + simulation.pvd layout;
                            vtu: parallel unstructured (.pvtu plus pieces)

time integration
  --dt=DURATION             timestep; a bare number is days, or use a unit
                            suffix: 1h, 2.5d, 3m, 1y (default 1h)
  --steps=INT               number of timesteps
  --t_end=DURATION          end time; overrides --steps.  As in the reference,
                            the run integrates while t <= t_end from t = 0,
                            i.e. floor(t_end/dt) + 1 steps

physics
  --G=FLOAT                 gravitational constant (default AU^3/(kg d^2))
  --softening=FLOAT         softening length epsilon
  --soften_potential=BOOL   use the softened distance in the potential energy
  --recenter_momentum=BOOL  remove the net momentum before the first step

Barnes-Hut
  --algorithm=barneshut|bruteforce
  --algorithm_type=0|1      the reference's spelling: 0 = bruteforce,
                            1 = barneshut
  --theta=FLOAT             opening parameter (0 = brute force)
  --leaf_capacity=INT       particles per leaf cell
  --max_depth=INT           maximum tree depth (<= 21)
  --max_branches=INT        cells published per partition: the coarse-tree
                            granularity and the tightness of the LET pruning
                            (0 = automatic, see branch_budget)
  --export_groups=INT       requester groups an owner exports separately for
                            (0 = automatic); more groups means tighter pruning
                            and less data moved, at more Dagger tasks
  --import_fanin=INT        bundles handed to one assemble_let task, i.e. owner
                            groups concatenated beforehand (0 = automatic)

Dagger execution
  --backend=distributed|mpi
  --workers=INT             addprocs count for the distributed backend
  --partitions=INT          partitions (default: one per Dagger CPU processor)
  --task_batch_size=INT     particles per interaction task; 0 = one task per
                            partition (see interaction_batch)     target particles per interaction task
  --use_master=BOOL         let process 1 own a partition (distributed backend)
  --check_uniformity=BOOL   enable Dagger's SPMD uniformity checks (MPI)

load balancing
  --load_balance=BOOL
  --rebalance_interval=INT  rebalance every INT steps (0 = never)

verification and reporting
  --energy_interval=INT     energy check every INT steps (0 = off)
  --verify_tree=BOOL        check tree invariants every step
  --disable_timers=LIST     comma separated phase names to stop timing, e.g.
                            "energy,let_exchange"; removes their
                            synchronisation points so the phases can overlap
  --max_tiles=INT           cap on tiles per partition (driver cost per tile)
  --parallel_sort=BOOL      sort each partition with a sample sort across its
                            threads; off by default, it loses on this machine
  --detailed_timers=BOOL    time the individual phases
  --warmup_steps=INT        steps excluded from the reported timings
  --quiet=BOOL
  --help
"""

config_help() = _CONFIG_HELP

_parse_value(::Type{String}, s::AbstractString) = String(s)
_parse_value(::Type{Symbol}, s::AbstractString) = Symbol(s)
_parse_value(::Type{Bool}, s::AbstractString) = s in ("1", "true", "yes", "on")
_parse_value(::Type{Int}, s::AbstractString) = parse(Int, s)
_parse_value(::Type{Float64}, s::AbstractString) = parse(Float64, s)

_coerce(::Type{String}, v) = String(v)
_coerce(::Type{Symbol}, v) = Symbol(v)
_coerce(::Type{Bool}, v) = Bool(v)
_coerce(::Type{Int}, v) = Int(v)
_coerce(::Type{Float64}, v) = Float64(v)

# Flag names of the C++ reference (`./simulate --help`), mapped onto ours so
# that a command line written for one runs on the other.
const CXX_ALIASES = Dict{Symbol,Symbol}(
    :file => :input,
    :algorithm_type => :algorithm,
)

# Fields that accept `1h` / `2.5d` / `1y` as well as a bare number of days.
const DURATION_FIELDS = (:dt, :t_end, :vs)

"Translate one `--key=value` pair, applying the C++ aliases."
function _alias(key::Symbol, val::String)
    key === :algorithm_type || return (get(CXX_ALIASES, key, key), val)
    # 0: Naive Brute-Force, 1: Barnes-Hut -- the reference's encoding.
    val == "0" && return (:algorithm, "bruteforce")
    val == "1" && return (:algorithm, "barneshut")
    error("--algorithm_type must be 0 (brute force) or 1 (Barnes-Hut), got \"$val\"")
end

"""
    parse_config(args) -> Config

Parse `--key=value` arguments, optionally seeded from a `--config=FILE` TOML
file.  Unknown keys are an error rather than being ignored, so a typo in a
benchmark script cannot silently change the run.
"""
function parse_config(args::Vector{String})
    cfg = Config()
    fields = fieldnames(Config)
    # A config file is applied first so that command line flags override it.
    for a in args
        if startswith(a, "--config=")
            path = a[10:end]
            for (k, v) in TOML.parsefile(path)
                sym = Symbol(k)
                sym in fields || error("unknown configuration key \"$k\" in $path")
                setfield!(cfg, sym, _coerce(fieldtype(Config, sym), v))
            end
        end
    end
    for a in args
        a == "--help" && return nothing
        startswith(a, "--config=") && continue
        startswith(a, "--") || error("unexpected argument \"$a\" (expected --key=value)")
        eq = findfirst('=', a)
        eq === nothing && error("expected --key=value, got \"$a\"")
        key, val = _alias(Symbol(a[3:(eq - 1)]), String(a[(eq + 1):end]))
        key in fields || error("unknown configuration key \"--$key\"")
        if key in DURATION_FIELDS
            setfield!(cfg, key, parse_duration(val))
        else
            setfield!(cfg, key, _parse_value(fieldtype(Config, key), val))
        end
    end
    validate!(cfg)
    return cfg
end

function validate!(cfg::Config)
    isempty(cfg.input) && cfg.generate <= 0 &&
        error("provide --input=PATH or --generate=N")
    cfg.dt > 0 || error("--dt must be positive")
    if cfg.t_end > 0
        # The reference integrates while `t <= t_end`, starting at t = 0, so it
        # performs one more step than `t_end / dt` whole steps.  Matching it
        # exactly matters: an off-by-one step changes the final state.
        cfg.steps = floor(Int, cfg.t_end / cfg.dt) + 1
    end
    cfg.steps >= 1 || error("--steps must be at least 1")
    # Snapshot cadence: --vs (a time interval, as in the reference) and
    # --vs_count (a number of snapshots) both express themselves as a number of
    # steps.  Giving both is an error rather than a silent precedence rule.
    cfg.vs > 0 && cfg.vs_count > 0 &&
        error("give either --vs (time between snapshots) or --vs_count " *
              "(number of snapshots), not both")
    if cfg.vs > 0
        interval = floor(Int, cfg.vs / cfg.dt)
        interval >= 1 ||
            error("--vs ($(cfg.vs) days) must be at least --dt ($(cfg.dt) days)")
        cfg.vtk_interval = interval
    elseif cfg.vs_count > 0
        cfg.vs_count <= cfg.steps ||
            error("--vs_count ($(cfg.vs_count)) exceeds the number of steps ($(cfg.steps))")
        cfg.vtk_interval = max(1, fld(cfg.steps, cfg.vs_count))
    end
    cfg.vtk_interval >= 0 || error("--vtk_interval must be non-negative")
    if cfg.vtk_interval > cfg.steps && !cfg.quiet
        # Not fatal -- but silently producing only the t = 0 snapshot is the
        # kind of surprise that is discovered after the job has run.
        @warn("--vtk_interval ($(cfg.vtk_interval)) exceeds the number of steps " *
              "($(cfg.steps)): only the initial snapshot will be written. " *
              "Use --vs_count=N for N snapshots, or --vs=DURATION.")
    end
    cfg.vtk_format in (:vtp, :vtu) || error("--vtk_format must be vtp or vtu")
    for t in disabled_timers(cfg)
        t in TIMER_PHASES ||
            error("--disable_timers: unknown phase \"$t\"; known phases are " *
                  join(TIMER_PHASES, ", "))
    end
    cfg.theta >= 0 || error("--theta must be non-negative")
    1 <= cfg.max_depth <= MAX_LEVEL || error("--max_depth must be in 1:$MAX_LEVEL")
    cfg.leaf_capacity >= 1 || error("--leaf_capacity must be at least 1")
    cfg.max_branches >= 0 || error("--max_branches must be non-negative")
    cfg.export_groups >= 0 || error("--export_groups must be non-negative")
    cfg.import_fanin >= 0 || error("--import_fanin must be non-negative")
    cfg.task_batch_size >= 0 || error("--task_batch_size must be non-negative")
    cfg.backend in (:distributed, :mpi) ||
        error("--backend must be distributed or mpi")
    cfg.algorithm in (:barneshut, :bruteforce) ||
        error("--algorithm must be barneshut or bruteforce")
    if cfg.algorithm === :bruteforce
        # The reference's brute force is parallel; this one is the serial O(N^2)
        # oracle.  Refuse rather than silently producing a serial "scaling" result.
        (cfg.workers > 1 || cfg.partitions > 1) &&
            error("--algorithm=bruteforce (--algorithm_type=0) is the serial O(N^2) " *
                  "reference solver; it ignores --workers/--partitions. Drop them to " *
                  "run it, or use --algorithm=barneshut with --theta=0 for the same " *
                  "forces computed by the distributed code.")
        cfg.vtk_interval > 0 &&
            error("--algorithm=bruteforce writes only final_state.csv; it has no " *
                  "snapshot output. Use --algorithm=barneshut for --vs/--vs_count.")
    end
    cfg.warmup_steps >= 0 || error("--warmup_steps must be non-negative")
    cfg.generate_kind in (:collision, :plummer) ||
        error("--generate_kind must be collision or plummer")
    if cfg.load_balance
        # Refuse rather than silently doing nothing: never fall back quietly.
        error("--load_balance=true (interval $(cfg.rebalance_interval)) is not " *
              "implemented yet -- periodic Morton repartitioning is Block 2 work. " *
              "Run with --load_balance=false: the initial distribution is balanced " *
              "(a global Morton sort with equal counts per partition) and stays " *
              "correct as particles drift, but it is not rebalanced.")
    end
    return cfg
end

"Phase names switched off individually with `--disable_timers`."
disabled_timers(cfg::Config) =
    [strip(t) for t in split(cfg.disable_timers, ',') if !isempty(strip(t))]

"Configuration as a `Dict` for the metrics file."
function config_dict(cfg::Config)
    d = Dict{String,Any}()
    for f in fieldnames(Config)
        v = getfield(cfg, f)
        d[String(f)] = v isa Symbol ? String(v) : v
    end
    return d
end
