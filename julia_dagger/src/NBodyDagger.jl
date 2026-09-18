"""
    NBodyDagger

Hybrid Barnes-Hut gravitational N-body solver in Julia, parallelised with
[Dagger.jl](https://github.com/JuliaParallel/Dagger.jl).
"""
module NBodyDagger

using Dagger
using Distributed
using Printf
using Random
using TOML
using WriteVTK

export Config, parse_config, run_simulation, config_help

include("morton.jl")
include("tree.jl")
include("treebuild.jl")
include("kernels.jl")
include("config.jl")
include("io.jl")
include("timers.jl")
include("layout.jl")
include("runtime.jl")
include("sort.jl")
include("simulation.jl")

end # module
