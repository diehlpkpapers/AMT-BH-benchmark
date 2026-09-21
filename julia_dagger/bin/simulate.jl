#!/usr/bin/env julia
# Entry point of the hybrid Barnes-Hut / Dagger simulation.
#
#   julia --project bin/simulate.jl --input=data.csv --steps=10 --workers=4 --threads=8
using NBodyDagger

cfg = parse_config(copy(ARGS))
if cfg === nothing
    println(config_help())
    exit(0)
end

# MPI.jl is a weak dependency; Dagger's MPI extension is loaded here, at top
# level, because `Base.require` defines its methods in a new world age and a
# function that was already entered cannot see them.
cfg.backend === :mpi && NBodyDagger.load_mpi!()

run_simulation(cfg)
