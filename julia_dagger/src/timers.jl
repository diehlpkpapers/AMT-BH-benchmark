"""
Phase timers.
"""
mutable struct Timers
    detailed::Bool
    total::Float64
    phases::Dict{String,Float64}
    calls::Dict{String,Int}
    disabled::Set{String}                 # phases switched off individually
    first_step::Dict{String,Float64}      # snapshot taken after the first step
end

Timers(detailed::Bool=true, disabled=String[]) =
    Timers(detailed, 0.0, Dict{String,Float64}(), Dict{String,Int}(),
           Set{String}(disabled), Dict{String,Float64}())

"""
    snapshot_first_step!(tm)

Record the per-phase cost of the first measured timestep.
"""
function snapshot_first_step!(tm::Timers)
    isempty(tm.first_step) || return tm
    for (k, v) in tm.phases
        k == "input" && continue          # input happens once, before any step
        tm.first_step[k] = v
    end
    return tm
end

const TIMER_PHASES = ["input", "tree_construction", "let_exchange", "particle_interactions",
                      "energy", "integration", "output", "load_balancing"]

"""
    reset!(tm)

Drop the measured phases, keeping the input timing: the warm-up iterations are
reset away, but reading and distributing the data happened once and stays
excluded from the step timings by being reported separately.
"""
function reset!(tm::Timers)
    tm.total = 0.0
    input = get(tm.phases, "input", nothing)
    input_calls = get(tm.calls, "input", nothing)
    empty!(tm.phases)
    empty!(tm.calls)
    if input !== nothing
        tm.phases["input"] = input
        tm.calls["input"] = input_calls
    end
    return tm
end

"""
    record!(tm, name, seconds)

Add `seconds` to a phase.  Always recorded for `input` and `output` (they have
to be excluded from the primary comparison, so they are timed even when the
detailed timers are off).
"""
function record!(tm::Timers, name::AbstractString, seconds::Float64)
    tm.phases[name] = get(tm.phases, name, 0.0) + seconds
    tm.calls[name] = get(tm.calls, name, 0) + 1
    return seconds
end

# `input` and `output` stay on even with the detailed timers off: the benchmark
# excludes them from the runtime comparison, so they have to be measurable in
# every configuration.  An explicit --disable_timers still switches them off.
always_timed(name::AbstractString) = name in ("input", "output", "total")

"""
    phase!(f, tm, name; sync=nothing)

Run `f` as timing phase `name`.  When the detailed timers are enabled, `sync`
is called after `f` to make the phase's asynchronous work complete before the
clock is stopped; when they are disabled the phase is not timed and `sync` is
skipped, so consecutive phases can overlap.
"""
function phase!(f::Function, tm::Timers, name::AbstractString; sync=nothing)
    if name in tm.disabled || (!tm.detailed && !always_timed(name))
        return f()
    end
    t0 = time_ns()
    result = f()
    sync === nothing || sync(result)
    record!(tm, name, (time_ns() - t0) / 1e9)
    return result
end

function timings_dict(tm::Timers)
    d = Dict{String,Any}("total" => tm.total)
    for p in TIMER_PHASES
        d[p] = get(tm.phases, p, 0.0)
    end
    for (k, v) in tm.phases
        d[k] = v
    end
    return d
end

"Per-phase cost of the first measured timestep, the benchmark's primary metric."
function first_step_dict(tm::Timers)
    d = Dict{String,Any}()
    for p in TIMER_PHASES
        p == "input" && continue
        haskey(tm.first_step, p) && (d[p] = tm.first_step[p])
    end
    return d
end
