"""
Input, output and reporting.
"""

"Struct-of-arrays particle set used for input, output and the serial reference."
mutable struct ParticleSet
    id::Vector{Int64}
    m::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    vx::Vector{Float64}
    vy::Vector{Float64}
    vz::Vector{Float64}
end

function ParticleSet(n::Integer)
    return ParticleSet(zeros(Int64, n), zeros(n), zeros(n), zeros(n), zeros(n),
                       zeros(n), zeros(n), zeros(n))
end

Base.length(p::ParticleSet) = length(p.id)

const CSV_HEADER = "id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z"

# The reference writes two extra descriptive columns:
#   id,name,class,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z
# Both shapes are accepted; `name` and `class` carry no physics.
const CSV_REQUIRED = ("id", "mass", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z")

# Accepted spellings per field, lower-cased.  Keplerian element files are not
# covered: they carry different physics, not different names.
const CSV_ALIASES = Dict{String,Vector{String}}(
    "id"    => ["id", "body_id"],
    "mass"  => ["mass", "m"],
    "pos_x" => ["pos_x", "x", "posx", "px"],
    "pos_y" => ["pos_y", "y", "posy", "py"],
    "pos_z" => ["pos_z", "z", "posz", "pz"],
    "vel_x" => ["vel_x", "vx", "velx"],
    "vel_y" => ["vel_y", "vy", "vely"],
    "vel_z" => ["vel_z", "vz", "velz"],
)

"Column index per required field, from a header line."
function _csv_columns(header::AbstractString, path::AbstractString)
    names = [lowercase(strip(replace(c, '"' => ""))) for c in split(header, ',')]
    idx = Dict{String,Int}()
    for (i, nm) in enumerate(names)
        haskey(idx, nm) || (idx[nm] = i)
    end
    cols = Vector{Int}(undef, length(CSV_REQUIRED))
    missing_cols = String[]
    for (k, field) in enumerate(CSV_REQUIRED)
        hit = findfirst(a -> haskey(idx, a), CSV_ALIASES[field])
        if hit === nothing
            push!(missing_cols, field)
        else
            cols[k] = idx[CSV_ALIASES[field][hit]]
        end
    end
    if !isempty(missing_cols)
        looks_keplerian = all(c -> any(==(c), names), ("a", "e", "ma"))
        hint = looks_keplerian ?
            "\nThis looks like a Keplerian orbital-element file (it has a, e, ma). " *
            "Those describe orbits, not state vectors: convert it first, e.g. run the " *
            "C++ reference once (./simulate --file=<elements> --t_end=1h --vs_dir=ic) " *
            "and read its ic/initial_state.csv here." : ""
        error("$path: header is missing the column(s) $(join(missing_cols, ", ")); " *
              "expected at least ($CSV_HEADER)$hint")
    end
    return (ntuple(k -> cols[k], length(CSV_REQUIRED)), length(names))
end

"""
    read_csv(path) -> ParticleSet

Read and validate a data set.
"""
function read_csv(path::AbstractString)
    isfile(path) || error("input file not found: $path")
    ids = Int64[]; ms = Float64[]
    xs = Float64[]; ys = Float64[]; zs = Float64[]
    vxs = Float64[]; vys = Float64[]; vzs = Float64[]
    cols = nothing          # column indices of the eight required fields
    ncol = 0
    lineno = 0
    open(path, "r") do io
        for line in eachline(io)
            lineno += 1
            s = strip(line)
            (isempty(s) || startswith(s, '#')) && continue
            if cols === nothing && !isnumeric(first(s)) && first(s) != '-' && first(s) != '+'
                cols, ncol = _csv_columns(s, path)
                continue
            end
            parts = split(s, ',')
            if cols === nothing
                # No header: fall back to the documented positional layouts.
                length(parts) in (8, 10) ||
                    error("$path:$lineno: a file without a header must hold 8 columns " *
                          "($CSV_HEADER) or the reference's 10 (with name and class), " *
                          "got $(length(parts))")
                cols = length(parts) == 8 ? (1, 2, 3, 4, 5, 6, 7, 8) : (1, 4, 5, 6, 7, 8, 9, 10)
                ncol = length(parts)
            end
            length(parts) == ncol ||
                error("$path:$lineno: expected $ncol columns, got $(length(parts))")
            try
                push!(ids, parse(Int64, strip(parts[cols[1]])))
                push!(ms, parse(Float64, strip(parts[cols[2]])))
                push!(xs, parse(Float64, strip(parts[cols[3]])))
                push!(ys, parse(Float64, strip(parts[cols[4]])))
                push!(zs, parse(Float64, strip(parts[cols[5]])))
                push!(vxs, parse(Float64, strip(parts[cols[6]])))
                push!(vys, parse(Float64, strip(parts[cols[7]])))
                push!(vzs, parse(Float64, strip(parts[cols[8]])))
            catch e
                error("$path:$lineno: could not parse \"$s\" ($e)")
            end
        end
    end
    isempty(ids) && error("$path: no particles found")
    ps = ParticleSet(ids, ms, xs, ys, zs, vxs, vys, vzs)
    validate_particles(ps, path)
    return ps
end

function validate_particles(ps::ParticleSet, source::AbstractString="input")
    n = length(ps)
    for i in 1:n
        isfinite(ps.m[i]) && ps.m[i] > 0 ||
            error("$source: particle $(ps.id[i]) has non-positive or non-finite mass $(ps.m[i])")
        (isfinite(ps.x[i]) && isfinite(ps.y[i]) && isfinite(ps.z[i])) ||
            error("$source: particle $(ps.id[i]) has a non-finite position")
        (isfinite(ps.vx[i]) && isfinite(ps.vy[i]) && isfinite(ps.vz[i])) ||
            error("$source: particle $(ps.id[i]) has a non-finite velocity")
    end
    length(unique(ps.id)) == n || error("$source: particle ids are not unique")
    for i in 1:n
        ps.id[i] >= 0 ||
            error("$source: particle id $(ps.id[i]) is negative (ids must be >= 0)")
    end
    return ps
end

"""
    write_csv(path, ps)

Write a particle set in the input format (used for the serial reference and for
merging the per-partition pieces of a distributed final state).
"""
function write_csv(path::AbstractString, ps::ParticleSet)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, CSV_HEADER)
        for i in 1:length(ps)
            @printf(io, "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    ps.id[i], ps.m[i], ps.x[i], ps.y[i], ps.z[i],
                    ps.vx[i], ps.vy[i], ps.vz[i])
        end
    end
    return path
end

"""
    write_csv_piece(path, id, m, x, y, z, vx, vy, vz)

Write one partition's particles without a header, for later concatenation.
Runs inside a Dagger task on the process that owns the partition.
"""
function write_csv_piece(path::AbstractString, id, m, x, y, z, vx, vy, vz)
    open(path, "w") do io
        for i in eachindex(id)
            @printf(io, "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    id[i], m[i], x[i], y[i], z[i], vx[i], vy[i], vz[i])
        end
    end
    return path
end

"Concatenate per-partition CSV pieces into one file with a header."
function merge_csv_pieces(path::AbstractString, pieces::Vector{String}; keep::Bool=false)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, CSV_HEADER)
        for p in pieces
            isfile(p) || continue
            open(p, "r") do pio
                write(io, read(pio))
            end
            keep || rm(p; force=true)
        end
    end
    return path
end

# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------

"""
    generate_particles(n; seed, kind) -> ParticleSet

Deterministic synthetic data set.  `:collision` (the default) places two
Plummer spheres on a collision course, which gives the clustered, evolving
density the benchmark asks for; `:plummer` produces a single sphere.
"""
function generate_particles(n::Integer; seed::Integer=20260918, kind::Symbol=:collision,
                            G::Float64=G_AU_DAY_KG,
                            total_mass::Float64=1.0e11 * 1.98847e30,
                            scale::Float64=2.0e5)
    n = Int(n)
    n >= 1 || error("--generate must be at least 1")
    rng = MersenneTwister(seed)
    ps = ParticleSet(n)
    if kind === :plummer || n == 1
        _plummer!(ps, 1:n, rng, total_mass, scale, G, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    elseif kind === :collision
        h = n ÷ 2
        sep = 6 * scale
        # Roughly parabolic approach velocity of the two clusters.
        vrel = 0.5 * sqrt(G * total_mass / sep)
        _plummer!(ps, 1:h, rng, total_mass / 2, scale, G,
                  (-sep / 2, -sep / 8, 0.0), (vrel, 0.0, 0.0))
        _plummer!(ps, (h + 1):n, rng, total_mass / 2, scale, G,
                  (sep / 2, sep / 8, 0.0), (-vrel, 0.0, 0.0))
    else
        error("unknown --generate kind $kind")
    end
    for i in 1:n
        ps.id[i] = i
    end
    return ps
end

function _plummer!(ps::ParticleSet, range, rng, mass::Float64, a::Float64, G::Float64,
                   centre::NTuple{3,Float64}, bulk::NTuple{3,Float64})
    n = length(range)
    n == 0 && return ps
    mpart = mass / n
    vscale = sqrt(G * mass / a)
    for i in range
        # Radius from the Plummer cumulative mass profile.
        u = rand(rng)
        u = clamp(u, 1e-12, 1 - 1e-12)
        r = a / sqrt(u^(-2 / 3) - 1)
        ct = 2 * rand(rng) - 1
        st = sqrt(max(0.0, 1 - ct * ct))
        phi = 2π * rand(rng)
        ps.m[i] = mpart
        ps.x[i] = centre[1] + r * st * cos(phi)
        ps.y[i] = centre[2] + r * st * sin(phi)
        ps.z[i] = centre[3] + r * ct
        # Velocity magnitude by Aarseth's rejection sampling of the Plummer
        # distribution function.
        q = 0.0
        while true
            q = rand(rng)
            g = rand(rng) * 0.1
            if g <= q * q * (1 - q * q)^3.5
                break
            end
        end
        v = q * sqrt(2.0) * vscale * (1 + (r / a)^2)^(-0.25)
        cvt = 2 * rand(rng) - 1
        svt = sqrt(max(0.0, 1 - cvt * cvt))
        pvt = 2π * rand(rng)
        ps.vx[i] = bulk[1] + v * svt * cos(pvt)
        ps.vy[i] = bulk[2] + v * svt * sin(pvt)
        ps.vz[i] = bulk[3] + v * cvt
    end
    return ps
end

"Remove the net momentum (mass-weighted mean velocity) from a particle set."
function recenter_momentum!(ps::ParticleSet)
    mtot = 0.0; px = 0.0; py = 0.0; pz = 0.0
    for i in 1:length(ps)
        mtot += ps.m[i]
        px += ps.m[i] * ps.vx[i]; py += ps.m[i] * ps.vy[i]; pz += ps.m[i] * ps.vz[i]
    end
    mtot == 0 && return ps
    for i in 1:length(ps)
        ps.vx[i] -= px / mtot; ps.vy[i] -= py / mtot; ps.vz[i] -= pz / mtot
    end
    return ps
end

# ---------------------------------------------------------------------------
# ParaView output
# ---------------------------------------------------------------------------

"""
    write_vtk_piece(dir, base, part, nparts, id, m, x, y, z, vx, vy, vz, ax, ay, az)

Write one partition's particles as a piece of a parallel VTK (`.pvtu`) data
set.  Called inside a Dagger task on the owning process, so no particle data
has to move for visualisation output.
"""
function write_vtk_piece(dir::AbstractString, base::AbstractString, part::Int, nparts::Int,
                         id, m, x, y, z, vx, vy, vz, ax, ay, az)
    mkpath(dir)
    points = zeros(Float64, 3, length(x))
    @inbounds for i in eachindex(x)
        points[1, i] = x[i]; points[2, i] = y[i]; points[3, i] = z[i]
    end
    # One VTK_VERTEX cell per particle: a point cloud ParaView can render, and
    # a cell type the parallel (.pvtu) writer supports.
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in eachindex(x)]
    vtk = pvtk_grid(joinpath(dir, base), points, cells; part=part, nparts=nparts,
                    append=true, compress=true)
    vtk["id"] = collect(id)
    vtk["mass"] = collect(m)
    vtk["velocity"] = permutedims(hcat(collect(vx), collect(vy), collect(vz)))
    vtk["acceleration"] = permutedims(hcat(collect(ax), collect(ay), collect(az)))
    return first(vtk_save(vtk))
end

"""
    write_vtp_piece(vs_dir, part, index, id, m, x, y, z, vx, vy, vz, ax, ay, az)

One partition's `.vtp` (VTK PolyData) file for one snapshot.
"""
function write_vtp_piece(vs_dir::AbstractString, part::Int, index::Int,
                         id, m, x, y, z, vx, vy, vz, ax, ay, az)
    dir = joinpath(vs_dir, "time_series", string(part - 1))
    mkpath(dir)
    n = length(x)
    points = zeros(Float64, 3, n)
    @inbounds for i in 1:n
        points[1, i] = x[i]; points[2, i] = y[i]; points[3, i] = z[i]
    end
    # The ".vtp" must be part of the name: WriteVTK inspects the extension, and
    # would otherwise read the snapshot index as one and warn on every write.
    vtk = vtk_grid(joinpath(dir, "sim.$(index).vtp"), points,
                   [MeshCell(PolyData.Verts(), 1:n)]; append=true, compress=true)
    vtk["body_id"] = collect(id)
    vtk["mass"] = collect(m)
    vtk["velocity"] = permutedims(hcat(collect(vx), collect(vy), collect(vz)))
    vtk["acceleration"] = permutedims(hcat(collect(ax), collect(ay), collect(az)))
    speed = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        speed[i] = sqrt(vx[i]^2 + vy[i]^2 + vz[i]^2)
    end
    vtk["speed"] = speed
    return first(vtk_save(vtk))
end

"""
    write_pvd(vs_dir, times, nparts) -> String

The ParaView collection tying every snapshot of every partition together, in
the reference's format: one `<DataSet>` per (time, part), carrying the physical
simulation time rather than the step number, so ParaView's animation runs on
the simulated clock.
"""
function write_pvd(vs_dir::AbstractString, times::Vector{Float64}, nparts::Int)
    mkpath(vs_dir)
    path = joinpath(vs_dir, "simulation.pvd")
    open(path, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\" ",
                    "compressor=\"vtkZLibDataCompressor\">")
        println(io, "\t<Collection>")
        for (step, t) in enumerate(times)
            for part in 0:(nparts - 1)
                println(io, "\t\t<DataSet timestep=\"$t\" group=\"\" part=\"$part\" ",
                            "file=\"time_series/$part/sim.$(step - 1).vtp\"/>")
            end
        end
        println(io, "\t</Collection>")
        print(io, "</VTKFile>")
    end
    return path
end

# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

"""
    write_metrics(path, sections)

Write a TOML report holding the effective configuration, the run environment,
the phase timings and the correctness data of a run.
"""
function write_metrics(path::AbstractString, sections::Dict{String,Any})
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, sections; sorted=true)
    end
    return path
end
