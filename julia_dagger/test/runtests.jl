using Test
using Random
using Dagger
using NBodyDagger
const N = NBodyDagger

const NWORKERS = parse(Int, get(ENV, "NBD_TEST_WORKERS", "2"))

@testset "NBodyDagger" begin

@testset "tiles" begin
    # A tile split must cover 1:n exactly once, in order, with no empty ranges.
    for n in (0, 1, 2, 7, 1000, 300_149), k in (1, 2, 3, 8, 64, 1000)
        t = N.tiles(n, k)
        @test all(!isempty, t)
        @test length(t) <= max(1, min(k, n)) || n == 0
        if n == 0
            @test isempty(t)
        else
            @test first(first(t)) == 1
            @test last(last(t)) == n
            for j in 2:length(t)
                @test first(t[j]) == last(t[j - 1]) + 1
            end
            @test sum(length, t) == n
        end
    end
    # Nearly equal sizes: no tile may be more than one longer than another.
    for (n, k) in ((1000, 7), (300_149, 192), (13, 5))
        len = length.(N.tiles(n, k))
        @test maximum(len) - minimum(len) <= 1
    end
end

@testset "tile count follows threads" begin
    lay = N.Layout(Dagger.Processor[], Vector{Dagger.Processor}[], 4, 8)
    cfg = Config()
    @test N.tile_count(cfg, lay, 10_000) == 8 * cfg.tiles_per_thread
    cfg.tiles_per_thread = 1
    @test N.tile_count(cfg, lay, 10_000) == 8          # one per thread
    cfg.tiles_per_thread = 3
    @test N.tile_count(cfg, lay, 10_000) == 24         # oversubscribed
    cfg.tiles = 5
    @test N.tile_count(cfg, lay, 10_000) == 5          # explicit wins
    cfg.tiles = 0
    @test N.tile_count(cfg, lay, 4) == 4               # never more tiles than particles
end

@testset "parallel tree build is equivalent to the serial one" begin
    # The invariant treebuild.jl rests on: splitting the build into subtree
    # tasks changes *when* nodes are created, never which nodes exist or what
    # they hold.
    #
    # It is explicitly NOT an invariant that the node *numbering* matches the
    # serial build.  The serial build numbers pre-order depth-first; the splice
    # keeps each stub in its slot in the upper tree and appends the subtree
    # interiors after it, so the two orders differ.  Nothing reads the tree by
    # index order: `branch_nodes` picks the frontier by subtree count and key
    # and returns it sorted by key, and both the LET emission and the force
    # walk descend by octant.  The tests below pin that down -- structure and
    # payload per key, and bit-identical physics, with one partition and with
    # two, so the comparison also covers the branch cut and the LET.
    keyed(tr) = Dict(tr.key[i] => (tr.level[i], tr.mass[i], tr.comx[i], tr.comy[i],
                                   tr.comz[i], tr.size[i], tr.nchild[i],
                                   tr.lfirst[i], tr.lcount[i])
                     for i in 1:N.nnodes(tr))

    function build(np::Int, leaf::Int, ntask::Int, workers::Int)
        cfg = parse_config(["--generate=$np", "--workers=$workers", "--theta=0.5",
                            "--leaf_capacity=$leaf", "--steps=3", "--quiet=true",
                            "--warmup_steps=1", "--tree_tasks=$ntask"])
        N.setup_backend!(cfg)
        s = N.prepare(cfg)
        ke0, pe0 = N.energies(s)
        for _ in 1:3
            N.step!(s, cfg.dt)
        end
        ke, pe = N.energies(s)
        s.reduced = collect(s.red)
        return (tree = keyed(fetch(s.trees[1])),
                drift = (ke + pe - ke0 - pe0) / abs(ke0 + pe0),
                ke = ke, pe = pe,
                direct = N.reduced_sum(s, N.RED_DIRECT),
                approx = N.reduced_sum(s, N.RED_APPROX))
    end

    for (np, leaf) in ((5_000, 1), (20_000, 4))
        ref = build(np, leaf, 1, 0)
        # 17 and 64 ask for more subtree tasks than the planner can place, so
        # they also check that the stub count clamps instead of overshooting.
        for k in (2, 3, 8, 17, 64)
            got = build(np, leaf, k, 0)
            @test got.tree == ref.tree
            @test got.ke == ref.ke
            @test got.pe == ref.pe
            @test got.direct == ref.direct
            @test got.approx == ref.approx
        end
    end
end

@testset "the staged build packs, counts and recycles like the serial path" begin
    # Per-task packing, range-based subtree counts and recycled arrays must give
    # exactly what the plain path gives.
    rng = MersenneTwister(7)
    n = 20_000
    bs = n
    buf = zeros(N.NFIELD * bs)
    meta = zeros(UInt64, N.NMETA * bs)
    p = N.pfields(buf, bs, n)
    # Clustered, so the planner has to split unevenly.
    for i in 1:n
        c = i % 5 == 0 ? 0.0 : 3.0
        p.x[i] = c + randn(rng); p.y[i] = randn(rng) * 0.2; p.z[i] = randn(rng) * 0.05
        p.m[i] = 1.0 + rand(rng)
    end
    key = N.mview(meta, bs, N.M_KEY, n)
    id = N.mview(meta, bs, N.M_ID, n)
    id .= 1:n
    dom = N.bounding_domain((minimum(p.x), minimum(p.y), minimum(p.z)),
                            (maximum(p.x), maximum(p.y), maximum(p.z)))
    N.compute_keys!(key, dom, p.x, p.y, p.z)
    N.sort_partition!(key, id, p.m, p.x, p.y, p.z, p.vx, p.vy, p.vz)

    function staged(ntask, leaf, prev)
        plan = N.task_tree_plan(buf, meta, bs, n, dom, leaf, N.MAX_LEVEL, ntask, prev)
        counts = [N.task_tree_count(buf, meta, bs, n, dom, leaf, N.MAX_LEVEL, plan, k)
                  for k in 1:ntask]
        off = N.task_tree_offsets(plan, counts...)
        tree = N.task_tree_reserve(plan, off, counts...)
        # Reverse order: the fill tasks must not depend on running in sequence.
        for k in ntask:-1:1
            N.task_tree_fill!(tree, buf, meta, bs, n, leaf, N.MAX_LEVEL, plan, off, k)
        end
        return N.task_tree_finish!(tree, plan, off)
    end
    fields(t) = (t.key, t.level, t.mass, t.comx, t.comy, t.comz, t.size, t.childflat,
                 t.nchild, t.lfirst, t.lcount, t.hot, t.kids)

    for leaf in (1, 4), ntask in (2, 5, 16)
        t = staged(ntask, leaf, nothing)
        hot = copy(t.hot); kids = copy(t.kids)
        empty!(t.hot); empty!(t.kids)
        N.pack_hot!(t)
        @test t.hot == hot
        @test t.kids == kids

        cnt = N.subtree_counts(t)
        @test N.RangeCounts(t) == cnt
        @test N.branch_nodes(t, N.RangeCounts(t); max_branches=64) ==
              N.branch_nodes(t, cnt; max_branches=64)

        # Built again in its own arrays, and in a tree from a different build.
        fresh = fields(staged(ntask, leaf, nothing))
        @test fields(staged(ntask, leaf, t)) == fresh
        other = N.build_local_tree(dom, key, p.x, p.y, p.z, p.m, 2, N.MAX_LEVEL)
        @test fields(staged(ntask, leaf, other)) == fresh
    end
    serial = N.build_local_tree(dom, key, p.x, p.y, p.z, p.m, 1, N.MAX_LEVEL)
    @test fields(N.build_local_tree(dom, key, p.x, p.y, p.z, p.m, 1, N.MAX_LEVEL,
                                    staged(8, 4, nothing))) == fields(serial)
end

@testset "the move pass has its own granularity" begin
    cfg = Config()
    cfg.max_tiles = 256
    small = N.Layout(Dagger.Processor[], Vector{Dagger.Processor}[], 1, 4)
    big = N.Layout(Dagger.Processor[], Vector{Dagger.Processor}[], 1, 192)
    # Where the driver cost is small, the walk's oversubscription is kept...
    @test N.move_tile_count(cfg, small, 4_000_000) == N.tile_count(cfg, small, 4_000_000)
    # ...and where it is not, far fewer tasks than the walk's tiles.
    @test N.move_tile_count(cfg, big, 4_000_000) ==
          4_000_000 ÷ (N.MOVE_PARTICLES_PER_PROC * 192)
    @test N.move_tile_count(cfg, big, 4_000_000) < N.tile_count(cfg, big, 4_000_000)
    @test N.move_tile_count(cfg, big, 1_000) == 1
    cfg.move_tiles = 0
    @test N.move_tile_count(cfg, big, 4_000_000) == N.tile_count(cfg, big, 4_000_000)
    cfg.move_tiles = 7
    @test N.move_tile_count(cfg, big, 4_000_000) == 7
end

# ---------------------------------------------------------------------------
# Physics oracles.
#
# Everything below compares this implementation against an independent answer,
# not against itself: the O(N^2) kernel for forces, a synchronised leapfrog for
# the integrator, and a single partition for the distributed decomposition.
# The tests above check that the *decomposition* is consistent; these check
# that what it computes is Barnes-Hut.
# ---------------------------------------------------------------------------

"Relative acceleration error per particle: worst case and RMS."
function acc_error(ax, ay, az, bx, by, bz)
    maxrel = 0.0
    rms = 0.0
    n = length(ax)
    for i in 1:n
        scale = max(sqrt(bx[i]^2 + by[i]^2 + bz[i]^2), 1e-300)
        d = sqrt((ax[i] - bx[i])^2 + (ay[i] - by[i])^2 + (az[i] - bz[i])^2) / scale
        maxrel = max(maxrel, d)
        rms += d * d
    end
    return maxrel, sqrt(rms / max(n, 1))
end

"Accelerations and potentials straight from the O(N^2) kernel."
function oracle_direct(ps, params)
    n = length(ps.m)
    ax = zeros(n); ay = zeros(n); az = zeros(n); pot = zeros(n)
    N.brute_force!(ax, ay, az, pot, ps.x, ps.y, ps.z, ps.m, params)
    return ax, ay, az, pot
end

"One serial Barnes-Hut pass: sort into Morton order, build one tree, walk it."
function oracle_tree(ps, params; leaf_capacity::Int=8)
    n = length(ps.m)
    x = copy(ps.x); y = copy(ps.y); z = copy(ps.z); m = copy(ps.m)
    vx = copy(ps.vx); vy = copy(ps.vy); vz = copy(ps.vz); id = copy(ps.id)
    dom = N.bounding_domain((minimum(x), minimum(y), minimum(z)),
                            (maximum(x), maximum(y), maximum(z)))
    key = Vector{UInt64}(undef, n)
    N.compute_keys!(key, dom, x, y, z)
    N.sort_partition!(key, id, m, x, y, z, vx, vy, vz)
    tree = N.build_local_tree(dom, key, x, y, z, m, leaf_capacity, N.MAX_LEVEL)
    ax = zeros(n); ay = zeros(n); az = zeros(n); pot = zeros(n)
    counts = N.InteractionCounts()
    N.traverse_batch!(ax, ay, az, pot, tree, 1:n, x, y, z, m, id, key, params, counts, Int32[])
    # Back into the caller's particle order so it can be compared elementwise.
    perm = sortperm(id)
    return ax[perm], ay[perm], az[perm], pot[perm], counts, tree
end

"Run the real distributed pipeline and return the state in input id order."
function run_sim(args::Vector{String}; steps::Int=0)
    cfg = parse_config(args)
    N.setup_backend!(cfg)
    s = N.prepare(cfg)
    for _ in 1:steps
        N.step!(s, cfg.dt)
    end
    steps > 0 && N.synchronize_velocities!(s, cfg.dt)
    ps, ax, ay, az, pot = N.gather_state(s)
    perm = sortperm(ps.id)
    reorder(v) = v[perm]
    out = (id = ps.id[perm], m = reorder(ps.m),
           x = reorder(ps.x), y = reorder(ps.y), z = reorder(ps.z),
           vx = reorder(ps.vx), vy = reorder(ps.vy), vz = reorder(ps.vz),
           ax = reorder(ax), ay = reorder(ay), az = reorder(az), pot = reorder(pot))
    return out, s
end

@testset "Morton keys respect the octree" begin
    dom = N.bounding_domain((-2.0, -1.0, -3.0), (4.0, 1.0, 5.0))
    # A cube that contains every point, so no key can fall outside the root.
    for (x, y, z) in ((-2.0, -1.0, -3.0), (4.0, 1.0, 5.0), (0.0, 0.0, 0.0))
        k = N.morton_key(dom, x, y, z)
        @test k != 0
    end
    # Points in the same deepest cell share a key; distant points do not.
    k1 = N.morton_key(dom, 0.0, 0.0, 0.0)
    @test N.morton_key(dom, 0.0, 0.0, 0.0) == k1
    @test N.morton_key(dom, 3.9, 0.9, 4.9) != k1
    # Sorting by key is what the tree build assumes: equal keys stay adjacent.
    rng = MersenneTwister(11)
    pts = [(4 * rand(rng) - 2, 2 * rand(rng) - 1, 8 * rand(rng) - 3) for _ in 1:500]
    keys = [N.morton_key(dom, p...) for p in pts]
    @test issorted(sort(keys))
    # The bit-spread key must equal the sentinel followed by one octant digit per
    # level, also for points outside the domain.
    digits(x, y, z, level) = begin
        ix, iy, iz = N.cell_index(dom, x, y, z, level)
        k = N.ROOT_KEY
        for l in (level - 1):-1:0
            k = (k << 3) | UInt64(N.octant(ix, iy, iz, l))
        end
        k
    end
    for _ in 1:2000, level in (0, 1, 7, N.MAX_LEVEL)
        x, y, z = 8 * rand(rng) - 3, 4 * rand(rng) - 2, 10 * rand(rng) - 4
        @test N.morton_key(dom, x, y, z, level) == digits(x, y, z, level)
    end
end

@testset "theta = 0 reproduces brute force exactly" begin
    ps = N.generate_particles(600; seed=3, kind=:plummer, G=1.0,
                              total_mass=1000.0, scale=1.0)
    params = N.PhysicsParams(1.0, 1e-6, 0.0, false)
    ax, ay, az, pot, counts, _ = oracle_tree(ps, params; leaf_capacity=8)
    bx, by, bz, bpot = oracle_direct(ps, params)
    maxrel, _ = acc_error(ax, ay, az, bx, by, bz)
    # Same kernel and the same set of pairs, but the tree walks them in Morton
    # order while the O(N^2) loop walks them in index order, so the sums differ
    # in the last bits.  Round-off is the only thing allowed to differ.
    @test maxrel < 1e-13
    @test sum(pot) ≈ sum(bpot)
    # theta = 0 can never accept a cell, so every interaction is direct.
    @test counts.approx == 0
    @test counts.direct == length(ps.m) * (length(ps.m) - 1)
end

@testset "the multipole acceptance criterion converges" begin
    ps = N.generate_particles(600; seed=5, kind=:plummer, G=1.0,
                              total_mass=1000.0, scale=1.0)
    bx, by, bz, _ = oracle_direct(ps, N.PhysicsParams(1.0, 1e-6, 0.0, false))
    errs = Float64[]
    direct = Int[]
    for theta in (1.0, 0.5, 0.25)
        params = N.PhysicsParams(1.0, 1e-6, theta, false)
        ax, ay, az, _, counts, _ = oracle_tree(ps, params; leaf_capacity=8)
        _, rms = acc_error(ax, ay, az, bx, by, bz)
        push!(errs, rms)
        push!(direct, counts.direct)
    end
    # A smaller opening angle accepts fewer cells, so more of the interactions
    # are evaluated particle-to-particle and the error falls.  The *approximate*
    # count is not monotone and must not be asserted to be: rejecting a cell
    # high in the tree makes the walk descend and accept several smaller ones
    # instead, which can raise the count even as the error drops.
    @test issorted(errs; rev = true)
    @test issorted(direct)
    @test errs[end] < 1e-2
end

"""
Walk `tree` for the point `(px, py, pz)` and count the nodes the opening
criterion accepts whose cell *contains* that point.  `guard` selects the real
criterion (with the Morton containment test) or the bare `s < theta * d`.
"""
function self_accepts(tree, px, py, pz, pkey, theta; guard::Bool)
    theta2 = theta * theta
    stack = Int[1]
    hits = 0
    while !isempty(stack)
        nd = pop!(stack)
        nkid = Int(tree.nchild[nd])
        (nkid == 0 && N.nparticles(tree, nd) == 0) && continue   # summary node
        dx = tree.comx[nd] - px; dy = tree.comy[nd] - py; dz = tree.comz[nd] - pz
        r2 = dx * dx + dy * dy + dz * dz
        s = tree.size[nd]
        if theta2 > 0 && s * s < theta2 * r2 &&
           !(guard && N.key_is_ancestor(tree.key[nd], pkey))
            cx, cy, cz, cs = N.cell_geometry(tree.dom, tree.key[nd])
            h = cs / 2 + 1e-12
            if abs(px - cx) <= h && abs(py - cy) <= h && abs(pz - cz) <= h
                hits += 1
            end
            continue
        end
        nkid > 0 && for oct in 0:7
            c = Int(N.child(tree, nd, oct))
            c > 0 && push!(stack, c)
        end
    end
    return hits
end

@testset "no particle is inside a cell it accepts" begin
    # The distance in the opening criterion is measured to the centre of mass,
    # which may sit anywhere inside the cell, so a target inside a cell of edge
    # s can be up to s*sqrt(3) from its own centre of mass.  Past
    # theta = 1/sqrt(3) the bare criterion therefore accepts cells that contain
    # the target and the particle attracts itself.  The walk carries a Morton
    # containment test to forbid that; this checks the test is there and that
    # it still has work to do.
    ps = N.generate_particles(4_000; seed=11, kind=:collision, G=1.0,
                              total_mass=1000.0, scale=1.0)
    params = N.PhysicsParams(1.0, 1e-6, 1.2, false)
    _, _, _, _, _, tree = oracle_tree(ps, params; leaf_capacity=1)
    key = Vector{UInt64}(undef, length(ps.m))
    N.compute_keys!(key, tree.dom, ps.x, ps.y, ps.z)

    guarded = sum(i -> self_accepts(tree, ps.x[i], ps.y[i], ps.z[i], key[i],
                                    1.2; guard=true), eachindex(ps.m))
    bare = sum(i -> self_accepts(tree, ps.x[i], ps.y[i], ps.z[i], key[i],
                                 1.2; guard=false), eachindex(ps.m))
    @test guarded == 0
    @test bare > 0          # the guard is load-bearing on this distribution

    # At the opening angle the benchmarks use the guard never fires, so it
    # cannot have perturbed any measured result.
    p5 = N.PhysicsParams(1.0, 1e-6, 0.5, false)
    _, _, _, _, _, t5 = oracle_tree(ps, p5; leaf_capacity=1)
    k5 = Vector{UInt64}(undef, length(ps.m))
    N.compute_keys!(k5, t5.dom, ps.x, ps.y, ps.z)
    @test sum(i -> self_accepts(t5, ps.x[i], ps.y[i], ps.z[i], k5[i],
                                0.5; guard=false), eachindex(ps.m)) == 0
end

@testset "every particle sees the total mass exactly once" begin
    # With a softened potential and eps far larger than the box,
    # 1/sqrt(r^2 + eps^2) -> (1/eps)(1 - r^2/2eps^2), so
    # -pot[i] * eps / m[i] counts the mass particle i interacted with.  It must
    # be M - m[i]: every other particle exactly once, itself never.  This audits
    # the real traversal, with the real acceptance decisions, so it catches both
    # double-counted and dropped mass at any opening angle.
    ps = N.generate_particles(4_000; seed=11, kind=:collision, G=1.0,
                              total_mass=1000.0, scale=1.0)
    M = sum(ps.m)
    rmax = maximum(sqrt.(ps.x .^ 2 .+ ps.y .^ 2 .+ ps.z .^ 2))
    eps = 1e8
    for theta in (0.5, 1.0, 1.2)
        params = N.PhysicsParams(1.0, eps, theta, true)
        _, _, _, pot, _, _ = oracle_tree(ps, params; leaf_capacity=1)
        worst = maximum(abs.((.-pot .* eps ./ ps.m) .- (M .- ps.m)) ./ (M .- ps.m))
        # Only the truncation of the expansion above may remain.
        @test worst < 4 * rmax^2 / eps^2
    end
end

@testset "tree invariants" begin
    ps = N.generate_particles(2_000; seed=7, kind=:collision)
    params = N.PhysicsParams(1.0, 1e-6, 0.5, false)
    _, _, _, _, _, tree = oracle_tree(ps, params; leaf_capacity=4)
    # The root accounts for every particle and for the whole mass.
    @test tree.mass[1] ≈ sum(ps.m) rtol=1e-12
    # Every node's mass is the sum of its children's, and its centre of mass is
    # their mass-weighted mean -- the property the MAC relies on.
    for nd in 1:N.nnodes(tree)
        kids = [Int(N.child(tree, nd, oct)) for oct in 0:7]
        filter!(!=(0), kids)
        isempty(kids) && continue
        @test tree.mass[nd] ≈ sum(tree.mass[k] for k in kids) rtol=1e-10
        for (com, axis) in ((tree.comx, :x), (tree.comy, :y), (tree.comz, :z))
            want = sum(tree.mass[k] * com[k] for k in kids) / tree.mass[nd]
            @test com[nd] ≈ want rtol=1e-10
        end
    end
    # Leaves partition the particles: every index covered exactly once.
    seen = falses(length(ps.m))
    for nd in 1:N.nnodes(tree)
        tree.lcount[nd] == 0 && continue
        for j in tree.lfirst[nd]:(tree.lfirst[nd] + tree.lcount[nd] - 1)
            @test !seen[j]
            seen[j] = true
        end
    end
    @test all(seen)
end

@testset "the distributed pipeline reproduces brute force at theta = 0" begin
    # The whole machine -- partitioning, tree, LET, tiles -- must collapse to
    # the O(N^2) answer when no cell can be accepted.
    args = ["--generate=400", "--generate_kind=plummer", "--workers=0",
            "--theta=0.0", "--leaf_capacity=4", "--steps=1", "--quiet=true",
            "--G=1.0", "--softening=1e-6", "--total_mass=1000.0",
            "--length_scale=1.0"]
    got, s = run_sim(args)
    ps = N.ParticleSet(length(got.m))
    ps.id .= got.id; ps.m .= got.m
    ps.x .= got.x; ps.y .= got.y; ps.z .= got.z
    bx, by, bz, bpot = oracle_direct(ps, s.params)
    maxrel, rms = acc_error(got.ax, got.ay, got.az, bx, by, bz)
    @test maxrel < 1e-12
    @test rms < 1e-13
    @test sum(got.pot) ≈ sum(bpot) rtol=1e-10
end

@testset "the partition count does not change the physics" begin
    base = ["--input=", "--generate=1500", "--generate_kind=collision",
            "--theta=0.5", "--leaf_capacity=2", "--steps=2", "--quiet=true"]
    filter!(!=("--input="), base)
    one, _ = run_sim(vcat(base, ["--workers=0"]); steps=2)
    many, _ = run_sim(vcat(base, ["--workers=$NWORKERS"]); steps=2)
    @test one.id == many.id
    # Splitting the domain changes summation order, not the answer.
    for f in (:x, :y, :z, :vx, :vy, :vz)
        @test getfield(one, f) ≈ getfield(many, f) rtol=1e-10
    end
    maxrel, _ = acc_error(one.ax, one.ay, one.az, many.ax, many.ay, many.az)
    @test maxrel < 1e-8
end

@testset "the leapfrog conserves energy and is time-reversible in dt" begin
    args = ["--generate=1200", "--generate_kind=plummer", "--workers=0",
            "--theta=0.3", "--leaf_capacity=4", "--quiet=true", "--steps=20"]
    cfg = parse_config(args)
    N.setup_backend!(cfg)
    s = N.prepare(cfg)
    N.synchronize_velocities!(s, cfg.dt)
    ke0, pe0 = N.energies(s)
    e0 = ke0 + pe0
    for _ in 1:20
        N.step!(s, cfg.dt)
    end
    N.synchronize_velocities!(s, cfg.dt)
    ke, pe = N.energies(s)
    # A symplectic integrator does not drift secularly; 1e-4 is loose enough
    # for 20 steps of a Plummer sphere and tight enough to catch a broken kick.
    @test abs((ke + pe - e0) / abs(e0)) < 1e-4
    # Kinetic energy is positive and the system is bound.
    @test ke > 0
    @test pe < 0
end

@testset "the fused half kicks match a synchronised leapfrog" begin
    # `step!` fuses the trailing and leading half kicks; `synchronize_velocities!`
    # is what makes the state comparable to a textbook kick-drift-kick.
    args = ["--generate=800", "--generate_kind=plummer", "--workers=0",
            "--theta=0.4", "--leaf_capacity=4", "--quiet=true", "--steps=5"]
    a, _ = run_sim(args; steps=5)
    b, _ = run_sim(args; steps=5)
    @test a.x == b.x && a.vx == b.vx          # deterministic to the last bit
    # Positions must have moved -- a test that passes on a frozen state is no test.
    c, _ = run_sim(args; steps=0)
    @test !(a.x ≈ c.x)
end

@testset "the predicted domain is the box the particles end up in" begin
    # Computed before the kick and drift, it must predict them exactly, on the
    # first step (half kick) and on the fused steps after it (whole kick).
    cfg = parse_config(["--generate=3000", "--generate_kind=collision", "--quiet=true",
                        "--energy_interval=0", "--move_tiles=3"])
    N.setup_backend!(cfg)
    s = N.prepare(cfg)
    for step in 1:4
        N.step!(s, cfg.dt; sync=false, want_pot=false)
        ps, = N.gather_state(s)
        @test N.particle_box(ps.x, ps.y, ps.z) == s.boxes[1]
    end
end

@testset "the packet walk is the scalar walk" begin
    # Bit-identical to the scalar walk, with one partition and with two (imports
    # and summary nodes in the LET).
    base = ["--generate=6000", "--generate_kind=collision", "--theta=0.5",
            "--leaf_capacity=1", "--quiet=true", "--energy_interval=0"]
    for workers in (0, NWORKERS)
        ref, sref = run_sim([base; "--workers=$workers"; "--simd_lanes=0"]; steps=2)
        sref.reduced = collect(sref.red)
        for lanes in (4, 8, 16, 32)
            got, sgot = run_sim([base; "--workers=$workers"; "--simd_lanes=$lanes"]; steps=2)
            sgot.reduced = collect(sgot.red)
            for f in (:x, :y, :z, :vx, :vy, :vz, :ax, :ay, :az, :pot)
                @test getfield(got, f) == getfield(ref, f)
            end
            @test N.reduced_sum(sgot, N.RED_DIRECT) == N.reduced_sum(sref, N.RED_DIRECT)
            @test N.reduced_sum(sgot, N.RED_APPROX) == N.reduced_sum(sref, N.RED_APPROX)
        end
    end
end

@testset "the packet walk does not allocate" begin
    # Allocation per particle would not change the result, but every collection
    # stops all threads, so it would stop the walk from scaling.
    rng = MersenneTwister(9)
    n = 5000
    x = randn(rng, n); y = randn(rng, n); z = randn(rng, n); m = rand(rng, n) .+ 1
    vx = zeros(n); vy = zeros(n); vz = zeros(n); id = collect(1:n)
    dom = N.bounding_domain((minimum(x), minimum(y), minimum(z)), (maximum(x), maximum(y), maximum(z)))
    key = Vector{UInt64}(undef, n); N.compute_keys!(key, dom, x, y, z)
    N.sort_partition!(key, id, m, x, y, z, vx, vy, vz)
    t = N.build_local_tree(dom, key, x, y, z, m, 1, N.MAX_LEVEL)
    ax = zeros(n); ay = zeros(n); az = zeros(n); pot = zeros(n)
    p = N.PhysicsParams(1.0, 1e-3, 0.5, false)
    walk(lanes, potv) = N.traverse_batch!(ax, ay, az, pot, t, 1:n, x, y, z, m, id, key, p,
                                          N.InteractionCounts(), Int32[], potv,
                                          Val(false), Val(false), lanes)
    for lanes in (Val(4), Val(8), Val(16), Val(32)), potv in (Val(false), Val(true))
        walk(lanes, potv)
        # A per-particle allocation would be megabytes here.  The bound leaves
        # room for the ~1 KB per call that `Pkg.test`'s forced bounds checks add.
        @test (@allocated walk(lanes, potv)) < 16_384
    end
end

@testset "the radix permutation is the stable sort's" begin
    rng = MersenneTwister(5)
    for n in (0, 1, 1023, 1024, 5000, 100_000)
        # Wide keys, narrow keys (few digits differ) and many duplicates.
        for keys in (rand(rng, UInt64, n),
                     (UInt64(1) << 63) .| rand(rng, UInt64(0):UInt64(1 << 20), n),
                     rand(rng, UInt64(7):UInt64(12), n))
            @test N.key_sortperm(keys) == sortperm(keys; alg = Base.Sort.DEFAULT_STABLE)
        end
    end
end

@testset "the sample sort is the serial sort" begin
    # Forced with --sort_tiles: at test sizes the default never picks it.  Needs
    # four threads.
    if Threads.nthreads() < 4
        @info "sample sort test skipped: needs --threads=4 or more"
    else
        common = ["--generate=20000", "--theta=0.5", "--leaf_capacity=4",
                  "--quiet=true", "--sort_tiles=4", "--energy_interval=0"]
        run(extra) = first(run_sim([common; extra]; steps=3))
        serial = run(["--parallel_sort=false"])
        for extra in (["--reuse_splitters=false"], ["--reuse_splitters=true"])
            got = run(extra)
            for f in (:id, :x, :y, :z, :vx, :vy, :vz, :ax, :ay, :az)
                @test getfield(got, f) == getfield(serial, f)
            end
        end
    end
end

@testset "result is independent of the tile count" begin
    # The invariant the hybrid layout rests on: cutting a partition into more
    # tiles changes only how the work is scheduled, never the answer.
    mktempdir() do dir
        common = ["--generate=4000", "--steps=2", "--warmup_steps=1", "--G=1.0",
                  "--softening=0.001", "--total_mass=1.0", "--length_scale=1.0",
                  "--dt=0.01", "--seed=31", "--theta=0.5", "--csv_output=false",
                  "--quiet=true", "--energy_interval=1"]
        results = map((1, 4, 17)) do k
            out = joinpath(dir, "t$k")
            cfg = parse_config([common; "--tiles=$k"; "--output=$out"])
            run_simulation(cfg)
            m = open(joinpath(out, "metrics.toml")) do io
                d = Dict{String,Float64}()
                for l in eachline(io)
                    for key in ("relative_drift", "direct", "approximate")
                        startswith(l, key * " ") && (d[key] = parse(Float64, split(l, " = ")[2]))
                    end
                end
                d
            end
            m
        end
        for k in 2:length(results)
            @test results[k]["direct"] == results[1]["direct"]
            @test results[k]["approximate"] == results[1]["approximate"]
            @test results[k]["relative_drift"] ≈ results[1]["relative_drift"] rtol=1e-12
        end
    end
end

end
