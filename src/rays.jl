# Which parts overlap, and by how much, from lines cast through their triangulated surfaces on
# the GPU. Raycore's ray queries are portable: on ray-tracing hardware they run through
# Mantle's hardware acceleration structure, elsewhere through Raycore's software BVH, and one
# kernel serves both. Each part has a structure of its own (a kernel reads one per dispatch),
# so the hits along a line are that part's alone and alternate between entering and leaving
# it. A part's pose moves the lines into the part, so nothing is rebuilt when a part moves.

"""
    Pose(R, t)

A rigid motion of a part: its point `p` goes to `R * p + t`.
"""
struct Pose
    R::Mat3f
    t::Vec3f
end
Pose() = Pose(Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1), Vec3f(0, 0, 0))
Pose(t::Vec3f) = Pose(Pose().R, t)

"A turn by `θ` about the line along x through profile point `c`."
function turn_x(θ, c::Point2d)
    s, k = sincos(Float32(θ))
    R = Mat3f(1, 0, 0, 0, k, s, 0, -s, k)
    a = Vec3f(0, c[1], c[2])
    return Pose(R, a - R * a)
end

Base.:∘(a::Pose, b::Pose) = Pose(a.R * b.R, a.R * b.t + a.t)

"""
    Traced(backend, mesh)

One part's surface in an acceleration structure of its own: the hardware one where `backend`
has ray-tracing hardware, Raycore's software BVH elsewhere. `lo` and `hi` bound the part.
"""
struct Traced{A}
    accel::A
    lo::Vec3f
    hi::Vec3f
end

function Traced(backend, m::GeometryBasics.Mesh)
    accel = Mantle.supports_hwtlas(backend) ? Mantle.HWTLAS{Raycore.Triangle{UInt32}}(backend) : Raycore.TLAS(backend)
    push!(accel, m; instance_id = UInt32(1))
    Raycore.sync!(accel)
    ps = coordinates(m)
    return Traced(accel, Vec3f(ntuple(i -> minimum(p -> p[i], ps), 3)), Vec3f(ntuple(i -> maximum(p -> p[i], ps), 3)))
end

"Box (lo, hi) of `t` moved by `pose`: around its eight moved corners."
function moved_box(t::Traced, pose::Pose)
    corners = [pose.R * Vec3f(x, y, z) + pose.t for x in (t.lo[1], t.hi[1]) for y in (t.lo[2], t.hi[2]) for z in (t.lo[3], t.hi[3])]
    return Vec3f(ntuple(i -> minimum(c -> c[i], corners), 3)), Vec3f(ntuple(i -> maximum(c -> c[i], corners), 3))
end

"""
The lines' direction: oblique to every axis. Nearly every face of the feeder is extruded along
x or lies across one of the axes; a line along one of them runs along such faces and hits
their facets over and over.
"""
const LINE = normalize(Vec3f(1, 0.41, 0.27))

"""
The grid's directions where the lines cross the plane x = const: turned by an odd angle, so
that no row of lines runs along an edge (most edges here run along an axis, and a whole row
grazing one edge would count it wrong all along).
"""
const GRID = let φ = 0.37f0
    (Vec2f(cos(φ), sin(φ)), Vec2f(-sin(φ), cos(φ)))
end

"""
A grid of parallel lines along `dir` (`LINE`, or x where a part slides along x), `h` apart
where they cross the plane x = `xc`: `ny` along `GRID[1]` and `nz` along `GRID[2]` from (`y0`,
`z0`). Each starts `s0` before that plane and is `length` long.
"""
struct Lines
    dir::Vec3f
    xc::Float32
    y0::Float32
    z0::Float32
    h::Float32
    ny::Int32
    nz::Int32
    s0::Float32
    length::Float32
end
Base.length(l::Lines) = Int(l.ny) * Int(l.nz)

"Where line `i` of `l` crosses the plane x = `l.xc`."
function crossing(l::Lines, i)
    j = Int32(i - 1) % l.ny
    k = Int32(i - 1) ÷ l.ny
    q = Vec2f(l.y0, l.z0) + (j * l.h) * GRID[1] + (k * l.h) * GRID[2]
    return Vec3f(l.xc, q[1], q[2])
end

"""
About `count` lines, but no closer than `finest` mm, that pass through the box (`lo`, `hi`): set
off the grid's round numbers so that no line runs along a mesh edge, each from 1 mm before
x = `x0` to 1 mm past `x1` (a line must start and end outside every part it is traced through).
"""
function Lines(lo::Vec3f, hi::Vec3f, x0, x1; count, finest, dir = LINE)
    xc = (lo[1] + hi[1]) / 2
    drift = (hi[1] - lo[1]) / 2 / dir[1]
    ylo, yhi = lo[2] - drift * abs(dir[2]), hi[2] + drift * abs(dir[2])
    zlo, zhi = lo[3] - drift * abs(dir[3]), hi[3] + drift * abs(dir[3])
    # The window's corners in the grid's directions.
    corners = [Vec2f(y, z) for y in (ylo, yhi) for z in (zlo, zhi)]
    a = extrema(q -> dot(q, GRID[1]), corners)
    b = extrema(q -> dot(q, GRID[2]), corners)
    h = max(Float32(finest), sqrt((a[2] - a[1]) * (b[2] - b[1]) / count))
    a0, b0 = a[1] + 0.0123f0 * h, b[1] + 0.0071f0 * h
    ny = max(0, floor(Int, (a[2] - a0) / h) + 1)
    nz = max(0, floor(Int, (b[2] - b0) / h) + 1)
    q0 = a0 * GRID[1] + b0 * GRID[2]
    return Lines(dir, xc, q0[1], q0[2], h, ny, nz, (xc - x0 + 1) / dir[1], (x1 - x0 + 2) / dir[1])
end

"""
Every hit of each line with the part in `accel`, moved by the pose (`R`, `t`), in order along
the line: `ts[k, i]` for line `i`, `counts[i]` of them (`K` means it may have more).
"""
function trace_lines!(ts, counts, accel, lines::Lines, R::Mat3f, t::Vec3f, K::Int32)
    i = KI.get_global_id(Int32).x
    i <= length(lines) || return nothing
    @inbounds begin
        Rt = transpose(R)
        start = crossing(lines, i) - lines.s0 * lines.dir
        o = Point3f(Rt * (start - t))
        d = Rt * lines.dir
        n = Int32(0)
        tmin = 0.0f0
        while n < K
            hit, _, th, _, _ = Raycore.closest_hit(accel, Raycore.Ray(o = o, d = d, t_min = tmin, t_max = lines.length))
            hit || break
            n += Int32(1)
            ts[n, i] = th
            # Just past this hit, a few floats on: the same triangle is not met again, and a
            # face met right after it (a line clipping a corner) still is.
            tmin = th + max(1.0f-5, th * 4.0f-7)
        end
        counts[i] = n
    end
    return nothing
end

"""
How long each line runs inside both parts, from their hits (`ta`, `ca` and `tb`, `cb`): each
part's hits alternate between entering and leaving it. `odd[i]` marks a line where a part's
hits do not pair up (it met an edge or grazed a face where the floats run out); it counts as
running in neither.
"""
function overlap_lines!(len, odd, ta, ca, tb, cb, n::Int32)
    i = KI.get_global_id(Int32).x
    i <= n || return nothing
    @inbounds begin
        na, nb = ca[i], cb[i]
        ia, ib = Int32(1), Int32(1)
        ina, inb = false, false
        last, acc = 0.0f0, 0.0f0
        while ia <= na || ib <= nb
            if ib > nb || (ia <= na && ta[ia, i] <= tb[ib, i])
                x = ta[ia, i]
                ina && inb && (acc += x - last)
                ina = !ina
                ia += Int32(1)
            else
                x = tb[ib, i]
                ina && inb && (acc += x - last)
                inb = !inb
                ib += Int32(1)
            end
            last = x
        end
        bad = isodd(na) || isodd(nb)
        len[i] = bad ? 0.0f0 : acc
        odd[i] = bad
    end
    return nothing
end

"""
The device buffers of the ray checks, for up to as many lines as `len` holds: two parts' hits
(`ta`, `ca` and `tb`, `cb`), each line's overlap and whether its hits pair up.
"""
mutable struct Scratch{T, C, L, O}
    ta::T
    ca::C
    tb::T
    cb::C
    len::L
    odd::O
end

function Scratch(backend, hits, n)
    z(T, dims...) = KI.zeros(backend, T, dims...)
    return Scratch(z(Float32, hits, n), z(Int32, n), z(Float32, hits, n), z(Int32, n), z(Float32, n), z(Bool, n))
end

"""
    Rays(backend; lines = 2^18, finest = 0.005, hits = 32)

What the ray checks run on: the device, how many `lines` to cast through where two parts'
boxes meet (a thin press fit needs many to be measured; no closer than `finest` mm), how many
hits a line may have in one part, and the buffers, kept from one check to the next.
"""
struct Rays{B, S}
    backend::B
    lines::Int
    finest::Float32
    hits::Int32
    scratch::S
end
Rays(backend = Mantle.defaultbackend(); lines = 2^18, finest = 0.005, hits = 32) =
    Rays(backend, lines, Float32(finest), Int32(hits), Scratch(backend, hits, 2lines))

"`r`'s buffers, grown to hold `n` lines."
function room!(r::Rays, n)
    s = r.scratch
    n <= length(s.len) && return s
    t = Scratch(r.backend, r.hits, nextpow(2, n))
    s.ta, s.ca, s.tb, s.cb, s.len, s.odd = t.ta, t.ca, t.tb, t.cb, t.len, t.odd
    return s
end

"Traces `lines` through part `p` at `pose` into the hit buffers `ts` and `counts`."
function trace!(ts, counts, r::Rays, p::Traced, pose::Pose, lines::Lines)
    launch!(r.backend, trace_lines!, (ts, counts, Adapt.adapt(r.backend, p.accel), lines, pose.R, pose.t, r.hits), length(lines))
    return nothing
end

"""
Sums the first `n` lines' overlaps into `out[1, p]`, counts their unpaired lines into
`out[2, p]` and keeps their most hits in one part in `out[3, p]`: one workgroup of 256.
"""
function reduce_lines!(out, len, odd, ca, cb, n::Int32, p::Int32)
    li = KI.get_local_id(Int32).x
    sv = KI.localmemory(Float32, Val((256,)), Val(1))
    sb = KI.localmemory(Float32, Val((256,)), Val(2))
    sm = KI.localmemory(Float32, Val((256,)), Val(3))
    v, bad, most = 0.0f0, 0.0f0, 0.0f0
    i = Int32(li)
    @inbounds while i <= n
        v += len[i]
        bad += odd[i] ? 1.0f0 : 0.0f0
        most = max(most, Float32(ca[i]), Float32(cb[i]))
        i += Int32(256)
    end
    @inbounds sv[li], sb[li], sm[li] = v, bad, most
    KI.barrier()
    # A tree over the workgroup, one step per halving, written out so no barrier sits in a loop.
    @inbounds li <= 128 && (sv[li] += sv[li + 128]; sb[li] += sb[li + 128]; sm[li] = max(sm[li], sm[li + 128]))
    KI.barrier()
    @inbounds li <= 64 && (sv[li] += sv[li + 64]; sb[li] += sb[li + 64]; sm[li] = max(sm[li], sm[li + 64]))
    KI.barrier()
    @inbounds li <= 32 && (sv[li] += sv[li + 32]; sb[li] += sb[li + 32]; sm[li] = max(sm[li], sm[li + 32]))
    KI.barrier()
    @inbounds li <= 16 && (sv[li] += sv[li + 16]; sb[li] += sb[li + 16]; sm[li] = max(sm[li], sm[li + 16]))
    KI.barrier()
    @inbounds li <= 8 && (sv[li] += sv[li + 8]; sb[li] += sb[li + 8]; sm[li] = max(sm[li], sm[li + 8]))
    KI.barrier()
    @inbounds li <= 4 && (sv[li] += sv[li + 4]; sb[li] += sb[li + 4]; sm[li] = max(sm[li], sm[li + 4]))
    KI.barrier()
    @inbounds li <= 2 && (sv[li] += sv[li + 2]; sb[li] += sb[li + 2]; sm[li] = max(sm[li], sm[li + 2]))
    KI.barrier()
    @inbounds if li == 1
        out[1, p] = sv[1] + sv[2]
        out[2, p] = sb[1] + sb[2]
        out[3, p] = max(sm[1], sm[2])
    end
    return nothing
end

"""
    shared(r, jobs; lines = r.lines)

Volume (mm³) each pair of parts in `jobs`, `(a, pose_a, b, pose_b)` each, shares: the length
about `lines` lines run inside both, through where their boxes meet, times the volume each
millimeter of line stands for. All of it runs on the device and comes back at once.
Lines whose hits do not pair up are left out; more than one in a thousand is an error (a part
whose mesh is not closed).
"""
function shared(r::Rays, jobs::AbstractVector; lines = r.lines)
    out = KI.zeros(r.backend, Float32, 3, length(jobs))
    unit = zeros(length(jobs))
    lines_of = zeros(Int, length(jobs))
    for (p, (a, pa, b, pb)) in enumerate(jobs)
        (alo, ahi), (blo, bhi) = moved_box(a, pa), moved_box(b, pb)
        lo, hi = max.(alo, blo), min.(ahi, bhi)
        any(lo .> hi) && continue
        grid = Lines(lo, hi, min(alo[1], blo[1]), max(ahi[1], bhi[1]); count = lines, finest = r.finest)
        h = grid.h
        n = length(grid)
        n == 0 && continue
        s = room!(r, n)
        trace!(s.ta, s.ca, r, a, pa, grid)
        trace!(s.tb, s.cb, r, b, pb, grid)
        launch!(r.backend, overlap_lines!, (s.len, s.odd, s.ta, s.ca, s.tb, s.cb, Int32(n)), n)
        launch!(r.backend, reduce_lines!, (out, s.len, s.odd, s.ca, s.cb, Int32(n), Int32(p)), 256)
        unit[p] = Float64(h)^2 * Float64(grid.dir[1])
        lines_of[p] = n
    end
    res = Array(out)
    maximum(res[3, :]; init = 0.0f0) < r.hits || error("a line has $(r.hits) hits or more in one part: raise `hits`")
    for p in eachindex(jobs)
        res[2, p] > lines_of[p] / 1000 && error("$(Int(res[2, p])) of $(lines_of[p]) lines do not pair up their hits: a part's mesh is not closed")
    end
    return Float64.(res[1, :]) .* unit
end

"Volume (mm³) parts `a` and `b` share at their poses (`shared(r, jobs)` for one pair)."
shared(r::Rays, a::Traced, pa::Pose, b::Traced, pb::Pose) = only(shared(r, [(a, pa, b, pb)]))

"""
How far each line's stretches of the moving part (`ta`, `ca`) go along it (toward +t for
`sense` 1, −t for −1) before they overlap the other part's stretches (`tb`, `cb`) more than
they do now: up to a stretch ahead, at once where one is already pressing deeper, `Inf` where
nothing is ahead. A line whose hits do not pair up gives `Inf` and is marked in `odd`.
"""
function advance_lines!(adv, odd, ta, ca, tb, cb, sense::Int32, n::Int32)
    i = KI.get_global_id(Int32).x
    i <= n || return nothing
    @inbounds begin
        na, nb = ca[i], cb[i]
        bad = isodd(na) || isodd(nb)
        best = Inf32
        if !bad
            for p in Int32(1):Int32(2):na, q in Int32(1):Int32(2):nb
                m0, m1 = sense > 0 ? (ta[p, i], ta[p + 1, i]) : (-ta[p + 1, i], -ta[p, i])
                s0, s1 = sense > 0 ? (tb[q, i], tb[q + 1, i]) : (-tb[q + 1, i], -tb[q, i])
                if s0 >= m1
                    best = min(best, s0 - m1)
                elseif s0 < m1 < s1 && m0 <= s0
                    best = 0.0f0
                end
            end
        end
        adv[i] = best
        odd[i] = bad
    end
    return nothing
end

"The smallest of the first `n` advances into `out[1, p]` and their unpaired lines into `out[2, p]`: one workgroup of 256."
function reduce_advance!(out, adv, odd, n::Int32, p::Int32)
    li = KI.get_local_id(Int32).x
    sv = KI.localmemory(Float32, Val((256,)), Val(1))
    sb = KI.localmemory(Float32, Val((256,)), Val(2))
    v, bad = Inf32, 0.0f0
    i = Int32(li)
    @inbounds while i <= n
        v = min(v, adv[i])
        bad += odd[i] ? 1.0f0 : 0.0f0
        i += Int32(256)
    end
    @inbounds sv[li], sb[li] = v, bad
    KI.barrier()
    @inbounds li <= 128 && (sv[li] = min(sv[li], sv[li + 128]); sb[li] += sb[li + 128])
    KI.barrier()
    @inbounds li <= 64 && (sv[li] = min(sv[li], sv[li + 64]); sb[li] += sb[li + 64])
    KI.barrier()
    @inbounds li <= 32 && (sv[li] = min(sv[li], sv[li + 32]); sb[li] += sb[li + 32])
    KI.barrier()
    @inbounds li <= 16 && (sv[li] = min(sv[li], sv[li + 16]); sb[li] += sb[li + 16])
    KI.barrier()
    @inbounds li <= 8 && (sv[li] = min(sv[li], sv[li + 8]); sb[li] += sb[li + 8])
    KI.barrier()
    @inbounds li <= 4 && (sv[li] = min(sv[li], sv[li + 4]); sb[li] += sb[li + 4])
    KI.barrier()
    @inbounds li <= 2 && (sv[li] = min(sv[li], sv[li + 2]); sb[li] += sb[li + 2])
    KI.barrier()
    @inbounds if li == 1
        out[1, p] = min(sv[1], sv[2])
        out[2, p] = sb[1] + sb[2]
    end
    return nothing
end

"""
    travel(r, jobs, sense; lines = r.lines)

How far (mm) the first part of each job in `jobs`, `(a, pose_a, b, pose_b)` each, slides along
x (toward +x for `sense` 1, −x for −1) before it runs into the second, its overlaps at the
start not counted: from about `lines` lines exactly along x through where their boxes meet
across x, the least any of them allows. `Inf` where nothing is ahead.
"""
function travel(r::Rays, jobs::AbstractVector, sense; lines = r.lines)
    out = KI.zeros(r.backend, Float32, 2, length(jobs))
    fill!(out, Inf32)
    lines_of = zeros(Int, length(jobs))
    along_x = Vec3f(1, 0, 0)
    for (p, (a, pa, b, pb)) in enumerate(jobs)
        (alo, ahi), (blo, bhi) = moved_box(a, pa), moved_box(b, pb)
        lo, hi = max.(alo, blo), min.(ahi, bhi)
        (lo[2] > hi[2] || lo[3] > hi[3]) && continue
        x0, x1 = min(alo[1], blo[1]), max(ahi[1], bhi[1])
        grid = Lines(Vec3f(x0, lo[2], lo[3]), Vec3f(x1, hi[2], hi[3]), x0, x1; count = lines, finest = r.finest, dir = along_x)
        n = length(grid)
        n == 0 && continue
        s = room!(r, n)
        trace!(s.ta, s.ca, r, a, pa, grid)
        trace!(s.tb, s.cb, r, b, pb, grid)
        launch!(r.backend, advance_lines!, (s.len, s.odd, s.ta, s.ca, s.tb, s.cb, Int32(sense), Int32(n)), n)
        launch!(r.backend, reduce_advance!, (out, s.len, s.odd, Int32(n), Int32(p)), 256)
        lines_of[p] = n
    end
    res = Array(out)
    for p in eachindex(jobs)
        res[2, p] > max(2, lines_of[p] / 1000) && error("$(Int(res[2, p])) of $(lines_of[p]) lines do not pair up their hits: a part's mesh is not closed")
    end
    return Float64.(res[1, :])
end

"A closed surface mesh of solids `build()` makes, in a Gmsh model of its own, elements at most `size` long."
function closed_surface(build; size = 1.0, curvature = 96)
    with_gmsh() do
        gmsh.model.add("traced")
        build()
        occ().synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
        gmsh.model.mesh.generate(2)
        m = closed_mesh()
        GeometryBasics.Mesh(Point3f.(coordinates(m)), faces(m))
    end
end

"Item `it` traced in its loaded pose, its solids fused where they overlap."
Traced(r::Rays, it::Item) = Traced(r.backend, closed_surface(() -> fuse_all!(it.build(true))))

"Every item in `list`, traced, by name."
traced(r::Rays, list) = Dict(it.name => Traced(r, it) for it in list)
