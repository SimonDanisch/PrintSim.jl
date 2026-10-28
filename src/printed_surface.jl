# As-printed surfaces on the GPU, straight from the G-code. A layer of a part is the union of
# its beads, each a capsule around its path (half the bead's width plus the swell), and of the
# voids inside the part that are no real holes: the gaps between sparse infill lines. The
# surface is contoured by dual marching cubes with Hermite data (GPUMeshing's `dualmc`) on a
# grid whose corner planes sit at the layers' mid-heights: an edge within a layer finds the
# capsules' boundary and its normal, an edge between two layers crosses at their shared face,
# straight up or down. So the walls stand vertical within each layer and the steps between
# layers stay sharp.
#
# Which voids are holes does not rest on the slicer's labels alone: Bambu Studio calls a bore's
# perimeter "Overhang wall" where it hangs over a wider bore below, and the bore then has no
# "Outer wall" around it in that layer. A void is a real hole when it reaches the outside of
# the layer, or when outer walls bound it (overhanging ones included), no fill bead does, and
# somewhere in it the nearest bead is farther than `hole`.

const BEAD_FILL = Int32(0)
const BEAD_INNER = Int32(1)
const BEAD_OUTER = Int32(2)

"""
The kind of bead a G-code feature lays: an outer wall (Bambu Studio's "Outer wall", and
"Overhang wall", a perimeter printed over air), an inner wall, or fill (everything else:
infill, top and bottom surfaces, bridges, gap fill).
"""
bead_kind(feature::AbstractString) = feature == "Outer wall" || feature == "Overhang wall" ? BEAD_OUTER :
                                     feature == "Inner wall" ? BEAD_INNER : BEAD_FILL

"""
The beads of one printed part as flat arrays in plate mm, for the device: `ends` `(4, n)`
holds each straight piece's ends `(ax, ay, bx, by)`, with its `width`, its `layer` (one-based
into `layers`) and its [`bead_kind`](@ref); `layers` `(2, nl)` holds each layer's bottom and
top. A layer is as high as its walls (combined infill may lie twice as high inside it), or
as its thinnest bead where it has none.
"""
struct BeadSegments
    ends::Matrix{Float32}
    width::Vector{Float32}
    layer::Vector{Int32}
    kind::Vector{Int32}
    layers::Matrix{Float32}
end

"The pieces of one object while the G-code is read."
struct SegmentLists
    ends::Vector{Float32}
    width::Vector{Float32}
    layer::Vector{Int32}
    kind::Vector{Int32}
    top::Vector{Float64}
    wall::Vector{Float64}
    least::Vector{Float64}
end
SegmentLists() = SegmentLists(Float32[], Float32[], Int32[], Int32[], Float64[], Float64[], Float64[])

"""
    read_segments(path; filament = 1.75, step = 0.3) -> Dict{Int, BeadSegments}

Every extrusion of a Bambu Studio G-code file ([`scan_beads`](@ref)) as
[`BeadSegments`](@ref), by object label id, collected while the file is read.
"""
function read_segments(path; filament = 1.75, step = 0.3)
    lists = Dict{Int, SegmentLists}()
    scan_beads(path; filament, step) do object, block, feature, a, b, z, h, w
        s = get!(SegmentLists, lists, object)
        if isempty(s.top) || s.top[end] != z
            isempty(s.top) || z > s.top[end] || error("object $object: a layer at z = $z after one at $(s.top[end])")
            push!(s.top, z)
            push!(s.wall, 0.0)
            push!(s.least, h)
        end
        kind = bead_kind(feature)
        kind == BEAD_FILL || (s.wall[end] = max(s.wall[end], h))
        s.least[end] = min(s.least[end], h)
        push!(s.ends, a[1], a[2], b[1], b[2])
        push!(s.width, w)
        push!(s.layer, length(s.top))
        push!(s.kind, kind)
    end
    return Dict(object => BeadSegments(reshape(s.ends, 4, :), s.width, s.layer, s.kind,
                                       Float32.(vcat(permutedims(s.top .- ifelse.(s.wall .> 0, s.wall, s.least)),
                                                     permutedims(s.top))))
                for (object, s) in lists)
end

"""
    printed_surface(dev, beads; cell = nothing, swell = 0.0, hole = 0.15) -> (V, F)

The closed surface of one printed part as its `beads` ([`BeadSegments`](@ref)) lay it, in
plate mm, as `Mantle.Buffer`s `(3, nv)` Float32 and `(3, nf)` one-based Int32, faces outward.
`cell` is the grid's side in the plate plane, by default [`gridcell`](@ref) of the part's
extent; `swell` widens every bead on both sides; `hole` is how far from every bead a void
must reach somewhere to be a real hole.
"""
function printed_surface(dev::Mantle.Device, b::BeadSegments; cell::Union{Nothing,Real} = nothing, swell::Real = 0.0,
                         hole::Real = 0.15)
    s = BeadGrid(dev, b; cell, swell)
    inside = beadfield(s, Float32(hole))
    V, F = beadcontour(s, inside)
    Mantle.free!(inside)
    Mantle.free!(s)
    return V, F
end

"""
    printed_surfaces(dev, beads; tolerance, budget = 2^24, cache = nothing, kw...) -> Vector{Tuple{V, F}}

The surfaces of many printed parts ([`printed_surface`](@ref), `kw` passed on), each
simplified within `tolerance` of its own (`GPUMeshing.simplify`). Parts go together into one
mesh of disjoint pieces, up to `budget` faces at a time, and simplify in one run: a round
costs about the same for one part as for fifty, and most parts are small. With a `cache`
directory each part's result is kept under the key of its beads, the settings and the code
that makes it ([`surfacekey`](@ref)), so a new design only rebuilds the parts it changed.
"""
function printed_surfaces(dev::Mantle.Device, beads::AbstractVector{BeadSegments}; tolerance::Real, budget::Int = 2^24,
                          cache::Union{Nothing,AbstractString} = nothing, kw...)
    out = Vector{Tuple{Mantle.Buffer{Float32,2},Mantle.Buffer{Int32,2}}}(undef, length(beads))
    keys = cache === nothing ? nothing : [surfacekey(b; tolerance, kw...) for b in beads]
    todo = Int[]
    for k in eachindex(beads)
        if keys !== nothing && isfile(joinpath(cache, keys[k] * ".f.i32"))
            out[k] = loadsurface(dev, joinpath(cache, keys[k]))
        else
            push!(todo, k)
        end
    end
    isempty(todo) && return out
    made = makesurfaces(dev, beads[todo]; tolerance, budget, kw...)
    for (k, mesh) in zip(todo, made)
        out[k] = mesh
        keys === nothing || savesurface(joinpath(cache, keys[k]), mesh)
    end
    return out
end

"""The source files a printed surface comes from: a change to any makes new cache keys."""
const SURFACE_SOURCES = (joinpath(@__DIR__, "printed_surface.jl"),
                         (joinpath(dirname(pathof(GPUMeshing)), f) for f in ("simplify.jl", "surface.jl", "dualmc.jl", "qef.jl", "dualgrid.jl", "parts.jl"))...)

"""
    surfacekey(beads; tolerance, kw...) -> String

The cache key of a part's simplified surface: SHA-256 of its beads, the tolerance and
settings, and the sources in [`SURFACE_SOURCES`](@ref).
"""
function surfacekey(b::BeadSegments; tolerance, kw...)
    ctx = SHA.SHA256_CTX()
    for a in (b.ends, b.width, b.layer, b.kind, b.layers)
        SHA.update!(ctx, reinterpret(UInt8, vec(a)))
    end
    SHA.update!(ctx, Vector{UInt8}(repr((Float64(tolerance), sort!(collect(pairs(kw)); by = first)))))
    for f in SURFACE_SOURCES
        SHA.update!(ctx, read(f))
    end
    return bytes2hex(SHA.digest!(ctx))
end

"""Writes a surface as `path.v.f32` (vertices, 3×n Float32) and `path.f.i32` (faces, 3×k one-based Int32)."""
function savesurface(path, (V, F))
    mkpath(dirname(path))
    write(path * ".v.f32", Array(Mantle.storage(V)))
    write(path * ".f.i32", Array(Mantle.storage(F)))
    return path
end

function loadsurface(dev::Mantle.Device, path)
    V = reshape(reinterpret(Float32, read(path * ".v.f32")), 3, :)
    F = reshape(reinterpret(Int32, read(path * ".f.i32")), 3, :)
    return Mantle.Buffer(dev, Matrix(V)), Mantle.Buffer(dev, Matrix(F))
end

function makesurfaces(dev::Mantle.Device, beads; tolerance, budget, kw...)
    surfaces = [printed_surface(dev, b; kw...) for b in beads]
    out = Vector{Tuple{Mantle.Buffer{Float32,2},Mantle.Buffer{Int32,2}}}(undef, length(beads))
    batch = Int[]
    function flush!()
        isempty(batch) && return
        V, F, P = GPUMeshing.joinparts(dev, surfaces[batch])
        V2, F2, P2 = GPUMeshing.simplify(dev, V, F; tolerance, part = P)
        for (k, piece) in zip(batch, GPUMeshing.splitparts(dev, V2, F2, P2, length(batch)))
            out[k] = piece
        end
        foreach(Mantle.free!, (V2, F2, P2))
        empty!(batch)
        return
    end
    for k in sortperm([size(F, 2) for (_, F) in surfaces])
        size(surfaces[k][2], 2) + sum(i -> size(surfaces[i][2], 2), batch; init = 0) > budget && flush!()
        push!(batch, k)
    end
    flush!()
    return out
end

"""
    gridcell(w, h; finest = 0.1, layer = 2.5e6, side = 1600)

The grid's cell for a part `w` by `h` on the plate: `finest`, unless a layer would hold more
than `layer` corners or a side more than `side` cells (the voxel hash takes 1625 along each
axis). Walls stand where the beads put them at any cell; the cell bounds the steps between
layers that come out smoothed (up to half a cell) and the narrowest slot that stays open.
"""
gridcell(w::Real, h::Real; finest::Real = 0.1, layer::Real = 2.5e6, side::Real = 1600) =
    max(finest, sqrt(w * h / layer), max(w, h) / side)

"""
A part's beads on the device and the grid they are contoured on: corners `(i, j)` at
`(lox + i cell, loy + j cell)`, `0 ≤ i ≤ nx`, `0 ≤ j ≤ ny`, in one plane per layer at its
mid-height (`planes`, with an empty plane below and above), and the beads of each layer in
bins of `nb` × `nb` cells (`off`, `lists`), each bin listing every bead within `reach` of it.
`faces[k + 1]` is where the layers of planes `k` and `k + 1` meet.
"""
struct BeadGrid
    dev::Mantle.Device
    ends::Mantle.Buffer{Float32,2}
    width::Mantle.Buffer{Float32,1}
    kind::Mantle.Buffer{Int32,1}
    planes::Mantle.Buffer{Float32,1}
    faces::Mantle.Buffer{Float32,1}
    off::Mantle.Buffer{Int32,1}
    lists::Mantle.Buffer{Int32,1}
    lox::Float32
    loy::Float32
    cell::Float32
    nx::Int32
    ny::Int32
    nl::Int32
    nb::Int32
    nbx::Int32
    nby::Int32
    swell::Float32
end

function Mantle.free!(s::BeadGrid)
    foreach(Mantle.free!, (s.ends, s.width, s.kind, s.planes, s.faces, s.off, s.lists))
    return nothing
end

geometry(s::BeadGrid) = (s.lox, s.loy, s.cell, s.nx, s.ny, s.nl)
bins(s::BeadGrid) = (s.nb, s.nbx, s.nby)

function BeadGrid(dev::Mantle.Device, b::BeadSegments; cell::Union{Nothing,Real} = nothing, swell::Real = 0.0)
    ns, nl = length(b.width), Int32(size(b.layers, 2))
    E, W, L, K = Mantle.Buffer(dev, b.ends), Mantle.Buffer(dev, b.width), Mantle.Buffer(dev, b.layer), Mantle.Buffer(dev, b.kind)
    lower, upper = Array(minimum(Mantle.storage(E); dims = 2)), Array(maximum(Mantle.storage(E); dims = 2))
    extent = (max(upper[1], upper[3]) - min(lower[1], lower[3]), max(upper[2], upper[4]) - min(lower[2], lower[4]))
    c, sw = Float32(cell === nothing ? gridcell(extent...) : cell), Float32(swell)
    # How far past a bead's edge its distance stays exact, and the margin around the part.
    reach = sw + 2c
    margin = maximum(Mantle.storage(W)) / 2 + sw + 3c
    lox, loy = min(lower[1], lower[3]) - margin, min(lower[2], lower[4]) - margin
    nx = ceil(Int32, (max(upper[1], upper[3]) + margin - lox) / c)
    ny = ceil(Int32, (max(upper[2], upper[4]) + margin - loy) / c)
    bottom, top = b.layers[1, :], b.layers[2, :]
    planes = vcat(bottom[1] - (top[1] - bottom[1]) / 2, (bottom .+ top) ./ 2, top[end] + (top[end] - bottom[end]) / 2)
    faces = vcat(bottom[1], (top[1:(end - 1)] .+ bottom[2:end]) ./ 2, top[end])
    nb = Int32(8)
    nbx, nby = nx ÷ nb + Int32(1), ny ÷ nb + Int32(1)
    nbins = Int(nbx) * Int(nby) * Int(nl)
    cnt = fill!(Mantle.Buffer(dev, Int32, nbins), 0)
    g = Mantle.Graph(dev)
    launch!(g, countbins!, (cnt, E, W, L, ns, lox, loy, c, nx, ny, nl, nb, nbx, nby, reach), ns)
    runonce!(g)
    off = prefixsum(dev, cnt)
    lists = Mantle.Buffer(dev, Int32, max(lastentry(off), 1))
    fill!(cnt, 0)
    g = Mantle.Graph(dev)
    launch!(g, fillbins!, (lists, off, cnt, E, W, L, ns, lox, loy, c, nx, ny, nl, nb, nbx, nby, reach), ns)
    runonce!(g)
    foreach(Mantle.free!, (cnt, L))
    return BeadGrid(dev, E, W, K, Mantle.Buffer(dev, planes), Mantle.Buffer(dev, faces), off, lists, lox, loy, c, nx, ny, nl,
                    nb, nbx, nby, sw)
end

"""
    beadfield(grid, hole; budget = 2^25) -> inside

Which corners of the grid's layers are inside the part, one bit each (corner `c` of
[`cornerindex`](@ref) at bit `(c - 1) % 32` of word `(c - 1) ÷ 32 + 1`): those within a bead,
and those in voids that are no real holes. The voids are the connected components of the
corners outside every bead, in each layer, found as Shiloach and Vishkin do: every corner
starts as its own root; a hooking pass hangs the larger of two neighbouring roots under the
smaller (compare-and-swap, so a root hangs once), and pointer jumping (`label[i] =
label[label[i]]`, pass after pass) flattens the trees again, until no pass hooks. Each
dispatch does a fixed amount of work per corner: walking parent chains instead runs for
seconds on a layer a thousand cells wide, past a display GPU's watchdog. Layers go in
batches of at most `budget` corners.
"""
function beadfield(s::BeadGrid, hole::Float32; budget::Int = 2^25)
    dev = s.dev
    per, nl = Int(s.nx + 1) * Int(s.ny + 1), Int(s.nl)
    inside = fill!(Mantle.Buffer(dev, UInt32, cld(per * nl, 32)), 0x00000000)
    L = clamp(budget ÷ per, 1, nl)
    D = Mantle.Buffer(dev, Float32, L * per)
    S = Mantle.Buffer(dev, Int32, L * per)
    label = Mantle.Buffer(dev, Int32, L * per)
    flags = Mantle.Buffer(dev, UInt32, L * per)
    apart = Mantle.Buffer(dev, Int32, 1)
    for first in 1:L:nl
        nc = min(L, nl - first + 1) * per
        fill!(flags, 0x00000000)
        g = Mantle.Graph(dev)
        launch!(g, beadfield!, (D, S, s.ends, s.width, s.off, s.lists, nc, Int32(first), geometry(s)..., bins(s)..., s.swell), nc)
        launch!(g, startvoids!, (label, D, nc), nc)
        runonce!(g)
        hooked = true
        while hooked
            fill!(apart, 0)
            g = Mantle.Graph(dev)
            launch!(g, hookvoids!, (apart, label, D, nc, s.nx, s.ny), nc)
            runonce!(g)
            hooked = only(Array(Mantle.storage(apart))) != 0
            jumped = true
            while jumped
                fill!(apart, 0)
                g = Mantle.Graph(dev)
                launch!(g, jumpvoids!, (apart, label, D, nc), nc)
                runonce!(g)
                jumped = only(Array(Mantle.storage(apart))) != 0
            end
        end
        g = Mantle.Graph(dev)
        launch!(g, voidflags!, (flags, label, D, S, s.kind, nc, s.nx, s.ny, s.cell, hole), nc)
        launch!(g, insidebits!, (inside, D, label, flags, nc, Int32((first - 1) * per)), nc)
        runonce!(g)
    end
    foreach(Mantle.free!, (D, S, label, flags, apart))
    return inside
end

"""
    beadcontour(grid, inside) -> (V, F)

The surface between the corners `inside` ([`beadfield`](@ref)) and the others: the voxels
between adjacent planes it passes through (found 32 to a word), their patches' vertices from
the beads' Hermite data, and [`dualmc`](@ref)'s quads around every crossed edge. A corner's
value for the asymptotic decider is its distance to the nearest bead's edge, held within two
cells, signed by `inside`.
"""
function beadcontour(s::BeadGrid, inside::Mantle.Buffer)
    dev, nx, ny, nl = s.dev, s.nx, s.ny, s.nl
    res = Int(max(nx, ny, nl + 1)) + 1
    res^3 < 2^32 || error("beadcontour: $(nx)×$(ny) cells and $nl layers exceed the voxel hash; take a larger cell")
    nvox = Int(nx) * Int(ny) * (Int(nl) + 1)
    nwords = cld(nvox, 32)
    words = Mantle.Buffer(dev, UInt32, nwords)
    counts = Mantle.Buffer(dev, Int32, nwords)
    g = Mantle.Graph(dev)
    launch!(g, activewords!, (words, counts, inside, nwords, Int32(nvox), nx, ny, nl), nwords)
    runonce!(g)
    offs = prefixsum(dev, counts)
    n = lastentry(offs)
    idx = Mantle.Buffer(dev, Int32, max(n, 1))
    g = Mantle.Graph(dev)
    launch!(g, emitvoxels!, (idx, words, offs, nwords), nwords)
    runonce!(g)
    foreach(Mantle.free!, (words, counts, offs))
    coords = Mantle.Buffer(dev, Int32, (3, max(n, 1)))
    keys = Mantle.Buffer(dev, Int32, max(n, 1))
    g = Mantle.Graph(dev)
    launch!(g, voxelkeys!, (coords, keys, idx, inside, s.ends, s.width, s.off, s.lists, n, geometry(s)..., bins(s)..., s.swell), n)
    runonce!(g)
    Mantle.free!(idx)
    grid = SparseGrid(dev, coords, n, res)
    dp = DualPatches(dev, keys)
    P = Mantle.Buffer(dev, Float32, (3, max(dp.count, 1)))
    g = Mantle.Graph(dev)
    launch_indexed_batches!(g, beadpatches!, (P, dp.pos, keys, dp.table, coords, inside, s.ends, s.width, s.off, s.lists,
                                              s.planes, s.faces, geometry(s)..., bins(s)..., s.swell), n)
    runonce!(g)
    V, F = dualmc(dev, grid, dp, P)
    foreach(Mantle.free!, (keys, P))
    Mantle.free!(dp)
    Mantle.free!(grid)
    return V, F
end

# ── kernels ──────────────────────────────────────────────────────────────────

"""Distance from `p` to the segment `a b` and the segment's point closest to `p`."""
@inline function segmentpoint(p::Vec2f, a::Vec2f, b::Vec2f)
    d = b - a
    l2 = dot(d, d)
    t = l2 > 0f0 ? clamp(dot(p - a, d) / l2, 0f0, 1f0) : 0f0
    q = a + t * d
    return norm(p - q), q
end

@inline segmentends(E, s) = @inbounds (Vec2f(E[1, s], E[2, s]), Vec2f(E[3, s], E[4, s]))

"""The bins piece `s` reaches into: those whose centre lies within its capsule plus `reach` and half the bin's diagonal."""
@inline function binrange(E, W, s, lox, loy, cell, nb, nbx, nby, reach)
    a, b = segmentends(E, s)
    r = @inbounds W[s] / 2f0 + reach
    side = cell * Float32(nb)
    bx0 = clamp(unsafe_trunc(Int32, floor((min(a[1], b[1]) - r - lox) / side)), Int32(0), nbx - Int32(1))
    bx1 = clamp(unsafe_trunc(Int32, floor((max(a[1], b[1]) + r - lox) / side)), Int32(0), nbx - Int32(1))
    by0 = clamp(unsafe_trunc(Int32, floor((min(a[2], b[2]) - r - loy) / side)), Int32(0), nby - Int32(1))
    by1 = clamp(unsafe_trunc(Int32, floor((max(a[2], b[2]) + r - loy) / side)), Int32(0), nby - Int32(1))
    return a, b, r + side * 0.7072f0, side, bx0, bx1, by0, by1
end

function countbins!(cnt, E, W, L, ns, lox, loy, cell, nx, ny, nl, nb, nbx, nby, reach)
    s = Int32(KI.get_global_id().x)
    s <= ns || return nothing
    a, b, r, side, bx0, bx1, by0, by1 = binrange(E, W, s, lox, loy, cell, nb, nbx, nby, reach)
    layer = @inbounds L[s]
    for by in by0:by1, bx in bx0:bx1
        centre = Vec2f(lox + (Float32(bx) + 0.5f0) * side, loy + (Float32(by) + 0.5f0) * side)
        first(segmentpoint(centre, a, b)) <= r || continue
        @inbounds Atomix.@atomic cnt[bx + nbx * (by + nby * (layer - Int32(1))) + Int32(1)] += Int32(1)
    end
    return nothing
end

function fillbins!(lists, off, cursor, E, W, L, ns, lox, loy, cell, nx, ny, nl, nb, nbx, nby, reach)
    s = Int32(KI.get_global_id().x)
    s <= ns || return nothing
    a, b, r, side, bx0, bx1, by0, by1 = binrange(E, W, s, lox, loy, cell, nb, nbx, nby, reach)
    layer = @inbounds L[s]
    for by in by0:by1, bx in bx0:bx1
        centre = Vec2f(lox + (Float32(bx) + 0.5f0) * side, loy + (Float32(by) + 0.5f0) * side)
        first(segmentpoint(centre, a, b)) <= r || continue
        k = bx + nbx * (by + nby * (layer - Int32(1))) + Int32(1)
        slot = @inbounds Atomix.@atomic cursor[k] += Int32(1)
        @inbounds lists[(k == Int32(1) ? Int32(0) : off[k - Int32(1)]) + slot] = s
    end
    return nothing
end

"""
`(d, s)`: how far `p` lies outside the nearest bead of layer `layer` (negative inside) and
that bead's piece, from the bin around `p`; `(Inf32, 0)` with no bead within reach. Of two
pieces equally near, the lower one, whatever order the bin lists them in.
"""
@inline function beaddistance(p::Vec2f, layer, E, W, off, lists, lox, loy, cell, nb, nbx, nby, swell)
    side = cell * Float32(nb)
    bx = clamp(unsafe_trunc(Int32, floor((p[1] - lox) / side)), Int32(0), nbx - Int32(1))
    by = clamp(unsafe_trunc(Int32, floor((p[2] - loy) / side)), Int32(0), nby - Int32(1))
    k = bx + nbx * (by + nby * (layer - Int32(1))) + Int32(1)
    best, which = Inf32, Int32(0)
    @inbounds for i in (k == Int32(1) ? Int32(1) : off[k - Int32(1)] + Int32(1)):off[k]
        s = lists[i]
        a, b = segmentends(E, s)
        d = first(segmentpoint(p, a, b)) - (W[s] / 2f0 + swell)
        (d < best || (d == best && s < which)) && ((best, which) = (d, s))
    end
    return best, which
end

"""The corner `(i, j)` of layer `l` (one-based) for flat corner index `idx`."""
@inline function cornerof(idx, nx, ny)
    m = idx - Int32(1)
    i = m % (nx + Int32(1))
    m ÷= nx + Int32(1)
    j = m % (ny + Int32(1))
    return i, j, m ÷ (ny + Int32(1)) + Int32(1)
end

@inline cornerindex(i, j, l, nx, ny) = i + (nx + Int32(1)) * (j + (ny + Int32(1)) * (l - Int32(1))) + Int32(1)

function beadfield!(D, S, E, W, off, lists, nc, first, lox, loy, cell, nx, ny, nl, nb, nbx, nby, swell)
    idx = Int32(KI.get_global_id().x)
    idx <= nc || return nothing
    i, j, l = cornerof(idx, nx, ny)
    d, s = beaddistance(Vec2f(lox + Float32(i) * cell, loy + Float32(j) * cell), first + l - Int32(1), E, W, off, lists, lox,
                        loy, cell, nb, nbx, nby, swell)
    @inbounds D[idx] = d
    @inbounds S[idx] = s
    return nothing
end

"""Each void corner starts as its own root; corners in beads get 0."""
function startvoids!(label, D, nc)
    idx = Int32(KI.get_global_id().x)
    idx <= nc || return nothing
    @inbounds label[idx] = D[idx] > 0f0 ? idx : Int32(0)
    return nothing
end

"""
Hangs the larger of the roots of a void corner and its void neighbour to the right, and above,
under the smaller, where it still is a root; sets `hooked` if one did. The trees are flat
when the pass starts, so a corner's label is its root.
"""
function hookvoids!(hooked, label, D, nc, nx, ny)
    idx = Int32(KI.get_global_id().x)
    idx <= nc || return nothing
    @inbounds D[idx] > 0f0 || return nothing
    i, j, _ = cornerof(idx, nx, ny)
    i < nx && hookvoid!(hooked, label, D, idx, idx + Int32(1))
    j < ny && hookvoid!(hooked, label, D, idx, idx + nx + Int32(1))
    return nothing
end

"""Hangs the larger root of void corners `a` and `b` under the smaller, if `b` is void and it still is a root."""
@inline function hookvoid!(hooked, label, D, a, b)
    @inbounds D[b] > 0f0 || return nothing
    ra, rb = @inbounds(label[a]), @inbounds(label[b])
    ra == rb && return nothing
    hi, lo = max(ra, rb), min(ra, rb)
    old = (@inbounds Atomix.@atomicreplace label[hi] hi => lo).old
    old == hi && @inbounds(hooked[1] = Int32(1))
    return nothing
end

"""One pointer jump: a void corner takes its parent's parent; sets `jumped` if one moved."""
function jumpvoids!(jumped, label, D, nc)
    idx = Int32(KI.get_global_id().x)
    idx <= nc || return nothing
    @inbounds D[idx] > 0f0 || return nothing
    p = @inbounds label[idx]
    pp = @inbounds label[p]
    if pp != p
        @inbounds label[idx] = pp
        @inbounds jumped[1] = Int32(1)
    end
    return nothing
end

const VOID_OUTSIDE = 0x00000001
const VOID_FILL = 0x00000002
const VOID_OUTER = 0x00000004
const VOID_DEEP = 0x00000008

"""Marks each void component, at its root: reaches the grid's edge, borders fill or outer walls, reaches past `hole`."""
function voidflags!(flags, label, D, S, K, nc, nx, ny, cell, hole)
    idx = Int32(KI.get_global_id().x)
    idx <= nc || return nothing
    d = @inbounds D[idx]
    d > 0f0 || return nothing
    i, j, _ = cornerof(idx, nx, ny)
    bits = 0x00000000
    (i == Int32(0) || j == Int32(0) || i == nx || j == ny) && (bits |= VOID_OUTSIDE)
    if d < 1.5f0 * cell
        kind = @inbounds K[S[idx]]
        kind == BEAD_FILL && (bits |= VOID_FILL)
        kind == BEAD_OUTER && (bits |= VOID_OUTER)
    end
    d > hole && (bits |= VOID_DEEP)
    bits == 0x00000000 || @inbounds Atomix.@atomic flags[label[idx]] |= bits
    return nothing
end

@inline isholeflags(f) = (f & VOID_OUTSIDE) != 0 ||
                         ((f & VOID_OUTER) != 0 && (f & VOID_FILL) == 0 && (f & VOID_DEEP) != 0)

"""Sets the bit of every corner of the batch inside the part: within a bead, or in a void that is no hole."""
function insidebits!(inside, D, label, flags, nc, offset)
    idx = Int32(KI.get_global_id().x)
    idx <= nc || return nothing
    d = @inbounds D[idx]
    d <= 0f0 || !isholeflags(@inbounds flags[label[idx]]) || return nothing
    c = offset + idx - Int32(1)
    @inbounds Atomix.@atomic inside[(c >> 5) + Int32(1)] |= UInt32(1) << (c & Int32(31))
    return nothing
end

"""Whether corner `(i, j)` of plane `k` is inside: the empty planes 0 and `nl + 1` are not."""
@inline function insideat(inside, i, j, k, nx, ny, nl)
    (k == Int32(0) || k == nl + Int32(1)) && return false
    c = cornerindex(i, j, k, nx, ny) - Int32(1)
    return (@inbounds(inside[(c >> 5) + Int32(1)]) >> (c & Int32(31))) & UInt32(1) != UInt32(0)
end

"""
The value at corner `(i, j)` of plane `k` for the asymptotic decider: the distance to the
nearest bead's edge, held between a millionth of a cell and two cells, negative inside.
"""
@inline function cornervalue(inside, E, W, off, lists, i, j, k, lox, loy, cell, nx, ny, nl, nb, nbx, nby, swell)
    (k == Int32(0) || k == nl + Int32(1)) && return cell
    d, _ = beaddistance(Vec2f(lox + Float32(i) * cell, loy + Float32(j) * cell), k, E, W, off, lists, lox, loy, cell, nb, nbx,
                        nby, swell)
    a = clamp(abs(d), 1f-6 * cell, 2f0 * cell)
    return insideat(inside, i, j, k, nx, ny, nl) ? -a : a
end

"""The voxel `(i, j, k)` (zero-based) for flat voxel index `v`."""
@inline function voxelof(v, nx, ny)
    m = v - Int32(1)
    i = m % nx
    m ÷= nx
    return i, m % ny, m ÷ ny
end

"""The marching cubes cube of voxel `(i, j, k)` from its corners' inside bits."""
@inline function voxelcube(inside, i, j, k, nx, ny, nl)
    cube = Int32(0)
    for c in Int32(0):Int32(7)
        insideat(inside, i + (c & Int32(1)), j + ((c >> 1) & Int32(1)), k + ((c >> 2) & Int32(1)), nx, ny, nl) &&
            (cube |= Int32(1) << c)
    end
    return cube
end

"""Thirty-two voxels per thread: which the surface passes through, and how many."""
function activewords!(words, counts, inside, nwords, nvox, nx, ny, nl)
    w = Int32(KI.get_global_id().x)
    w <= nwords || return nothing
    bits = UInt32(0)
    for b in Int32(0):Int32(31)
        v = Int32(32) * (w - Int32(1)) + b + Int32(1)
        v <= nvox || break
        i, j, k = voxelof(v, nx, ny)
        cube = voxelcube(inside, i, j, k, nx, ny, nl)
        (cube == Int32(0) || cube == Int32(255)) || (bits |= UInt32(1) << b)
    end
    @inbounds words[w] = bits
    @inbounds counts[w] = Int32(count_ones(bits))
    return nothing
end

function emitvoxels!(idx, words, offs, nwords)
    w = Int32(KI.get_global_id().x)
    w <= nwords || return nothing
    bits = @inbounds words[w]
    slot = w == Int32(1) ? Int32(0) : @inbounds(offs[w - Int32(1)])
    for b in Int32(0):Int32(31)
        if (bits >> b) & UInt32(1) != UInt32(0)
            slot += Int32(1)
            @inbounds idx[slot] = Int32(32) * (w - Int32(1)) + b + Int32(1)
        end
    end
    return nothing
end

function voxelkeys!(coords, keys, idx, inside, E, W, off, lists, n, lox, loy, cell, nx, ny, nl, nb, nbx, nby, swell)
    t = Int32(KI.get_global_id().x)
    t <= n || return nothing
    i, j, k = voxelof(@inbounds(idx[t]), nx, ny)
    @inbounds coords[1, t], coords[2, t], coords[3, t] = i, j, k
    f(dx, dy, dz) = cornervalue(inside, E, W, off, lists, i + dx, j + dy, k + dz, lox, loy, cell, nx, ny, nl, nb, nbx, nby, swell)
    o, l = Int32(0), Int32(1)
    @inbounds keys[t] = cellkey(f(o, o, o), f(l, o, o), f(o, l, o), f(l, l, o), f(o, o, l), f(l, o, l), f(o, l, l), f(l, l, l))
    return nothing
end

"""
Where the bead edge of layer `layer` crosses the grid edge from `p0` to `p0 + step`, and the
bead's outward normal there, `(point, normal)` in the plate plane: by bisection on the beads'
distance, `inside` telling which side `p0` is on.
"""
@inline function beadedge(p0::Vec2f, step::Vec2f, layer, inside, E, W, off, lists, lox, loy, cell, nb, nbx, nby, swell)
    t0, t1 = 0f0, 1f0
    for _ in Int32(1):Int32(16)
        t = (t0 + t1) / 2f0
        d, _ = beaddistance(p0 + t * step, layer, E, W, off, lists, lox, loy, cell, nb, nbx, nby, swell)
        (d < 0f0) == inside ? (t0 = t) : (t1 = t)
    end
    p = p0 + (t0 + t1) / 2f0 * step
    _, s = beaddistance(p, layer, E, W, off, lists, lox, loy, cell, nb, nbx, nby, swell)
    sa, sb = segmentends(E, s)
    _, q = segmentpoint(p, sa, sb)
    return p, normalize(p - q)
end

"""
Each patch's vertex: the QEF of its crossings, kept in its voxel. An edge between two planes
crosses at their layers' shared face (`Zf`), straight up or down; an edge within a plane
(`Zp`) at its layer's bead edge ([`beadedge`](@ref)).
"""
function beadpatches!(P, pos, keys, table, coords, inside, E, W, off, lists, Zp, Zf, lox, loy, cell, nx, ny, nl, nb, nbx, nby,
                      swell, first, count)
    g = Int32(KI.get_global_id().x)
    g <= count || return nothing
    t = first + g - Int32(1)
    @inbounds begin
        i, j, k = coords[1, t], coords[2, t], coords[3, t]
        code = table[keys[t] + Int32(1)]
        np = patchcount(code)
        base = pos[t] - np
        lo = Vec3f(lox + Float32(i) * cell, loy + Float32(j) * cell, Zp[k + Int32(1)])
        hi = Vec3f(lo[1] + cell, lo[2] + cell, Zp[k + Int32(2)])
        for p in Int32(1):np
            a11 = a12 = a13 = a22 = a23 = a33 = 0f0
            b = Vec3f(0f0)
            m = Vec3f(0f0)
            count = 0f0
            for a in Int32(1):Int32(3), u in Int32(0):Int32(1), w in Int32(0):Int32(1)
                # The edge's low corner: the two offsets across axis `a`.
                ox, oy, oz = a == Int32(1) ? (Int32(0), u, w) : a == Int32(2) ? (u, Int32(0), w) : (u, w, Int32(0))
                patchof(code, GPUMeshing.localedge(a, ox, oy, oz)) == p || continue
                ins = insideat(inside, i + ox, j + oy, k + oz, nx, ny, nl)
                p0 = Vec2f(lox + Float32(i + ox) * cell, loy + Float32(j + oy) * cell)
                if a == Int32(3)
                    x = Vec3f(p0[1], p0[2], Zf[k + Int32(1)])
                    nn = Vec3f(0f0, 0f0, ins ? 1f0 : -1f0)
                else
                    step = a == Int32(1) ? Vec2f(cell, 0f0) : Vec2f(0f0, cell)
                    q, n2 = beadedge(p0, step, k + oz, ins, E, W, off, lists, lox, loy, cell, nb, nbx, nby, swell)
                    x = Vec3f(q[1], q[2], Zp[k + oz + Int32(1)])
                    nn = Vec3f(n2[1], n2[2], 0f0)
                end
                a11 += nn[1] * nn[1]; a12 += nn[1] * nn[2]; a13 += nn[1] * nn[3]
                a22 += nn[2] * nn[2]; a23 += nn[2] * nn[3]; a33 += nn[3] * nn[3]
                b += nn * dot(nn, x)
                m += x
                count += 1f0
            end
            v = qefsolve((a11, a12, a13, a22, a23, a33), b, m / count)
            v = Vec3f(clamp(v[1], lo[1], hi[1]), clamp(v[2], lo[2], hi[2]), clamp(v[3], lo[3], hi[3]))
            P[1, base + p], P[2, base + p], P[3, base + p] = v
        end
    end
    return nothing
end
