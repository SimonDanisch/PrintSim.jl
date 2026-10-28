# The simulated world as Newton lays it out: a `ModelBuilder` on the host collects what there
# is, `Model` holds it on the device and is never written during a run, `State` holds what
# changes. Units are SI; the device works in Float32.

"""
    SheetMaterial(; young, poisson, thickness, grammage, stretch_damping, bending_damping)

A thin sheet: in-plane Young's modulus (Pa) and Poisson ratio, thickness (m), mass per area
(kg/m²). The damping terms are relative: a force of `damping` times the elastic force for the
same change, per step. Defaults are 80 g/m² office paper.
"""
Base.@kwdef struct SheetMaterial
    young::Float64 = 4.0e9
    poisson::Float64 = 0.3
    thickness::Float64 = 1.0e-4
    grammage::Float64 = 0.080
    stretch_damping::Float64 = 0.0
    bending_damping::Float64 = 0.0
end

"Plate bending stiffness D = E t³ / 12(1 − ν²), in N·m."
bending_stiffness(m::SheetMaterial) = m.young * m.thickness^3 / (12 * (1 - m.poisson^2))

"""
    SheetGrid(length, width, spacing)

A rectangular sheet as a grid of vertices about `spacing` apart: `nu` along its length (the
sheet's u axis), `nv` across its width (v).
"""
struct SheetGrid
    length::Float64
    width::Float64
    nu::Int
    nv::Int
end
SheetGrid(length, width, spacing) = SheetGrid(length, width, round(Int, length / spacing) + 1, round(Int, width / spacing) + 1)

"Where vertex (i, j) lies on the flat sheet, in m from its corner."
flat(g::SheetGrid, i, j) = Vec2((i - 1) * g.length / (g.nu - 1), (j - 1) * g.width / (g.nv - 1))

"""
One sheet in a model: its grid and the index of its first vertex. Vertex (i, j) is
`first + (j - 1) * nu + i - 1`, u running fastest.
"""
struct SheetPiece
    first::Int
    grid::SheetGrid
end
vertex(s::SheetPiece, i, j) = s.first + (j - 1) * s.grid.nu + i - 1
vertices(s::SheetPiece) = s.first:(s.first + s.grid.nu * s.grid.nv - 1)

"""
Sheets stacked in order, bottom first, and how they touch: `static` and `kinetic` friction of
paper on paper, `stiffness` the contact pressure per depth (Pa/m).
"""
struct Stacking
    sheets::Vector{Int}           # indices into the builder's sheets, bottom first
    static::Float64
    kinetic::Float64
    stiffness::Float64
end

"""
A triangle mesh that sheets touch, fixed in the world: its vertices in meters, in Float64 so
that a flat face stays flat to the last bit (two triangles of a tilted face rounded to Float32
meet at a 3 nm kink, which a sliding sheet feels as a jump of 70 times a vertex's weight), its
triangles counterclockwise seen from outside, and how it grips paper.
"""
struct MeshPiece
    vertices::Vector{Vec3{Float64}}
    faces::Vector{NTuple{3, Int32}}
    static::Float64
    kinetic::Float64
    stiffness::Float64          # Pa/m: contact pressure per depth
end

"An affine body's mesh, center of mass, principal frame, length scale and generalized masses."
struct AffinePiece
    mesh::Int
    meshes::Vector{Int}
    center::Vec3{Float64}
    axes::Mat{3, 3, Float64, 9}
    scale::Float64
    mass::Vec{4, Float64}
    orthogonal::Float64
    velocity::Vec3{Float64}
    spin::Vec3{Float64}
    fixed::Bool
end

"A joint only where the assembly has no modeled bearing geometry (for example a motor housing)."
struct JointPiece
    a::Int32
    b::Int32
    anchor_a::Vec3f
    anchor_b::Vec3f
    axis_a::Vec3f
    axis_b::Vec3f
    reference_a::Vec3f
    reference_b::Vec3f
    kind::UInt8
    linear::Float32
    angular::Float32
    damping::Float32
end

"A full-step motor's rotor, housing, local axis/reference and measured available torque."
struct StepperPiece
    rotor::Int32
    housing::Int32
    axis_a::Vec3f
    axis_b::Vec3f
    reference_a::Vec3f
    reference_b::Vec3f
    torque::Float32
    steps::Int32
end

"A tetrahedral elastic part and the surface vertices shared with its volume mesh."
struct TetPiece
    mesh::Int
    x::Vector{Vec3{Float64}}
    initial::Vector{Vec3{Float64}}
    tets::Vector{NTuple{10, Int32}}
    surface_nodes::Vector{Int32}
    young::Float64
    poisson::Float64
    density::Float64
    fixed::Vector{UInt8}
end

"An ideal glued attachment between an affine point and a tetrahedral surface triangle."
struct BondPiece
    body::Int32
    nodes::NTuple{3, Int32}
    weights::Vec3f
    coordinate::Position
    stiffness::Float32
end

"""
    ModelBuilder(; gravity)

Collects particles and the elements between them on the host. `add_sheet!` adds sheets,
`add_mesh!` meshes they touch; `Model(backend, builder)` puts the result on the device.
"""
struct ModelBuilder
    x::Vector{Vec3{Float64}}
    v::Vector{Vec3f}
    mass::Vector{Float32}
    area::Vector{Float32}
    fixed::Vector{UInt8}
    tris::Vector{NTuple{3, Int32}}
    springs::Vector{NTuple{2, Int32}}
    spring_rest::Vector{Float64}                # m, to the device as a pair: `precise.jl`
    spring_stiffness::Vector{Float32}           # N/m
    spring_damping::Vector{Float32}
    hinges::Vector{NTuple{4, Int32}}            # opposite 1, opposite 2, edge start, edge end
    hinge_rest::Vector{Float32}
    hinge_stiffness::Vector{Float32}            # energy ½ k (θ − θ₀)², N·m
    hinge_damping::Vector{Float32}
    sheets::Vector{SheetPiece}
    stacks::Vector{Stacking}
    meshes::Vector{MeshPiece}
    bodies::Vector{AffinePiece}
    joints::Vector{JointPiece}
    motors::Vector{StepperPiece}
    deformables::Vector{TetPiece}
    bonds::Vector{BondPiece}
    excluded::Vector{NTuple{2, Int32}}
    gravity::Base.RefValue{Vec3f}
end

ModelBuilder(; gravity = Vec3f(0, 0, -9.81)) = ModelBuilder(Vec3{Float64}[], Vec3f[], Float32[], Float32[], UInt8[], NTuple{3, Int32}[],
    NTuple{2, Int32}[], Float64[], Float32[], Float32[], NTuple{4, Int32}[], Float32[], Float32[], Float32[],
    SheetPiece[], Stacking[], MeshPiece[], AffinePiece[], JointPiece[], StepperPiece[], TetPiece[], BondPiece[], NTuple{2, Int32}[], Ref(Vec3f(gravity)))

"""
    add_mesh!(builder, mesh; static = 0.3, kinetic = static, stiffness = 2e10) -> index

`mesh` (GeometryBasics, coordinates in meters, triangles counterclockwise from outside) as
something sheets touch. Build it in Float64 (`Point3d`): its faces stay flat only then.
`stiffness` is the contact pressure per depth once the surfaces are pressed into each other,
2·10¹⁰ Pa/m (20 kPa at 1 µm, paper compressed across its thickness); how they take up load
before, on their asperities, is `normal_force`'s.
"""
function add_mesh!(b::ModelBuilder, m::GeometryBasics.Mesh; static = 0.3, kinetic = static, stiffness = 2.0e10)
    vertices = [Vec3{Float64}(p) for p in GeometryBasics.coordinates(m)]
    faces = [NTuple{3, Int32}(Int32.(Tuple(GeometryBasics.value.(f)))) for f in GeometryBasics.decompose(GLTriangleFace, m)]
    push!(b.meshes, MeshPiece(vertices, faces, static, kinetic, stiffness))
    return length(b.meshes)
end

"""
The triangles of a grid: every cell split along the same diagonal, counterclockwise in the
flat sheet's (u, v) plane. Every vertex then sits in the same neighborhood, so a sheet bent
evenly is in equilibrium: a grid whose diagonals alternate gives its vertices of four and of
eight neighbors opposite forces under even bending, and the sheet buckles into a zigzag.
"""
function grid_triangles(g::SheetGrid)
    tris = NTuple{3, Int}[]
    id(i, j) = (j - 1) * g.nu + i
    for j in 1:(g.nv - 1), i in 1:(g.nu - 1)
        a, b, c, d = id(i, j), id(i + 1, j), id(i, j + 1), id(i + 1, j + 1)
        push!(tris, (a, b, d), (a, d, c))
    end
    return tris
end

"""
Every interior edge with the triangles on both sides, as (opposite 1, opposite 2, edge start,
edge end): the edge runs start to end in the first triangle's counterclockwise order, so both
normals point the same way on a flat sheet.
"""
function grid_hinges(tris)
    seen = Dict{Tuple{Int, Int}, Int}()
    hinges = NTuple{4, Int}[]
    for (a, b, c) in tris, (p, q, o) in ((a, b, c), (b, c, a), (c, a, b))
        other = pop!(seen, (q, p), 0)
        if other == 0
            seen[(p, q)] = o
        else
            push!(hinges, (other, o, q, p))
        end
    end
    return hinges
end

"""
The springs of a grid with their stiffness, for in-plane stiffness E·t: along u and v each
spring carries its share of the cross section, the diagonals take the shear.
"""
function grid_springs(g::SheetGrid, Et)
    hu, hv = g.length / (g.nu - 1), g.width / (g.nv - 1)
    id(i, j) = (j - 1) * g.nu + i
    springs = Tuple{Int, Int, Float64}[]
    share(k, n) = k == 1 || k == n ? 0.5 : 1.0
    for j in 1:g.nv, i in 1:(g.nu - 1)
        push!(springs, (id(i, j), id(i + 1, j), Et * share(j, g.nv) * hv / hu))
    end
    for j in 1:(g.nv - 1), i in 1:g.nu
        push!(springs, (id(i, j), id(i, j + 1), Et * share(i, g.nu) * hu / hv))
    end
    for j in 1:(g.nv - 1), i in 1:(g.nu - 1)
        push!(springs, (id(i, j), id(i + 1, j + 1), 0.5 * Et * hu * hv / (hu^2 + hv^2)))
    end
    return springs
end

"Dihedral angle of a hinge, signed, as the solver measures it."
function dihedral(x0, x1, x2, x3)
    n1 = normalize(cross(x2 - x0, x3 - x0))
    n2 = normalize(cross(x3 - x1, x2 - x1))
    return atan(dot(cross(n1, n2), normalize(x3 - x2)), dot(n1, n2))
end

"|e|²/(A₁ + A₂) of a hinge on the flat sheet."
function hinge_weight(P, o1, o2, s, t)
    a1 = norm(cross(P[s] - P[o1], P[t] - P[o1])) / 2
    a2 = norm(cross(P[s] - P[o2], P[t] - P[o2])) / 2
    return norm(P[t] - P[s])^2 / (a1 + a2)
end

"""
The factor that makes the hinges' energy ½ c D |e|²/(A₁ + A₂) θ² of a grid bent into a
cylinder equal the plate's ½ D κ² per area: along the sheet and across it.
"""
function bending_calibration(g::SheetGrid, hinges; κ = 1.0)
    g.nu>=2 && g.nv>=2 && isfinite(g.length) && isfinite(g.width) &&
        g.length>0 && g.width>0 && isfinite(κ) && κ>0 ||
        throw(ArgumentError("positive sheet dimensions/curvature and at least two vertices per axis required"))
    if min(g.nu,g.nv)<3
        # Two coordinate samples cannot resolve cylindrical curvature: all
        # sampled points of that cylinder lie in one plane, making the old
        # calibration divide by zero. Use the standard discrete-shell hinge
        # energy instead, not a clamp of the resulting infinite stiffness.
        # Codim-IPC 9c6cbe3, FEM/Shell/DISCRETE_SHELL.h and BENDING.h:
        # k=D/2, hbar=(A1+A2)/(3|e|), E=k*theta^2*|e|/hbar.
        # Thus the stiffness multiplying theta^2/2 is 3D*hinge_weight.
        return 3.0,3.0
    end
    P = [Vec3(p[1], p[2], 0.0) for p in (flat(g, i, j) for j in 1:g.nv for i in 1:g.nu)]
    function energy(bend)
        X = [bend(p) for p in P]
        return sum(0.5 * hinge_weight(P, h...) * dihedral(X[[h...]]...)^2 for h in hinges)
    end
    plate = 0.5 * κ^2 * g.length * g.width
    along = energy(p -> Vec3(sin(κ * p[1]) / κ, p[2], (1 - cos(κ * p[1])) / κ))
    across = energy(p -> Vec3(p[1], sin(κ * p[2]) / κ, (1 - cos(κ * p[2])) / κ))
    isfinite(along) && isfinite(across) && along>0 && across>0 ||
        throw(ArgumentError("sheet cylinder calibration is degenerate; choose resolved dimensions/curvature"))
    return plate / along, plate / across
end

"""
    add_sheet!(builder, grid, material; shape, place) -> SheetPiece

A sheet whose vertex at flat position p (m, from its corner) rests at `shape(p)`: a flat
sheet by default, a curled or wavy one if `shape` says so. Its rest angles come from that
shape, its lengths from the flat sheet (paper curls without stretching). It starts at
`place(shape(p))`.
"""
function add_sheet!(b::ModelBuilder, g::SheetGrid, m::SheetMaterial; shape = p -> Vec3(p[1], p[2], 0.0), place = identity)
    g.nu>=2 && g.nv>=2 && isfinite(g.length) && isfinite(g.width) &&
        g.length>0 && g.width>0 || throw(ArgumentError("positive sheet dimensions and at least two vertices per axis required"))
    first = length(b.x) + 1
    flatp = [flat(g, i, j) for j in 1:g.nv for i in 1:g.nu]
    P = [Vec3(p[1], p[2], 0.0) for p in flatp]
    rest = [Vec3{Float64}(shape(p)) for p in flatp]
    tris = grid_triangles(g)
    hinges = grid_hinges(tris)
    cu, cv = bending_calibration(g, hinges)
    D = bending_stiffness(m)
    mass = zeros(length(flatp))
    for (a, c, d) in tris
        area = norm(cross(P[c] - P[a], P[d] - P[a])) / 2
        push!(b.tris, Int32.((a, c, d) .+ (first - 1)))
        for k in (a, c, d)
            mass[k] += m.grammage * area / 3
        end
    end
    for (p, q, k) in grid_springs(g, m.young * m.thickness)
        push!(b.springs, Int32.((p, q) .+ (first - 1)))
        push!(b.spring_rest, norm(P[q] - P[p]))
        push!(b.spring_stiffness, k)
        push!(b.spring_damping, m.stretch_damping)
    end
    for h in hinges
        push!(b.hinges, Int32.(h .+ (first - 1)))
        push!(b.hinge_rest, dihedral(rest[[h...]]...))
        push!(b.hinge_stiffness, (cu + cv) / 2 * D * hinge_weight(P, h...))
        push!(b.hinge_damping, m.bending_damping)
    end
    append!(b.x, [Vec3{Float64}(place(p)) for p in rest])
    append!(b.v, zeros(Vec3f, length(rest)))
    append!(b.mass, mass)
    append!(b.area, mass ./ m.grammage)
    append!(b.fixed, zeros(UInt8, length(rest)))
    piece = SheetPiece(first, g)
    push!(b.sheets, piece)
    return piece
end

"Holds vertex `i` where it is: it no longer moves."
fix!(b::ModelBuilder, i::Integer) = (b.fixed[i] = 0x01; b)

"Compressed lists of what touches each vertex: for vertex i, entries `start[i]:start[i+1]-1`."
function incidence(n, elements)
    counts = zeros(Int, n)
    for el in elements, i in el
        counts[i] += 1
    end
    total = sum(counts)
    total < typemax(Int32) || throw(ArgumentError("incidence exceeds Int32 capacity"))
    start = Vector{Int32}(undef, n + 1)
    start[1] = 1
    for i in 1:n
        start[i + 1] = start[i] + counts[i]
    end
    cursor = copy(start[1:n])
    ids, corners = Vector{Int32}(undef, total), Vector{Int32}(undef, total)
    for (e, el) in enumerate(elements), (k, i) in enumerate(el)
        at = cursor[i]
        ids[at], corners[at] = Int32(e), Int32(k - 1)
        cursor[i] += 1
    end
    return start, ids, corners
end

# --- on the device -----------------------------------------------------------------------------

"""
    Model(backend, builder)

The builder's world on the device (on the host for `Array`): never written during a run.
Kernels take it whole.
"""
struct Model{F, R, U, I, T2, T3, T4}
    mass::F
    area::F
    fixed::U
    tris::T3
    tri_edges::U              # per triangle the edges it owns (bit k: corner k to k + 1): each edge one triangle
    springs::T2
    spring_rest::R            # rest lengths as pairs (hi, lo): Float32 rounds 2 mm to 0.1 nm
    spring_stiffness::F
    spring_damping::F
    hinges::T4
    hinge_rest::F
    hinge_stiffness::F
    hinge_damping::F
    vspring_start::I
    vspring::I
    vspring_order::I
    vhinge_start::I
    vhinge::I
    vhinge_order::I
    vtri_start::I             # the triangles at each vertex, and which corner of each it is
    vtri::I
    vtri_order::I
    one::F                    # a 1 the compilers cannot see: `precise.jl`
    gravity::Vec3f
end
Adapt.@adapt_structure Model

"`r` as a pair of Float32, hi + lo."
float_pair(r::Float64) = (h = Float32(r); (h, Float32(r - h)))

"""
`a` on `backend`, or on the host for `Array`. A vector gets at least one element: an empty
buffer is not something every driver allocates.
"""
upload(backend, a::AbstractVector) = (d = KI.allocate(backend, eltype(a), max(length(a), 1)); isempty(a) || copyto!(d, a); d)
upload(backend, a::AbstractArray) = (d = KI.allocate(backend, eltype(a), size(a)); isempty(a) || copyto!(d, a); d)
upload(::Type{Array}, a::AbstractVector) = isempty(a) ? similar(a, 1) : copy(a)
upload(::Type{Array}, a::AbstractArray) = copy(a)

function Model(backend, b::ModelBuilder)
    up(a) = upload(backend, a)
    n = length(b.x)
    ss, sl, so = incidence(n, b.springs)
    hs, hl, ho = incidence(n, b.hinges)
    ts, tl, to = incidence(n, b.tris)
    return Model(up(b.mass), up(b.area), up(b.fixed), up(b.tris), up(owned_edges(b.tris)), up(b.springs), up(float_pair.(b.spring_rest)), up(b.spring_stiffness),
        up(b.spring_damping), up(b.hinges), up(b.hinge_rest), up(b.hinge_stiffness), up(b.hinge_damping), up(ss), up(sl), up(so), up(hs), up(hl), up(ho),
        up(ts), up(tl), up(to), up(Float32[1]), b.gravity[])
end

"Per triangle the edges it owns, bit k for corner k to k + 1: each edge belongs to the lowest triangle at it."
function owned_edges(tris)
    owner = Dict{Tuple{Int32, Int32}, Int}()
    for (t, f) in enumerate(tris), k in 1:3
        e = minmax(f[k], f[mod1(k + 1, 3)])
        owner[e] = min(get(owner, e, t), t)
    end
    mask = zeros(UInt8, max(length(tris), 1))
    for (t, f) in enumerate(tris), k in 1:3
        owner[minmax(f[k], f[mod1(k + 1, 3)])] == t && (mask[t] |= UInt8(1) << (k - 1))
    end
    return mask
end

"""
    State(backend, builder)

What changes: particle positions (double-float, `precise.jl`) and velocities.
"""
struct State{P, V}
    x::P
    v::V
end
function State(backend, b::ModelBuilder)
    x = KI.zeros(backend, Position, max(length(b.x), 1))
    v = KI.zeros(backend, Position, max(length(b.v), 1))
    isempty(b.x) || copyto!(x, Position.(b.x))
    isempty(b.v) || copyto!(v, Position.(b.v))
    return State(x, v)
end
