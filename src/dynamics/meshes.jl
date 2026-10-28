# Triangle meshes that sheets touch, and how a vertex finds the triangles near it.
#
# Every mesh gets its own bounding volume hierarchy, built once on the host over its triangles
# as they rest. Meshes that move or deform later keep its topology and refit its boxes every
# step, which runs in a recorded graph without sorting anything. A query walks the meshes'
# root boxes, then the hierarchy of each mesh whose box comes near, with a fixed-size stack per
# thread. The hierarchy only finds candidates (Float32 boxes, a margin around them); the gap to
# the chosen triangle is then taken in double-float (`triangle_gap`).

"Triangles in a leaf of a mesh's hierarchy, at most."
const LEAF = 4

"""
Nodes a walk through a hierarchy keeps on its stack: at most as many as the hierarchy has
levels, which median splits keep at ⌈log₂(triangles / `LEAF`)⌉ + 1 (19 for a million
triangles); `Colliders` checks it.
"""
const STACK = 48

"A mesh hierarchy's nodes as the host builds them, every mesh's appended to the arrays."
struct BVHTables
    lo::Vector{Vec3f}
    hi::Vector{Vec3f}
    left::Vector{Int32}         # interior: child node; leaf: −(first index into `order`)
    right::Vector{Int32}        # interior: child node; leaf: how many triangles
    order::Vector{Int32}        # triangles, leaf by leaf
    root::Vector{Int32}         # per mesh
end
BVHTables() = BVHTables(Vec3f[], Vec3f[], Int32[], Int32[], Int32[], Int32[])

"The Float32 box around the Float64 box (`lo`, `hi`): rounded outward, so it holds every point."
outer(lo, hi) = (Vec3f(prevfloat.(Float32.(lo))), Vec3f(nextfloat.(Float32.(hi))))

"""
Adds the hierarchy over triangles `ids` (with Float64 boxes `boxes` and centers `centers`) to
`t`, splitting at the median center along the longest extent; returns its root node.
"""
function add_bvh!(t::BVHTables, ids::Vector{Int32}, boxes, centers)
    lo = reduce((a, b) -> min.(a, b), (boxes[i][1] for i in ids))
    hi = reduce((a, b) -> max.(a, b), (boxes[i][2] for i in ids))
    l, h = outer(lo, hi)
    push!(t.lo, l); push!(t.hi, h); push!(t.left, 0); push!(t.right, 0)
    node = length(t.lo)
    if length(ids) <= LEAF
        t.left[node] = -(length(t.order) + 1)
        t.right[node] = length(ids)
        append!(t.order, ids)
    else
        clo = reduce((a, b) -> min.(a, b), (centers[i] for i in ids))
        chi = reduce((a, b) -> max.(a, b), (centers[i] for i in ids))
        axis = argmax(chi - clo)
        sorted = sort(ids; by = i -> centers[i][axis])
        half = length(sorted) ÷ 2
        t.left[node] = add_bvh!(t, sorted[1:half], boxes, centers)
        t.right[node] = add_bvh!(t, sorted[(half + 1):end], boxes, centers)
    end
    return Int32(node)
end

"Levels of the hierarchy below `node` in `t`, counting it."
levels(t::BVHTables, node) = t.left[node] < 0 ? 1 : 1 + max(levels(t, t.left[node]), levels(t, t.right[node]))

"""
Which corners and edges of a mesh's triangles can press into a sheet's face: convex ones (an
edge whose two faces turn away from each other, or that bounds the mesh; a corner on such an
edge). An edge or corner inside a flat face cannot, and counted there it would stiffen a flat
contact by how finely the face is meshed. Each feature belongs to one triangle, its lowest:
bits 1 to 3 of `mask` mark its corners, bits 4 to 6 its edges (corner k to k + 1).
"""
function features(vertices::Vector{Vec3{Float64}}, faces::Vector{NTuple{3, Int32}})
    edges = Dict{Tuple{Int32, Int32}, Vector{Tuple{Int, Int}}}()
    for (t, f) in enumerate(faces), k in 1:3
        a, b = f[k], f[mod1(k + 1, 3)]
        push!(get!(edges, minmax(a, b), Tuple{Int, Int}[]), (t, k))
    end
    normal(f) = normalize(cross(vertices[f[2]] - vertices[f[1]], vertices[f[3]] - vertices[f[1]]))
    mask = zeros(UInt8, length(faces))
    sharp = falses(length(vertices))
    for ((a, b), sides) in edges
        convex = length(sides) != 2 || begin
            (t1, _), (t2, _) = sides
            f1, f2 = faces[t1], faces[t2]
            far = only(v for v in f2 if v != a && v != b)
            # The second face's far corner below the first face's plane, by more than rounding.
            dot(normal(f1), vertices[far] - vertices[a]) < -1.0e-9 * norm(vertices[b] - vertices[a])
        end
        convex || continue
        t, k = minimum(sides)
        mask[t] |= UInt8(1) << (2 + k)
        sharp[a] = sharp[b] = true
    end
    owner = fill(typemax(Int), length(vertices))
    for (t, f) in enumerate(faces), k in 1:3
        sharp[f[k]] && t < owner[f[k]] && (owner[f[k]] = t)
    end
    for (t, f) in enumerate(faces), k in 1:3
        owner[f[k]] == t && (mask[t] |= UInt8(1) << (k - 1))
    end
    return mask
end

"""
The builder's meshes on the device: every vertex as a pair, every triangle (global vertex
indices), the mesh it belongs to and its features (`features`), the hierarchies, and per mesh
how it grips paper.
"""
struct Colliders{P, T, I, U, V, F, E, X}
    points::P
    point_normal::V
    point_owner::I
    point_tri_start::I
    point_tri::I
    other_face::E
    faces::T
    owner::I
    features::U
    excluded::X
    lo::V
    hi::V
    left::I
    right::I
    order::I
    root::I
    height::I           # leaf: 0; interior: one plus the largest child height
    depth::Int
    grip::F             # (3, meshes): static, kinetic friction, stiffness (Pa/m)
    meshes::Int
end
Adapt.@adapt_structure Colliders

function Colliders(backend, b::ModelBuilder)
    up(a) = upload(backend, a)
    points, normals, point_owner, faces, owner, mask = Position[], Vec3f[], Int32[], NTuple{3, Int32}[], Int32[], UInt8[]
    t = BVHTables()
    grip = zeros(Float32, 3, max(length(b.meshes), 1))
    for (m, piece) in enumerate(b.meshes)
        offset = Int32(length(points))
        append!(points, Position.(piece.vertices))
        append!(normals, vertex_normals(piece))
        append!(point_owner, fill(Int32(m), length(piece.vertices)))
        first = length(faces)
        append!(faces, [f .+ offset for f in piece.faces])
        append!(owner, fill(Int32(m), length(piece.faces)))
        append!(mask, features(piece.vertices, piece.faces))
        corners(f) = (piece.vertices[f[1]], piece.vertices[f[2]], piece.vertices[f[3]])
        boxes = Dict(Int32(first + k) => (min.(corners(f)...), max.(corners(f)...)) for (k, f) in enumerate(piece.faces))
        centers = Dict(Int32(first + k) => sum(corners(f)) / 3 for (k, f) in enumerate(piece.faces))
        push!(t.root, add_bvh!(t, Int32.(first .+ (1:length(piece.faces))), boxes, centers))
        levels(t, t.root[end]) <= STACK || error("mesh $m: its hierarchy has $(levels(t, t.root[end])) levels, a walk keeps at most $STACK nodes")
        grip[:, m] .= (piece.static, piece.kinetic, piece.stiffness)
    end
    isempty(t.lo) && (push!(t.lo, zero(Vec3f)); push!(t.hi, zero(Vec3f)); push!(t.left, 0); push!(t.right, 0))
    height = zeros(Int32, length(t.left))
    for node in reverse(eachindex(t.left))
        t.left[node] > 0 && (height[node] = 1 + max(height[t.left[node]], height[t.right[node]]))
    end
    ts, tl, _ = incidence(length(points), faces)
    adjacent = zeros(Int32, 3, max(length(faces), 1))
    edges = Dict{Tuple{Int32, Int32}, Tuple{Int32, Int32}}()
    for (triangle, f) in enumerate(faces), k in 1:3
        edge = minmax(f[k], f[mod1(k + 1, 3)])
        if haskey(edges, edge)
            previous, corner = edges[edge]
            adjacent[k, triangle], adjacent[corner, previous] = previous, triangle
        else
            edges[edge] = (Int32(triangle), Int32(k))
        end
    end
    excluded = zeros(UInt8, max(length(b.meshes), 1), max(length(b.meshes), 1))
    for (a, c) in b.excluded
        excluded[a, c] = excluded[c, a] = 1
    end
    return Colliders(up(points), up(normals), up(point_owner), up(ts), up(tl), up(adjacent), up(faces), up(owner), up(mask), up(excluded), up(t.lo), up(t.hi), up(t.left), up(t.right), up(t.order), up(t.root), up(height), Int(maximum(height)), up(grip), length(b.meshes))
end

"Exclude mesh IDs for a modeled glue bond or an internal bearing replaced by a joint."
function exclude_contacts!(b::ModelBuilder, a::Integer, c::Integer)
    1 <= a <= length(b.meshes) && 1 <= c <= length(b.meshes) && a != c || throw(ArgumentError("invalid excluded mesh pair"))
    pair = Int32.(minmax(a, c))
    pair in b.excluded || push!(b.excluded, pair)
    return b
end

function vertex_normals(piece::MeshPiece)
    normals = zeros(Vec3{Float64}, length(piece.vertices))
    for f in piece.faces
        n = normalize(cross(piece.vertices[f[2]] - piece.vertices[f[1]], piece.vertices[f[3]] - piece.vertices[f[1]]))
        for k in 1:3
            a = piece.vertices[f[mod1(k + 1, 3)]] - piece.vertices[f[k]]
            c = piece.vertices[f[mod1(k + 2, 3)]] - piece.vertices[f[k]]
            angle = acos(clamp(dot(a, c) / (norm(a) * norm(c)), -1, 1))
            normals[f[k]] += angle * n
        end
    end
    return Vec3f[norm(n) > 0 ? normalize(n) : zero(Vec3f) for n in normals]
end

"Refit the existing BVH topology after meshes move; each launch reads only completed children."
function refit!(target, C::Colliders)
    for height in 0:C.depth
        launch!(target, refit_nodes!, (C, Int32(height)), length(C.left))
    end
    return C
end

function refit_nodes!(C, height::Int32)
    node = KI.get_global_id(Int32).x
    node <= length(C.left) || return nothing
    @inbounds if C.height[node] == height
        left = C.left[node]
        if left < 0
            lo, hi = Vec3f(Inf32), Vec3f(-Inf32)
            for k in -left:(-left + C.right[node] - Int32(1))
                f = C.faces[C.order[k]]
                a, b, c = approximate(C.points[f[1]]), approximate(C.points[f[2]]), approximate(C.points[f[3]])
                lo, hi = min.(lo, a, b, c), max.(hi, a, b, c)
            end
            # Outward margin includes rounding in the hi+lo conversion, without requiring
            # floating point nextafter in a device compiler.
            margin = 4eps(Float32) * max.(abs.(lo), abs.(hi)) .+ 1.0f-10
            C.lo[node], C.hi[node] = lo - margin, hi + margin
        elseif left > 0
            right = C.right[node]
            C.lo[node], C.hi[node] = min.(C.lo[left], C.lo[right]), max.(C.hi[left], C.hi[right])
        end
    end
    return nothing
end

"Squared distance from `q` to the box (`lo`, `hi`)."
@inline box_distance2(q, lo, hi) = (d = max.(lo - q, 0.0f0) + max.(q - hi, 0.0f0); dot(d, d))

"""
Unsigned distance from the origin to the triangle with corners `a`, `b`, `c` (Float32, relative
to the point), and the triangle's unit normal: what ranks candidates.
"""
@inline function rough_distance(a::Vec3f, b::Vec3f, c::Vec3f)
    w, d, n = projection(a, b, c)
    minimum(w) >= 0 && return abs(d), n
    return sqrt(min(segment_distance2(a, b), segment_distance2(b, c), segment_distance2(c, a))), n
end

"Squared distance from the origin to segment `a`–`b`."
@inline function segment_distance2(a::Vec3f, b::Vec3f)
    e = b - a
    t = clamp(-dot(a, e) / max(dot(e, e), 1.0f-30), 0.0f0, 1.0f0)
    q = a + t * e
    return dot(q, q)
end

"""
    triangle_gap(p, a, b, c, r, side, one) -> (g, n, w)

Signed distance from `p` to the triangle with corners `a`, `b`, `c` (pairs), positive on the side
`side` (+1: the one its counterclockwise normal points to, −1: the other), less `r`, to the
precision of the result: `r` comes off before anything is rounded (for a vertex resting half a
sheet above a face, the distance alone rounds to 3 pm). `n` is the unit direction from the closest point to `p`, `w` the closest
point's barycentric weights. Where `p` projects into the face the distance is taken along the
face's normal from the foot point (a normal good to 10⁻⁷ moves a corner a millimeter away by
0.1 nm across it), its length corrected (`unit_excess`); elsewhere from the closest point on
the nearest edge.
"""
@inline function triangle_gap(p::Position, a::Position, b::Position, c::Position, r::Float32, side::Float32, one)
    w, _, nf = projection(a - p, b - p, c - p)
    nf = side * nf
    if minimum(w) >= 0
        dh, dl = dot_pair(nf, difference(p, a, one), one)
        bh, bl = dot_pair(nf, difference(b, a, one), one)
        ch, cl = dot_pair(nf, difference(c, a, one), one)
        dl -= w[2] * (bh + bl) + w[3] * (ch + cl) + unit_excess(nf, one) / 2 * dh
        return (dh - r) + dl, nf, w
    end
    t1, r1 = segment_offset(p, a, b, one)
    t2, r2 = segment_offset(p, b, c, one)
    t3, r3 = segment_offset(p, c, a, one)
    l1, l2, l3 = length_pair(r1, one), length_pair(r2, one), length_pair(r3, one)
    if l1 <= l2 && l1 <= l3
        e, l, ww = r1, l1, Vec3f(1 - t1, t1, 0)
    elseif l2 <= l3
        e, l, ww = r2, l2, Vec3f(0, 1 - t2, t2)
    else
        e, l, ww = r3, l3, Vec3f(t3, 0, 1 - t3)
    end
    n = (e.hi + e.lo) / max(l, 1.0f-30)
    outside = dot(nf, n) >= 0
    g = ifelse(outside, stretch(e, (r, 0.0f0), one), -l - r)
    return g, ifelse(outside, n, -n), ww
end

"Corners of triangle `k` of the colliders, as pairs."
@inline corners(C::Colliders, k) = @inbounds (f = C.faces[k]; (C.points[f[1]], C.points[f[2]], C.points[f[3]]))

"""
The triangles of mesh `m` within `radius` of `p`: the closest, and the closest whose normal
turns more than 25° from its (a vertex in a concave corner touches both faces). Returns their
indices (0: none), rough distances and normals.
"""
@inline function near_triangles(C::Colliders, m, p::Position, radius)
    @inbounds begin
        q = approximate(p)
        best1, best2 = Int32(0), Int32(0)
        d1, d2 = Inf32, Inf32
        n1, n2 = zero(Vec3f), zero(Vec3f)
        stack = MVector{STACK, Int32}(undef)
        top = Int32(1)
        stack[1] = C.root[m]
        r2 = radius^2
        while top > Int32(0)
            node = stack[top]
            top -= Int32(1)
            if box_distance2(q, C.lo[node], C.hi[node]) <= r2
                l = C.left[node]
                if l < 0
                    k = -l
                    while k < -l + C.right[node]
                        tri = C.order[k]
                        a, b, c = corners(C, tri)
                        d, n = rough_distance(a - p, b - p, c - p)
                        if d <= radius
                            if d < d1
                                if best1 != 0 && dot(n, n1) < 0.9063f0
                                    best2, d2, n2 = best1, d1, n1
                                end
                                best1, d1, n1 = tri, d, n
                            elseif d < d2 && dot(n, n1) < 0.9063f0
                                best2, d2, n2 = tri, d, n
                            end
                        end
                        k += Int32(1)
                    end
                else
                    stack[top + Int32(1)] = l
                    stack[top + Int32(2)] = C.right[node]
                    top += Int32(2)
                end
            end
        end
        # A second triangle only if it still turns away from the final first one.
        if best2 != 0 && dot(n2, n1) >= 0.9063f0
            best2, d2 = Int32(0), Inf32
        end
        return best1, d1, best2, d2
    end
end

"""
    box_mesh(center, (u, v, n), half) -> GeometryBasics.Mesh

A box around `center` with axes `u`, `v`, `n` (orthonormal and right-handed, Float64) and half
extents `half` along them, meshed in Float64, triangles counterclockwise from outside.
"""
function box_mesh(center, axes, half)
    m = GeometryBasics.mesh(Rect3d(Point3d(-1, -1, -1) .* half, Point3d(2, 2, 2) .* half); pointtype = Point3d, facetype = GLTriangleFace)
    u, v, n = axes
    points = [Point3d(center + p[1] * u + p[2] * v + p[3] * n) for p in GeometryBasics.coordinates(m)]
    return GeometryBasics.Mesh(points, GeometryBasics.faces(m))
end

"A bought torus meshed in Float64, with at most `tolerance` meters chord error in total."
function torus_mesh(center, axis, radial, major, minor; tolerance = 1e-6)
    major > minor > 0 && 0 < tolerance < minor || throw(ArgumentError("invalid torus radii or tolerance"))
    axis, radial = normalize(Vec3{Float64}(axis)), normalize(Vec3{Float64}(radial))
    abs(dot(axis, radial)) < 1e-10 || throw(ArgumentError("torus axes must be perpendicular"))
    other = cross(axis, radial)
    # Split the error budget between the two principal parameter directions.
    segments(radius) = max(8, 4cld(ceil(Int, π / acos(1 - tolerance / (2radius))), 4))
    n, m = segments(major + minor), segments(minor)
    points = Point3d[]
    for j in 0:(m - 1), i in 0:(n - 1)
        θ, φ = 2π * i / n, 2π * j / m
        push!(points, Point3d(center + minor * sin(φ) * axis + (major + minor * cos(φ)) * (cos(θ) * radial + sin(θ) * other)))
    end
    id(i, j) = mod(i, n) + n * mod(j, m) + 1
    faces = GLTriangleFace[]
    for j in 0:(m - 1), i in 0:(n - 1)
        a, b, c, d = id(i, j), id(i + 1, j), id(i, j + 1), id(i + 1, j + 1)
        push!(faces, GLTriangleFace(a, b, c), GLTriangleFace(b, d, c))
    end
    return GeometryBasics.Mesh(points, faces)
end
