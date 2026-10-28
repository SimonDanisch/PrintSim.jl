# A fine collision surface carried by a coarse quadratic tetrahedral cage: each surface
# point moves with the T10 displacement field of the element holding it,
# x = Σ N_k(λ) x_k (Ferguson et al., High-Order Incremental Potential Contact, 2023). The
# elasticity lives on the cage, contacts on the surface; `surface_map_*` apply W and W'.

"T10 basis at barycentric `λ`: corners, then edges 12, 13, 14, 23, 24, 34 (`add_deformable!` order)."
function quadratic_tet_weights(λ)
    corners = ntuple(i -> λ[i] * (2λ[i] - 1), 4)
    edges = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    return (corners..., ntuple(k -> 4λ[edges[k][1]] * λ[edges[k][2]], 6)...)
end

function tet_barycentric(p, a, b, c, d)
    T = StaticArrays.SMatrix{3, 3, Float64}(b - a..., c - a..., d - a...)
    μ = T \ StaticArrays.SVector{3, Float64}(p - a)
    return (1 - μ[1] - μ[2] - μ[3], μ[1], μ[2], μ[3])
end

"""
Weights on a 2^-40 grid with the largest absorbing the remainder: every partial sum is
exact in Float64, so the row sums to exactly one (as `SurfaceEmbedding` requires).
"""
function exact_partition(w)
    q = collect(round.(Float64.(w) .* 2.0^40) ./ 2.0^40)
    k = argmax(abs.(q))
    q[k] = 0.0
    q[k] = 1 - sum(q)
    sum(q) == 1 || error("weights do not form an exact partition of unity")
    return Tuple(q)
end

"Distance from `p` to a tetrahedron (zero inside) and its smallest barycentric coordinate."
function tet_distance(p, a, b, c, d)
    λ = tet_barycentric(p, a, b, c, d)
    m = minimum(λ)
    m >= 0 && return (0.0, m, λ)
    distance = sqrt(min(host_triangle_distance2(p, b, c, d), host_triangle_distance2(p, a, c, d),
        host_triangle_distance2(p, a, b, d), host_triangle_distance2(p, a, b, c)))
    return (distance, m, λ)
end

"""
    embed_surfaces!(b::ModelBuilder, surfaces; max_distance = 1e-4)

Replace the collision surface of each deformable `part => mesh` in `surfaces` by `mesh`
(rest frame, meters), embedded into the part's tetrahedra: each point takes the T10 weights
of the element holding it, or, outside the cage, of the nearest element (extrapolated), at
most `max_distance` away. Returns the `surface_maps` for `WorldSolver`/`Recording`.
"""
function embed_surfaces!(b::ModelBuilder, surfaces; max_distance = 1e-4)
    maps = Dict{Int, Any}()
    replacements = Pair{Int, MeshPiece}[]
    for (part, mesh) in surfaces
        1 <= part <= length(b.deformables) || throw(ArgumentError("unknown FE part $part"))
        piece = b.deformables[part]
        x, tets = piece.x, piece.tets
        boxes = [(reduce((p, q) -> min.(p, q), (x[t[k]] for k in 1:4)), reduce((p, q) -> max.(p, q), (x[t[k]] for k in 1:4))) for t in tets]
        tree = BVHTables()
        root = add_bvh!(tree, Int32.(eachindex(tets)), boxes, [(lo + hi) / 2 for (lo, hi) in boxes])
        points = Vec3{Float64}.(GeometryBasics.coordinates(mesh))
        nodes = Vector{NTuple{10, Int32}}(undef, length(points))
        weights = Vector{NTuple{10, Float64}}(undef, length(points))
        stack = Int32[]
        for (i, p) in enumerate(points)
            best, distance, inside, λbest = 0, Inf, -Inf, (0.0, 0.0, 0.0, 0.0)
            grow = 1e-7
            # Grow the search box until the nearest element found lies inside it.
            while true
                empty!(stack)
                push!(stack, root)
                while !isempty(stack)
                    node = pop!(stack)
                    (any(p .< Float64.(tree.lo[node]) .- grow) || any(p .> Float64.(tree.hi[node]) .+ grow)) && continue
                    if tree.left[node] < 0
                        first = -tree.left[node]
                        for at in first:(first + tree.right[node] - 1)
                            t = tets[tree.order[at]]
                            d, m, λ = tet_distance(p, x[t[1]], x[t[2]], x[t[3]], x[t[4]])
                            if d < distance || (d == distance && m > inside)
                                best, distance, inside, λbest = tree.order[at], d, m, λ
                            end
                        end
                    else
                        push!(stack, tree.left[node], tree.right[node])
                    end
                end
                (best != 0 && distance <= grow) || grow > max_distance ? break : (grow *= 4)
            end
            best != 0 && distance <= max_distance ||
                error("surface point $i of part $part lies $(distance) m outside its cage")
            nodes[i] = tets[best]
            weights[i] = exact_partition(quadratic_tet_weights(λbest))
        end
        apply(state, k) = sum(weights[k][j] * state[nodes[k][j]] for j in 1:10)
        rest = [apply(x, k) for k in eachindex(points)]
        initial = [apply(piece.initial, k) for k in eachindex(points)]
        maximum(norm.(rest .- points)) <= 1e-9 || error("embedding does not reproduce the rest surface of part $part")
        faces = [NTuple{3, Int32}(convert(Int32, v) for v in f) for f in GeometryBasics.faces(mesh)]
        old = b.meshes[piece.mesh]
        push!(replacements, piece.mesh => MeshPiece(initial, faces, old.static, old.kinetic, old.stiffness))
        maps[piece.mesh] = (; nodes, weights, points = rest, initial, faces)
    end
    for (mesh, replacement) in replacements
        b.meshes[mesh] = replacement
    end
    return maps
end

"""
    elastic_cage(mesh; envelope = 40e-6, size = 1e-3, executable = ftetwild_executable())

Coarse, well-shaped linear tetrahedra for the elasticity of a part: fTetWild (`--coarsen`)
fills `mesh`, keeping its boundary within `envelope` of `mesh`. Contacts use the finer
surface embedded into the cage (`embed_surfaces!`).
"""
function elastic_cage(mesh; envelope = 40e-6, size = 1e-3, executable = ftetwild_executable(), max_iterations = 80)
    points, tets = tetrahedralize_exact(mesh, size, envelope, executable, max_iterations; coarsen = true)
    return (; points, tets)
end
