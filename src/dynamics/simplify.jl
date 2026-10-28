import StaticArrays

# Symmetric 4×4 plane quadric (Garland & Heckbert 1997), upper triangle row by row:
# (aa, ab, ac, ad, bb, bc, bd, cc, cd, dd) for the plane a x + b y + c z + d = 0.
const Quadric = StaticArrays.SVector{10, Float64}

function plane_quadric(n, d)
    a, b, c = n
    return Quadric(a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d)
end

quadric_matrix(q) = StaticArrays.SMatrix{3, 3, Float64}(q[1], q[2], q[3], q[2], q[5], q[6], q[3], q[6], q[8])
quadric_vector(q) = StaticArrays.SVector(q[4], q[7], q[9])
quadric_value(q, x) = dot(x, quadric_matrix(q) * x) + 2dot(quadric_vector(q), x) + q[10]

"Quadric minimizer, kept at `m` along directions the merged planes leave free."
function quadric_position(q, m)
    A = quadric_matrix(q)
    r = -(A * m + quadric_vector(q))
    e = eigen(Symmetric(A))
    limit = 1e-6 * max(abs(e.values[end]), floatmin(Float64))
    step = zero(m)
    for k in 1:3
        λ = e.values[k]
        abs(λ) > limit || continue
        v = e.vectors[:, k]
        step += v * (dot(v, r) / λ)
    end
    return m + step
end

triangle_normal(a, b, c) = cross(b - a, c - a)

"Mean-ratio triangle quality: 1 for equilateral, 0 for degenerate."
function triangle_quality(a, b, c)
    s = sum(abs2, b - a) + sum(abs2, c - b) + sum(abs2, a - c)
    s > 0 || return 0.0
    return 2sqrt(3.0) * norm(triangle_normal(a, b, c)) / s
end

struct SimplifyHeap
    items::Vector{Tuple{Float64, Int32, Int32, Int32, Int32}}
end

function Base.push!(h::SimplifyHeap, item)
    items = h.items
    push!(items, item)
    i = length(items)
    while i > 1
        p = i >> 1
        items[p][1] <= items[i][1] && break
        items[p], items[i] = items[i], items[p]
        i = p
    end
    return h
end

function Base.pop!(h::SimplifyHeap)
    items = h.items
    top = items[1]
    last = pop!(items)
    n = length(items)
    if n > 0
        items[1] = last
        i = 1
        while true
            l, r = 2i, 2i + 1
            s = i
            l <= n && items[l][1] < items[s][1] && (s = l)
            r <= n && items[r][1] < items[s][1] && (s = r)
            s == i && break
            items[s], items[i] = items[i], items[s]
            i = s
        end
    end
    return top
end

Base.isempty(h::SimplifyHeap) = isempty(h.items)

"Indices of the triangles of an indexed host surface whose boxes overlap `lo`..`hi`."
function host_surface_box!(out, surface, lo, hi)
    tree = surface.tree
    empty!(out)
    stack = Int32[surface.root]
    while !isempty(stack)
        node = pop!(stack)
        (any(Float64.(tree.lo[node]) .> hi) || any(Float64.(tree.hi[node]) .< lo)) && continue
        if tree.left[node] < 0
            first = -tree.left[node]
            append!(out, @view tree.order[first:(first + tree.right[node] - 1)])
        else
            push!(stack, tree.left[node], tree.right[node])
        end
    end
    return out
end

"Whether any triangle of an indexed host surface lies within `radius` of `point`."
function host_surface_within(surface, point, radius)
    tree, faces, points = surface.tree, surface.faces, surface.points
    r2 = radius^2
    stack = Int32[surface.root]
    while !isempty(stack)
        node = pop!(stack)
        sum(abs2, max.(Float64.(tree.lo[node]) - point, point - Float64.(tree.hi[node]), 0.0)) > r2 && continue
        if tree.left[node] < 0
            first = -tree.left[node]
            for at in first:(first + tree.right[node] - 1)
                a, b, c = faces[tree.order[at]]
                host_triangle_distance2(point, points[a], points[b], points[c]) <= r2 && return true
            end
        else
            push!(stack, tree.left[node], tree.right[node])
        end
    end
    return false
end

"""
    simplify_surface(mesh; tolerance, max_edge = Inf, max_turn = deg2rad(45), min_quality = 0.05)

Edge-collapse simplification of a closed, consistently oriented triangle mesh, placing
vertices by area-weighted quadrics (Garland & Heckbert 1997). A collapse is accepted only if

- every original vertex and face centroid assigned to the collapsed star stays within
  `tolerance` of the new star (the samples move to their nearest new triangle),
- the new vertex, the new edges' midpoints and the new faces' centroids lie within
  `tolerance` of the original surface,
- the surface stays a closed 2-manifold (link condition) and no new triangle crosses a
  current one (candidates: the current faces listed at the original faces near it),
- no remaining face turns its normal by more than `max_turn`, falls below `min_quality`
  (unless it was already a sliver) or gets an edge longer than `max_edge`.

Vertices marked in `locked` keep their place. A collapse whose star holds more than
`max_samples` tracked samples is refused. Returns `(; mesh, collapses, flips, rejected)`.
"""
function simplify_surface(mesh; tolerance, max_edge = Inf, max_turn = deg2rad(45), min_quality = 0.05, rounds = 16,
        locked = falses(length(GeometryBasics.coordinates(mesh))), max_samples = 4096)
    tolerance > 0 && isfinite(tolerance) || throw(ArgumentError("positive finite tolerance required"))
    P = Vec3{Float64}.(GeometryBasics.coordinates(mesh))
    F = [NTuple{3, Int32}(Int32(convert(Int, v)) for v in f) for f in GeometryBasics.faces(mesh)]
    nv = length(P)
    original = host_surface_index(copy(P), [Tuple(Int.(f)) for f in F])
    falive = trues(length(F))
    valive = trues(nv)
    version = zeros(Int32, nv)
    VF = [Int32[] for _ in 1:nv]
    Q = fill(zero(Quadric), nv)
    S = copy(P)
    FS = [Int32[] for _ in F]
    for (i, f) in enumerate(F)
        a, b, c = P[f[1]], P[f[2]], P[f[3]]
        n = triangle_normal(a, b, c)
        l = norm(n)
        l > 0 || throw(ArgumentError("degenerate input triangle $i"))
        q = plane_quadric(n / l, -dot(n / l, a)) * (l / 2)
        for v in f
            push!(VF[v], Int32(i))
            Q[v] += q
        end
        push!(S, (a + b + c) / 3)
        push!(FS[i], Int32(nv + i))
    end
    for v in 1:nv
        push!(FS[VF[v][1]], Int32(v))
    end
    # Current faces listed at the original faces their boxes, grown by `reach`, overlap.
    # Where a current face crosses a new triangle it lies within `reach` of the original
    # surface, so an original face near the new triangle lists it. Faces start listed at
    # themselves; lists drop dead and moved-away faces when they outgrow `limit`.
    reach = 2tolerance
    registry = [Int32[f] for f in eachindex(F)]
    limit = fill(Int32(8), length(F))
    function face_box(f, grow)
        a, b, c = P[F[f][1]], P[F[f][2]], P[F[f][3]]
        return min.(a, b, c) .- grow, max.(a, b, c) .+ grow
    end
    function register!(o, f)
        list = registry[o]
        f in list && return
        if length(list) >= limit[o]
            a, b, c = (original.points[k] for k in original.faces[o])
            olo, ohi = min.(a, b, c), max.(a, b, c)
            filter!(list) do g
                falive[g] || return false
                glo, ghi = face_box(g, reach)
                return all(glo .<= ohi) && all(olo .<= ghi)
            end
            limit[o] = max(Int32(8), Int32(2length(list)))
        end
        push!(list, f)
        return
    end
    mark = zeros(Int32, length(F))
    stamp = Int32(0)
    reached = Vector{Int32}[]
    candidates = Int32[]
    limit2 = tolerance^2
    turn = cos(max_turn)
    heap = SimplifyHeap(Tuple{Float64, Int32, Int32, Int32, Int32}[])
    function candidate(u, v)
        q = Q[u] + Q[v]
        w = quadric_position(q, (P[u] + P[v]) / 2)
        # Mean squared plane distance over the merged area, shortest edges first on ties.
        area = q[1] + q[5] + q[8]
        cost = max(quadric_value(q, w), 0.0) / max(area, floatmin(Float64))
        return (cost + (1e-3 * norm(P[u] - P[v]))^2, w)
    end
    function push_edge!(u, v)
        (locked[u] || locked[v]) && return
        c = candidate(u, v)
        push!(heap, (c[1], Int32(u), Int32(v), version[u], version[v]))
    end
    function neighbours!(out, u)
        empty!(out)
        for f in VF[u]
            falive[f] || continue
            for x in F[f]
                x != u && !(x in out) && push!(out, x)
            end
        end
        return out
    end
    for f in F, j in 1:3
        u, v = f[j], f[mod1(j + 1, 3)]
        u < v && push_edge!(u, v)
    end
    nu, nw = Int32[], Int32[]
    star = Int32[]
    triangles = NTuple{3, Vec3{Float64}}[]
    corners = NTuple{3, Int32}[]
    owners = Int32[]
    assignment = Int32[]
    remaining = length(F)
    collapses = 0
    rejected = zeros(Int, 7) # unused, shared edge, link, normal turn, quality/length, distance, crossing
    # Edges refused for the shape of their star may pass once neighbours have moved:
    # retry them in rounds while collapses still happen.
    deferred = Tuple{Int32, Int32}[]
    for _ in 1:rounds
    before = collapses
    while !isempty(heap)
        _, u, v, vu, vv = pop!(heap)
        valive[u] && valive[v] && version[u] == vu && version[v] == vv || continue
        w = candidate(u, v)[2]
        # The edge's two faces and the link condition of a closed manifold.
        shared = 0
        for f in VF[u]
            falive[f] && v in F[f] && (shared += 1)
        end
        shared == 2 || (rejected[2] += 1; continue)
        neighbours!(nu, u)
        neighbours!(nw, v)
        common = count(x -> x in nw, nu)
        common == 2 || (rejected[3] += 1; push!(deferred, (u, v)); continue)
        remaining > 4 || break
        empty!(star); empty!(triangles); empty!(owners); empty!(corners)
        ok = true
        for (x, y) in ((u, v), (v, u))
            for f in VF[x]
                falive[f] || continue
                f in star || push!(star, f)
                t = F[f]
                y in t && continue
                a, b, cc = (P[k] for k in t)
                old = triangle_normal(a, b, cc)
                p = ntuple(k -> t[k] == x ? w : P[t[k]], 3)
                new = triangle_normal(p...)
                # Printed layer strips start as slivers; later collapses and flips remove
                # them. Only triangles above the quality floor must stay above it.
                quality = triangle_quality(p...)
                floor = triangle_quality(a, b, cc) < min_quality ? 0.0 : min_quality
                if dot(old, new) <= turn * norm(old) * norm(new)
                    rejected[4] += 1
                    ok = false
                    break
                elseif quality < floor || max(norm(p[2] - p[1]), norm(p[3] - p[2]), norm(p[1] - p[3])) > max_edge
                    rejected[5] += 1
                    ok = false
                    break
                end
                push!(triangles, p)
                push!(corners, map(k -> k == x ? u : k, t))
                push!(owners, f)
            end
            ok || break
        end
        ok || (push!(deferred, (u, v)); continue)
        # New geometry near the original surface.
        if !host_surface_within(original, w, tolerance)
            ok = false
        else
            for p in triangles
                host_surface_within(original, (p[1] + p[2] + p[3]) / 3, tolerance) || (ok = false; break)
            end
            if ok
                for x in nu
                    x == v && continue
                    host_surface_within(original, (w + P[x]) / 2, tolerance) || (ok = false; break)
                end
                for x in nw
                    (x == u || x in nu) && continue
                    host_surface_within(original, (w + P[x]) / 2, tolerance) || (ok = false; break)
                end
            end
        end
        # Original samples near the new star. Very large flat faces gather many samples;
        # past `max_samples` the collapse is refused rather than re-checking them all.
        empty!(assignment)
        ok && sum(f -> length(FS[f]), star) > max_samples && (ok = false)
        if ok
            for f in star, sample in FS[f]
                best, at = Inf, Int32(0)
                for (k, p) in enumerate(triangles)
                    d = host_triangle_distance2(S[sample], p...)
                    d < best && ((best, at) = (d, owners[k]))
                end
                best <= limit2 || (ok = false; break)
                push!(assignment, at)
            end
        end
        ok || (rejected[6] += 1; continue)
        # No new triangle may cross a current face outside the star.
        for (k, p) in enumerate(triangles)
            stamp += Int32(1)
            empty!(candidates)
            lo = min.(p[1], p[2], p[3]) .- reach
            hi = max.(p[1], p[2], p[3]) .+ reach
            k > length(reached) && push!(reached, Int32[])
            host_surface_box!(reached[k], original, lo, hi)
            for o in reached[k], c in registry[o]
                mark[c] == stamp && continue
                mark[c] = stamp
                push!(candidates, c)
            end
            for c in candidates
                (falive[c] && !(c in star)) || continue
                fc = F[c]
                any(in(corners[k]), fc) && continue
                clo, chi = face_box(c, 0.0)
                (all(clo .<= hi) && all(lo .<= chi)) || continue
                q = (P[fc[1]], P[fc[2]], P[fc[3]])
                crossing = false
                for e in 1:3
                    crossing |= segment_crosses_triangle(p[e], p[mod1(e + 1, 3)], q...; depth = tolerance / 100)
                    crossing |= segment_crosses_triangle(q[e], q[mod1(e + 1, 3)], p...; depth = tolerance / 100)
                end
                crossing && (ok = false; break)
            end
            ok || break
        end
        ok || (rejected[7] += 1; push!(deferred, (u, v)); continue)
        moved = Int32[]
        for f in star
            append!(moved, FS[f])
            empty!(FS[f])
        end
        for (sample, f) in zip(moved, assignment)
            push!(FS[f], sample)
        end
        for f in VF[u]
            falive[f] && v in F[f] && (falive[f] = false; remaining -= 1)
        end
        for f in VF[v]
            falive[f] || continue
            F[f] = map(k -> k == v ? u : k, F[f])
            push!(VF[u], f)
        end
        filter!(f -> falive[f], VF[u])
        empty!(VF[v])
        P[u] = w
        Q[u] += Q[v]
        valive[v] = false
        version[u] += Int32(1)
        collapses += 1
        for (k, f) in enumerate(owners), o in reached[k]
            register!(o, f)
        end
        for x in neighbours!(nu, u)
            push_edge!(u, x)
        end
    end
    collapses == before && break
    for (u, v) in deferred
        valive[u] && valive[v] && push_edge!(u, v)
    end
    empty!(deferred)
    end
    keep = findall(valive)
    index = zeros(Int32, nv)
    index[keep] .= Int32.(1:length(keep))
    faces = [NTuple{3, Int32}(index[k] for k in F[f]) for f in eachindex(F) if falive[f]]
    points = P[keep]
    flips = flip_flat_edges!(points, faces; deviation = tolerance / 10, max_turn)
    out = GeometryBasics.Mesh(Point3d.(points), [GLTriangleFace(f...) for f in faces])
    return (; mesh = out, collapses, flips, rejected)
end

"Shortest distance between segments pq and rs."
function segment_distance(p, q, r, s)
    d1, d2, w = q - p, s - r, p - r
    a, e, b, c, f = dot(d1, d1), dot(d2, d2), dot(d1, d2), dot(d1, w), dot(d2, w)
    den = a * e - b^2
    sN = den > 1e-30 * a * e ? clamp((b * f - c * e) / den, 0.0, 1.0) : 0.0
    t = e > 0 ? clamp((b * sN + f) / e, 0.0, 1.0) : 0.0
    sN = a > 0 ? clamp((b * t - c) / a, 0.0, 1.0) : 0.0
    return norm(w + sN * d1 - t * d2)
end

"""
Flip edges between nearly coplanar faces where that raises the pair's worse quality. The
surface moves by at most `deviation` (distance between the old and new diagonal), and no
new triangle may cross another face by more than `depth` (see `segment_crosses_triangle`):
opposite walls closer than `deviation` stay apart.
"""
function flip_flat_edges!(P, F; deviation, depth = deviation / 10, max_turn = deg2rad(45), rounds = 8)
    turn = cos(max_turn)
    flips = 0
    nearby = Int32[]
    candidates = Int32[]
    function crosses(t, c)
        any(in(c), t) && return false
        p, q = (P[t[1]], P[t[2]], P[t[3]]), (P[c[1]], P[c[2]], P[c[3]])
        for e in 1:3
            segment_crosses_triangle(p[e], p[mod1(e + 1, 3)], q...; depth) && return true
            segment_crosses_triangle(q[e], q[mod1(e + 1, 3)], p...; depth) && return true
        end
        return false
    end
    for _ in 1:rounds
        edges = Dict{Tuple{Int32, Int32}, Int32}()
        for (i, f) in enumerate(F), j in 1:3
            edges[(f[j], f[mod1(j + 1, 3)])] = Int32(i)
        end
        # Faces as they were at the start of the round. A face flipped since covers, with
        # its `partner`, the same quad as before, so the pair is found through either box.
        index = host_surface_index(P, F)
        partner = zeros(Int32, length(F))
        touched = falses(length(F))
        changed = 0
        for (i, f) in enumerate(F), j in 1:3
            touched[i] && continue
            a, b, c = f[j], f[mod1(j + 1, 3)], f[mod1(j + 2, 3)]
            a < b || continue
            k = get(edges, (b, a), Int32(0))
            (k == 0 || touched[k]) && continue
            g = F[k]
            d = g[findfirst(x -> x != a && x != b, g)]
            (c == d || haskey(edges, (c, d)) || haskey(edges, (d, c))) && continue
            old = min(triangle_quality(P[a], P[b], P[c]), triangle_quality(P[b], P[a], P[d]))
            new = min(triangle_quality(P[a], P[d], P[c]), triangle_quality(P[d], P[b], P[c]))
            new > 1.2old || continue
            segment_distance(P[a], P[b], P[c], P[d]) <= deviation || continue
            n0 = triangle_normal(P[a], P[b], P[c]) + triangle_normal(P[b], P[a], P[d])
            n1, n2 = triangle_normal(P[a], P[d], P[c]), triangle_normal(P[d], P[b], P[c])
            dot(n0, n1) > turn * norm(n0) * norm(n1) && dot(n0, n2) > turn * norm(n0) * norm(n2) || continue
            lo = min.(P[a], P[b], P[c], P[d]) .- deviation
            hi = max.(P[a], P[b], P[c], P[d]) .+ deviation
            host_surface_box!(nearby, index, lo, hi)
            empty!(candidates)
            for o in nearby
                push!(candidates, o)
                partner[o] > 0 && push!(candidates, partner[o])
            end
            any(o -> o != i && o != k && (crosses((a, d, c), F[o]) || crosses((d, b, c), F[o])), candidates) && continue
            F[i] = (a, d, c)
            F[k] = (d, b, c)
            touched[i] = touched[k] = true
            partner[i], partner[k] = k, i
            for e in ((a, b), (b, a))
                delete!(edges, e)
            end
            edges[(a, d)] = i; edges[(d, c)] = i; edges[(c, a)] = i
            edges[(d, b)] = k; edges[(b, c)] = k; edges[(c, d)] = k
            changed += 1
        end
        flips += changed
        changed == 0 && break
    end
    return flips
end

"""
Segment pq crosses triangle abc by more than `depth`: both endpoints lie at least `depth`
off the triangle's plane on opposite sides and the crossing point at least `depth` inside
its edges. With `depth = 0` any proper crossing counts.
"""
function segment_crosses_triangle(p, q, a, b, c; depth = 0.0)
    n = cross(b - a, c - a)
    l = norm(n)
    l > 0 || return false
    n /= l
    dp, dq = dot(p - a, n), dot(q - a, n)
    (dp > depth && dq < -depth) || (dp < -depth && dq > depth) || return false
    x = p + (dp / (dp - dq)) * (q - p)
    for (e0, e1) in ((a, b), (b, c), (c, a))
        e = e1 - e0
        m = cross(n, e)
        dot(x - e0, m) > depth * norm(m) || return false
    end
    return true
end

"""
    surface_self_crossings(points, faces)

Pairs of triangles without a shared vertex whose edges cross each other (sweep along x,
bounding-box culled, threaded).
"""
function surface_self_crossings(points, faces; depth = 0.0)
    lo = [min.(points[f[1]], points[f[2]], points[f[3]]) for f in faces]
    hi = [max.(points[f[1]], points[f[2]], points[f[3]]) for f in faces]
    order = sortperm(lo; by = first)
    function sweep(range)
        found = Tuple{Int, Int}[]
        for k in range
            i = order[k]
            fi = faces[i]
            for m in (k + 1):length(order)
                j = order[m]
                lo[j][1] > hi[i][1] && break
                (lo[j][2] > hi[i][2] || lo[i][2] > hi[j][2] || lo[j][3] > hi[i][3] || lo[i][3] > hi[j][3]) && continue
                fj = faces[j]
                any(in(fj), fi) && continue
                hit = false
                for e in 1:3
                    hit |= segment_crosses_triangle(points[fi[e]], points[fi[mod1(e + 1, 3)]], points[fj[1]], points[fj[2]], points[fj[3]]; depth)
                    hit |= segment_crosses_triangle(points[fj[e]], points[fj[mod1(e + 1, 3)]], points[fi[1]], points[fi[2]], points[fi[3]]; depth)
                end
                hit && push!(found, minmax(i, j))
            end
        end
        return found
    end
    chunks = Iterators.partition(eachindex(order), cld(length(order), 4Threads.nthreads()))
    crossings = fetch.([Threads.@spawn sweep(range) for range in chunks])
    return reduce(vcat, crossings)
end

"""
    simplify_surface_clean(mesh; tolerance, attempts = 8, kw...)

`simplify_surface`, repeated with the original vertices around each self-crossing of the
result locked in place until the simplified surface has none.
"""
function simplify_surface_clean(mesh; tolerance, attempts = 8, kw...)
    original = Vec3{Float64}.(GeometryBasics.coordinates(mesh))
    locked = falses(length(original))
    for attempt in 1:attempts
        result = simplify_surface(mesh; tolerance, locked, kw...)
        points = Vec3{Float64}.(GeometryBasics.coordinates(result.mesh))
        faces = [Tuple(convert(Int, v) for v in f) for f in GeometryBasics.faces(result.mesh)]
        crossings = surface_self_crossings(points, faces; depth = tolerance / 100)
        isempty(crossings) && return (; result..., attempts = attempt, locked = count(locked))
        # Lock every original vertex inside the crossing triangles' bounding boxes,
        # grown by the tolerance, so the next pass keeps that region as printed.
        for (i, j) in crossings, f in (faces[i], faces[j])
            lo = min.(points[f[1]], points[f[2]], points[f[3]]) .- 2tolerance
            hi = max.(points[f[1]], points[f[2]], points[f[3]]) .+ 2tolerance
            for (v, p) in enumerate(original)
                all(lo .<= p .<= hi) && (locked[v] = true)
            end
        end
    end
    error("simplified surface still crosses itself after $attempts attempts")
end
