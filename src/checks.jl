# Checks of an assembly on the GPU, from its items traced by `Rays`: what overlaps in place,
# what a move runs into, whether the parts go together in order, how far each slides.

"Whether boxes (lo, hi) meet, or come within `margin` of each other."
overlaps((lo1, hi1), (lo2, hi2); margin = 0.0) = all(lo1 .- margin .<= hi2) && all(lo2 .- margin .<= hi1)

"""
    RayChecks(items; rays = Rays())

The `items` of an assembly, each triangulated into a structure of its own on `rays`' device
(`Traced`), by name, in the items' order: what `interference`, `collisions`,
`sequence_check` and `axial_check` run on. Triangulating takes a few minutes for a hundred
parts; the checks then take seconds.
"""
struct RayChecks{R <: Rays, T <: Traced}
    rays::R
    names::Vector{String}
    parts::Dict{String, T}
end

RayChecks(list::AbstractVector{<:Item}; rays = Rays()) = RayChecks(rays, [it.name for it in list], traced(rays, list))

"Box of item `name` at `pose`."
box(c::RayChecks, name, pose = Pose()) = moved_box(c.parts[name], pose)

"Volumes (mm³) each pair in `pairs`, `(a, pose_a, b, pose_b)` with item names, shares."
shared(c::RayChecks, pairs; kw...) = shared(c.rays, [(c.parts[a], pa, c.parts[b], pb) for (a, pa, b, pb) in pairs]; kw...)

"""
    interference(c::RayChecks; tolerance = 0.5)

Every pair of items whose boxes meet, in place: the pairs that share more than `tolerance`
mm³, largest first. In a design that fits only intended press fits (O-rings in their grooves,
pawl tips on the ratchets, pins and pegs) show up.
"""
function interference(c::RayChecks; tolerance = 0.5)
    I = Pose()
    pairs = [(a, I, b, I) for (i, a) in enumerate(c.names) for b in c.names[(i + 1):end] if overlaps(box(c, a), box(c, b))]
    v = shared(c, pairs)
    hits = [(a = a, b = b, volume = x) for ((a, _, b, _), x) in zip(pairs, v) if x > tolerance]
    return sort!(hits; by = h -> -h.volume)
end

"""
The items `moving` (name => pose) against all others but `ignore`, in place: the items they
share more than `tolerance` mm³ with, and how much, largest first.
"""
function collisions(c::RayChecks, moving::Dict{String, Pose}; ignore = String[], tolerance = 0.5)
    I = Pose()
    pairs = [(m, p, s, I) for (m, p) in moving for s in c.names if !haskey(moving, s) && !(s in ignore) && overlaps(box(c, m, p), box(c, s))]
    hit = Dict{String, Float64}()
    for ((_, _, s, _), x) in zip(pairs, shared(c, pairs))
        hit[s] = get(hit, s, 0.0) + x
    end
    return sort!([s => x for (s, x) in hit if x > tolerance]; by = h -> -h.second)
end

"""
    box_window(box, other, u, L)

The stretch of a move along `u` (from `L` before its place to its place) during which the
moving bounding box `box` (taken at the place) meets `other`: `(near, far)`, or `nothing`.
"""
function box_window((lo, hi), (olo, ohi), u, L)
    near, far = 0.0, L
    for k in 1:3
        # lo + t u ≤ ohi and olo ≤ hi + t u, for t in [0, L]
        if abs(u[k]) < 1.0e-12
            (lo[k] <= ohi[k] && olo[k] <= hi[k]) || return nothing
        else
            a, e = (ohi[k] - lo[k]) / u[k], (olo[k] - hi[k]) / u[k]
            near, far = max(near, min(a, e)), min(far, max(a, e))
        end
    end
    return near <= far ? (near, far) : nothing
end

"""
    sequence_check(c::RayChecks, plan; steps = all, step = 0.5, tolerance = 0.5)

Builds the assembly in the order of `plan` (`AssemblyStep`s) and makes every move
its steps name (`AssemblyStep.moves`): the items of a move come in along their way, from where
it starts to their place, against everything already in place (earlier steps, the step's base
items and the moves before it in the same step), `step` mm at a time over the stretch where
their boxes meet. A press fit overlaps its mate at its place; on the way it must never overlap
more than there. Returns, per move, each item it hits with the largest overlap beyond the one
at its place (mm³) and the distances before its place between which that happens (mm).
"""
function sequence_check(c::RayChecks, plan::AbstractVector{AssemblyStep}; steps = nothing, step = 0.5, tolerance = 0.5)
    done = Set{String}()
    results = Pair{String, Vector{@NamedTuple{with::String, excess::Float64, from::Float64, to::Float64}}}[]
    I = Pose()
    for (k, s) in enumerate(plan)
        if isnothing(steps) || k in steps
            placed = [n for n in c.names if (!s.apart && n in done) || occursin(s.base, n)]
            for (pattern, way) in s.moves
                group = [n for n in c.names if occursin(pattern, n) && !(n in placed)]
                isempty(group) && continue
                L, u = norm(way), Vec3f(normalize(way))
                pairs, at = Tuple{String, Pose, String, Pose}[], Tuple{String, Float64}[]
                for m in group, p in placed
                    w = box_window(box(c, m), box(c, p), u, L)
                    isnothing(w) && continue
                    for t in unique(vcat(0.0, range(first(w), last(w); step)))
                        push!(pairs, (m, Pose(Float32(t) * u), p, I))
                        push!(at, (p, t))
                    end
                end
                volume = Dict{Tuple{String, Float64}, Float64}()
                for (key, x) in zip(at, shared(c, pairs))
                    volume[key] = get(volume, key, 0.0) + x
                end
                found = @NamedTuple{with::String, excess::Float64, from::Float64, to::Float64}[]
                for p in unique(first.(at))
                    place = get(volume, (p, 0.0), 0.0)
                    over = [(t, x - place) for ((q, t), x) in volume if q == p && t > 0 && x - place > tolerance]
                    isempty(over) && continue
                    push!(found, (with = p, excess = round(maximum(last, over); digits = 1), from = round(maximum(first, over); digits = 1),
                        to = round(minimum(first, over); digits = 1)))
                end
                append!(placed, group)
                push!(results, "$k. $(s.title): $(join(group, ", "))" => sort!(found; by = h -> -h.excess))
                @info "sequence_check" last(results)
            end
        end
        s.kept && union!(done, [n for n in c.names if first(step_role(s, n)) != :none])
    end
    return results
end

"""
The first contact of the items `moving` (name => shift along x) as they slide on along `sense`:
how much further (mm) and with which other item; `(Inf, "")` within `reach`. Overlaps they
have at the start (pawl tips on their ratchets, O-rings on the cork) do not count until they
would press deeper (`travel`).
"""
function first_contact(c::RayChecks, moving::Dict{String, Float64}, sense; reach, lines)
    shifted(m) = Pose(Vec3f(moving[m], 0, 0))
    # Only what the moving boxes sweep through on their way.
    function swept(m, s)
        (lo, hi), (slo, shi) = box(c, m, shifted(m)), box(c, s)
        return all(lo[2:3] .<= shi[2:3]) && all(slo[2:3] .<= hi[2:3]) && lo[1] - reach <= shi[1] && slo[1] <= hi[1] + reach
    end
    near = [(m, s) for m in keys(moving) for s in c.names if !haskey(moving, s) && swept(m, s)]
    isempty(near) && return (Inf, "")
    t = travel(c.rays, [(c.parts[m], shifted(m), c.parts[s], Pose()) for (m, s) in near], sense; lines)
    k = argmin(t)
    return t[k] > reach ? (Inf, "") : (t[k], near[k][2])
end

"""
    axial_check(c::RayChecks, bodies, parts; reach = 5.0, lines = 2^18)

How far each body that moves along x slides, each way, until the frame stops it, and how far
each part slides on its shaft between its neighbours. `bodies` and `parts` name them, each a
title and the pattern of its items' names (`"feed shaft" => r"^feed_(shaft|hub)\$"`); every item
in no body stands, as part of the frame. A body pushes the free bodies it meets ahead of it
until one of them meets a part of the frame. Overlaps a part has at its place (pawl tips on
their ratchets, press fits) only count once they would grow. Returns both tables: how far each
way, what stops it, and for the bodies whom it pushes; `Inf` if nothing stops it within
`reach` mm.
"""
function axial_check(c::RayChecks, body_patterns, part_patterns; reach = 5.0, lines = 2^18)
    bodies = Dict(body => [n for n in c.names if occursin(pattern, n)] for (body, pattern) in body_patterns)
    body_of = Dict(n => body for (body, names) in bodies for n in names)
    contact(moving, sense) = first_contact(c, moving, sense; reach, lines)
    cut(t) = t > reach ? Inf : round(t; digits = 2)
    function travel_of(start, sense)
        moving = Dict(n => 0.0 for n in bodies[start])
        joined, total = [start], 0.0
        while true
            t, hit = contact(moving, sense)
            total += t
            total > reach && return (Inf, "", joined[2:end])
            foreach(n -> moving[n] += sense * t, keys(moving))
            next = get(body_of, hit, nothing)
            isnothing(next) && return (total, hit, joined[2:end])
            push!(joined, next)
            foreach(n -> moving[n] = 0.0, bodies[next])
        end
    end
    body_rows = map(body_patterns) do (body, _)
        (m, mby, mvia), (p, pby, pvia) = travel_of(body, -1), travel_of(body, 1)
        (body = body, minus = cut(m), minus_by = mby, minus_pushes = join(mvia, ", "), plus = cut(p), plus_by = pby, plus_pushes = join(pvia, ", "))
    end
    part_rows = map(part_patterns) do (part, pattern)
        moving = Dict(n => 0.0 for n in c.names if occursin(pattern, n))
        (m, mby), (p, pby) = contact(moving, -1), contact(moving, 1)
        (part = part, minus = cut(m), minus_by = mby, plus = cut(p), plus_by = pby)
    end
    return (bodies = body_rows, parts = part_rows)
end

