# Where printed parts meet, checked on the parts as printed: cuts across x through the parts
# rebuilt from the slicer's G-code (`printed_parts`), in the profile plane.

"A region turned by `θ` about `c` in the profile plane."
function turned(g, c::Point2d, θ)
    s, k = sincos(θ)
    return map_region(g, v -> [c[1] + k * (v[1] - c[1]) - s * (v[2] - c[2]), c[2] + s * (v[1] - c[1]) + k * (v[2] - c[2])])
end

overlap(a, b) = GEOS.area(GEOS.intersection(a, b))
gap(a, b) = GEOS.distance(a, b)

"""
A bought part where it meets a printed one, as its drawing gives it: `region(x, axis)` is its
cross section at `x` in the profile plane. It does not shrink or drift.
"""
struct BoughtSection{F}
    name::String
    region::F
end

"""
A place where two parts meet, cut at the planes `xs`. Both parts are lined up where they meet:
at `axis`, or without one at the middle of the moving part's cut. `kind` says what it has to do:

- `:turns`: `moving` turns about `axis` in `host` and must not touch it at any angle;
- `:holds`: a clamp or press fit, `moving` must overlap `host` a little;
- `:slides`: `moving` goes into `host` and must not overlap it;
- `:keyed`: `moving` slides onto the D-shaft `host` without overlapping it, and the flat drives
  it (`key_play`);
- `:seated`: a clip in its groove: it goes where the groove ended up after cooling, along x
  too, and must not overlap it;
- `:sits`: a peg or pin, `moving` needs `host` material all around it;
- `:rests`: a foot or block on a face, `moving` (cut at `xs`) needs `host` (cut at `host_xs`)
  under all of it.
"""
struct Joint
    name::String
    moving::Union{String, BoughtSection}
    host::Union{String, BoughtSection}
    kind::Symbol
    axis::Union{Point2d, Nothing}
    xs::Vector{Float64}
    host_xs::Vector{Float64}
    only::Union{Vector{Point2d}, Nothing}   # the part of the moving section that counts, if not all of it
end
Joint(name, moving, host, kind, axis, xs; only = nothing) = Joint(name, moving, host, kind, axis, xs, xs, only)
Joint(name, moving, host, kind, axis, xs, host_xs) = Joint(name, moving, host, kind, axis, xs, host_xs, nothing)

"Section of a part at `x`: a printed part as printed, a bought part as drawn."
part_section(printed::Dict{String, PrintedPart}, name::String, x, axis::Point2d) = section(printed[name], x)
part_section(::Dict{String, PrintedPart}, p::BoughtSection, x, axis::Point2d) = p.region(x, axis)

"How far cooling moved a part at (`x`, `axis`) in the profile plane; bought parts stay."
function profile_drift(printed::Dict{String, PrintedPart}, name::String, x, axis::Point2d)
    v = drift(printed[name], Vec3d(x, axis[1], axis[2]))
    return Vec2d(v[2], v[3])
end
profile_drift(::Dict{String, PrintedPart}, ::BoughtSection, x, axis::Point2d) = Vec2d(0, 0)

"A region moved by `v` in the profile plane."
moved(g, v::Vec2d) = map_region(g, q -> [q[1] + v[1], q[2] + v[2]])

"""
    check(joint, printed; angles = 0:10:350)

What the printed parts do at `joint`, the worst over its planes: for `:turns` the smallest gap
at any angle (negative: the overlap area in mm² where it binds), for `:holds` the overlap
(positive: it grips) or the gap, for `:slides` the gap, for `:sits` and `:rests` how much of the
moving part has material around or under it.
"""
function check(j::Joint, printed::Dict{String, PrintedPart}; angles = 0:10:350)
    worst = j.kind in (:sits, :rests) ? 1.0 : Inf
    for (x, xh) in zip(j.xs, j.host_xs)
        # The moving part sits where the host's feature ended up after cooling.
        a0 = part_section(printed, j.moving, x, something(j.axis, Point2d(0, 0)))
        j.only === nothing || (a0 = GEOS.intersection(a0, GEOS.Polygon([[[q[1], q[2]] for q in vcat(j.only, j.only[1:1])]])))
        at = something(j.axis, Point2d(GEOS.GeoInterface.coordinates(GEOS.centroid(a0))...))
        if j.kind == :seated
            q = Vec3d(x, at[1], at[2])
            xh = x + drift(printed[j.host], q)[1] - drift(printed[j.moving], q)[1]
        end
        dh = profile_drift(printed, j.host, xh, at)
        a = moved(j.axis === nothing ? a0 : part_section(printed, j.moving, x, at), dh - profile_drift(printed, j.moving, x, at))
        h = part_section(printed, j.host, xh, at)
        axis = at + dh
        GEOS.isEmpty(a) && error("$(j.moving) has no material at x = $x ($(j.name))")
        if j.kind == :turns
            for θ in angles
                t = turned(a, axis, deg2rad(θ))
                o = overlap(t, h)
                worst = min(worst, o > 1.0e-4 ? -o : gap(t, h))
            end
        elseif j.kind == :holds
            o = overlap(a, h)
            worst = min(worst, o > 1.0e-4 ? o : -gap(a, h))
        elseif j.kind in (:slides, :keyed, :seated)
            o = overlap(a, h)
            worst = min(worst, o > 1.0e-4 ? -o : gap(a, h))
        elseif j.kind == :sits
            ring = GEOS.difference(GEOS.buffer(a, 0.6, 8), a)
            worst = min(worst, overlap(ring, h) / GEOS.area(ring))
        else
            # Its own pegs fill the holes under it.
            own = moved(part_section(printed, j.moving, xh, at), dh - profile_drift(printed, j.moving, xh, at))
            worst = min(worst, overlap(a, GEOS.union(h, own)) / GEOS.area(a))
        end
    end
    return worst
end

"""
    joint_report(printed, joints)

Every joint checked on the printed parts, with a verdict: turning parts need a gap at every
angle, fits that hold need some overlap (clamps, press fits) but not more than printed parts
give (0.3 mm² per mm of contact is a hard press), slides need a gap, pegs and feet need
material around or under at least 80 % of them.
"""
function joint_report(printed::Dict{String, PrintedPart}, joints::AbstractVector{Joint})
    map(joints) do j
        v = check(j, printed)
        ok = j.kind == :turns ? v > 0.02 : j.kind == :holds ? v > 0 : j.kind in (:slides, :keyed, :seated) ? v >= 0 : v >= 0.8
        (joint = j.name, kind = j.kind, value = round(v; digits = 3), ok = ok)
    end
end

"""
    key_play(j, printed; step = 0.25, limit = 30.0)

How far (degrees) the part of a `:keyed` joint turns on its D-shaft, both ways together, before
its bore meets the shaft's flat: the play with which the flat drives it. The largest over the
joint's planes; `2limit` if it turns that far without meeting the flat.
"""
function key_play(j::Joint, printed::Dict{String, PrintedPart}; step = 0.25, limit = 30.0)
    worst = 0.0
    for (x, xh) in zip(j.xs, j.host_xs)
        at = j.axis
        dh = profile_drift(printed, j.host, xh, at)
        a = moved(part_section(printed, j.moving, x, at), dh - profile_drift(printed, j.moving, x, at))
        h = part_section(printed, j.host, xh, at)
        axis = at + dh
        way(sense) = something(findfirst(θ -> overlap(turned(a, axis, sense * deg2rad(θ)), h) > 1.0e-4, step:step:limit), length(step:step:limit)) * step
        worst = max(worst, way(1) + way(-1))
    end
    return worst
end

"Every `:keyed` joint's play (`key_play`), degrees."
key_report(printed::Dict{String, PrintedPart}, joints::AbstractVector{Joint}) =
    [(joint = j.name, play = key_play(j, printed)) for j in joints if j.kind == :keyed]

"Smallest and largest distance from `c` of the outline of a region."
function radii(g, c::Point2d)
    rs = Float64[]
    for poly in polygons(g), v in GEOS.GeoInterface.coordinates(GEOS.exteriorRing(poly))
        push!(rs, hypot(v[1] - c[1], v[2] - c[2]))
    end
    return extrema(rs)
end

disk(c::Point2d, r) = GEOS.buffer(GEOS.Point(c[1], c[2]), r, 32)

"""
    printed_interference(printed; step = 1.0, tolerance = 0.05)

Every pair of printed parts that overlap, from cuts across the whole feeder every `step` mm
along x: the volume they share (mm³, cut area times step) and the deepest overlap (mm, the
largest overlap area per mm of its outline). Parts stay where they were printed for, each
cooled toward its own middle. Pairs that are meant to press together show up too.
"""
function printed_interference(printed::Dict{String, PrintedPart}; step = 1.0, tolerance = 0.05)
    names = collect(keys(printed))
    extent = Dict(n => x_extent(printed[n]) for n in names)
    lo, hi = minimum(first(e) for e in values(extent)), maximum(last(e) for e in values(extent))
    shared = Dict{Tuple{String, String}, Vector{Float64}}()
    for x in (lo + step / 2 + 0.037):step:hi
        here = [n for n in names if extent[n][1] < x < extent[n][2]]
        cuts = Dict(n => section(printed[n], x) for n in here)
        boxes = Dict(n => GEOS.envelope(cuts[n]) for n in here)
        for i in eachindex(here), j in (i + 1):length(here)
            a, b = here[i], here[j]
            GEOS.intersects(boxes[a], boxes[b]) || continue
            o = GEOS.intersection(cuts[a], cuts[b])
            area = GEOS.area(o)
            area > 1.0e-3 || continue
            depth = 2area / max(GEOS.geomLength(o), 1.0e-9)
            key = a < b ? (a, b) : (b, a)
            v = get!(shared, key, [0.0, 0.0])
            v[1] += area * step
            v[2] = max(v[2], depth)
        end
    end
    hits = [(a = k[1], b = k[2], volume = round(v[1]; digits = 2), depth = round(v[2]; digits = 3)) for (k, v) in shared if v[1] > tolerance]
    return sort!(hits; by = h -> -h.volume)
end

"An axis-aligned rectangle from `lo` to `hi` as a LibGEOS polygon."
rect(lo::Point2d, hi::Point2d) = GEOS.Polygon([[[lo[1], lo[2]], [hi[1], lo[2]], [hi[1], hi[2]], [lo[1], hi[2]], [lo[1], lo[2]]]])

