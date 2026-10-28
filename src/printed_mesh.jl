function printed_rings(region)
    rings = Vector{Point2d}[]
    for poly in polygons(region)
        for (k, ring) in enumerate([GEOS.exteriorRing(poly); collect(GEOS.interiorRings(poly))])
            points = Point2d[Point2d(p...) for p in GEOS.GeoInterface.coordinates(ring)]
            !isempty(points) && first(points) == last(points) && pop!(points)
            length(points) < 3 && continue
            area = sum(points[i][1] * points[mod1(i + 1, length(points))][2] -
                points[i][2] * points[mod1(i + 1, length(points))][1] for i in eachindex(points))
            (k == 1 ? area < 0 : area > 0) && reverse!(points)
            push!(rings, points)
        end
    end
    return rings
end

"""
    printed_mesh(part; unit=1e-3)

Construct the cooled layer-union surface using planar region differences and
constrained triangulation through LibGEOS. Shared layer edges are subdivided at
all incident vertices so the output contains neither internal caps nor T-junctions.
"""
function printed_mesh(part::PrintedPart; unit = 1e-3, precision = 1e-5)
    unit > 0 || throw(ArgumentError("unit must be positive"))
    precision > 0 || throw(ArgumentError("precision must be positive"))
    isempty(part.layers) && throw(ArgumentError("a printed mesh needs layers"))
    # Coordinates in slicer units (mm). This grid only removes roundoff at Boolean
    # intersections; its displacement is less than 10⁻¹² m after transformation.
    canonical(p) = Point2d(round(p[1]; digits = 10), round(p[2]; digits = 10))
    # Slicer text can describe a nominally shared plane as, e.g., 17.799999
    # and 17.8 mm. Normalize z as well as the contours so those do not become
    # artificial one-nanometer volumes at a layer seam.
    layers = [(; z0 = round(l.z0; digits = 5), z1 = round(l.z1; digits = 5), region = l.region)
        for l in part.layers if round(l.z1; digits = 5) > round(l.z0; digits = 5)]
    # Normalize the planar topology before differencing adjacent layers. A 10 nm
    # grid stays below the 0.1 µm volume-mesh envelope and eliminates numerical
    # ribbons between otherwise coincident slicer perimeters.
    regions = [GEOS.setPrecision(GEOS.setPrecision(l.region, precision), 0.0) for l in layers]
    rings = [map(r -> canonical.(r), printed_rings(region)) for region in regions]
    caps = Dict{Float64, Vector{Tuple{NTuple{3, Point2d}, Bool}}}()
    plane_points = Dict{Float64, Set{Point2d}}()
    for (l, rr) in zip(layers, rings), z in (l.z0, l.z1), r in rr
        union!(get!(Set{Point2d}, plane_points, z), r)
    end
    function cap!(region, z, upward)
        GEOS.isEmpty(region) && return
        for polygon in polygons(GEOS.constrainedDelaunayTriangulation(region))
            r = only(printed_rings(polygon))
            length(r) == 3 || error("cap triangulation did not return triangles")
            triangle = Tuple(canonical.(r))
            length(unique(triangle)) == 3 || continue
            a, b, c = map(p -> (round(Int128, p[1] * 1e10), round(Int128, p[2] * 1e10)), triangle)
            (b[1] - a[1]) * (c[2] - a[2]) == (b[2] - a[2]) * (c[1] - a[1]) && continue
            push!(get!(Vector{Tuple{NTuple{3, Point2d}, Bool}}, caps, z), (triangle, upward))
            union!(get!(Set{Point2d}, plane_points, z), triangle)
        end
    end
    for (i, l) in enumerate(layers)
        below = i > 1 && layers[i - 1].z1 == l.z0 ? regions[i - 1] : nothing
        above = i < length(layers) && layers[i + 1].z0 == l.z1 ? regions[i + 1] : nothing
        cap!(below === nothing ? regions[i] : GEOS.difference(regions[i], below), l.z0, false)
        cap!(above === nothing ? regions[i] : GEOS.difference(regions[i], above), l.z1, true)
    end
    indexes = Dict(z => (sort!(collect(points); by = first), sort!(collect(points); by = last)) for (z, points) in plane_points)
    function edge_points(a, b, z)
        a == b && return Point2d[a]
        d = b - a
        axis = abs(d[1]) < abs(d[2]) ? 1 : 2
        points = indexes[z][axis]
        lo, hi = minmax(a[axis], b[axis])
        first_id = searchsortedfirst(points, lo - 2e-10; by = p -> p isa Number ? p : p[axis])
        last_id = searchsortedlast(points, hi + 2e-10; by = p -> p isa Number ? p : p[axis])
        result = Tuple{Float64, Point2d}[(0.0, a), (1.0, b)]
        dd = dot(d, d)
        for j in first_id:last_id
            p = points[j]
            t = dot(p - a, d) / dd
            0 < t < 1 || continue
            norm(p - a - t * d) <= 2e-10 || continue
            push!(result, (t, p))
        end
        sort!(result; by = first)
        return unique!(last.(result))
    end
    positions, triangles = Point3d[], GLTriangleFace[]
    lookup = Dict{Point3d, Int}()
    function vertex!(p)
        get!(lookup, p) do
            push!(positions, p)
            length(positions)
        end
    end
    function triangle!(a, b, c)
        norm(cross(b - a, c - a)) > 0 || return
        push!(triangles, GLTriangleFace(vertex!(a), vertex!(b), vertex!(c)))
    end
    lift(p, z) = Point3d(p[1], p[2], z)
    function triangulate_boundary!(boundary, z, upward)
        xy = [(round(Int128, p[1] * 1e10), round(Int128, p[2] * 1e10)) for p in boundary]
        orient(a, b, c) = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])
        sign = upward ? 1 : -1
        ids = collect(eachindex(boundary))
        area = sign * sum(xy[i][1] * xy[mod1(i + 1, length(xy))][2] -
            xy[i][2] * xy[mod1(i + 1, length(xy))][1] for i in eachindex(xy))
        while length(ids) > 3
            found = false
            for at in eachindex(ids)
                a, b, c = ids[mod1(at - 1, length(ids))], ids[at], ids[mod1(at + 1, length(ids))]
                ear = sign * orient(xy[a], xy[b], xy[c])
                0 < ear < area || continue
                # Keep collinear boundary vertices in the triangulation: a point
                # on the proposed diagonal blocks the ear just like an interior one.
                blocked = any(ids) do v
                    v in (a, b, c) && return false
                    sign * orient(xy[a], xy[b], xy[v]) >= 0 &&
                        sign * orient(xy[b], xy[c], xy[v]) >= 0 &&
                        sign * orient(xy[c], xy[a], xy[v]) >= 0
                end
                blocked && continue
                triangle!(lift(boundary[a], z), lift(boundary[b], z), lift(boundary[c], z))
                deleteat!(ids, at)
                area -= ear
                found = true
                break
            end
            found || error("could not triangulate the conforming printed cap without slivers")
        end
        triangle!((lift(boundary[i], z) for i in ids)...)
    end
    for (z, cc) in caps, (triangle, upward) in cc
        boundary = Point2d[]
        for k in 1:3
            append!(boundary, edge_points(triangle[k], triangle[mod1(k + 1, 3)], z)[1:(end - 1)])
        end
        unique!(boundary)
        # Anchoring avoids cancellation for tiny seam triangles far from (0, 0).
        origin = boundary[1]
        area = sum((boundary[k][1] - origin[1]) * (boundary[mod1(k + 1, length(boundary))][2] - origin[2]) -
            (boundary[k][2] - origin[2]) * (boundary[mod1(k + 1, length(boundary))][1] - origin[1]) for k in eachindex(boundary))
        (area > 0) == upward || reverse!(boundary)
        if length(boundary) == 3
            triangle!((lift(p, z) for p in boundary)...)
        else
            triangulate_boundary!(boundary, z, upward)
        end
    end
    for (l, rr) in zip(layers, rings), ring in rr, k in eachindex(ring)
        a, b = ring[k], ring[mod1(k + 1, length(ring))]
        a == b && continue
        bottom, top = edge_points(a, b, l.z0), edge_points(a, b, l.z1)
        # Zip the two subdivided edges without adding a coplanar center node.
        # Such centers made the volume mesher create nearly zero-height tets.
        i, j = 1, 1
        d = b - a
        parameter(p) = dot(p - a, d)
        while i < length(bottom) || j < length(top)
            if i < length(bottom) && (j == length(top) || parameter(bottom[i + 1]) <= parameter(top[j + 1]))
                triangle!(lift(bottom[i], l.z0), lift(bottom[i + 1], l.z0), lift(top[j], l.z1))
                i += 1
            else
                triangle!(lift(bottom[i], l.z0), lift(top[j + 1], l.z1), lift(top[j], l.z1))
                j += 1
            end
        end
    end
    edges = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    for face in triangles, k in 1:3
        a, b = convert(Int, face[k]), convert(Int, face[mod1(k + 1, 3)])
        count, orientation = get(edges, minmax(a, b), (0, 0))
        edges[minmax(a, b)] = count + 1, orientation + (a < b ? 1 : -1)
    end
    bad = filter(pair -> last(pair) != (2, 0), collect(edges))
    isempty(bad) || error("$(part.item.name): $(length(bad)) unmatched boundary edges: " *
        repr([(positions[a], positions[b], counts) for ((a, b), counts) in first(bad, min(8, length(bad)))]))
    transformed = [Point3d(unit * assembled(part, Vec3d(p))) for p in positions]
    return GeometryBasics.Mesh(transformed, triangles)
end
