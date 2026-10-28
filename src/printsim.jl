# The parts as they come off the printer. Bambu Studio slices the same project that gets
# printed; every extrusion in its G-code lays a bead, and the beads of one layer of one part,
# united, are what that layer of the part is made of. LibGEOS does the 2D geometry. Cooling
# shrinks each part toward its middle by a fraction per material, a range that the tolerance
# coupon narrows down after its first print.

const GEOS = LibGEOS

"""
One extrusion path in the G-code: plate points (mm) laid at layer top `z`, layer height `h`,
bead width `w`. `outer` marks outer walls, the beads that make a part's visible faces.
"""
struct BeadPath
    points::Vector{Vec2d}
    z::Float64
    h::Float64
    w::Float64
    outer::Bool
end

"G-code word `letter` of a command line, or `nothing`."
function word(cmd::AbstractString, letter::Char)
    for token in eachsplit(cmd)
        first(token) == letter && return parse(Float64, SubString(token, 2))
    end
    return nothing
end

"""
    scan_beads(lay, path; filament = 1.75, step = 0.3)

Every extrusion of a Bambu Studio G-code file, handed piece by piece to
`lay(object, block, feature, a, b, z, h, w)`: the unique label id Bambu Studio writes around
each object's moves (`100 + k` for the project's k-th object), a counter that grows at every
new layer, feature and object, the feature's name, the piece from `a` to `b` (plate mm), the
layer top `z`, layer height `h` and bead width `w`. Arcs (G2/G3) become straight pieces at
most `step` long. A bead's width follows from the filament it lays, with Slic3r's bead of a
rectangle with round sides: w = volume / (length · h) + h (1 - π/4). Brims, skirts and
supports are left out: they come off after printing.
"""
function scan_beads(lay, path; filament = 1.75, step = 0.3)
    area = π * filament^2 / 4
    x, y, z, h = 0.0, 0.0, 0.0, 0.2
    relative, e_last = true, 0.0
    object, feature, removed, block = 0, "", false, 0
    function piece(a::Vec2d, b::Vec2d, de)
        L = norm(b - a)
        (object == 0 || removed || de <= 0 || L < 1.0e-6) && return
        lay(object, block, feature, a, b, z, h, de * area / (L * h) + h * (1 - π / 4))
        return
    end
    for line in eachline(path)
        isempty(line) && continue
        if first(line) == ';'
            if startswith(line, "; Z_HEIGHT:")
                block += 1
                z = parse(Float64, strip(line[12:end]))
            elseif startswith(line, "; LAYER_HEIGHT:")
                h = parse(Float64, strip(line[16:end]))
            elseif startswith(line, "; FEATURE:")
                block += 1
                feature = String(strip(line[11:end]))
                removed = feature in ("Brim", "Skirt", "Prime tower") || startswith(feature, "Support")
            elseif startswith(line, "; start printing object, unique label id:") || startswith(line, "; OBJECT_ID:")
                # A plate with a single object has no start and stop marks, only the object's id per layer.
                block += 1
                object = parse(Int, strip(split(line, ':')[end]))
            elseif startswith(line, "; stop printing object")
                block += 1
                object = 0
            end
            continue
        end
        k = findfirst(';', line)
        cmd = strip(k === nothing ? line : line[1:(k - 1)])
        isempty(cmd) && continue
        g = first(eachsplit(cmd))
        if g == "M83"
            relative = true
        elseif g == "M82"
            relative = false
        elseif g == "G92"
            e = word(cmd, 'E')
            e === nothing || (e_last = e)
        elseif g in ("G0", "G1", "G2", "G3")
            nx, ny, e = word(cmd, 'X'), word(cmd, 'Y'), word(cmd, 'E')
            to = Vec2d(nx === nothing ? x : nx, ny === nothing ? y : ny)
            de = e === nothing ? 0.0 : relative ? e : e - e_last
            e === nothing || relative || (e_last = e)
            from = Vec2d(x, y)
            if g in ("G2", "G3") && (word(cmd, 'I') !== nothing || word(cmd, 'J') !== nothing)
                i, j = something(word(cmd, 'I'), 0.0), something(word(cmd, 'J'), 0.0)
                c = from + Vec2d(i, j)
                r = norm(from - c)
                a0 = atan(from[2] - c[2], from[1] - c[1])
                a1 = atan(to[2] - c[2], to[1] - c[1])
                sweep = g == "G3" ? mod(a1 - a0, 2π) : -mod(a0 - a1, 2π)
                sweep == 0 && (sweep = g == "G3" ? 2π : -2π)
                n = max(1, ceil(Int, abs(sweep) * r / step))
                prev = from
                for m in 1:n
                    φ = a0 + sweep * m / n
                    p = m == n ? to : c + r * Vec2d(cos(φ), sin(φ))
                    piece(prev, p, de / n)
                    prev = p
                end
            else
                piece(from, to, de)
            end
            x, y = to
        end
    end
    return nothing
end

"""
    read_beads(path; filament = 1.75, step = 0.3)

Every extrusion of a Bambu Studio G-code file ([`scan_beads`](@ref)) as bead paths, by object
label id. Consecutive pieces of about the same width in one feature join into one path.
"""
function read_beads(path; filament = 1.75, step = 0.3)
    out = Dict{Int, Vector{BeadPath}}()
    current, owner, last_block = nothing, 0, -1
    function finish!()
        current === nothing && return
        length(current.points) >= 2 && push!(get!(out, owner, BeadPath[]), current)
        current = nothing
        return
    end
    scan_beads(path; filament, step) do object, block, feature, a, b, z, h, w
        block == last_block || finish!()
        last_block = block
        outer = feature == "Outer wall"
        if current !== nothing && current.points[end] == a && abs(current.w - w) < 0.03 && current.z == z && current.outer == outer
            push!(current.points, b)
        else
            finish!()
            current = BeadPath([a, b], z, h, w, outer)
            owner = object
        end
    end
    finish!()
    return out
end

"""
One layer of a printed part between `z0` and `z1` (plate coordinates, as sliced): the region
its beads cover, and a prepared copy for fast point tests.
"""
struct PrintedLayer
    z0::Float64
    z1::Float64
    region::GEOS.AbstractGeometry
    prepared::GEOS.PreparedGeometry
end

polygon_coords(ring::Vector{Vec2d}) = [[q[1], q[2]] for q in vcat(ring, ring[1:1])]

"""
    layer_region(paths)

The region one layer's beads cover: each path buffered by half its width with round ends, all
united. Gaps inside a part, between sparse infill lines, count as filled: only holes bounded by
an outer wall are real holes of the part.
"""
function layer_region(paths::Vector{BeadPath})
    pieces = GEOS.Polygon[]
    for p in paths
        line = GEOS.LineString([[q[1], q[2]] for q in p.points])
        append!(pieces, polygons(GEOS.buffer(line, p.w / 2, 6)))
    end
    all = GEOS.unaryUnion(GEOS.MultiPolygon(pieces))
    walls = [GEOS.LineString([[q[1], q[2]] for q in p.points]) for p in paths if p.outer]
    isempty(walls) && return all
    loops = outer_wall_loops(paths)
    filled = [GEOS.Polygon(GEOS.exteriorRing(poly)) for poly in polygons(all)]
    solid = length(filled) == 1 ? only(filled) : GEOS.unaryUnion(GEOS.MultiPolygon(filled))
    # Work on the global void, not each disconnected bead ring separately: filling
    # sparse infill must not overwrite a real bore bounded by another bead ring.
    void = GEOS.difference(solid, all)
    holes = GEOS.intersection(void, outer_wall_holes(loops;exterior=solid,
        margin=maximum(p.w for p in paths)/2))
    return GEOS.difference(solid, holes)
end

"Closed outer-perimeter centerlines, including the slicer's short seam gap."
function outer_wall_loops(paths)
    walls = [p for p in paths if p.outer]
    isempty(walls) && return GEOS.Polygon[]
    lines = GEOS.lineMerge(GEOS.MultiLineString([GEOS.LineString([[q[1], q[2]] for q in p.points]) for p in walls]))
    chains = lines isa GEOS.LineString ? [lines] : [GEOS.getGeometry(lines, i) for i in 1:GEOS.numGeometries(lines)]
    loops = GEOS.Polygon[]
    seam = maximum(p.w for p in walls) / 2
    for line in chains
        points = Vec2d[Vec2d(q...) for q in GEOS.GeoInterface.coordinates(line)]
        length(points) >= 3 && norm(first(points) - last(points)) <= seam || continue
        polygon = GEOS.Polygon(GEOS.LinearRing(polygon_coords(points)))
        GEOS.isValid(polygon) && GEOS.area(polygon) > 0 && push!(loops, polygon)
    end
    return loops
end

"Hole domains from nested perimeter loops; material islands inside holes stay material."
function outer_wall_holes(loops;exterior=nothing,margin=0.0)
    depths=[count(other->GEOS.covers(other,loop),loops) for loop in loops]
    # Bambu may emit an exterior overhang as Bridge rather than Outer wall.
    # A closed bore wall then has no tagged exterior ancestor. The deposited
    # union's exterior still supplies that ancestor: a root strictly farther
    # from that boundary than any bead's half-width cannot be its perimeter.
    # Restore the missing nesting level for that root and its material islands.
    interior_roots=exterior===nothing ? eltype(loops)[] :
        [loop for (loop,depth) in zip(loops,depths) if depth==1 &&
            GEOS.covers(exterior,loop) &&
            GEOS.distance(GEOS.boundary(loop),GEOS.boundary(exterior))>margin+1e-8]
    holes = GEOS.readgeom("POLYGON EMPTY")
    for loop in sort(loops; by = GEOS.area, rev = true)
        depth = count(other -> GEOS.covers(other, loop), loops)
        depth += count(root->GEOS.covers(root,loop),interior_roots)
        if iseven(depth)
            holes = GEOS.union(holes, loop)
        elseif depth > 1
            holes = GEOS.difference(holes, loop)
        end
    end
    return holes
end

"The polygons of a (multi)polygon."
polygons(g::GEOS.Polygon) = [g]
polygons(g::GEOS.MultiPolygon) = [GEOS.getGeometry(g, i) for i in 1:GEOS.numGeometries(g)]
polygons(g::GEOS.GeometryCollection) = reduce(vcat, [polygons(GEOS.getGeometry(g, i)) for i in 1:GEOS.numGeometries(g)]; init = GEOS.Polygon[])
polygons(::Union{GEOS.LineString, GEOS.MultiLineString, GEOS.Point, GEOS.MultiPoint}) = GEOS.Polygon[]

"""
A printed part: its layers as sliced, in plate coordinates, and how they sit in the assembly.
A point `p` of the assembly is at `rotation * p + offset` on the plate before cooling; cooling
shrinks the part toward `center` by `shrink` across the layers (and not along z, where each
new layer lands on one that has cooled already). `swell` is how far every layer's edges stand
out in its plane beyond the beads' planned width (squeezed plastic bulging out): holes come
out smaller, pegs bigger; the layers' heights stay (`reprint` makes the swollen layers).
"""
struct PrintedPart
    item::Item
    layers::Vector{PrintedLayer}
    rotation::Mat3d
    offset::Vec3d
    center::Vec3d
    shrink::Float64
    swell::Float64
end

"""
    PrintedPart(object, placement, paths; shrink)

The part `object` laid out at `placement` (as in the project Bambu Studio sliced), built from
its G-code bead paths.
"""
function PrintedPart(o::PrintObject, p::Placement, paths::Vector{BeadPath}; shrink, swell = 0.0)
    T = turn_z(p.turned ? π / 2 : 0.0)
    c = Vec3d(p.center[1], p.center[2], 0.0)
    rotation = T * o.rotation
    offset = -T * o.center + Vec3d(p.center[1], p.center[2], o.size[3] / 2)
    layers = PrintedLayer[]
    zs = sort!(unique(q.z for q in paths))
    for z in zs
        here = filter(q -> q.z == z, paths)
        region = layer_region(here)
        # Combined infill can be extruded 0.4 mm high inside a 0.2 mm wall layer.
        # It does not make the entire outer contour twice as thick. The boundary
        # layer follows the walls; without any wall, retain the thinnest path height.
        walls = filter(q -> q.outer, here)
        h = isempty(walls) ? minimum(q -> q.h, here) : maximum(q -> q.h, walls)
        # G-code heights are decimal. Reuse their precision after subtraction so
        # adjacent 0.2 mm slabs share one plane exactly, rather than overlap by an
        # ulp and leave zero-thickness Boolean facets at their interface.
        push!(layers, PrintedLayer(round(z - h; digits = 8), round(z; digits = 8), region, GEOS.prepareGeom(region)))
    end
    return PrintedPart(o.item, layers, rotation, offset, c, shrink, swell)
end

"Plate point as sliced of the assembly point `q` on the cooled part."
function sliced(part::PrintedPart, q::Vec3d)
    p = part.rotation * q + part.offset
    s = 1 - part.shrink
    return Vec3d(part.center[1] + (p[1] - part.center[1]) / s, part.center[2] + (p[2] - part.center[2]) / s, p[3])
end

"Assembly point of a plate point as sliced, after cooling."
function assembled(part::PrintedPart, p::Vec3d)
    s = 1 - part.shrink
    cooled = Vec3d(part.center[1] + (p[1] - part.center[1]) * s, part.center[2] + (p[2] - part.center[2]) * s, p[3])
    return part.rotation' * (cooled - part.offset)
end

"""
The layer at height `z` (as sliced), or nothing above or below the part. Slicers round layer
heights, so a height on a boundary, or within a hair of one, goes to the nearest layer.
"""
function layer_at(part::PrintedPart, z; hair = 1.0e-3)
    ls = part.layers
    (z < ls[1].z0 - hair || z > ls[end].z1 + hair) && return nothing
    k = searchsortedfirst(ls, z; by = l -> l isa PrintedLayer ? l.z1 : l)
    k > length(ls) && return ls[end]
    (ls[k].z0 <= z || k == 1) && return ls[k]
    return z - ls[k - 1].z1 < ls[k].z0 - z ? ls[k - 1] : ls[k]
end

"Whether the assembly point `q` lies in the printed part's material."
function Base.in(q::Vec3d, part::PrintedPart)
    p = sliced(part, q)
    l = layer_at(part, p[3])
    l === nothing && return false
    return GEOS.prepcontains(l.prepared, GEOS.Point(p[1], p[2]))
end

"Whether the part's layers lie across the assembly's x axis (it printed standing on an x face)."
layers_across_x(part::PrintedPart) = abs(part.rotation[3, 1]) > 0.99

"""
    section(part, x)

The printed part cut by the assembly plane at `x`, as a LibGEOS region in the profile plane
(y, z). A part that printed standing on an x face gives one layer's region; a part that
printed lying gives a stack of strips, one per layer, which shows its stair steps.
"""
function section(part::PrintedPart, x)
    R = part.rotation
    to_profile(p::Vec3d) = (q = assembled(part, p); [q[2], q[3]])
    if layers_across_x(part)
        z = R[3, 1] * x + part.offset[3]
        l = layer_at(part, z)
        l === nothing && return GEOS.Polygon([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
        return map_region(l.region, v -> to_profile(Vec3d(v[1], v[2], z)))
    end
    abs(R[3, 1]) < 1.0e-6 || error("$(part.item.name) printed tilted against x; only x-aligned layers are supported")
    # The plane is upright on the plate: points whose assembly x is `x`.
    e = Vec2d(R[1, 1], R[2, 1])
    strips = GEOS.Polygon[]
    for l in part.layers
        q0 = sliced(part, Vec3d(x, 0, 0))
        k = dot(e, Vec2d(q0[1], q0[2]))
        a = e * k
        t = Vec2d(-e[2], e[1])
        line = GEOS.LineString([collect(a - 2000t), collect(a + 2000t)])
        cut = GEOS.intersection(l.region, line)
        GEOS.isEmpty(cut) && continue
        for seg in segments(cut)
            p1, p2 = seg
            corners = [to_profile(Vec3d(p1[1], p1[2], l.z0)), to_profile(Vec3d(p2[1], p2[2], l.z0)),
                to_profile(Vec3d(p2[1], p2[2], l.z1)), to_profile(Vec3d(p1[1], p1[2], l.z1))]
            # A bead's side is round: full width only at mid-height, half a layer less at its top and bottom.
            r = (l.z1 - l.z0) / 2 - 1.0e-3
            rect = GEOS.Polygon([vcat(corners, corners[1:1])])
            append!(strips, polygons(GEOS.buffer(GEOS.buffer(rect, -r, 8), r, 8)))
        end
    end
    isempty(strips) && return GEOS.Polygon([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    return GEOS.unaryUnion(GEOS.MultiPolygon(strips))
end


"Endpoint pairs of a line or lines."
segments(g::GEOS.LineString) = (c = GEOS.GeoInterface.coordinates(g); [(c[1], c[end])])
segments(g::GEOS.MultiLineString) = reduce(vcat, [segments(GEOS.getGeometry(g, i)) for i in 1:GEOS.numGeometries(g)])
segments(g::GEOS.GeometryCollection) = reduce(vcat, [segments(GEOS.getGeometry(g, i)) for i in 1:GEOS.numGeometries(g)
    if GEOS.getGeometry(g, i) isa Union{GEOS.LineString, GEOS.MultiLineString}]; init = Tuple{Vector{Float64}, Vector{Float64}}[])
segments(::Union{GEOS.Point, GEOS.MultiPoint}) = Tuple{Vector{Float64}, Vector{Float64}}[]

"A region with every vertex mapped by `f` (which keeps orientation or flips it; rings are rebuilt)."
function map_region(g, f)
    polys = map(polygons(g)) do poly
        ext = [f(v) for v in GEOS.GeoInterface.coordinates(GEOS.exteriorRing(poly))]
        holes = [[f(v) for v in GEOS.GeoInterface.coordinates(r)] for r in GEOS.interiorRings(poly)]
        GEOS.Polygon(vcat([ext], holes))
    end
    isempty(polys) && return GEOS.Polygon([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    return length(polys) == 1 ? only(polys) : GEOS.MultiPolygon(polys)
end

"""
    printed_parts(objects, placements, dir; shrink)

Every printed part of the project `objects` laid out at `placements`, rebuilt from Bambu
Studio's G-code for it in `dir` (`plate_1.gcode`, ...). `shrink` gives the cooling shrinkage
per material across the layers.
"""
function printed_parts(objects::Vector{<:PrintObject}, placements::Vector{Placement}, dir;
        shrink = Dict(:PLA => 0.003, :PETG => 0.004))
    parts = Dict{String, PrintedPart}()
    for plate in 1:maximum(p -> p.plate, placements)
        beads = read_beads(joinpath(dir, "plate_$plate.gcode"))
        for (label, paths) in beads
            k = label - 100
            o, p = objects[k], placements[k]
            p.plate == plate || error("$(o.item.name) is placed on plate $(p.plate) but was sliced on plate $plate")
            parts[o.item.name] = PrintedPart(o, p, paths; shrink = shrink[o.item.material])
        end
    end
    return parts
end

"""
    reprint(part; shrink, swell)

The same printed part with a different `shrink` or `swell`. Swell grows every layer in its own
plane: squeezed plastic bulges sideways, layer heights stay. It starts from the part as sliced
(no swell), so parts of one swell are best made once and then reprinted with other shrinks.
"""
function reprint(p::PrintedPart; shrink = p.shrink, swell = p.swell)
    swell == p.swell && return PrintedPart(p.item, p.layers, p.rotation, p.offset, p.center, shrink, swell)
    p.swell == 0 || error("$(p.item.name): reprint swell from the part as sliced, not from one swollen by $(p.swell)")
    layers = map(p.layers) do l
        r = GEOS.buffer(l.region, swell, 8)
        PrintedLayer(l.z0, l.z1, r, GEOS.prepareGeom(r))
    end
    return PrintedPart(p.item, layers, p.rotation, p.offset, p.center, shrink, swell)
end

"""
    drift(part, q)

Where cooling moved the part's material that sat at the assembly point `q` before: the shrunk
part's point minus `q`. Shafts follow the bores they sit in, so fits are compared after taking
out each part's drift; distances within one part, such as gear centers in a plate, keep it.
"""
drift(part::PrintedPart, q::Vec3d) = assembled(part, part.rotation * q + part.offset) - q

"Whether the part printed lying flat in the assembly: its layers horizontal there."
layers_across_z(part::PrintedPart) = abs(part.rotation[3, 3]) > 0.99

"""
    plan(part, z)

A part that printed flat, cut by the horizontal assembly plane at `z`: a LibGEOS region in the
assembly's (x, y).
"""
function plan(part::PrintedPart, z)
    layers_across_z(part) || error("$(part.item.name) did not print flat; plan sections need horizontal layers")
    zp = part.rotation[3, 3] * z + part.offset[3]
    l = layer_at(part, zp)
    l === nothing && return GEOS.Polygon([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    return map_region(l.region, v -> (q = assembled(part, Vec3d(v[1], v[2], zp)); [q[1], q[2]]))
end

"""
    z_extent(part, x, y)

Lowest and highest assembly z of a flat-printed part's material on the vertical line through
(x, y), from the layers that cover it; nothing where there is none.
"""
function z_extent(part::PrintedPart, x, y)
    layers_across_z(part) || error("$(part.item.name) did not print flat")
    p = sliced(part, Vec3d(x, y, 0.0))
    pt = GEOS.Point(p[1], p[2])
    R33, o3 = part.rotation[3, 3], part.offset[3]
    zs = [(z - o3) / R33 for l in part.layers if GEOS.prepcontains(l.prepared, pt) for z in (l.z0, l.z1)]
    return isempty(zs) ? nothing : extrema(zs)
end

"Extent along assembly x of a printed part (from its layers' bounding boxes)."
function x_extent(part::PrintedPart)
    lo, hi = Inf, -Inf
    for l in part.layers, poly in polygons(l.region)
        for v in GEOS.GeoInterface.coordinates(GEOS.exteriorRing(poly))
            for z in (l.z0, l.z1)
                x = assembled(part, Vec3d(v[1], v[2], z))[1]
                lo, hi = min(lo, x), max(hi, x)
            end
        end
    end
    return lo, hi
end
