# Parts as solids in Gmsh's OpenCASCADE kernel, and the meshes made from them. All in millimeters.

"""
    Item(name, material, color, build; down, size, settings, group)

One item of an assembly. `build(loaded)` adds its solids to the current Gmsh model in the
assembled position; springs differ between `loaded` (as assembled) and unloaded (as printed).
`material` is `:PLA`, `:PETG`, `:steel` (bought steel parts: shafts, bearings, fasteners), `:bought` or `:household`. `down` is the direction (assembly
coordinates) that faces the print bed, `size` the mesh size (mm) for display, `settings` the
slicer settings it needs (`"key" => "value"`), and `group` what it belongs to (exports and
checks pick items by group).
"""
struct Item{F}
    name::String
    material::Symbol
    color::Symbol
    build::F
    down::Vec3d
    size::Float64
    settings::Vector{Pair{String, String}}
    group::Symbol
end

Item(name, material, color, build; down = Vec3d(-1, 0, 0), size = 2.0, settings = Pair{String, String}[], group = :frame) =
    Item(name, material, color, build, down, size, settings, group)

"Whether item `i` is printed."
printed(i::Item) = i.material in (:PLA, :PETG)

"""
    parts(items; dir = nothing)

Every item as a display mesh in its assembled place. With `dir`, each printed item is also
written (in assembly coordinates, springs unloaded) as STEP into `dir/step`, for editing in
CAD, and as binary STL into `dir/stl`, which GitHub shows in 3D.
"""
function parts(list::AbstractVector{<:Item}; dir = nothing)
    dir === nothing || foreach(sub -> mkpath(joinpath(dir, sub)), ("step", "stl"))
    with_gmsh() do
        gmsh.option.setNumber("Mesh.Binary", 1)
        map(list) do it
            m = solid(() -> it.build(true), it.name; size = it.size)
            if dir !== nothing && printed(it)
                generate_mesh(() -> it.build(false), it.name; size = it.size)
                gmsh.write(joinpath(dir, "step", it.name * ".step"))
                gmsh.write(joinpath(dir, "stl", it.name * ".stl"))
                gmsh.model.remove()
            end
            Part(it.name, m, it.color, it.material)
        end
    end
end

"One part of the assembly: its surface mesh, display color and material (`:PLA`, `:PETG`, `:bought` or `:household`)."
struct Part
    name::String
    mesh::GeometryBasics.Mesh
    color::RGBAf
    material::Symbol
end

Part(name, mesh, color::Union{Symbol, Tuple}, material::Symbol) = Part(name, mesh, Makie.to_color(color), material)

"Runs `f` in an initialized Gmsh session and finalizes the session afterwards."
function with_gmsh(f)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)
    try
        return f()
    finally
        gmsh.finalize()
    end
end

"""
Surface mesh of the current Gmsh model, or of its `surfaces`. Every CAD face gets its own
vertices, so normals are smooth within a face and sharp along its edges.
"""
function surface_mesh(surfaces = gmsh.model.getEntities(2))
    positions = Point3f[]
    faces = GLTriangleFace[]
    # Coordinates from the global node list: a surface's own node list can miss nodes it shares
    # with degenerate edges (a torus seam, a wedge's apex).
    all_tags, coords, _ = gmsh.model.mesh.getNodes()
    node = Dict(t => i for (i, t) in enumerate(all_tags))
    for (_, tag) in surfaces
        types, _, element_nodes = gmsh.model.mesh.getElements(2, tag)
        k = findfirst(==(2), types)
        k === nothing && continue
        tri = element_nodes[k]
        local_tags = unique(tri)
        offset = length(positions)
        index = Dict(t => offset + i for (i, t) in enumerate(local_tags))
        append!(positions, [Point3f(coords[3node[t] - 2], coords[3node[t] - 1], coords[3node[t]]) for t in local_tags])
        append!(faces, [GLTriangleFace(index[tri[3j - 2]], index[tri[3j - 1]], index[tri[3j]]) for j in 1:(length(tri) ÷ 3)])
    end
    return GeometryBasics.mesh(positions, faces; normal = face_normals_per_vertex(positions, faces))
end

"Area-weighted vertex normals; vertices are not shared across CAD faces, so edges stay sharp."
function face_normals_per_vertex(positions, faces)
    normals = zeros(Vec3f, length(positions))
    for f in faces
        a, b, c = positions[f[1]], positions[f[2]], positions[f[3]]
        n = cross(b - a, c - a)
        for i in f
            normals[i] += n
        end
    end
    return normalize.(normals)
end

"""
Watertight surface mesh of the current Gmsh model, for printing: CAD faces share the vertices
along their edges, and every triangle faces out of the solid.
"""
function closed_mesh()
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    triangles = NTuple{3, UInt64}[]
    for (_, volume) in gmsh.model.getEntities(3)
        gmsh.model.mesh.setOutwardOrientation(volume)
        for (_, surface) in gmsh.model.getBoundary([(3, volume)], false, false, false)
            types, _, element_nodes = gmsh.model.mesh.getElements(2, surface)
            tri = element_nodes[findfirst(==(2), types)]
            append!(triangles, [(tri[3j - 2], tri[3j - 1], tri[3j]) for j in 1:(length(tri) ÷ 3)])
        end
    end
    # Only the nodes the triangles use, numbered from 1.
    used = sort!(unique!(reduce(vcat, [collect(t) for t in triangles])))
    index = Dict(tag => i for (i, tag) in enumerate(used))
    node = Dict(tag => i for (i, tag) in enumerate(node_tags))
    positions = [Point3d(coords[3node[t] - 2], coords[3node[t] - 1], coords[3node[t]]) for t in used]
    return GeometryBasics.Mesh(positions, [GLTriangleFace(index[a], index[b], index[c]) for (a, b, c) in triangles])
end

"""
    generate_mesh(build, name; size = 3.0, curvature = 24)

Builds one part in a fresh Gmsh model with `build()` and meshes its surface with elements at
most `size` long and `curvature` elements per full circle.
"""
function generate_mesh(build, name; size = 3.0, curvature = 24)
    gmsh.model.add(name)
    build()
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
    gmsh.model.mesh.generate(2)
    return
end

"""
    solid(build, name; size = 3.0, dir = nothing)

Builds one part in a fresh Gmsh model with `build()`, meshes its surface and returns the mesh
for display. With `dir`, also writes `name.stl` and `name.step` there.
"""
function solid(build, name; size = 3.0, dir = nothing)
    generate_mesh(build, name; size)
    if dir !== nothing
        gmsh.write(joinpath(dir, name * ".stl"))
        gmsh.write(joinpath(dir, name * ".step"))
    end
    m = surface_mesh()
    gmsh.model.remove()
    return m
end

# --- profile to solids ---------------------------------------------------------------------

occ() = gmsh.model.occ

"Point tags for profile points at `x`, reusing a tag for points that coincide."
struct PointCache
    x::Float64
    tags::Dict{NTuple{2, Int}, Int32}
end
PointCache(x) = PointCache(x, Dict{NTuple{2, Int}, Int32}())

function Base.getindex(c::PointCache, p::Point2d)
    key = (round(Int, p[1] * 1.0e4), round(Int, p[2] * 1.0e4))
    return get!(() -> occ().addPoint(c.x, p[1], p[2]), c.tags, key)
end

"Extrudes a closed 2D curve loop (given as curve tags) across x from `x0` by `width`."
function extrude_loop(curves, width)
    loop = occ().addCurveLoop(curves)
    surface = occ().addPlaneSurface([loop])
    out = occ().extrude([(2, surface)], width, 0, 0)
    return [dt for dt in out if dt[1] == 3]
end

"A polygon in the profile plane, extruded across x."
function prism(outline::Vector{Point2d}, x0, width)
    pts = PointCache(x0)
    n = length(outline)
    curves = [occ().addLine(pts[outline[i]], pts[outline[mod1(i + 1, n)]]) for i in 1:n]
    return extrude_loop(curves, width)
end

cylinder_x(c::Point2d, r, x0, width) = [(3, occ().addCylinder(x0, c[1], c[2], width, 0, 0, r))]
box3(lo, size) = [(3, occ().addBox(lo..., size...))]

cut!(object, tools) = first(occ().cut(object, tools))
fuse!(object, tools) = first(occ().fuse(object, tools))


"Convex hull of 2D points (Andrew's monotone chain), counterclockwise."
function convex_hull(points::Vector{Point2d})
    pts = sort(unique(points))
    length(pts) < 3 && return pts
    turn(o, a, b) = cross(a - o, b - o)
    lower, upper = Point2d[], Point2d[]
    for q in pts
        while length(lower) >= 2 && turn(lower[end - 1], lower[end], q) <= 0
            pop!(lower)
        end
        push!(lower, q)
    end
    for q in reverse(pts)
        while length(upper) >= 2 && turn(upper[end - 1], upper[end], q) <= 0
            pop!(upper)
        end
        push!(upper, q)
    end
    return vcat(lower[1:(end - 1)], upper[1:(end - 1)])
end


"The solids `tags`, fused into one where they overlap: a closed surface of their union for `Traced`."
fuse_all!(tags) = length(tags) <= 1 ? tags : first(occ().fuse(tags[1:1], tags[2:end]))
