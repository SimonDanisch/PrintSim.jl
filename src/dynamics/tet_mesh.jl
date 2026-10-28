"The exact-envelope fTetWild CLI: `PRINTSIM_FTETWILD` if set, else the one built into PrintSim's deps."
ftetwild_executable() = get(ENV, "PRINTSIM_FTETWILD", joinpath(dirname(pathof(@__MODULE__)), "..", "deps", "usr", "bin", "FloatTetwild_bin"))

"""
    tetrahedralize(mesh; size=1e-3)

Fill an outward closed surface mesh (meters) with linear tetrahedra, preserving its
existing boundary triangles. This host preparation uses a discrete Gmsh volume;
`add_deformable!` upgrades its elements to quadratic tetrahedra for dynamics.
"""
function tetrahedralize(mesh; size = 1e-3, algorithm = 1, verbose = false, envelope = 1e-7,
        executable = ftetwild_executable(), optimize = true, max_iterations = 80,
        extend_boundary = false, coarsen = false)
    size > 0 || throw(ArgumentError("tetrahedral mesh size must be positive"))
    algorithm == :exact && return tetrahedralize_exact(mesh, size, envelope, executable, max_iterations; coarsen)
    # Gmsh's absolute geometry tolerance is expressed in model units. Work in
    # millimeters, like the original printed solids, rather than feeding SI
    # coordinates to a tolerance intended for ordinary-sized CAD models.
    points = [Point3d(p * 1e3) for p in GeometryBasics.coordinates(mesh)]
    faces = GeometryBasics.faces(mesh)
    isempty(points) && throw(ArgumentError("empty surface mesh"))
    return with_gmsh() do
        if verbose
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.option.setNumber("General.Verbosity", 4)
        end
        gmsh.option.setNumber("Geometry.Tolerance", 1e-12)
        gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-14)
        gmsh.model.add("printed_volume")
        surface = gmsh.model.addDiscreteEntity(2)
        gmsh.model.mesh.addNodes(2, surface, collect(1:length(points)), reduce(vcat, collect.(points)))
        connectivity = Int[convert(Int, i) for face in faces for i in face]
        gmsh.model.mesh.addElementsByType(surface, 2, collect(1:length(faces)), connectivity)
        loop = gmsh.model.geo.addSurfaceLoop([surface])
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size * 1e3)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", extend_boundary ? 1 : 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm)
        gmsh.option.setNumber("Mesh.Optimize", optimize ? 1 : 0)
        gmsh.model.mesh.generate(3)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        types, _, elements = gmsh.model.mesh.getElements(3)
        k = findfirst(==(4), types)
        k === nothing && error("Gmsh produced no tetrahedra")
        raw = elements[k]
        used = sort!(unique(raw))
        lookup = Dict(t => i for (i, t) in enumerate(tags))
        renumber = Dict(t => Int32(i) for (i, t) in enumerate(used))
        nodes = [Vec3{Float64}(coords[3lookup[t] - 2], coords[3lookup[t] - 1], coords[3lookup[t]]) * 1e-3 for t in used]
        tets = NTuple{4, Int32}[(renumber[raw[4j - 3]], renumber[raw[4j - 2]], renumber[raw[4j - 1]], renumber[raw[4j]]) for j in 1:(length(raw) ÷ 4)]
        return compact_tetrahedra(nodes, tets)
    end
end

"Run an fTetWild CLI built with FLOAT_TETWILD_WITH_EXACT_ENVELOPE=ON."
function tetrahedralize_exact(mesh, size, envelope, executable, max_iterations = 80; coarsen = false)
    isfile(executable) || throw(ArgumentError("no exact-envelope fTetWild executable at $executable"))
    0 < envelope < size || throw(ArgumentError("the surface envelope must be positive and below the tet size"))
    points, faces = GeometryBasics.coordinates(mesh), GeometryBasics.faces(mesh)
    low, high = reduce((a, b) -> min.(a, b), points), reduce((a, b) -> max.(a, b), points)
    center, scale = (low + high) / 2, norm(high - low)
    # Keep failed candidates for diagnosis; remove the directory only after the
    # output was read successfully.
    dir = mktempdir(; prefix = "printsim-exact-tet-", cleanup = false)
    try
        input, output = joinpath(dir, "surface.obj"), joinpath(dir, "volume.mesh")
        open(input, "w") do io
            for p in points
                x = (p - center) / scale
                println(io, "v ", x[1], ' ', x[2], ' ', x[3])
            end
            for f in faces
                println(io, "f ", convert(Int, f[1]), ' ', convert(Int, f[2]), ' ', convert(Int, f[3]))
            end
        end
        args = String[executable, "-i", input, "-o", output, "-a", string(size / scale), "-e", string(envelope / scale),
            "--manifold-surface", "--use-input-for-wn", "--no-binary", "--level", "2", "--max-threads", "4", "--max-its", string(max_iterations)]
        coarsen && push!(args, "--coarsen")
        process = run(ignorestatus(addenv(Cmd(args), "OMP_NUM_THREADS" => "1", "OPENBLAS_NUM_THREADS" => "1")))
        success(process) || error("exact-envelope mesher exited with code $(process.exitcode); retained input is $(dirname(input))")
        nodes, tets = read_medit_tetrahedra(output)
        result = compact_tetrahedra([Vec3{Float64}(p * scale + center) for p in nodes], tets)
        rm(dir; recursive = true)
        return result
    catch err
        @error "Exact-envelope meshing failed; diagnostic files retained" directory = dir
        rethrow()
    end
end

function read_medit_tetrahedra(path)
    lines = strip.(readlines(path))
    vertices, tetrahedra = findfirst(==("Vertices"), lines), findfirst(==("Tetrahedra"), lines)
    (vertices === nothing || tetrahedra === nothing) && error("missing volume sections in $path")
    n, t = parse(Int, lines[vertices + 1]), parse(Int, lines[tetrahedra + 1])
    n > 0 && t > 0 || error("empty output volume in $path")
    points = [Vec3{Float64}(parse.(Float64, split(lines[vertices + 1 + i])[1:3])) for i in 1:n]
    tets = NTuple{4, Int32}[Tuple(parse.(Int32, split(lines[tetrahedra + 1 + i])[1:4])) for i in 1:t]
    return points, tets
end

"Drop duplicate/zero-volume mesher elements, without deleting any positive-volume cell."
function compact_tetrahedra(points, tetrahedra)
    seen = Set{NTuple{4, Int32}}()
    tets = NTuple{4, Int32}[]
    for t in tetrahedra
        a, b, c, d = t
        determinant = dot(points[b] - points[a], cross(points[c] - points[a], points[d] - points[a]))
        isfinite(determinant) || error("nonfinite tetrahedron from mesher")
        determinant == 0 && continue
        key = Tuple(sort(collect(t)))
        key in seen && continue
        push!(seen, key)
        push!(tets, t)
    end
    isempty(tets) && error("mesher returned no positive volume")
    used = sort!(unique!(reduce(vcat, [collect(t) for t in tets])))
    remap = zeros(Int32, length(points))
    remap[used] .= Int32.(eachindex(used))
    return points[used], [Tuple(remap[i] for i in t) for t in tets]
end

# The host surface index that `simplify_surface` tracks its error against.
function host_surface_index(points, faces)
    boxes = [(reduce((a, b) -> min.(a, b), (points[i] for i in f)),
        reduce((a, b) -> max.(a, b), (points[i] for i in f))) for f in faces]
    centers = [(a + b) / 2 for (a, b) in boxes]
    tree = BVHTables()
    root = add_bvh!(tree, Int32.(eachindex(faces)), boxes, centers)
    return (; points, faces, tree, root)
end

function host_triangle_distance2(p, a, b, c)
    u, v, d = b - a, c - a, p - a
    uu, uv, vv = dot(u, u), dot(u, v), dot(v, v)
    det = uu * vv - uv^2
    if det > 1e-14 * uu * vv
        s = (dot(d, u) * vv - dot(d, v) * uv) / det
        t = (dot(d, v) * uu - dot(d, u) * uv) / det
        s >= 0 && t >= 0 && s + t <= 1 && return sum(abs2, d - s * u - t * v)
    end
    segment(a, b) = begin
        e = b - a
        s = clamp(dot(p - a, e) / max(sum(abs2, e), floatmin(Float64)), 0.0, 1.0)
        sum(abs2, p - a - s * e)
    end
    return min(segment(a, b), segment(b, c), segment(c, a))
end
