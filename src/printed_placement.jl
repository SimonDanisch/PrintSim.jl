# Where printed parts go in the assembly, recovered from the project Bambu Studio sliced: the
# project names each part's plate, G-code label and plate position, and holds its mesh in
# print orientation. That mesh is the CAD mesh turned and centred vertex for vertex, so the
# CAD mesh, meshed again the same way, gives the rigid motion between assembly and plate.

"""
Where a printed part's plate points go in the assembly: a plate point `p` as sliced lies at
`rotation * q + offset` for the assembly point `q` before cooling; cooling shrinks the part
toward `center` by `shrink` in the plate plane (not along z). `label` is the part's object
label in the G-code, `plate` its plate. The same motion as [`PrintedPart`](@ref)'s.
"""
struct PartPlacement
    label::Int
    plate::Int
    rotation::Mat3d
    offset::Vec3d
    center::Vec3d
    shrink::Float64
end

"""
    printed_placements(dev, items, project; bed = Bed(), curvature = 72, shrink, cache = nothing) -> Dict{String, PartPlacement}

The placement of every printed item of `items` in `project`, a project `export_3mf` wrote
with `curvature`: plate, label and plate position from the project's settings, rotation and
translation from its saved mesh. The CAD mesh is meshed again as `PrintObject` does, so its
vertices come in the saved mesh's order, and the rigid motion between the two point sets
follows by Kabsch's method on the device: centroids and cross-covariance by reductions, the
3×3 singular value decomposition on the host. Every saved point must lie within 2 µm of the
moved CAD point. With a `cache` directory the placements are kept under the SHA-256 of the
project, the items' names and the settings: the CAD is meshed again only for a new project.
"""
function printed_placements(dev::Mantle.Device, list, path; bed = Bed(), curvature = 72,
                            shrink = Dict(:PLA => 0.003, :PETG => 0.004), cache::Union{Nothing,AbstractString} = nothing)
    bytes = read(path)
    names = sort([it.name for it in list if printed(it)])
    key = bytes2hex(SHA.sha256(vcat(bytes, Vector{UInt8}(repr((names, bed, curvature, sort!(collect(shrink))))))))
    file = cache === nothing ? nothing : joinpath(cache, "placements-$key.jls")
    file !== nothing && isfile(file) && return Serialization.deserialize(file)
    out = placementsof(dev, list, ZipReader(bytes); bed, curvature, shrink)
    if file !== nothing
        mkpath(cache)
        Serialization.serialize(file, out)
    end
    return out
end

function placementsof(dev::Mantle.Device, list, archive; bed, curvature, shrink)
    settings = zip_readentry(archive, "Metadata/model_settings.config", String)
    main = zip_readentry(archive, "3D/3dmodel.model", String)
    ids = Dict{String,Int}()
    for object in eachmatch(r"<object id=\"(\d+)\">(.*?)</object>"s, settings)
        ids[project_metadata(object.captures[2], "name")] = parse(Int, object.captures[1])
    end
    plates, labels = Dict{Int,Int}(), Dict{Int,Int}()
    for block in eachmatch(r"<plate>(.*?)</plate>"s, settings)
        plate = parse(Int, project_metadata(block.captures[1], "plater_id"))
        for instance in eachmatch(r"<model_instance>(.*?)</model_instance>"s, block.captures[1])
            id = parse(Int, project_metadata(instance.captures[1], "object_id"))
            plates[id], labels[id] = plate, parse(Int, project_metadata(instance.captures[1], "identify_id"))
        end
    end
    transforms = Dict{Int,Vector{Float64}}()
    for item in eachmatch(r"<item objectid=\"(\d+)\"[^>]*transform=\"([^\"]+)\"", main)
        transforms[parse(Int, item.captures[1])] = parse.(Float64, split(item.captures[2]))
    end
    nplates = maximum(values(plates))
    out = Dict{String,PartPlacement}()
    with_gmsh() do
        for it in list
            printed(it) || continue
            id = get(ids, it.name, 0)
            id == 0 && throw(ArgumentError("$(it.name) is not in the project"))
            xml = zip_readentry(archive, "3D/Objects/object_$(id ÷ 2).model", String)
            saved = Float32[parse(Float32, m.captures[k]) for m in eachmatch(r"<vertex x=\"([^\"]+)\" y=\"([^\"]+)\" z=\"([^\"]+)\"", xml)
                            for k in 1:3]
            generate_mesh(() -> it.build(false), it.name; it.size, curvature)
            cad = closed_mesh()
            gmsh.model.remove()
            P = Mantle.Buffer(dev, Float32.(reinterpret(reshape, Float64, coordinates(cad))))
            S = Mantle.Buffer(dev, reshape(saved, 3, :))
            size(P) == size(S) || throw(ArgumentError("$(it.name): CAD no longer matches the sliced project"))
            M, t, err, height = rigidmotion(P, S)
            foreach(Mantle.free!, (P, S))
            err < 2e-3 || throw(ArgumentError("$(it.name): saved mesh $(err) mm off the CAD mesh moved"))
            transform = transforms[id]
            xy = Vec2d(transform[10], transform[11]) - plate_origin(bed, plates[id], nplates)
            out[it.name] = PartPlacement(labels[id], plates[id], M, t + Vec3d(xy[1], xy[2], height / 2),
                                         Vec3d(xy[1], xy[2], 0), shrink[it.material])
        end
    end
    return out
end

"""
`(M, t, err, height)`: the rotation and translation taking the points `P` onto `S` (both
`(3, n)` Float32 on the device, in order) by Kabsch's method, the largest distance left, and
the height of `S`. The sums run on the device, the decomposition on the host.
"""
function rigidmotion(P::Mantle.Buffer, S::Mantle.Buffer)
    p, s = Mantle.storage(P), Mantle.storage(S)
    n = size(p, 2)
    pm, sm = vec(Array(sum(p; dims = 2))) ./ n, vec(Array(sum(s; dims = 2))) ./ n
    H = [Float64(sum((p[i:i, :] .- Float32(pm[i])) .* (s[j:j, :] .- Float32(sm[j])))) for i in 1:3, j in 1:3]
    U, _, V = svd(H)
    d = sign(det(V * U'))
    M = Mat3d(V * Diagonal([1, 1, d]) * U')
    t = Vec3d(sm) - M * Vec3d(pm)
    m, u = Float32.(Matrix(M)), Float32.(t)
    e = (m[1, 1] .* p[1:1, :] .+ m[1, 2] .* p[2:2, :] .+ m[1, 3] .* p[3:3, :] .+ u[1] .- s[1:1, :]) .^ 2 .+
        (m[2, 1] .* p[1:1, :] .+ m[2, 2] .* p[2:2, :] .+ m[2, 3] .* p[3:3, :] .+ u[2] .- s[2:2, :]) .^ 2 .+
        (m[3, 1] .* p[1:1, :] .+ m[3, 2] .* p[2:2, :] .+ m[3, 3] .* p[3:3, :] .+ u[3] .- s[3:3, :]) .^ 2
    d > 0 || throw(ArgumentError("the saved mesh is a mirror image of the CAD mesh"))
    return M, t, sqrt(Float64(maximum(e))), Float64(maximum(s[3:3, :]) - minimum(s[3:3, :]))
end

"""
    assembled!(dev, V, placement; unit = 1e-3) -> V

The plate-frame vertices `V` `(3, n)` Float32 (mm, as sliced) of a part moved into the
assembly in place: cooled toward the placement's centre in the plate plane, then taken back
through its rotation and offset, and scaled by `unit` (metres by default).
"""
function assembled!(dev::Mantle.Device, V::Mantle.Buffer, q::PartPlacement; unit::Real = 1e-3)
    R = Float32.(Matrix(q.rotation))
    g = Mantle.Graph(dev)
    launch!(g, assemble!, (V, size(V, 2), Mat3f(R'), Vec3f(q.offset), Vec2f(q.center[1], q.center[2]), Float32(1 - q.shrink),
                           Float32(unit)), size(V, 2))
    runonce!(g)
    return V
end

function assemble!(V, n, Rt, offset, center, keep, unit)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds p = Vec3f(V[1, i], V[2, i], V[3, i])
    cooled = Vec3f(center[1] + (p[1] - center[1]) * keep, center[2] + (p[2] - center[2]) * keep, p[3])
    q = Rt * (cooled - offset) * unit
    @inbounds V[1, i], V[2, i], V[3, i] = q
    return nothing
end
