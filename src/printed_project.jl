# Recover the actual saved placement, not a new packing that might move a part between
# plates. This reader deliberately accepts PrintSim's own exported project format only.
function project_metadata(block, key)
    found = match(Regex("<metadata key=\"$key\" value=\"([^\"]*)\""), block)
    found === nothing && throw(ArgumentError("project is missing metadata $key"))
    return replace(found.captures[1], "&quot;" => "\"", "&apos;" => "'", "&lt;" => "<", "&gt;" => ">", "&amp;" => "&")
end

"""
    printed_project(items, project, gcode_dir; names=nothing, shrink, bed=Bed())

Read selected parts from the saved project written by `export_3mf`, with the G-codes
produced from that very project. Saved labels, plates, positions and quarter-turns are
used, not a fresh packing. Each selected CAD print mesh is checked against the saved
mesh to reject stale designs or unsupported external project edits.
"""
function printed_project(list, path, dir; names = nothing, bed = Bed(),
        shrink = Dict(:PLA => 0.003, :PETG => 0.004))
    archive = ZipReader(read(path))
    settings = zip_readentry(archive, "Metadata/model_settings.config", String)
    main = zip_readentry(archive, "3D/3dmodel.model", String)
    ids = Dict{String, Int}()
    for object in eachmatch(r"<object id=\"(\d+)\">(.*?)</object>"s, settings)
        name = project_metadata(object.captures[2], "name")
        haskey(ids, name) && throw(ArgumentError("duplicate project item $name"))
        ids[name] = parse(Int, object.captures[1])
    end
    plates, labels = Dict{Int, Int}(), Dict{Int, Int}()
    for block in eachmatch(r"<plate>(.*?)</plate>"s, settings)
        plate = parse(Int, project_metadata(block.captures[1], "plater_id"))
        for instance in eachmatch(r"<model_instance>(.*?)</model_instance>"s, block.captures[1])
            id = parse(Int, project_metadata(instance.captures[1], "object_id"))
            plates[id], labels[id] = plate, parse(Int, project_metadata(instance.captures[1], "identify_id"))
        end
    end
    transforms = Dict{Int, Vector{Float64}}()
    for item in eachmatch(r"<item objectid=\"(\d+)\"[^>]*transform=\"([^\"]+)\"", main)
        transforms[parse(Int, item.captures[1])] = parse.(Float64, split(item.captures[2]))
    end
    selected = filter(it -> printed(it) && (names === nothing || it.name in names), list)
    names === nothing || Set(it.name for it in selected) == Set(names) || throw(ArgumentError("unknown requested printed item"))
    objects = with_gmsh() do
        [PrintObject(it) for it in selected]
    end
    nplates = maximum(values(plates))
    beads = Dict{Int, Dict{Int, Vector{BeadPath}}}()
    out = Dict{String, PrintedPart}()
    for object in objects
        name = object.item.name
        id = get(ids, name, 0)
        id != 0 && iseven(id) || throw(ArgumentError("$name is not a PrintSim project object"))
        xml = zip_readentry(archive, "3D/Objects/object_$(id ÷ 2).model", String)
        saved = Point3d[Point3d(parse.(Float64, m.captures)...) for m in
            eachmatch(r"<vertex x=\"([^\"]+)\" y=\"([^\"]+)\" z=\"([^\"]+)\"", xml)]
        points = GeometryBasics.coordinates(object.mesh)
        length(saved) == length(points) || throw(ArgumentError("$name CAD no longer matches the sliced project"))
        errors = [maximum(norm(turn_z(angle) * point - q) for (point, q) in zip(points, saved)) for angle in (0.0, π / 2)]
        minimum(errors) < 2e-5 || throw(ArgumentError("$name CAD/print orientation differs from the sliced project (error $(minimum(errors)) mm)"))
        turned = argmin(errors) == 2
        transform = transforms[id]
        length(transform) == 12 && transform[1:9] == [1, 0, 0, 0, 1, 0, 0, 0, 1] ||
            throw(ArgumentError("external build transforms are not supported"))
        plate = plates[id]
        xy = Point2d(transform[10], transform[11]) - plate_origin(bed, plate, nplates)
        placement = Placement(plate, Point2d(xy), turned)
        paths = get!(beads, plate) do
            read_beads(joinpath(dir, "plate_$plate.gcode"))
        end
        haskey(paths, labels[id]) || throw(ArgumentError("$name has no extrusions on its saved plate"))
        out[name] = PrintedPart(object, placement, paths[labels[id]]; shrink = shrink[object.item.material])
    end
    return out
end
