# Print files for Bambu Lab printers: every printed part in its print orientation, packed onto
# plates and written as an unsliced Bambu Studio project (3MF), so orientation, placement and
# all slicer settings stay editable before printing.

"""
    Bed(; size, area, gap)

A Bambu Lab print bed, in millimeters. `size` is the whole plate, which also spaces the plates
of a project in Bambu Studio's scene. Parts go inside `area`. The default is the region of the
H2D's plate that both nozzles reach, without the strip at the back used by the nozzle wrapping
check. `gap` is the free space between parts, which also takes the slicer's automatic brim
and tree supports reaching past a part; an explicit brim comes on top.
"""
Base.@kwdef struct Bed
    size::Vec3d = Vec3d(350, 320, 325)
    area::Rect2d = Rect2d(25, 0, 300, 308)
    gap::Float64 = 12.0
end

skew(v::Vec3d) = Mat3d(0, v[3], -v[2], -v[3], 0, v[1], v[2], -v[1], 0)
turn_z(φ) = Mat3d(cos(φ), sin(φ), 0, -sin(φ), cos(φ), 0, 0, 0, 1)
turn(φ) = Mat2d(cos(φ), sin(φ), -sin(φ), cos(φ))

"Rotation that turns the unit vector `a` onto the unit vector `b`."
function rotation(a::Vec3d, b::Vec3d)
    c = dot(a, b)
    if c < -1 + 1.0e-9   # opposite: half a turn around any axis normal to `a`
        n = normalize(cross(a, abs(a[1]) < 0.9 ? Vec3d(1, 0, 0) : Vec3d(0, 1, 0)))
        return one(Mat3d) + 2 * skew(n) * skew(n)
    end
    K = skew(cross(a, b))
    return one(Mat3d) + K + K * K / (1 + c)
end

"Turn about z that gives the points' convex hull the smallest bounding rectangle."
function smallest_footprint(points::Vector{Point3d}; step = 0.5)
    hull = convex_hull([Point2d(p[1], p[2]) for p in points])
    area(φ) = prod(widths(Rect2d([turn(φ) * h for h in hull])))
    return turn_z(argmin(area, deg2rad.(0:step:(90 - step))))
end

"""
A printed item ready for the bed: a watertight mesh in print orientation, centered on the
origin. The mesh's points are `rotation * p - center` of the item's points `p` in the assembly.
"""
struct PrintObject{M <: GeometryBasics.Mesh}
    item::Item
    mesh::M
    size::Vec3d
    rotation::Mat3d
    center::Vec3d
end

"""
    PrintObject(item; curvature = 72)

Meshes a printed item as it prints (springs unloaded) in the current Gmsh session, with
`curvature` elements per full circle, so bores stay within about 0.01 mm of round. The part is
turned so `item.down` faces the bed, then about z to the smallest footprint.
"""
function PrintObject(q::Item; curvature = 72)
    generate_mesh(() -> q.build(false), q.name; q.size, curvature)
    m = closed_mesh()
    gmsh.model.remove()
    down = rotation(normalize(q.down), Vec3d(0, 0, -1))
    flat = [down * p for p in coordinates(m)]
    R = smallest_footprint(flat)
    turned = [R * p for p in flat]
    box = Rect3d(turned)
    center = Vec3d(minimum(box) + widths(box) / 2)
    return PrintObject(q, GeometryBasics.Mesh([Point3d(p - center) for p in turned], faces(m)), Vec3d(widths(box)), R * down, center)
end

"Where an object goes: its plate, counted from 1, its center on the plate, and whether it turns a quarter turn about z."
struct Placement
    plate::Int
    center::Point2d
    turned::Bool
end

"""
    arrange(objects, bed)

Packs the objects' footprints, largest first, onto as few plates as the guillotine packer
manages. A footprint is the part with the gap and its brim around it, but no more than the
whole area: a part that only just fits gets the area's width to itself, and nothing can sit
beside it that way. Throws if a part does not fit on the bed at all.
"""
function arrange(objects::Vector{<:PrintObject}, bed::Bed)
    W, H = floor.(Int, widths(bed.area))
    brim(o) = parse(Float64, get(Dict(o.item.settings), "brim_width", "0"))
    function footprint(o)
        w, h = ceil.(Int, Vec2d(o.size[1], o.size[2]) .+ (bed.gap + 2brim(o)))
        # Which way the part goes is up to the packer: clamp each side to the side of the area it fits in.
        o.size[1] <= W && o.size[2] <= H ? (min(w, W), min(h, H)) : (min(w, H), min(h, W))
    end
    packers = GuillotinePacker{Int}[]
    placements = Vector{Placement}(undef, length(objects))
    for i in sortperm(objects; by = o -> prod(footprint(o)), rev = true)
        o = objects[i]
        w, h = footprint(o)
        fits = (w <= W && h <= H) || (h <= W && w <= H)
        (fits && o.size[3] <= bed.size[3]) || error("$(o.item.name) ($(o.size) mm) does not fit on the bed")
        # First plate with room; `push!` returns the placed rectangle, or nothing without packing.
        rect, k = nothing, 0
        while rect === nothing
            k += 1
            k > length(packers) && push!(packers, GuillotinePacker(W, H))
            rect = push!(packers[k], Rect2(0, 0, w, h))
        end
        turned = w != h && widths(rect)[1] == h
        center = Point2d(minimum(bed.area)) + Vec2d(minimum(rect)) + Vec2d(widths(rect)) / 2
        placements[i] = Placement(k, center, turned)
    end
    return placements
end

"""
Origin of plate `k` (from 1) of `n` in Bambu Studio's scene: the plates fill rows of ⌈√n⌉
columns, spaced 1.2 plate sizes apart, with further rows toward -y.
"""
function plate_origin(bed::Bed, k, n)
    cols = ceil(Int, sqrt(n))
    row, col = divrem(k - 1, cols)
    return Vec2d(col * 1.2 * bed.size[1], -row * 1.2 * bed.size[2])
end

number(x) = string(round(x; digits = 5))
hex(k, pad) = string(k; base = 16, pad)

function model_header(io)
    println(io, """<?xml version="1.0" encoding="UTF-8"?>""")
    println(io, """<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" \
        xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" \
        xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">""")
    return
end

"The mesh of object `k` as its own model file, like Bambu Studio stores it."
function object_model(io, o::PrintObject, k, turned)
    model_header(io)
    println(io, """ <metadata name="BambuStudio:3mfVersion">1</metadata>\n <resources>""")
    println(io, """  <object id="$(2k - 1)" p:UUID="$(hex(k, 4))0000-81cb-4c03-9d28-80fed5dfa1dc" type="model">""")
    println(io, "   <mesh>\n    <vertices>")
    R = turn_z(turned ? π / 2 : 0.0)
    for p in coordinates(o.mesh)
        v = R * p
        println(io, """     <vertex x="$(number(v[1]))" y="$(number(v[2]))" z="$(number(v[3]))"/>""")
    end
    println(io, "    </vertices>\n    <triangles>")
    for f in faces(o.mesh)
        a, b, c = GeometryBasics.raw.(f)   # 3MF counts vertices from 0
        println(io, """     <triangle v1="$a" v2="$b" v3="$c"/>""")
    end
    println(io, "    </triangles>\n   </mesh>\n  </object>\n </resources>\n <build/>\n</model>")
    return
end

"The project's main model: one object per part referencing its mesh file, placed by a build item."
function main_model(io, objects, placements, bed)
    model_header(io)
    println(io, """ <metadata name="Application">BambuStudio-02.08.02.61</metadata>""")
    println(io, """ <metadata name="BambuStudio:3mfVersion">1</metadata>""")
    println(io, """ <metadata name="Title">Paperhawk sheet feeder</metadata>""")
    println(io, " <resources>")
    for k in eachindex(objects)
        println(io, """  <object id="$(2k)" p:UUID="$(hex(k, 8))-61cb-4c03-9d28-80fed5dfa1dc" type="model">""")
        println(io, """   <components>\n    <component p:path="/3D/Objects/object_$k.model" objectid="$(2k - 1)" \
            p:UUID="$(hex(k, 4))0000-b206-40ff-9872-83e8017abed1" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>""")
        println(io, "   </components>\n  </object>")
    end
    println(io, """ </resources>\n <build p:UUID="2c7c17d8-22b5-4d84-8835-1976022ea369">""")
    n = maximum(p -> p.plate, placements)
    for (k, (o, p)) in enumerate(zip(objects, placements))
        x, y = p.center + plate_origin(bed, p.plate, n)
        println(io, """  <item objectid="$(2k)" p:UUID="$(hex(2k, 8))-b1ec-4553-aec9-835e5b724bb4" \
            transform="1 0 0 0 1 0 0 0 1 $(number(x)) $(number(y)) $(number(o.size[3] / 2))" printable="1"/>""")
    end
    println(io, " </build>\n</model>")
    return
end

"Bambu Studio's per-object settings and the plates with the objects on them."
function model_settings(io, objects, placements)
    println(io, """<?xml version="1.0" encoding="UTF-8"?>\n<config>""")
    for (k, o) in enumerate(objects)
        println(io, """  <object id="$(2k)">""")
        println(io, """    <metadata key="name" value="$(o.item.name)"/>""")
        println(io, """    <metadata key="extruder" value="$(filament(o.item))"/>""")
        for (key, value) in o.item.settings
            println(io, """    <metadata key="$key" value="$value"/>""")
        end
        println(io, """    <part id="$(2k - 1)" subtype="normal_part">""")
        println(io, """      <metadata key="name" value="$(o.item.name)"/>""")
        println(io, """      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>""")
        println(io, """      <mesh_stat face_count="$(length(faces(o.mesh)))" edges_fixed="0" degenerate_facets="0" \
            facets_removed="0" facets_reversed="0" backwards_edges="0"/>""")
        println(io, "    </part>\n  </object>")
    end
    for plate in 1:maximum(p -> p.plate, placements)
        println(io, """  <plate>\n    <metadata key="plater_id" value="$plate"/>""")
        println(io, """    <metadata key="plater_name" value=""/>\n    <metadata key="locked" value="false"/>""")
        for k in findall(p -> p.plate == plate, placements)
            println(io, """    <model_instance>\n      <metadata key="object_id" value="$(2k)"/>""")
            println(io, """      <metadata key="instance_id" value="0"/>\n      <metadata key="identify_id" value="$(100 + k)"/>""")
            println(io, "    </model_instance>")
        end
        println(io, "  </plate>")
    end
    println(io, "  <assemble>\n  </assemble>\n</config>")
    return
end

xml(f, args...) = (io = IOBuffer(); f(io, args...); take!(io))

"""
    write_3mf(path, objects, placements, bed, settings)

Writes a Bambu Studio project with `settings` (the contents of a `project_settings.config`)
as its printer, process and filament configuration.
"""
function write_3mf(path, objects, placements, bed::Bed, settings::Vector{UInt8})
    rels(targets) = string("""<?xml version="1.0" encoding="UTF-8"?>\n""",
        """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n""",
        join([""" <Relationship Target="$t" Id="rel-$i" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n"""
              for (i, t) in enumerate(targets)]),
        "</Relationships>\n")
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
        <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
         <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
         <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
         <Default Extension="png" ContentType="image/png"/>
         <Default Extension="gcode" ContentType="text/x.gcode"/>
        </Types>
        """
    ZipWriter(path) do w
        zip_writefile(w, "[Content_Types].xml", codeunits(content_types))
        zip_writefile(w, "_rels/.rels", codeunits(rels(["/3D/3dmodel.model"])))
        zip_newfile(w, "3D/3dmodel.model"; compress = true)
        write(w, xml(main_model, objects, placements, bed))
        zip_writefile(w, "3D/_rels/3dmodel.model.rels", codeunits(rels(["/3D/Objects/object_$k.model" for k in eachindex(objects)])))
        for (k, (o, p)) in enumerate(zip(objects, placements))
            zip_newfile(w, "3D/Objects/object_$k.model"; compress = true)
            write(w, xml(object_model, o, k, p.turned))
        end
        zip_newfile(w, "Metadata/model_settings.config"; compress = true)
        write(w, xml(model_settings, objects, placements))
        zip_newfile(w, "Metadata/project_settings.config"; compress = true)
        write(w, settings)
    end
    return path
end

"Filament slot of an item in the project: 1 is PLA, 2 is PETG."
filament(it::Item) = it.material == :PETG ? 2 : 1

"""
    arrange(objects, bed) by material

Packs the PLA and the PETG items onto separate plates, so no plate changes filament.
"""
function arrange_by_material(objects::Vector{<:PrintObject}, bed::Bed)
    placements = Vector{Placement}(undef, length(objects))
    first_plate = 0
    for material in (:PLA, :PETG)
        idx = findall(o -> o.item.material == material, objects)
        isempty(idx) && continue
        ps = arrange(objects[idx], bed)
        for (k, i) in enumerate(idx)
            placements[i] = Placement(ps[k].plate + first_plate, ps[k].center, ps[k].turned)
        end
        first_plate += maximum(p -> p.plate, ps)
    end
    return placements
end

"""
    export_3mf(items, path; bed = Bed(), settings, curvature = 72)

Writes the printed ones of `items` as an unsliced Bambu Studio project: each item a separate,
named object in print orientation, PLA and PETG on separate plates, with brim, supports or
solid infill where an item needs them. Orientation, placement and every setting stay editable
in Bambu Studio.

`settings` holds the printer, process and filament presets. The default was exported by
Bambu Studio 02.08.02.61 from the fully resolved system presets "Bambu Lab H2D 0.4 nozzle",
"0.20mm Standard @BBL H2D", "Bambu PLA Basic @BBL H2D" and "Bambu PETG Basic @BBL H2D". Two
values were set by hand: the textured PEI plate the H2D ships with, and one standard nozzle per
extruder, which Bambu Studio otherwise only learns from the connected printer. Returns the
objects and their placements.
"""
function export_3mf(list::AbstractVector{<:Item}, path; bed = Bed(),
        settings = joinpath(dirname(@__DIR__), "data", "bambu_h2d.config"), curvature = 72)
    objects = with_gmsh() do
        [PrintObject(it; curvature) for it in list if printed(it)]
    end
    placements = arrange_by_material(objects, bed)
    write_3mf(path, objects, placements, bed, read(settings))
    return objects, placements
end
