# Illustrated assembly instructions, raytraced with RayMakie: the parts in the materials they
# are made of, on a desk in a photo studio lit by an HDRI, each step's own parts in full color and
# pulled out along their way in with an arrow, the parts built in before paler.

"""
One assembly step: what it shows and says. `base` matches items shown in place and in color,
`new` pairs a name pattern with the offset (mm) the items are pulled out along their way in.
Items of earlier steps are toned down; with `apart` the step is done on the table, apart from
what was built so far, and shows and moves only its own items. The camera looks from `eye`
toward `lookat`, which picture frames itself along that direction (`framing`), with `up` up. `ground` is the real up
direction while the step is done, what the table lies across: `-x` while the frame lies on
its right plate. `moves` are the moves the hands make, in order, each a name pattern with the
way the items come in (`sequence_check` follows them): by default `new`, but parts that go in
together as one unit move as one. With `kept = false` the step only prepares parts (O-rings
onto hubs), which then show up in later pictures only once a step builds them in. `leads`
names the `Lead`s the picture draws.
"""
struct AssemblyStep
    title::String
    text::String
    base::Regex
    new::Vector{Pair{Regex, Vec3f}}
    moves::Vector{Pair{Regex, Vec3f}}
    eye::Vec3f
    lookat::Vec3f
    up::Vec3f
    ground::Vec3f
    kept::Bool
    apart::Bool
    leads::Vector{String}
end

AssemblyStep(title, text; base = r"^$", new = Pair{Regex, Vec3f}[], moves = new, eye, lookat, up = Vec3f(0, 0, 1), ground = up,
    kept = true, apart = false, leads = String[]) =
    AssemblyStep(title, text, base, new, moves, Vec3f(eye), Vec3f(lookat), Vec3f(up), Vec3f(ground), kept, apart, leads)

"Pulls `name`'s offset out of a step: base items stay, new items move by their pattern's offset."
function step_role(s::AssemblyStep, name)
    occursin(s.base, name) && return :base, Vec3f(0)
    for (pattern, offset) in s.new
        occursin(pattern, name) && return :new, offset
    end
    return :none, Vec3f(0)
end

"""
    insertion_arrow(mesh, offset; obstacles = (), gap = 4, shortest = 20, longest = 70)

Where the arrow for a part pulled out by `offset` goes: from the part's leading face (the one
that moves first on the way in) toward where that face ends up, stopping `gap` short of the
first of the `obstacles` (bounding boxes of the other parts as drawn) in its way. Where that
leaves no room for an arrow `shortest` mm long (its 10 mm tip and some shaft), the arrow
pushes the part from behind instead. Returns start point and direction.
"""
function insertion_arrow(m::GeometryBasics.Mesh, offset::Vec3f; obstacles = (), gap = 4.0f0, shortest = 20.0f0, longest = 70.0f0)
    L = norm(offset)
    u = offset / L                                     # the way out; the part goes in along -u
    pts = coordinates(m)
    along = [dot(Vec3f(q), u) for q in pts]
    lo, hi = extrema(along)
    c = Vec3f(sum(pts) / length(pts))
    side = c - dot(c, u) * u                           # the centroid's position across the way in
    free = L - (hi - lo)                               # empty way between the pulled-out part and its place
    start = side + (lo + L - gap) * u                  # just off the leading face, toward the place
    blocked = minimum((ray_entry(start, -u, box) for box in obstacles); init = Inf32)
    len = min(free - 2gap, longest, blocked - gap)
    len >= shortest && return start, -len * u
    tail = side + (hi + L + gap) * u                   # behind the trailing face, pushing
    blocked = minimum((ray_entry(tail, u, box) for box in obstacles); init = Inf32)
    len = max(shortest, min(L / 2, longest, blocked - gap))
    return tail + len * u, -len * u
end

"Up to `n` of mesh `m`'s vertices, moved by `offset`."
sample_points(m::GeometryBasics.Mesh, offset = Vec3f(0); n = 3000) =
    [Vec3f(q) + offset for q in coordinates(m)[1:max(1, length(coordinates(m)) ÷ n):end]]

"""
    view_fit(s, points; fov, aspect, fill)

Look-at point and eye distance that keep step `s`'s viewing direction and up and fit `points`
into the middle `fill` of the picture (`fov` vertical in degrees, `aspect` width over height):
the look-at point moves until the points' projection is centered, the eye as close as it may.
"""
function view_fit(s::AssemblyStep, points; fov, aspect, fill)
    back = normalize(s.eye - s.lookat)
    right = normalize(cross(-back, s.up))
    up = cross(right, -back)
    ty = Float32(tand(fov / 2) * fill)
    tx = ty * Float32(aspect)
    fit(lookat) = maximum(max(abs(dot(q - lookat, right)) / tx, abs(dot(q - lookat, up)) / ty) + dot(q - lookat, back) for q in points)
    lookat = sum(points) / length(points)
    for _ in 1:8
        D = fit(lookat)
        xs = extrema(dot(q - lookat, right) / ((D - dot(q - lookat, back)) * tx) for q in points)
        ys = extrema(dot(q - lookat, up) / ((D - dot(q - lookat, back)) * ty) for q in points)
        lookat += D * (sum(xs) / 2 * tx * right + sum(ys) / 2 * ty * up)
    end
    return lookat, fit(lookat)
end

"""
    framing(s, focus, context; fov, aspect, fill = 0.85, least = 0.65)

Step `s` with its eye and look-at point set for the picture: centered on the `focus` points
and close enough to fill the picture with them, but at least `least` of the distance that
shows `focus` and `context` together, so a close-up still shows where it is on the whole.
"""
function framing(s::AssemblyStep, focus, context; fov, aspect, fill = 0.85, least = 0.65)
    lookat, D = view_fit(s, focus; fov, aspect, fill)
    isempty(context) || (D = max(D, least * last(view_fit(s, vcat(focus, context); fov, aspect, fill))))
    eye = lookat + D * normalize(s.eye - s.lookat)
    return AssemblyStep(s.title, s.text, s.base, s.new, s.moves, eye, lookat, s.up, s.ground, s.kept, s.apart, s.leads)
end

"A tube of radius `r` along `points`, as a lead is drawn: a cylinder per segment and a ball at every bend."
function tube(points, r)
    segments = [normal_mesh(Cylinder(Point3f(p), Point3f(q), Float32(r))) for (p, q) in zip(points[1:(end - 1)], points[2:end])]
    bends = [normal_mesh(Sphere(Point3f(p), Float32(r))) for p in points[2:(end - 1)]]
    return merge(vcat(segments, bends))
end

"""
An arrow from `start` along `v` as one mesh: a shaft of radius `shaft`, a cone `tip` long with
radius `head`. The cone's mantle is shaded smooth around, its base flat: both sides of the edge
between them keep their own normals.
"""
function arrow_mesh(start, v; shaft = 1.4f0, head = 3.6f0, tip = 10.0f0, n = 32)
    L = norm(v)
    u = Vec3f(v / L)
    base = Point3f(start + (L - tip) * u)
    apex = Point3f(start + v)
    a = normalize(cross(u, abs(u[3]) < 0.9f0 ? Vec3f(0, 0, 1) : Vec3f(1, 0, 0)))
    b = cross(u, a)
    radial = [cos(φ) * a + sin(φ) * b for φ in range(0, 2π; length = n + 1)[1:n]]
    mantle(r) = normalize(tip * r + head * u)                 # normal of the cone's side above `r`
    points, normals, faces = Point3f[], Vec3f[], GLTriangleFace[]
    for i in 1:n
        j = mod1(i + 1, n)
        k = length(points)
        append!(points, [base + head * radial[i], base + head * radial[j], apex])
        append!(normals, [mantle(radial[i]), mantle(radial[j]), mantle(normalize(radial[i] + radial[j]))])
        push!(faces, GLTriangleFace(k + 1, k + 2, k + 3))
        append!(points, [base + head * radial[j], base + head * radial[i], base])
        append!(normals, fill(-u, 3))
        push!(faces, GLTriangleFace(k + 4, k + 5, k + 6))
    end
    cone = GeometryBasics.Mesh(points, faces; normal = normals)
    return merge([normal_mesh(Cylinder(Point3f(start), base, Float32(shaft))), cone])
end

"How far a ray from `p` along `d` goes before it enters the box `(lo, hi)`; `Inf` if it misses it or starts inside."
function ray_entry(p, d, (lo, hi))
    t0, t1 = -Inf32, Inf32
    for k in 1:3
        if abs(d[k]) < 1.0f-9
            lo[k] <= p[k] <= hi[k] || return Inf32
        else
            e, f = (lo[k] - p[k]) / d[k], (hi[k] - p[k]) / d[k]
            t0, t1 = max(t0, min(e, f)), min(t1, max(e, f))
        end
    end
    return t0 <= t1 && t0 > 0 ? t0 : Inf32
end

"The names of the items step `s` builds in."
step_items(s::AssemblyStep, shown) = [q.name for q in shown if first(step_role(s, q.name)) != :none]

"""
    Lead(name, points, near)

A cable as the pictures draw it: a tube along `points`, framed together with the items named
in `near` (what it runs between).
"""
struct Lead
    name::String
    points::Vector{Vec3f}
    near::Vector{String}
end

"""
    step_scene(s, shown, done, leads)

What the picture of step `s` holds, independent of how it is drawn: `drawn` parts, each with its
offset and whether it belongs to the step (`own`) or an earlier step built it in (`done`);
`arrows` and `leads` as meshes; the points the camera frames (`focus`: the step's parts where
they are drawn and where they go, the arrows, each lead `s` names with what it runs between) and
the points it keeps partly in view (`context`: what was built before).
"""
function step_scene(s::AssemblyStep, shown, done, leads::AbstractVector{Lead})
    drawn = @NamedTuple{part::Part, offset::Vec3f, own::Bool}[]
    arrows = GeometryBasics.Mesh[]
    focus, context = Vec3f[], Vec3f[]
    # Boxes of the step's parts as drawn, pulled out: an arrow stops short of the next one.
    boxes = Dict{String, Tuple{Vec3f, Vec3f}}()
    for q in shown
        role, offset = step_role(s, q.name)
        (role == :none || offset == Vec3f(0)) && continue
        pts = sample_points(q.mesh, offset)
        boxes[q.name] = (reduce((p, r) -> min.(p, r), pts) .- 3.6f0, reduce((p, r) -> max.(p, r), pts) .+ 3.6f0)
    end
    for q in shown
        role, offset = step_role(s, q.name)
        if role == :none
            !s.apart && q.name in done || continue
            push!(drawn, (part = q, offset, own = false))
            append!(context, sample_points(q.mesh; n = 300))
            continue
        end
        push!(drawn, (part = q, offset, own = true))
        append!(focus, sample_points(q.mesh, offset))
        if offset != Vec3f(0)
            append!(focus, sample_points(q.mesh))
            push!(arrows, arrow_mesh(insertion_arrow(q.mesh, offset; obstacles = [box for (n, box) in boxes if n != q.name])...))
        end
    end
    foreach(a -> append!(focus, sample_points(a)), arrows)
    tubes = GeometryBasics.Mesh[]
    for lead in leads
        lead.name in s.leads || continue
        push!(tubes, tube(lead.points, 2.5))
        append!(focus, lead.points)
        for q in shown
            q.name in lead.near && append!(focus, sample_points(q.mesh))
        end
    end
    return (; drawn, arrows, leads = tubes, focus, context)
end

"""
    write_assembly_markdown(path, steps, images; title, intro)

The assembly instructions as Markdown: `title`, `intro`, then each step with its text and
picture (linked relative to `path`).
"""
function write_assembly_markdown(path, steps, images; title = "Assembly", intro = "")
    open(path, "w") do io
        println(io, "# $title\n")
        isempty(intro) || println(io, intro, "\n")
        for (k, (s, img)) in enumerate(zip(steps, images))
            println(io, "## $k. $(s.title)\n")
            println(io, s.text, "\n")
            println(io, "![Step $k]($(relpath(img, dirname(path))))\n")
        end
    end
    return path
end

"Downloads `url` into the cache folder `folder` the first time; returns its local path."
function downloaded(url, folder)
    dir = mkpath(joinpath(homedir(), ".cache", "PrintSim", folder))
    path = joinpath(dir, basename(url))
    isfile(path) || Downloads.download(url, path)
    return path
end

"Linear reflectance of an sRGB color channel; Hikari's reflectances are linear."
linear(x) = x <= 0.04045f0 ? x / 12.92f0 : ((x + 0.055f0) / 1.055f0)^2.4f0

"Printed plastic of sRGB color `c`, kept below 85 % reflectance like real filament."
plastic(c, roughness) = Hikari.CoatedDiffuse(reflectance = 0.85f0 .* linear.((c.r, c.g, c.b)), roughness = Float32(roughness))

"""
The solids of a STEP file as render meshes, each with the color the file gives it, moved by
`place` (a function of a point). `color(volume, mesh)` overrides it.
"""
function step_meshes(path, place; size = 0.8, color = nothing)
    with_gmsh() do
        gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
        gmsh.model.add("bought")
        occ().importShapes(path)
        occ().synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        gmsh.model.mesh.generate(2)
        out = Pair{GeometryBasics.Mesh, RGBf}[]
        for (_, v) in gmsh.model.getEntities(3)
            surfaces = gmsh.model.getBoundary([(3, v)], false, false, false)
            m = surface_mesh(surfaces)
            ps = [Point3f(place(Vec3d(q))) for q in coordinates(m)]
            fs = faces(m)
            placed = GeometryBasics.mesh(ps, fs; normal = face_normals_per_vertex(ps, fs))
            if isnothing(color)
                r, g, b_, _ = gmsh.model.getColor(2, last(first(surfaces)))
                push!(out, placed => RGBf(r / 255, g / 255, b_ / 255))
            else
                append!(out, color(m, placed))
            end
        end
        out
    end
end

"""
A material for a model's solid of sRGB color `c`: light grays are bare metal, yellows gold
plating, everything else plastic or solder mask.
"""
function finish(c::RGBf)
    hi, lo = max(c.r, c.g, c.b), min(c.r, c.g, c.b)
    hi - lo < 0.08 && hi > 0.55 && return Hikari.Conductor(eta = (1.6f0, 1.6f0, 1.52f0), k = (1.6f0, 1.6f0, 1.52f0), roughness = 0.2f0)
    c.r > 0.6 && c.g > 0.45 && c.b < 0.35 && return Hikari.Conductor(eta = (0.18f0, 0.42f0, 1.37f0), k = (3.4f0, 2.35f0, 1.77f0), roughness = 0.2f0)
    return Hikari.CoatedDiffuse(reflectance = 0.85f0 .* linear.((c.r, c.g, c.b)), roughness = 0.25f0)
end

"The direction toward pixel `(row, col)` of an equirectangular map (Hikari's layout: +y up, row 1 the zenith)."
function direction(sky::AbstractMatrix, row, col)
    h, w = size(sky)
    θ = (row - 0.5f0) / h * Float32(π)
    φ = (col - 0.5f0) / w * 2.0f0 * Float32(π) - Float32(π)
    return Vec3f(sin(θ) * cos(φ), cos(θ), sin(θ) * sin(φ))
end

"The equirectangular map's radiance toward `d`."
function radiance(sky::AbstractMatrix, d)
    h, w = size(sky)
    θ = acos(clamp(d[2], -1.0f0, 1.0f0))
    φ = atan(d[3], d[1])
    col = clamp(floor(Int, (φ + Float32(π)) / (2.0f0 * Float32(π)) * w) + 1, 1, w)
    row = clamp(floor(Int, θ / Float32(π) * h) + 1, 1, h)
    return sky[row, col]
end

luminance(c) = 0.2126f0 * c.r + 0.7152f0 * c.g + 0.0722f0 * c.b

"Where the map's light comes from: the luminance-weighted direction of its brightest `share` of pixels."
function brightest(sky::AbstractMatrix; share = 0.002)
    h, w = size(sky)
    cut = partialsort(vec(luminance.(sky)), round(Int, (1 - share) * length(sky)))
    total = zero(Vec3f)
    for row in 1:h, col in 1:w
        L = luminance(sky[row, col])
        L > cut || continue
        total += L * sin((row - 0.5f0) / h * Float32(π)) * direction(sky, row, col)
    end
    return normalize(total)
end

"""
    Studio(; samples = 512, px_per_unit = 2, hdri, wood, exposure = 2, desk = (1600, 1000), aperture = 0.002, ...)

How the pictures are rendered. `samples` per pixel, `px_per_unit` pixels per figure unit.
`hdri` is an equirectangular environment (lighting, reflections and the room behind), turned
in every picture so its main light comes from above, behind and left of the camera, scaled by
`exposure`. The parts stand on a `desk` (mm) topped with the `wood` texture. The camera's lens
radius is `aperture` times its distance to the look-at point, for the same little depth of
field at every distance. `pla` and `petg` are the printed plastics' roughness. Parts built in
before move `fade_amount` of the way toward `fade` and turn matte (`matte`). `arrow` and `lead`
color the arrows and the leads. `bought` gives the materials of bought and household parts (a
name pattern and a Hikari material each), `models` CAD models drawn instead of an item's own
mesh (`step_meshes`), by item name. The desk lies under `center`.
"""
struct Studio
    samples::Int
    px_per_unit::Int
    sky::Matrix{RGBf}
    light::Vec3f
    exposure::Float32
    wood::Matrix{Hikari.RGBSpectrum}
    desk::Tuple{Float32, Float32}
    aperture::Float32
    pla::Float32
    petg::Float32
    fade::RGBf
    fade_amount::Float32
    matte::Float32
    arrow::RGBf
    lead::RGBf
    bought::Vector{Pair{Regex, Any}}
    models::Dict{String, Vector{Pair{GeometryBasics.Mesh, RGBf}}}
    center::Vec3f
end

function Studio(; samples = 512, px_per_unit = 2,
        hdri = downloaded("https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/2k/brown_photostudio_02_2k.exr", "polyhaven"),
        wood = downloaded("https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/plywood/plywood_diff_2k.jpg", "polyhaven"),
        exposure = 2.0, desk = (1600, 1000), aperture = 0.002, pla = 0.45, petg = 0.12, fade = RGBf(0.86, 0.85, 0.83),
        fade_amount = 0.6, matte = 0.6, arrow = RGBf(0.85, 0.12, 0.08), lead = RGBf(0.12, 0.12, 0.14), bought = Pair{Regex, Any}[],
        models = Dict{String, Vector{Pair{GeometryBasics.Mesh, RGBf}}}(), center = Vec3f(0, 0, 0))
    sky = RGBf.(Makie.FileIO.load(hdri))
    texture = map(c -> Hikari.RGBSpectrum(linear(Float32(c.r)), linear(Float32(c.g)), linear(Float32(c.b))), Makie.FileIO.load(wood))
    return Studio(samples, px_per_unit, sky, brightest(sky), exposure, texture, Float32.(desk), aperture, pla, petg, fade,
        fade_amount, matte, arrow, lead, bought, models, Vec3f(center))
end

"Part `q` in the material it is made of."
function material(st::Studio, q::Part)
    for (pattern, m) in st.bought
        occursin(pattern, q.name) && return m
    end
    return plastic(q.color, q.material == :PETG ? st.petg : st.pla)
end

"Color `c` when an earlier step built its part in: faded and matte, so the step's own parts stand out."
function faded(st::Studio, c::Makie.Colorant)
    c, f, t = RGBf(c), st.fade, st.fade_amount
    return plastic(RGBf(c.r + t * (f.r - c.r), c.g + t * (f.g - c.g), c.b + t * (f.b - c.b)), st.matte)
end
faded(st::Studio, q::Part) = faded(st, q.color)

"""
How part `q` is drawn: its meshes with their materials, its own (`own`) or faded. Bought parts
with a CAD model (`st.models`) show the model.
"""
function looks(st::Studio, q::Part, own::Bool)
    haskey(st.models, q.name) || return [q.mesh => (own ? material(st, q) : faded(st, q))]
    return [m => (own ? finish(c) : faded(st, c)) for (m, c) in st.models[q.name]]
end

"Where the main light should come from: above, behind and left of the camera, so the faces it sees are lit."
function light_direction(s::AssemblyStep)
    view = normalize(s.lookat - s.eye)
    right = normalize(cross(view, s.up))
    up = normalize(cross(right, view))
    return normalize(-0.7f0 * view + up - 0.45f0 * right)
end

"The rotation from the map's frame into the scene: its +y onto `s.ground`, its main light's azimuth onto `light_direction(s)`'s."
function sky_rotation(st::Studio, s::AssemblyStep)
    up = normalize(s.ground)
    t = light_direction(s)
    to = normalize(t - dot(t, up) * up)
    from = normalize(Vec3f(st.light[1], 0, st.light[3]))
    y = Vec3f(0, 1, 0)
    return Mat3f(hcat(to, up, cross(to, up)) * transpose(hcat(from, y, cross(from, y))))
end

"""
    environment(st, s; n = 1024)

The light of a picture looking along step `s`'s direction: the HDRI turned by `sky_rotation`,
as the equal-area map Hikari samples (pixel `[i, j]` holds the direction at
`uv = ((j - ½) / n, (i - ½) / n)`).
"""
function environment(st::Studio, s::AssemblyStep; n = 1024)
    R = transpose(sky_rotation(st, s))
    image = Matrix{RGBf}(undef, n, n)
    Threads.@threads for i in 1:n
        for j in 1:n
            d = Hikari.equal_area_square_to_sphere(Point2f((j - 0.5f0) / n, (i - 0.5f0) / n))
            image[i, j] = st.exposure * radiance(st.sky, R * d)
        end
    end
    return Makie.EnvironmentLight(1.0f0, image)
end

"""
Points the camera of `ax` as the framed step `s` says, focused on its look-at point, and puts
the desk under everything `placed` (meshes with their offsets), across `s.ground`, its top at
their lowest point.
"""
function camera_and_desk!(st::Studio, ax, s::AssemblyStep, placed; fov)
    cam = Makie.cam3d!(ax.scene; eyeposition = s.eye, s.lookat, upvector = s.up, fov, center = false)
    update_cam!(ax.scene, cam, s.eye, s.lookat, s.up)
    cam.focal_distance[] = norm(s.lookat - s.eye)
    cam.lens_radius[] = st.aperture * cam.focal_distance[]
    up = normalize(s.ground)
    low = minimum(minimum(dot(Vec3f(q) + offset, up) for q in coordinates(m)) for (m, offset) in placed)
    x = Vec3f(1, 0, 0)
    a = abs(dot(x, up)) > 0.9f0 ? Vec3f(0, 0, 1) : normalize(x - dot(x, up) * up)
    b = cross(up, a)
    c = st.center + (low - dot(st.center, up) - 0.2f0) * up
    w, d = st.desk ./ 2
    corners = [Point3f(c + i * w * a + j * d * b) for (i, j) in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    uv = [Vec2f(0, 0), Vec2f(1, 0), Vec2f(1, 1), Vec2f(0, 1)]
    top = GeometryBasics.Mesh(corners, [GLTriangleFace(1, 2, 3), GLTriangleFace(1, 3, 4)]; normal = fill(up, 4), uv)
    mesh!(ax, top; material = Hikari.CoatedDiffuse(reflectance = Hikari.Texture(st.wood), roughness = 0.35f0))
    edges = vcat(corners .- Ref(0.5f0 * up), corners .- Ref(30.0f0 * up))
    mesh!(ax, normal_mesh(Rect3f(minimum(edges), maximum(edges) - minimum(edges))); material = Hikari.Diffuse(Kd = linear.((0.62f0, 0.5f0, 0.36f0))))
    return nothing
end

"Renders `fig` on the device `dev` and writes it to `path` (JPEG): through the screen itself, since `save` does not hand `px_per_unit` on to RayMakie."
function save_picture(dev::Mantle.Device, st::Studio, path, fig)
    Makie.update_state_before_display!(fig)
    screen = RayMakie.Screen(fig.scene; samples = st.samples, device = Mantle.backend(dev))
    img = colorbuffer(screen; figure = fig, px_per_unit = st.px_per_unit)
    close(screen)
    save(path, img)
    return path
end

"""
    step_figure(st, k, s, scene; size, fov)

The picture of step `k` from its `step_scene`: the step's own parts in their materials, pulled
out along their way in with a red arrow each, earlier parts faded, the leads `s` names, framed
on what the step is about.
"""
function step_figure(st::Studio, k, s::AssemblyStep, scene; size, fov)
    fig = Figure(; size, backgroundcolor = :white)
    Label(fig[0, 1], "Step $k: $(s.title)"; fontsize = 30, halign = :left, tellwidth = false)
    ax = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = [environment(st, s)]))
    for (; part, offset, own) in scene.drawn, (m, look) in looks(st, part, own)
        translate!(mesh!(ax, m; material = look), offset)
    end
    isempty(scene.arrows) || mesh!(ax, merge(scene.arrows); material = plastic(st.arrow, 0.25f0))
    foreach(lead -> mesh!(ax, lead; material = plastic(st.lead, 0.25f0)), scene.leads)
    framed = framing(s, scene.focus, scene.context; fov, aspect = /(widths(viewport(ax.scene)[])...))
    camera_and_desk!(st, ax, framed, [(d.part.mesh, d.offset) for d in scene.drawn]; fov)
    return fig
end

"""
    assembly_guide(dev, st, dir, plan, shown; leads = Lead[], steps = all, size = (1400, 1050), fov = 32, markdown, title, intro)

Renders on the device `dev` one picture per step of `plan` (`AssemblyStep`s) into `dir` (`step_01.jpg`, …) from the
parts `shown` (`Part`s), and writes the instructions with the steps and their pictures to
`markdown` (`ASSEMBLY.md` next to `dir`). `steps` picks which pictures are rendered; the
Markdown always lists all of them.
"""
function assembly_guide(dev::Mantle.Device, st::Studio, dir, plan::AbstractVector{AssemblyStep}, shown::AbstractVector{Part}; leads = Lead[], steps = nothing,
        size = (1400, 1050), fov = 32, markdown = joinpath(dirname(dir), "ASSEMBLY.md"), title = "Assembly", intro = "")
    mkpath(dir)
    done = Set{String}()
    images = String[]
    for (k, s) in enumerate(plan)
        path = joinpath(dir, "step_$(lpad(k, 2, '0')).jpg")
        push!(images, path)
        if isnothing(steps) || k in steps
            save_picture(dev, st, path, step_figure(st, k, s, step_scene(s, shown, done, leads); size, fov))
        end
        s.kept && union!(done, step_items(s, shown))
    end
    write_assembly_markdown(markdown, plan, images; title, intro)
    return images
end
