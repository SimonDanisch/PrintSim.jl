# Readback is for observations, never for the mechanical step or motor control.
struct SparseObservation{T} <: AbstractVector{T}
    count::Int
    indices::Vector{Int32}
    values::Vector{T}
end
Base.size(a::SparseObservation) = (a.count,)
Base.IndexStyle(::Type{<:SparseObservation}) = IndexLinear()
function Base.getindex(a::SparseObservation{T}, i::Int) where {T}
    @boundscheck checkbounds(a, i)
    at = searchsortedfirst(a.indices, i)
    return at <= length(a.indices) && a.indices[at] == i ? a.values[at] : zero(T)
end
function sparse_observation(values::Vector{T}) where {T}
    ids = findall(!iszero, values)
    SparseObservation(length(values), Int32.(ids), values[ids])
end
Base.maximum(a::SparseObservation{T}; init = zero(T)) where {T} = max(init, maximum(a.values; init = zero(T)))

struct DynamicsFrame
    t::Float64
    sheets::Vector{Position}
    bodies::Vector{Position}
    deformables::Vector{Position}
    surface::Vector{Point3f}
    normal_force::Union{Vector{Float32}, SparseObservation{Float32}}
    sliding::Union{Vector{UInt8}, SparseObservation{UInt8}}
    strain::Vector{Mat3f}  # Green strain at the tetrahedra's four quadrature points
    motor_angle::Vector{Float32}
    motor_command::Vector{Int32}
    motor_torque::Vector{Float32}
end

"An append-only frame store; geometry stays in Recording, large DOFs stay on disk."
struct DiskFrames <: AbstractVector{DynamicsFrame}
    directory::String
    files::Vector{String}
    times::Vector{Float64}
end
Base.size(frames::DiskFrames) = (length(frames.files),)
Base.IndexStyle(::Type{DiskFrames}) = IndexLinear()
Base.getindex(frames::DiskFrames, i::Int) = open(Serialization.deserialize, frames.files[i])::DynamicsFrame
function Base.push!(frames::DiskFrames, frame::DynamicsFrame)
    path = joinpath(frames.directory, lpad(string(length(frames.files) + 1), 8, '0') * ".jls")
    isfile(path) && error("refusing to overwrite an existing dynamics frame: $path")
    temporary, io = mktemp(frames.directory)
    Serialization.serialize(io, frame)
    close(io)
    mv(temporary, path)
    push!(frames.files, path)
    push!(frames.times, frame.t)
    return frames
end

struct Recording
    faces::Vector{GLTriangleFace}
    sheet_vertices::Int
    mesh_vertices::Vector{UnitRange{Int}}
    names::Vector{String}
    colors::Vector{RGBAf}
    rest_surface::Vector{Point3f}
    mesh_body::Vector{Int32}
    coordinates::Vector{Vec3f}
    fem_node::Vector{Int32}
    compact::Bool
    frames::Union{Vector{DynamicsFrame}, DiskFrames}
    series::Dict{String, Vector{Float64}}
end

# Preserve the serialized Recording layout: older recordings remain readable.
# Proxy geometry has its own immutable FE map, not a misleading one-node ID.
struct MappedRecording{T}
    base::Recording
    transfer::T
end
@inline function Base.getproperty(rec::MappedRecording,name::Symbol)
    name===:base && return getfield(rec,:base)
    name===:transfer && return getfield(rec,:transfer)
    return getproperty(getfield(rec,:base),name)
end

"Geometry and observations for replay of a coupled world, without rerunning its physics."
function Recording(b::ModelBuilder; names = ["mesh $i" for i in eachindex(b.meshes)],
        colors = fill(:gray70, length(b.meshes)), compact = false, directory = nothing,surface_maps=nothing)
    compact && require_surface_maps(b,surface_maps)
    length(names) == length(colors) == length(b.meshes) || throw(ArgumentError("one name and color per collider required"))
    faces = GLTriangleFace[GLTriangleFace(t...) for t in b.tris]
    ranges = UnitRange{Int}[]
    offset = length(b.x)
    for mesh in b.meshes
        push!(ranges, (offset + 1):(offset + length(mesh.vertices)))
        append!(faces, (GLTriangleFace((Int(v) + offset for v in f)...) for f in mesh.faces))
        offset += length(mesh.vertices)
    end
    rest = compact ? Point3f[Point3f(p) for mesh in b.meshes for p in mesh.vertices] : Point3f[]
    mesh_body, coordinates, fem_node = Int32[], Vec3f[], Int32[]
    if compact
        owner = zeros(Int32, length(b.meshes))
        for (id, body) in enumerate(b.bodies), mesh in body.meshes
            owner[mesh] = id
        end
        offsets = cumsum([0; length.(getfield.(b.meshes, :vertices))])
        append!(fem_node, zeros(Int32, offsets[end]))
        for (mesh, piece) in enumerate(b.meshes)
            body = owner[mesh]
            push!(mesh_body, body)
            append!(coordinates, (body == 0 ? zero(Vec3f) :
                Vec3f(b.bodies[body].axes' * (p - b.bodies[body].center) / b.bodies[body].scale) for p in piece.vertices))
        end
        first = 0
        for piece in b.deformables
            surface_maps===nothing || continue
            for (i, node) in enumerate(piece.surface_nodes)
                fem_node[offsets[piece.mesh] + i] = first + node
            end
            first += length(piece.x)
        end
    end
    frames = if directory === nothing
        DynamicsFrame[]
    else
        mkpath(directory)
        # These are durable simulation observations, not disposable Julia
        # temporaries: normal worker teardown must not remove the history.
        DiskFrames(mktempdir(directory;prefix="frames-",cleanup=false),String[],Float64[])
    end
    rec=Recording(faces, length(b.x), ranges, String.(names), Makie.to_color.(colors), rest, mesh_body, coordinates, fem_node, compact,
        frames, Dict{String, Vector{Float64}}())
    return compact && surface_maps!==nothing ? MappedRecording(rec,surface_transfer_data(b,surface_maps)) : rec
end

recording_times(rec::Recording) = rec.frames isa DiskFrames ? rec.frames.times : [f.t for f in rec.frames]
recording_times(rec::MappedRecording)=recording_times(rec.base)

"Rebuild display vertices from the recorded DOFs; static geometry is stored just once."
function frame_surface(rec::Recording, frame::DynamicsFrame)
    rec.compact || return frame.surface
    ns = rec.sheet_vertices
    points = Point3f[Point3f(approximate(p)) for p in frame.sheets]
    append!(points, rec.rest_surface)
    for (mesh, range) in enumerate(rec.mesh_vertices), i in range
        at = i - ns
        body, node = rec.mesh_body[mesh], rec.fem_node[at]
        if node != 0
            points[i] = Point3f(approximate(frame.deformables[node]))
        elseif body != 0
            points[i] = Point3f(approximate(affine_point(frame.bodies, Int32(body), rec.coordinates[at], 1.0f0)))
        end
    end
    return points
end

function frame_surface(rec::MappedRecording,frame::DynamicsFrame)
    points=frame_surface(rec.base,frame)
    map=rec.transfer.embedding
    length(frame.deformables)==map.nodes || throw(ArgumentError("frame FE node count differs from recording embedding"))
    for point in eachindex(rec.transfer.point_row)
        row=rec.transfer.point_row[point]
        row==0 && continue
        p=surface_map_position(map,row,frame.deformables,1f0)
        points[rec.sheet_vertices+point]=Point3f(approximate(p))
    end
    return points
end

function record!(rec::MappedRecording,world::WorldSolver,state::WorldState,t,series::Pair...)
    world.deformables.model isa MappedFEMModel || throw(ArgumentError("mapped recording requires a mapped FE world"))
    record!(rec.base,world,state,t,series...)
    return rec
end

"""
    record!(recording, world, state, time, series...)

Read back one completed step, including exact double-float DOFs, contact loads (N),
slip states, FE strains and actual/commanded motor rotation. Optional `name => value`
observations become time series; missing samples are NaN.
After an asynchronous `run!(stepper)`, call `waitfor!(stepper)` first.
"""
function record!(rec::Recording, world::WorldSolver, state::WorldState, t, series::Pair...)
    isempty(rec.frames) || t > last(recording_times(rec)) || throw(ArgumentError("frame times must increase"))
    s, b, f = world.sheets, world.bodies, world.deformables
    ns = rec.sheet_vertices
    ns == s.n || throw(ArgumentError("recording belongs to a different world"))
    nf, nb = f.model.nodes, 4b.model.bodies
    nf > 0 && !rec.compact && launch!(world.backend, fem_linearize!, (f.force, f.gradients, f.block, f.strain, state.deformables, f.model, f.one), f.model.elements)
    rec.compact || world_surface!(world.backend, world, state)
    sheet, body, fem = Array(state.sheets.x)[1:ns], Array(state.bodies.q)[1:nb], Array(state.deformables.x)[1:nf]
    surface = Point3f[Point3f(approximate(p)) for p in sheet]
    points = rec.compact || b.colliders.meshes == 0 ? Position[] : Array(b.colliders.points)
    append!(surface, (Point3f(approximate(p)) for p in points))
    count = rec.compact ? ns + length(rec.rest_surface) : length(surface)
    force, sliding = zeros(Float32, count), zeros(UInt8, count)
    if ns > 0
        λ, slip = Array(s.touches.lagged), Array(s.touches.sliding)
        links, lslip = Array(s.partners.lagged), Array(s.partners.sliding)
        for i in 1:ns
            force[i] = sum(@view λ[:, i]) + sum(@view links[:, i])
            sliding[i] = UInt8(any(!iszero, @view slip[:, i]) || any(!iszero, @view lslip[:, i]))
        end
        λf, wf, sf, triangles = Array(s.faces.lagged), Array(s.faces.weights), Array(s.faces.sliding), Array(s.model.tris)
        for triangle in 1:s.triangles, slot in 1:FACE_SLOTS, k in 1:3
            vertex = triangles[triangle][k]
            force[vertex] += λf[slot, triangle] * wf[slot, triangle][k]
            λf[slot, triangle] > 0 && sf[slot, triangle] != 0 && (sliding[vertex] = 1)
        end
    end
    if b.colliders.meshes > 0
        λ, slip = Array(b.contacts.lagged), Array(b.contacts.sliding)
        for i in 1:(count - ns)
            force[ns + i] = sum(@view λ[:, i])
            sliding[ns + i] = UInt8(any(!iszero, @view slip[:, i]))
        end
    end
    nm = b.motors.motors
    angle, command, torque = Array(b.motor_state.angle)[1:nm], Array(b.motor_state.commanded)[1:nm], Array(b.motor_state.torque)[1:nm]
    rec.compact && empty!(surface)
    force = rec.compact ? sparse_observation(force) : force
    sliding = rec.compact ? sparse_observation(sliding) : sliding
    push!(rec.frames, DynamicsFrame(Float64(t), sheet, body, fem, surface, force, sliding,
        rec.compact ? Mat3f[] : Array(f.strain)[1:f.model.elements], angle, command, torque))
    observations = Dict{String, Float64}(String(name) => Float64(value) for (name, value) in series)
    steps = Array(b.motors.steps)
    for motor in 1:nm
        observations["motor $motor missed steps"] = command[motor] - angle[motor] * steps[motor] / (2π)
        observations["motor $motor torque (N m)"] = torque[motor]
    end
    for (name, value) in observations
        samples = get!(rec.series, name) do
            fill(NaN, length(rec.frames) - 1)
        end
        push!(samples, value)
    end
    for samples in values(rec.series)
        length(samples) < length(rec.frames) && push!(samples, NaN)
    end
    return rec
end

struct DiskRecordingIndex
    version::Int
    geometry::String
    geometry_digest::String
    frames::DiskFrames
    series::Dict{String,Vector{Float64}}
end
const RECORDING_GEOMETRY_DIGESTS=Dict{String,String}()
const RECORDING_ARCHIVE_LOCK=ReentrantLock()

function recording_observations(rec::Recording,frames,series)
    values=ntuple(fieldcount(Recording)) do k
        name=fieldname(Recording,k)
        name===:frames ? frames : name===:series ? series : getfield(rec,k)
    end
    return Recording(values...)
end
recording_observations(rec::MappedRecording,frames,series)=
    MappedRecording(recording_observations(rec.base,frames,series),rec.transfer)

function disk_recording_index(rec)
    rec.frames isa DiskFrames || throw(ArgumentError("external geometry requires disk-backed frames"))
    geometry=abspath(joinpath(rec.frames.directory,"geometry.jls"))
    digest=lock(RECORDING_ARCHIVE_LOCK) do
        if !isfile(geometry)
            empty_frames=DiskFrames(rec.frames.directory,String[],Float64[])
            fixed=recording_observations(rec,empty_frames,Dict{String,Vector{Float64}}())
            temporary,io=mktemp(dirname(geometry))
            try
                Serialization.serialize(io,fixed)
            finally
                close(io)
            end
            mv(temporary,geometry)
            delete!(RECORDING_GEOMETRY_DIGESTS,geometry)
        end
        get!(RECORDING_GEOMETRY_DIGESTS,geometry) do
            bytes2hex(open(SHA.sha256,geometry))
        end
    end
    frames=DiskFrames(rec.frames.directory,copy(rec.frames.files),copy(rec.frames.times))
    return DiskRecordingIndex(1,geometry,digest,frames,deepcopy(rec.series))
end

"Fork an append-only observation history without modifying earlier frames."
function branch_recording(rec::Union{Recording,MappedRecording},directory)
    index=disk_recording_index(rec)
    mkpath(directory)
    frame_directory=mktempdir(directory;prefix="frames-",cleanup=false)
    # Geometry is immutable in the external archive. A branch owns its new
    # frame directory but refers to the same geometry and earlier frame files.
    geometry=joinpath(frame_directory,"geometry.jls")
    symlink(index.geometry,geometry)
    lock(RECORDING_ARCHIVE_LOCK) do
        RECORDING_GEOMETRY_DIGESTS[abspath(geometry)]=index.geometry_digest
    end
    frames=DiskFrames(frame_directory,copy(index.frames.files),copy(index.frames.times))
    return recording_observations(rec,frames,deepcopy(index.series))
end

"""
Serialize observations from trusted files, using the same Julia/package versions.
With `external_geometry=true`, disk-backed recordings commit their immutable
geometry/FE map once beside the frames; subsequent saves replace only a small
index. Do not mutate a recording's geometry after opting into this format.
Legacy recordings remain readable and remain the default write format.
"""
function save_recording(path, rec::Union{Recording,MappedRecording};external_geometry=false)
    data=external_geometry ? disk_recording_index(rec) : rec
    temporary, io = mktemp(dirname(abspath(path)))
    try
        Serialization.serialize(io, data)
    finally
        close(io)
    end
    mv(temporary, path; force = true)
    return path
end
function load_recording(path)
    data=open(Serialization.deserialize,path)
    return load_recording_data(data)
end
function load_recording_data(data)
    data isa DiskRecordingIndex || return data::Union{Recording,MappedRecording}
    data.version==1 || throw(ArgumentError("unsupported disk recording index"))
    bytes2hex(open(SHA.sha256,data.geometry))==data.geometry_digest ||
        throw(ArgumentError("recording geometry checksum differs"))
    length(data.frames.files)==length(data.frames.times) &&
        all(isfinite,data.frames.times) && all(>(0),diff(data.frames.times)) ||
        throw(ArgumentError("invalid recording frame index"))
    all(isfile,data.frames.files) || throw(ArgumentError("a recording frame is missing"))
    all(v->length(v)==length(data.frames),values(data.series)) ||
        throw(ArgumentError("recording series and frames differ"))
    fixed=open(Serialization.deserialize,data.geometry)::Union{Recording,MappedRecording}
    isempty(fixed.frames) || throw(ArgumentError("recording geometry unexpectedly contains frames"))
    return recording_observations(fixed,data.frames,data.series)
end

"The world's meshes, a frame slider, contact coloring and synchronized time series."
function player(rec::Union{Recording,MappedRecording}; color = :material, series = sort!(collect(keys(rec.series))),
        frame_index = nothing, camera = nothing,device=Makie.automatic)
    isempty(rec.frames) && throw(ArgumentError("record at least one frame before playback"))
    color in (:material, :force, :slip) || throw(ArgumentError("color must be :material, :force or :slip"))
    fig = Figure(; size = (1200, 800))
    Makie.colsize!(fig.layout, 1, Makie.Relative(isempty(series) ? 1.0 : 0.65))
    Makie.rowsize!(fig.layout, 1, Makie.Relative(0.85))
    scene = LScene(fig[1, 1]; show_axis = true)
    slider = Slider(fig[2, 1]; range = 1:length(rec.frames), startvalue = 1)
    index = frame_index === nothing ? slider.value : frame_index
    frame = Makie.lift(i -> rec.frames[i], index)
    render_backend=device isa Mantle.Device ? Mantle.backend(device) : device
    surface=render_backend isa KI.Backend ? RecordingSurface(render_backend,rec,frame[]) : nothing
    geometry = Makie.lift(frame) do f
        surface===nothing ? GeometryBasics.Mesh(frame_surface(rec,f),rec.faces) :
            GeometryBasics.Mesh(update_recording_surface!(surface,f),surface.faces)
    end
    if color == :material
        colors = fill(RGBAf(0.95, 0.95, 0.88, 1), length(GeometryBasics.coordinates(geometry[])))
        for (range, c) in zip(rec.mesh_vertices, rec.colors)
            colors[range] .= Ref(c)
        end
        mesh!(scene, geometry; color = colors)
    else
        quantity = Makie.lift(f -> color == :force ? f.normal_force : Float32.(f.sliding), frame)
        limit = color == :force ? max(maximum(maximum(f.normal_force; init = 0) for f in rec.frames), 1f-10) : 1.0f0
        mesh!(scene, geometry; color = quantity, colorrange = (0, limit), colormap = :viridis)
    end
    Label(fig[3, 1], Makie.lift(f -> "t = $(round(f.t; digits=4)) s", frame))
    times = recording_times(rec)
    for (row, name) in enumerate(series)
        axis = Axis(fig[row, 2]; title = name, xlabel = "time (s)")
        lines!(axis, times, rec.series[name])
        vlines!(axis, Makie.lift(f -> [f.t], frame); color = :red)
    end
    camera === nothing || Makie.update_cam!(scene.scene, camera...)
    return fig
end

"""
    save_video(path, recording; framerate=30, speed=1, rasterize=true, device=Makie.automatic)

Render recorded simulation states. Playback selects existing frames by their actual
timestamps; it never prescribes, interpolates or reruns mechanical motion.
Uses RayMakie's real-time raster path by default, independently of Makie's
currently active backend. GPU rendering must share the simulation's reservation;
pass its already-created `device` to avoid creating an independent context.
"""
function save_video(path, rec::Union{Recording,MappedRecording}; framerate = 30, speed = 1.0, camera = nothing,
        rasterize::Bool=true,device=Makie.automatic)
    length(rec.frames) >= 2 || throw(ArgumentError("video requires at least two completed simulation states"))
    speed > 0 && framerate > 0 || throw(ArgumentError("positive speed and framerate required"))
    index = Observable(1)
    figure = player(rec; frame_index = index, series = String[], camera,device)
    times = recording_times(rec)
    count = max(2, ceil(Int, (last(times) - first(times)) * framerate / speed) + 1)
    render_device=device isa Mantle.Device ? Mantle.backend(device) : device
    Makie.record(figure, path, 0:(count - 1); framerate,backend=RayMakie,rasterize,device=render_device,visible=false) do frame
        time = min(first(times) + frame * speed / framerate, last(times))
        selected=clamp(searchsortedlast(times,time),1,length(times))
        # A 30 fps export can select the same 10 Hz observation three times.
        # Do not reload its large disk frame and rebuild the FE display map
        # merely to encode another copy of the already displayed state.
        index[]==selected || (index[]=selected)
    end
    return path
end
