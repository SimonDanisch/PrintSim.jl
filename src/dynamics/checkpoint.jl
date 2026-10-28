# A Recording is an observation, not a restart: velocity, friction memory,
# contact acceleration and controller counters must survive a cold restart too.

struct SparseCheckpointArray{T,N,I<:Integer} <: AbstractArray{T,N}
    dimensions::NTuple{N,Int}
    indices::Vector{I}
    values::Vector{T}
end
Base.size(a::SparseCheckpointArray)=a.dimensions
Base.IndexStyle(::Type{<:SparseCheckpointArray})=IndexLinear()
function Base.getindex(a::SparseCheckpointArray{T},i::Int) where T
    @boundscheck checkbounds(a,i)
    at=searchsortedfirst(a.indices,i)
    return at<=length(a.indices) && a.indices[at]==i ? a.values[at] : zero(T)
end
function Base.Array(a::SparseCheckpointArray{T}) where T
    result=zeros(T,size(a))
    result[a.indices]=a.values
    return result
end

function checkpoint_array_storage(array;min_entries=256)
    host=Array(array)
    length(host)>=min_entries || return host
    # isequal preserves signed zero; NaNs are rejected by the finite audit.
    indices=findall(v->!isequal(v,zero(eltype(host))),vec(host))
    I=length(host)<=typemax(Int32) ? Int32 : Int64
    length(indices)*(sizeof(I)+sizeof(eltype(host)))<sizeof(eltype(host))*length(host) || return host
    return SparseCheckpointArray(size(host),I.(indices),host[indices])
end

checkpoint_array_finite(a::Array)=all(checkpoint_finite,a)
checkpoint_array_finite(a::SparseCheckpointArray)=all(checkpoint_finite,a.values)

function validate_sparse_checkpoint(a::SparseCheckpointArray)
    all(>=(0),size(a)) && length(a.indices)==length(a.values) &&
        all(i->1<=i<=length(a),a.indices) &&
        all(k->a.indices[k]>a.indices[k-1],2:length(a.indices)) ||
        throw(ArgumentError("invalid sparse checkpoint indices/dimensions"))
    return nothing
end

function checkpoint_roots(world::WorldSolver,state::WorldState,control)
    sheet=world.sheets
    # Only persistent history: force/Hessian workspaces and incidence are
    # overwritten before use next step and are prohibitively large to archive.
    body=world.bodies.contacts
    body_history=(;body.mesh,body.normal,body.lagged,body.spring,body.sliding)
    history=if sheet isa SheetSolver
        t,f,p=sheet.touches,sheet.faces,sheet.partners
        (;sheet.acceleration,
            touches=(;t.mesh,t.normal,t.lagged,t.spring,t.sliding),
            faces=(;f.kind,f.feature,f.edge,f.side,f.weights,f.lagged,f.spring,f.sliding),
            partners=(;p.cell,p.side,p.lagged,p.spring,p.sliding))
    else
        nothing
    end
    model=world.deformables.model
    guard=model isa MappedFEMModel ? model.transfer.guard : nothing
    return (;state,sheet_history=history,body_contacts=body_history,
        motor_state=world.bodies.motor_state,world.bodies.load_force,world.bodies.load_torque,
        external=world.deformables.external,control,proxy_maximum=guard===nothing ? nothing : guard.maximum)
end

function checkpoint_array_bindings!(bindings,value,path=())
    if value isa AbstractArray
        isbitstype(eltype(value)) || throw(ArgumentError("checkpoint arrays require isbits elements: $path"))
        bindings[path]=value
    elseif value!==nothing && !isbitstype(typeof(value))
        for name in fieldnames(typeof(value))
            checkpoint_array_bindings!(bindings,getfield(value,name),(path...,name))
        end
    end
    return bindings
end
checkpoint_array_bindings(roots)=checkpoint_array_bindings!(Dict{Tuple,Any}(),roots)

checkpoint_finite(x::Number)=isfinite(x)
checkpoint_finite(x::Position)=all(isfinite,x.hi)&&all(isfinite,x.lo)
checkpoint_finite(x)=all(checkpoint_finite,x)

function checkpoint_digest(value)
    value isa AbstractString && occursin(r"^[0-9a-f]{64}$",value) ||
        throw(ArgumentError("checkpoint provenance must be a lowercase SHA-256 digest"))
    return String(value)
end

checkpoint_settings(world)=(;timestep=world.deformables.timestep,newton=world.newton,
    iterations=world.iterations,cg_tolerance=world.cg_tolerance,newton_tolerance=world.newton_tolerance,
    continuous_collisions=world.ccd!==nothing,roughness=world.bodies.roughness,
    tangential=world.bodies.tangential,sheet_radius=world.sheets.radius,
    elastic_schwarz=world.deformables.schwarz!==nothing,
    elastic_coarse=string(nameof(typeof(world.deformables.coarse))))

function checkpoint_control(world,control)
    control===nothing && return nothing
    control isa MotorSequence && control.timestep==world.deformables.timestep &&
        control.motors==world.bodies.motors.motors ||
        throw(ArgumentError("checkpoint requires a matching device MotorSequence"))
    return nothing
end

"""
    state_checkpoint(world, state; time, step, model_digest, engine_digest, control=nothing)

Capture a completed timestep, including velocities, friction history, contact
acceleration, loads and device controller state. Immutable model/solver caches
are rebuilt, not duplicated. Digests must identify the exact immutable model
and execution engine. This captures state; it does not replace the step audit.
"""
function state_checkpoint(world::WorldSolver,state::WorldState;time,step,
        model_digest,engine_digest,control=nothing,sparse=true)
    isfinite(time) && time>=0 && step isa Integer && step>=0 || throw(ArgumentError("invalid checkpoint time/step"))
    model_digest=checkpoint_digest(model_digest)
    engine_digest=checkpoint_digest(engine_digest)
    checkpoint_control(world,control)
    bindings=checkpoint_array_bindings(checkpoint_roots(world,state,control))
    arrays=Dict(path=>(sparse ? checkpoint_array_storage(array) : Array(array)) for (path,array) in bindings)
    all(checkpoint_array_finite,values(arrays)) || error("refusing a nonfinite dynamics checkpoint")
    return (;version=2,time=Float64(time),step=Int(step),settings=checkpoint_settings(world),
        model_digest,engine_digest,arrays)
end

"Atomically save a restart, refusing to replace prior evidence. Trusted files only."
function save_state_checkpoint(path,world::WorldSolver,state::WorldState;kw...)
    isfile(path) && throw(ArgumentError("refusing to replace an existing dynamics checkpoint"))
    checkpoint=state_checkpoint(world,state;kw...)
    mkpath(dirname(abspath(path)))
    temporary,io=mktemp(dirname(abspath(path)))
    try
        Serialization.serialize(io,checkpoint)
    finally
        close(io)
    end
    mv(temporary,path)
    return path
end

function validate_checkpoint_arrays(bindings,arrays)
    Set(keys(bindings))==Set(keys(arrays)) || throw(ArgumentError("checkpoint state/history layout differs"))
    for (path,destination) in bindings
        source=arrays[path]
        source isa Union{Array,SparseCheckpointArray} && eltype(source)===eltype(destination) && size(source)==size(destination) ||
            throw(ArgumentError("checkpoint array type/shape differs: $path"))
        source isa SparseCheckpointArray && validate_sparse_checkpoint(source)
        checkpoint_array_finite(source) || throw(ArgumentError("nonfinite checkpoint array: $path"))
    end
    return nothing
end

"""
    restore_state_checkpoint!(world, state, checkpoint; model_digest, engine_digest, control=nothing)

Restore into a freshly constructed/primed solver with the same immutable model
and settings. Supply the matching device controller. All provenance and arrays
are checked before any copy; the restored collision geometry is then refitted.
`checkpoint` may be a trusted serialized path or the value from `state_checkpoint`.
"""
function restore_state_checkpoint!(world::WorldSolver,state::WorldState,checkpoint;
        model_digest,engine_digest,control=nothing)
    data=checkpoint isa AbstractString ? open(Serialization.deserialize,checkpoint) : checkpoint
    checkpoint_control(world,control)
    data.version in (1,2) && data.model_digest==checkpoint_digest(model_digest) &&
        data.engine_digest==checkpoint_digest(engine_digest) && data.settings==checkpoint_settings(world) ||
        throw(ArgumentError("checkpoint provenance/settings differ"))
    isfinite(data.time) && data.time>=0 && data.step isa Integer && data.step>=0 ||
        throw(ArgumentError("invalid checkpoint time/step"))
    bindings=checkpoint_array_bindings(checkpoint_roots(world,state,control))
    validate_checkpoint_arrays(bindings,data.arrays)
    for (path,destination) in bindings
        # Densify one sparse history array at a time, not the entire checkpoint.
        # The existing upload path avoids a new device scatter dependency.
        source=data.arrays[path]
        copyto!(destination,source isa Array ? source : Array(source))
    end
    # step! captures the previous collider surface for CCD before prediction.
    # A cold solver still holds the builder's initial surface until this refit.
    world_surface!(world.backend,world,state)
    return (;time=data.time,step=data.step)
end
