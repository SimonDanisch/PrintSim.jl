# Assembly manipulation is an external actuator, not a prescribed part pose.
# This zero-order-held force controller is sampled ONLY at accepted timestep
# boundaries. Existing FEM external-load work is therefore consistent throughout
# every Newton solve and backtracking trial; no hidden moving penalty is added.

struct NodalAssemblyGrip{I,W,P}
    nodes::I
    weights::W
    reference::P
    targets::P
    stiffness::Float32
    force_limit::Float32
    last_node::Int
end

"""A force-limited grip patch, with a total force budget in newtons.

`reference` contains world positions for the selected nodes at grip acquisition.
Weights must be positive and sum to one. Nonoverlapping patches can control a
part's orientation without fixing any of its material nodes. This is actuator
infrastructure, not an assembly-success or collision certificate.
"""
function NodalAssemblyGrip(backend,nodes,reference;weights=fill(1/length(nodes),length(nodes)),
        stiffness,force_limit)
    !isempty(nodes) && length(nodes)==length(reference)==length(weights) ||
        throw(ArgumentError("nonempty matching grip nodes, positions and weights required"))
    all(i->i isa Integer && 0<i<=typemax(Int32),nodes) && length(unique(nodes))==length(nodes) ||
        throw(ArgumentError("unique positive node IDs required"))
    all(w->isfinite(w) && w>0,weights) && isapprox(sum(weights),1;rtol=1e-6,atol=0) ||
        throw(ArgumentError("positive normalized grip weights required"))
    all(p->length(p)==3 && all(isfinite,p),reference) ||
        throw(ArgumentError("finite three-dimensional reference positions required"))
    all(v->isfinite(v) && 0<v<=floatmax(Float32),(stiffness,force_limit)) ||
        throw(ArgumentError("finite positive representable stiffness and force limit required"))
    k,f=Float32(stiffness),Float32(force_limit)
    k>0 && f>0 || throw(ArgumentError("controller coefficients underflow Float32"))
    w=Float32.(weights); w ./= sum(w)
    all(>(0),w) || throw(ArgumentError("grip weights underflow Float32"))
    p=Position.(reference)
    return NodalAssemblyGrip(upload(backend,Int32.(nodes)),upload(backend,w),
        upload(backend,p),upload(backend,p),k,f,Int(maximum(nodes)))
end

@inline function assembly_grip_force(delta,weight,k,limit)
    # Normalize before multiplying by stiffness: a large finite displacement
    # must not overflow an intermediate k*delta and create a NaN actuator.
    scale=max(abs(delta[1]),abs(delta[2]),abs(delta[3]))
    scale==0 && return zero(Vec3f)
    unit=delta/scale
    magnitude=sqrt(dot(unit,unit))
    amplitude=min(k*scale,limit/magnitude)*weight
    return amplitude*unit
end

function assembly_grip_targets!(targets,reference,translation,one)
    i=KI.get_global_id(Int32).x
    i<=length(targets) || return nothing
    @inbounds targets[i]=moved(reference[i],translation,one[1])
    return nothing
end

function freeze_assembly_grip_forces!(external,base,x,nodes,weights,targets,k,limit,engaged::Bool)
    j=KI.get_global_id(Int32).x
    j<=length(nodes) || return nothing
    @inbounds begin
        i=nodes[j]
        force=engaged ? assembly_grip_force(targets[j]-x[i],weights[j],k,limit) : zero(Vec3f)
        external[i]=base[j]+force
    end
    return nothing
end

"""Record one sampled grip update before prediction, never during Newton.

`base` is the other external force at each grip node, not an alias of `external`.
The caller must synchronize before editing targets and must reject overlapping
patches. Releasing (`engaged=false`) restores these base loads exactly.
"""
function sample_assembly_grip!(target,external,base,x,grip::NodalAssemblyGrip;engaged=true)
    length(base)==length(grip.nodes) || throw(DimensionMismatch("one base load per grip node required"))
    base===external && throw(ArgumentError("base forces must not alias the updated external loads"))
    length(x)==length(external) || throw(DimensionMismatch("position and load counts differ"))
    grip.last_node<=length(x) || throw(BoundsError(x,grip.last_node))
    launch!(target,freeze_assembly_grip_forces!,
        (external,base,x,grip.nodes,grip.weights,grip.targets,grip.stiffness,grip.force_limit,Bool(engaged)),
        length(grip.nodes))
    return nothing
end
