# Leaf-vs-tree traversal reuses PrintSim's BVH topology. Narrow phase is the
# Tight Inclusion port in ccd.jl. Unlike the force contact lists, this traversal
# considers ALL swept overlapping triangle pairs and all nonincident VF/EE
# primitives, including sheet self-contact and non-neighboring sheets.
struct SweptCollision{P, V, I, T, U, X, F, Q}
    start::P
    finish::P
    lo::V
    hi::V
    left::I
    right::I
    order::I
    height::I
    queries::I
    faces::T
    owner::I
    dynamic::I
    rigid::I
    excluded::X
    radius::F
    bound::U
    status::U
    blocking::I
    blocking_reason::U
    refinement::Q
    mesh_points::Int
    query_count::Int
    root::Int32
    depth::Int
end
Adapt.@adapt_structure SweptCollision

function SweptCollision(backend, b::ModelBuilder; sheet_radius = 0.5f-4)
    points = Vec3{Float64}[]
    faces = NTuple{3, Int32}[]
    owner = Int32[]
    for (id, mesh) in enumerate(b.meshes)
        first = Int32(length(points))
        append!(points, mesh.vertices)
        append!(faces, (f .+ first for f in mesh.faces))
        append!(owner, fill(Int32(id), length(mesh.faces)))
    end
    mesh_points = length(points)
    append!(points, b.x)
    append!(faces, (f .+ Int32(mesh_points) for f in b.tris))
    for f in b.tris
        sheet = findfirst(s -> f[1] in vertices(s), b.sheets)
        sheet === nothing && error("sheet triangle has no owning sheet")
        push!(owner, Int32(length(b.meshes) + sheet))
    end
    objects = length(b.meshes) + length(b.sheets)
    dynamic, rigid = zeros(Int32, objects), zeros(Int32, objects)
    for (id, body) in enumerate(b.bodies), mesh in body.meshes
        rigid[mesh] = id
        dynamic[mesh] = !body.fixed
    end
    for part in b.deformables
        dynamic[part.mesh] = 2
    end
    dynamic[(length(b.meshes) + 1):end] .= 3
    radius = zeros(Float32, objects)
    radius[(length(b.meshes) + 1):end] .= sheet_radius
    excluded = zeros(UInt8, objects, objects)
    for (a, c) in b.excluded
        excluded[a, c] = excluded[c, a] = 1
    end
    tree = BVHTables()
    root = Int32(0)
    if !isempty(faces)
        boxes = [(min.((points[i] for i in f)...), max.((points[i] for i in f)...)) for f in faces]
        centers = [(lo + hi) / 2 for (lo, hi) in boxes]
        root = add_bvh!(tree, Int32.(eachindex(faces)), boxes, centers)
    end
    height = zeros(Int32, length(tree.left))
    for node in reverse(eachindex(tree.left))
        tree.left[node] > 0 && (height[node] = 1 + max(height[tree.left[node]], height[tree.right[node]]))
    end
    depth = isempty(height) ? 0 : maximum(height)
    depth + 1 <= STACK || error("swept collision BVH exceeds traversal stack")
    # Static triangles are targets only. Every dynamic-static pair is visited
    # from its dynamic side, and dynamic-dynamic pairs are ordered by ID.
    queries = Int32[i for i in eachindex(faces) if dynamic[owner[i]] != 0]
    up(a) = upload(backend, a)
    # Metal has no Float64: keep that shader path entirely free of wide
    # arithmetic. Vulkan's capability is explicit, and Array supports it too.
    refinement=Val(backend===Array || Mantle.supports_float64(Mantle.todevice(backend)))
    return SweptCollision(up(Position.(points)), up(Position.(points)), up(tree.lo), up(tree.hi),
        up(tree.left), up(tree.right), up(tree.order), up(height), up(queries), up(faces), up(owner),
        up(dynamic), up(rigid), up(excluded), up(radius), up(UInt32[reinterpret(UInt32, 1f0)]), up(UInt32[0]),
        up(zeros(Int32,length(queries))),up(zeros(UInt32,length(queries))),refinement,
        mesh_points, length(queries), root, Int(depth))
end

function ccd_capture!(ccd, C, sheet, start::Bool)
    i = KI.get_global_id(Int32).x
    i <= length(ccd.start) || return nothing
    @inbounds begin
        value = i <= ccd.mesh_points ? C.points[i] : sheet[i - ccd.mesh_points]
        if start
            ccd.start[i] = value
        else
            ccd.finish[i] = value
        end
    end
    return nothing
end

@inline function swept_face_box(ccd, face)
    lo, hi = Vec3f(Inf32), Vec3f(-Inf32)
    @inbounds for i in face
        a, b = approximate(ccd.start[i]), approximate(ccd.finish[i])
        lo, hi = min.(lo, a, b), max.(hi, a, b)
    end
    margin = 8eps(Float32) * max.(abs.(lo), abs.(hi)) .+ 1f-10
    return lo - margin, hi + margin
end

function swept_refit!(ccd, level::Int32)
    node = KI.get_global_id(Int32).x
    node <= length(ccd.left) || return nothing
    @inbounds if ccd.height[node] == level
        left = ccd.left[node]
        if left < 0
            lo, hi = Vec3f(Inf32), Vec3f(-Inf32)
            for at in -left:(-left + ccd.right[node] - Int32(1))
                face = ccd.order[at]
                a, b = swept_face_box(ccd, ccd.faces[face])
                r = ccd.radius[ccd.owner[face]]
                lo, hi = min.(lo, a .- r), max.(hi, b .+ r)
            end
            ccd.lo[node], ccd.hi[node] = lo, hi
        else
            right = ccd.right[node]
            ccd.lo[node] = min.(ccd.lo[left], ccd.lo[right])
            ccd.hi[node] = max.(ccd.hi[left], ccd.hi[right])
        end
    end
    return nothing
end

@inline function swept_primitive(ccd, ids, kind, separation::Float32 = 0f0,initial_check::Bool=false)
    # Translation relative to the primitive's first starting point preserves
    # double-float cancellation accuracy and reduces the inclusion error filter.
    @inbounds origin = ccd.start[ids[1]]
    @inbounds x0 = ntuple(k -> ccd.start[ids[k]] - origin, Val(4))
    @inbounds x1 = ntuple(k -> ccd.finish[ids[k]] - origin, Val(4))
    # Rigid common translation cannot introduce an intersection.
    d = x1[1] - x0[1]
    common = true
    for k in Int32(2):Int32(4)
        common &= x1[k] - x0[k] == d
    end
    common && !initial_check && return (; hit = false, time = 1f0, resolved = true)
    result=tight_ccd(x0, x1, kind, separation, 1f-8, Int32(512))
    result=settle_unresolved(ccd,ids,kind,separation,result,x0,x1,ccd.refinement)
    # Nanometre-separated nonincident features can lie inside Float32's
    # inclusion error filter. They need upstream-style double precision, not
    # a false positive accepted as an actual initial intersection.
    return refine_swept_result(ccd,ids,kind,separation,result,ccd.refinement)
end

@inline refine_swept_result(ccd,ids,kind,separation,result,::Val{false})=result

"Settle a query Tight Inclusion left unresolved with additive CCD, keeping the later safe time."
@inline function settle_unresolved(ccd,ids,kind,separation,result,x0,x1,::Val{false})
    (!result.hit || result.resolved) && return result
    settled=additive_ccd(x0,x1,kind,separation,Int32(4096))
    settled.hit || return settled
    return (;hit=true,time=max(result.time,settled.time),resolved=settled.resolved)
end
@inline function settle_unresolved(ccd,ids,kind,separation,result,x0,x1,::Val{true})
    (!result.hit || result.resolved) && return result
    @inbounds origin=ccd.start[ids[1]]
    @inbounds w0=ntuple(k->wide_ccd_relative(ccd.start[ids[k]],origin),Val(4))
    @inbounds w1=ntuple(k->wide_ccd_relative(ccd.finish[ids[k]],origin),Val(4))
    settled=additive_ccd(w0,w1,kind,Float64(separation),Int32(4096))
    settled.hit || return (;hit=false,time=1f0,resolved=true)
    time=Float32(settled.time)
    # Round the safe time DOWN when narrowing to the device scalar.
    Float64(time)>settled.time && (time=reinterpret(Float32,reinterpret(UInt32,time)-UInt32(1)))
    return (;hit=true,time=max(result.time,time),resolved=settled.resolved)
end

@inline function wide_ccd_relative(p,origin)
    # Convert each leading/trailing component BEFORE subtraction/summation.
    # Converting p-origin would already have discarded the trailing precision.
    return (Vec3{Float64}(p.hi)-Vec3{Float64}(origin.hi))+
        (Vec3{Float64}(p.lo)-Vec3{Float64}(origin.lo))
end

@inline function refine_swept_result(ccd,ids,kind,separation,result,::Val{true})
    (!result.hit || result.time>0f0) && return result
    @inbounds origin=ccd.start[ids[1]]
    @inbounds x0=ntuple(k->wide_ccd_relative(ccd.start[ids[k]],origin),Val(4))
    @inbounds x1=ntuple(k->wide_ccd_relative(ccd.finish[ids[k]],origin),Val(4))
    refined=tight_ccd(x0,x1,kind,Float64(separation),1e-12,Int32(8192))
    time=Float32(refined.time)
    # Round the conservative time DOWN when narrowing to the device scalar.
    Float64(time)>refined.time && (time=reinterpret(Float32,reinterpret(UInt32,time)-UInt32(1)))
    return (;hit=refined.hit,time,resolved=refined.resolved)
end

@inline function swept_accumulate!(ccd, ids, kind, separation,initial_check)
    result = swept_primitive(ccd, ids, kind, separation,initial_check)
    flags=UInt32(0)
    if result.hit
        # Stay strictly before the conservative impact bound. An unresolved
        # query is reported separately, even if it supplies a usable bound.
        value = reinterpret(UInt32, 0.9f0 * result.time)
        Atomix.@atomic min(ccd.bound[1], value)
        result.resolved || (flags |= UInt32(1))
        result.time > 0 || (flags |= UInt32(2))
        flags==0 || (Atomix.@atomic ccd.status[1] |= flags)
    end
    return flags
end

@inline function initial_edge_face_result(scene,::Val{false})
    origin=scene.start[1]
    x0=ntuple(k->scene.start[k]-origin,Val(4))
    x1=ntuple(k->scene.finish[k]-origin,Val(4))
    return tight_ccd(x0,x1,Val(true),0f0,1f-8,Int32(512))
end
@inline function initial_edge_face_result(scene,::Val{true})
    origin=scene.start[1]
    x0=ntuple(k->wide_ccd_relative(scene.start[k],origin),Val(4))
    x1=ntuple(k->wide_ccd_relative(scene.finish[k],origin),Val(4))
    return tight_ccd(x0,x1,Val(true),0.,1e-12,Int32(8192))
end

# Motion VF/EE queries assume a disjoint starting surface: two already-crossed
# triangles need not have any coincident vertex/face or edge/edge feature.
# Test a STATIC edge against a STATIC face by using its edge parameter as the
# time variable of the existing point/triangle inclusion kernel. A hit anywhere
# along that fictitious trajectory means physical time ZERO, not a safe prefix.
@inline initial_edge_face_separated(scene,::Val{false})=false
@inline function initial_edge_face_separated(scene,::Val{true})
    ids=(Int32(1),Int32(2),Int32(3),Int32(4))
    origin=scene.start[2]
    u=wide_ccd_relative(scene.start[3],origin)
    v=wide_ccd_relative(scene.start[4],origin)
    normal=cross(u,v)
    path=wide_ccd_relative(scene.finish[1],scene.start[1])
    # Static segment/triangle SAT candidates. Arbitrary rounded axes are safe:
    # the existing outward arithmetic must still prove STRICT separation of
    # the exact hi+lo endpoints. These include in-plane coplanar axes and the
    # segment-edge cross axes absent from the generic VF trajectory prefilter.
    ccd_projection_axis_separated(scene,ids,Val(true),0f0,cross(path,normal)) && return true
    for k in Int32(1):Int32(3)
        edge=k==1 ? u : k==2 ? v-u : -v
        ccd_projection_axis_separated(scene,ids,Val(true),0f0,cross(edge,normal)) && return true
        ccd_projection_axis_separated(scene,ids,Val(true),0f0,cross(path,edge)) && return true
    end
    return false
end

@inline function initial_edge_face_flags!(ccd,aa,ab,face)
    (aa in face || ab in face) && return UInt32(0)
    @inbounds start=(ccd.start[aa],ccd.start[face[1]],ccd.start[face[2]],ccd.start[face[3]])
    @inbounds finish=(ccd.start[ab],start[2],start[3],start[4])
    scene=(;start,finish)
    ids=(Int32(1),Int32(2),Int32(3),Int32(4))
    ccd_projected_separated(scene,ids,Val(true),0f0,ccd.refinement) && return UInt32(0)
    initial_edge_face_separated(scene,ccd.refinement) && return UInt32(0)
    result=initial_edge_face_result(scene,ccd.refinement)
    result.hit || return UInt32(0)
    flags=result.resolved ? UInt32(2) : UInt32(3)
    Atomix.@atomic min(ccd.bound[1],UInt32(0))
    Atomix.@atomic ccd.status[1]|=flags
    return flags
end

@inline function swept_triangle_pair!(ccd, a, b, separation::Float32 = 0f0,initial_check::Bool=false)
    flags=UInt32(0)
    if initial_check
        for i in Int32(1):Int32(3)
            j=i==3 ? Int32(1) : i+Int32(1)
            flags|=initial_edge_face_flags!(ccd,a[i],a[j],b)
            flags|=initial_edge_face_flags!(ccd,b[i],b[j],a)
        end
        flags & UInt32(2)!=0 && return flags
    end
    for k in Int32(1):Int32(3)
        p = a[k]
        if !(p in b)
            flags |= swept_accumulate!(ccd, (p, b...), Val(true), separation,initial_check)
        end
        p = b[k]
        if !(p in a)
            flags |= swept_accumulate!(ccd, (p, a...), Val(true), separation,initial_check)
        end
    end
    for i in Int32(1):Int32(3), j in Int32(1):Int32(3)
        aa, ab = a[i], a[i == 3 ? Int32(1) : i + Int32(1)]
        ba, bb = b[j], b[j == 3 ? Int32(1) : j + Int32(1)]
        (aa == ba || aa == bb || ab == ba || ab == bb) && continue
        flags |= swept_accumulate!(ccd, (aa, ab, ba, bb), Val(false), separation,initial_check)
    end
    return flags
end

@inline function record_swept_blocker!(ccd,query,other,flags)
    # Each query has exactly one writer. Preserve a zero-time failure across
    # later Newton sweeps, in preference to a positive-time unresolved query.
    @inbounds if flags!=0 && (ccd.blocking[query]==0 ||
            (flags & UInt32(2)!=0 && ccd.blocking_reason[query] & UInt32(2)==0))
        ccd.blocking[query]=other
        ccd.blocking_reason[query]=flags
    end
    return nothing
end

"Host-only bounded report of the triangle pairs behind sticky CCD failures."
function ccd_host_range(a,range::AbstractUnitRange)
    # Array(view(::GPUArray,...)) may first materialize the whole parent on
    # the host. Use the supported offset copy API: one bounded transfer, no
    # temporary device array or view-copy kernel on the failing device.
    result=Vector{eltype(a)}(undef,length(range))
    isempty(range) || copyto!(result,1,a,first(range),length(range))
    return result
end

function ccd_failure_report(ccd;limit=16,query_range=eachindex(ccd.blocking),geometry=ccd)
    limit>=0 || throw(ArgumentError("nonnegative CCD report limit required"))
    report=NamedTuple[]
    limit==0 && return report
    query_range isa AbstractUnitRange || throw(ArgumentError("CCD report needs a contiguous query range"))
    isempty(query_range) && return report
    1<=first(query_range)<=last(query_range)<=length(ccd.blocking) ||
        throw(ArgumentError("CCD report query range is out of bounds"))
    blocking=ccd_host_range(ccd.blocking,query_range)
    reason=ccd_host_range(ccd.blocking_reason,query_range)
    # Initialization already has immutable host geometry. Otherwise fetch only
    # the few reported metadata entries, never whole multi-million-face arrays.
    value(a,i)=only(ccd_host_range(a,i:i))
    # Zero-time blockers take priority when the report is truncated.
    for zero_time in (true,false),localquery in eachindex(blocking)
        length(report)>=limit && break
        other=blocking[localquery]
        other==0 && continue
        (reason[localquery] & UInt32(2)!=0)==zero_time || continue
        query=first(query_range)+localquery-1
        face=value(geometry.queries,query)
        push!(report,(;query,face,target_face=other,
            owners=(value(geometry.owner,face),value(geometry.owner,other)),
            vertices=(value(geometry.faces,face),value(geometry.faces,other)),flags=reason[localquery]))
    end
    return report
end

function swept_collisions!(ccd,initial_check::Bool=false,first::Int32=Int32(1),count::Int32=Int32(ccd.query_count))
    lane = KI.get_global_id(Int32).x
    lane<=count || return nothing
    query = first+lane-Int32(1)
    query <= ccd.query_count && ccd.root != 0 || return nothing
    @inbounds begin
        triangle = ccd.queries[query]
        a = ccd.faces[triangle]
        own = ccd.owner[triangle]
        alo, ahi = swept_face_box(ccd, a)
        alo, ahi = alo .- ccd.radius[own], ahi .+ ccd.radius[own]
        stack = MVector{STACK, Int32}(undef)
        count = Int32(1)
        stack[1] = ccd.root
        while count > 0
            node = stack[count]
            count -= Int32(1)
            all(ahi .>= ccd.lo[node]) && all(alo .<= ccd.hi[node]) || continue
            left = ccd.left[node]
            if left > 0
                stack[count + 1], stack[count + 2] = left, ccd.right[node]
                count += Int32(2)
            else
                for at in -left:(-left + ccd.right[node] - Int32(1))
                    other = ccd.order[at]
                    other == triangle && continue
                    owner = ccd.owner[other]
                    ccd.dynamic[owner] != 0 && other < triangle && continue
                    ccd.excluded[own, owner] != 0 && continue
                    ccd.rigid[own] != 0 && ccd.rigid[own] == ccd.rigid[owner] && continue
                    b = ccd.faces[other]
                    blo, bhi = swept_face_box(ccd, b)
                    blo, bhi = blo .- ccd.radius[owner], bhi .+ ccd.radius[owner]
                    all(ahi .>= blo) && all(alo .<= bhi) || continue
                    flags=swept_triangle_pair!(ccd, a, b, ccd.radius[own] + ccd.radius[owner],initial_check)
                    record_swept_blocker!(ccd,query,other,flags)
                end
            end
        end
    end
    return nothing
end

"Two-pass BVH candidate generation, separated from expensive primitive CCD."
function swept_pair_candidates!(work,ccd,first::Int32,queries::Int32,::Val{STORE}) where STORE
    lane=KI.get_global_id(Int32).x
    lane<=queries || return nothing
    @inbounds begin
        query=first+lane-Int32(1)
        triangle=ccd.queries[query]
        a=ccd.faces[triangle]
        own=ccd.owner[triangle]
        alo,ahi=swept_face_box(ccd,a)
        alo,ahi=alo.-ccd.radius[own],ahi.+ccd.radius[own]
        stack=MVector{STACK,Int32}(undef)
        pending=Int32(1)
        stack[1]=ccd.root
        found=Int32(0)
        while pending>0
            node=stack[pending]
            pending-=Int32(1)
            all(ahi.>=ccd.lo[node]) && all(alo.<=ccd.hi[node]) || continue
            left=ccd.left[node]
            if left>0
                stack[pending+1],stack[pending+2]=left,ccd.right[node]
                pending+=Int32(2)
            else
                for at in -left:(-left+ccd.right[node]-Int32(1))
                    other=ccd.order[at]
                    other==triangle && continue
                    owner=ccd.owner[other]
                    ccd.dynamic[owner]!=0 && other<triangle && continue
                    ccd.excluded[own,owner]!=0 && continue
                    ccd.rigid[own]!=0 && ccd.rigid[own]==ccd.rigid[owner] && continue
                    blo,bhi=swept_face_box(ccd,ccd.faces[other])
                    blo,bhi=blo.-ccd.radius[owner],bhi.+ccd.radius[owner]
                    all(ahi.>=blo) && all(alo.<=bhi) || continue
                    found+=Int32(1)
                    if STORE
                        index=work.offset[lane]+found
                        if found<=work.count[lane] && index<=length(work.pairs)
                            work.pairs[index]=(query,other)
                        else
                            Atomix.@atomic work.invalid[1]+=Int32(1)
                        end
                    end
                end
            end
        end
        if STORE
            found==work.count[lane] || (Atomix.@atomic work.invalid[1]+=Int32(1))
        else
            work.count[lane]=found
        end
    end
    return nothing
end

function swept_pair_checks!(work,ccd,first::Int32,count::Int32,initial_check::Bool)
    lane=KI.get_global_id(Int32).x
    lane<=count || return nothing
    @inbounds begin
        query,other=work.pairs[first+lane-Int32(1)]
        triangle=ccd.queries[query]
        a,b=ccd.faces[triangle],ccd.faces[other]
        radius=ccd.radius[ccd.owner[triangle]]+ccd.radius[ccd.owner[other]]
        work.flags[first+lane-Int32(1)]=swept_triangle_pair!(ccd,a,b,radius,initial_check)
    end
    return nothing
end

function swept_pair_reports!(work,ccd,queries::Int32,first::Int32,count::Int32)
    lane=KI.get_global_id(Int32).x
    lane<=queries || return nothing
    @inbounds begin
        lower=max(work.offset[lane]+Int32(1),first)
        upper=min(work.offset[lane]+work.count[lane],first+count-Int32(1))
        # One writer per query, as in the fused traversal. Pair threads never
        # race on the diagnostic target/reason tuple.
        for index in lower:upper
            query,other=work.pairs[index]
            record_swept_blocker!(ccd,query,other,work.flags[index])
        end
    end
    return nothing
end

"Bounded candidate storage; no physics or primitive-pair test differs from the fused path."
function swept_pair_workspace(backend,queries,capacity)
    0<queries<=typemax(Int32) && 0<capacity<=typemax(Int32) ||
        throw(ArgumentError("invalid CCD pair workspace bounds"))
    return (;count=upload(backend,zeros(Int32,queries)),offset=upload(backend,zeros(Int32,queries)),
        pairs=upload(backend,fill((Int32(0),Int32(0)),capacity)),flags=upload(backend,zeros(UInt32,capacity)),
        invalid=upload(backend,zeros(Int32,1)))
end

"Portable host collision state, without solver matrices, device handles or recorded plans."
function ccd_snapshot(ccd::SweptCollision)
    names=fieldnames(typeof(ccd))
    data=NamedTuple{names}(map(names) do name
        value=getfield(ccd,name)
        value isa AbstractArray ? Array(value) : value
    end)
    return (;version=1,data)
end

"Restore a trusted local snapshot for a collision-only diagnostic."
function restore_ccd_snapshot(backend,snapshot)
    snapshot.version==1 || throw(ArgumentError("unsupported CCD snapshot version"))
    fieldnames(typeof(snapshot.data))==fieldnames(Base.unwrap_unionall(SweptCollision)) ||
        throw(ArgumentError("CCD snapshot fields do not match this implementation"))
    data=snapshot.data
    length(data.start)==length(data.finish) && length(data.faces)==length(data.owner) &&
        length(data.blocking)==length(data.blocking_reason) &&
        0<=data.query_count<=length(data.blocking) && data.query_count<=length(data.queries) ||
        throw(ArgumentError("inconsistent CCD snapshot dimensions"))
    restored=map(fieldnames(typeof(data))) do name
        value=getfield(data,name)
        name==:refinement && return Val(backend===Array || Mantle.supports_float64(Mantle.todevice(backend)))
        return value isa AbstractArray ? upload(backend,value) : value
    end
    return SweptCollision(restored...)
end

"Save collision input atomically, refusing to replace earlier diagnostic evidence."
function save_ccd_snapshot(path,ccd;metadata=nothing)
    isfile(path) && throw(ArgumentError("refusing to overwrite CCD snapshot $path"))
    snapshot=ccd_snapshot(ccd)
    directory=dirname(abspath(path))
    mkpath(directory)
    temporary,io=mktemp(directory)
    try
        Serialization.serialize(io,(;snapshot...,metadata))
    finally
        close(io)
    end
    mv(temporary,path)
    return path
end

"Capture/refit before an initial audit; a callback may save this state before expensive queries."
function prepare_initial_collisions!(world,state)
    ccd=world.ccd
    ccd===nothing && return nothing
    ccd.root==0 && return nothing
    backend=world.backend
    world_surface!(backend,world,state;refit=false)
    launch!(backend,zero!,(ccd.status,),1)
    launch!(backend,zero!,(ccd.blocking,),ccd.query_count)
    launch!(backend,zero!,(ccd.blocking_reason,),ccd.query_count)
    launch!(backend,reset_step_bound!,(ccd.bound,),1)
    for start in (true,false)
        launch!(backend,ccd_capture!,(ccd,world.bodies.colliders,state.sheets.x,start),length(ccd.start))
    end
    for level in 0:ccd.depth
        launch!(backend,swept_refit!,(ccd,Int32(level)),length(ccd.left))
    end
    return ccd
end

"Synchronous, bounded initialization batches; graph-time CCD remains a separate path."
function initial_ccd_batches!(backend,ccd;batch=128,on_batch=nothing,first_query=1,max_seconds=Inf,
        last_query=ccd.query_count,pairwise=true,pair_batch=2048,pair_capacity=1_048_576)
    0<batch<=typemax(Int32) || throw(ArgumentError("positive bounded CCD batch required"))
    1<=first_query<=ccd.query_count+1 || throw(ArgumentError("invalid initial CCD resume query"))
    0<=last_query<=ccd.query_count && first_query<=last_query+1 || throw(ArgumentError("invalid initial CCD end query"))
    max_seconds>=0 || throw(ArgumentError("nonnegative initial CCD time budget required"))
    pairwise && return initial_ccd_pair_batches!(backend,ccd;batch,on_batch,first_query,last_query,max_seconds,pair_batch,pair_capacity)
    started=time_ns()
    processed=first_query-1
    for first in first_query:batch:last_query
        (time_ns()-started)/1e9>=max_seconds && break
        count=min(batch,last_query-first+1)
        launch!(backend,swept_collisions!,(ccd,true,Int32(first),Int32(count)),count;group=min(batch,128))
        # One completed dispatch at a time: do not accumulate expensive wide
        # inclusion queries or GPU resource holds behind an unread fence.
        Mantle.waitidle(Mantle.todevice(backend))
        processed=first+count-1
        status=only(Array(ccd.status))
        on_batch===nothing || on_batch((;first,last=processed,total=ccd.query_count,status))
        # A single zero-time blocker already invalidates the assembly. Keep its
        # evidence and stop; success still requires examining every query.
        status & UInt32(2)==0 || break
    end
    return processed
end

"""
Load-balanced initialization CCD: count/gather BVH candidates, then run the
unchanged inclusion tests in bounded pair-parallel dispatches. StiffGIPC also
separates BVH pair production from pair processing (collision/mlbvh.cu); this
uses a deterministic count/prefix/gather instead of an overflowing append list.
No candidate is dropped when capacity is exceeded: shrink the query batch or
fail closed if even one query does not fit the explicit storage budget.
"""
function initial_ccd_pair_batches!(backend,ccd;batch,on_batch,first_query,last_query,max_seconds,
        pair_batch=2048,pair_capacity=1_048_576)
    0<pair_batch<=typemax(Int32) || throw(ArgumentError("positive CCD pair batch required"))
    processed=first_query-1
    processed==last_query && return processed
    started=time_ns()
    expired()=(time_ns()-started)/1e9>=max_seconds
    expired() && return processed
    work=swept_pair_workspace(backend,min(batch,last_query-processed),pair_capacity)
    device=Mantle.todevice(backend)
    while processed<last_query
        expired() && return processed
        first=processed+1
        count=min(batch,last_query-processed)
        host_counts=Int32[]
        total=0
        while true
            launch!(backend,swept_pair_candidates!,(work,ccd,Int32(first),Int32(count),Val(false)),count;group=min(count,128))
            Mantle.waitidle(device)
            host_counts=ccd_host_range(work.count,1:count)
            all(>=(0),host_counts) || error("CCD candidate count overflow")
            total=sum(Int64,host_counts)
            total<=pair_capacity && break
            count>1 || error("single CCD query exceeds pair capacity ($total > $pair_capacity)")
            count=cld(count,2)
            expired() && return processed
        end
        offset=Vector{Int32}(undef,count)
        prefix=0
        for i in 1:count
            offset[i]=Int32(prefix)
            prefix+=host_counts[i]
        end
        copyto!(work.offset,1,offset,1,count)
        launch!(backend,swept_pair_candidates!,(work,ccd,Int32(first),Int32(count),Val(true)),count;group=min(count,128))
        Mantle.waitidle(device)
        only(Array(work.invalid))==0 || error("CCD count/gather changed or exceeded storage")
        status=only(Array(ccd.status))
        for pair_first in 1:pair_batch:total
            expired() && return processed
            pairs=min(pair_batch,total-pair_first+1)
            launch!(backend,swept_pair_checks!,(work,ccd,Int32(pair_first),Int32(pairs),true),pairs;group=min(pairs,128))
            launch!(backend,swept_pair_reports!,(work,ccd,Int32(count),Int32(pair_first),Int32(pairs)),count;group=min(count,128))
            Mantle.waitidle(device)
            status=only(Array(ccd.status))
            status & UInt32(2)==0 || break
        end
        processed=first+count-1
        on_batch===nothing || on_batch((;first,last=processed,total=ccd.query_count,status,candidate_pairs=total))
        status & UInt32(2)==0 || break
    end
    return processed
end

"Validate the starting configuration rather than assuming it is intersection free."
function validate_initial_collisions!(world,state;batch=128,on_prepared=nothing,on_batch=nothing)
    ccd=prepare_initial_collisions!(world,state)
    ccd===nothing && return nothing
    on_prepared===nothing || on_prepared(ccd)
    processed=initial_ccd_batches!(world.backend,ccd;batch,on_batch)
    status=Array(ccd.status)
    status[1] & UInt32(2)==0 || error("initial assembly is intersecting, violates sheet thickness, or has an unresolved zero-time CCD query (status=$status, processed=$processed/$(ccd.query_count), pairs=$(ccd_failure_report(ccd)))")
    return nothing
end

function world_ccd_filter!(target, world, state, sheet_base, body_base, fem_base)
    ccd = world.ccd
    ccd === nothing && return nothing
    ccd.root == 0 && return nothing
    world_surface!(target, world, state; refit = false)
    launch!(target, ccd_capture!, (ccd, world.bodies.colliders, state.sheets.x, false), length(ccd.start))
    launch!(target, reset_step_bound!, (ccd.bound,), 1)
    for level in 0:ccd.depth
        launch!(target, swept_refit!, (ccd, Int32(level)), length(ccd.left))
    end
    queued_swept_collisions!(target,world.ccd_pairs,ccd)
    launch!(target, bounded_step!, (state.sheets.x, sheet_base, ccd.bound, world.bodies.one), world.sheets.n)
    launch!(target, bounded_step!, (state.bodies.q, body_base, ccd.bound, world.bodies.one), 4world.bodies.model.bodies)
    launch!(target, bounded_step!, (state.deformables.x, fem_base, ccd.bound, world.bodies.one), world.deformables.model.nodes)
    world_surface!(target, world, state)
    return nothing
end
