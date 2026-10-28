# Recorded moving CCD: the same BVH count/gather and inclusion narrow phase as
# initialization, with a device-resident hierarchical scan and pair-work loop.
# StiffGIPC's separate candidate/CCD queues are the architecture reference;
# no upstream source is copied. Overflow fails closed, never truncates pairs.
struct RuntimeCCDQueue{W,S,I,L}
    work::W
    scan::S
    valid::I
    chunks::I
    total::I
    iteration::I
    overflow::L
    capacity::Int
    pair_batch::Int
end

function RuntimeCCDQueue(backend,ccd;capacity=nothing,pair_batch=8192,track_minimum::Bool=false)
    n=ccd.query_count
    capacity===nothing && (capacity=max(1,min(Int64(typemax(Int32)),16Int64(n))))
    0<capacity<=typemax(Int32) && 0<pair_batch<=typemax(Int32) ||
        throw(ArgumentError("invalid runtime CCD queue capacity/batch"))
    scan=DevicePrefixScan(backend,n)
    z(T,n)=KI.zeros(backend,T,max(1,n))
    work=(;count=z(Int32,n),offset=scan.offsets[1],
        pairs=upload(backend,fill((Int32(0),Int32(0)),capacity)),
        flags=z(UInt32,capacity),invalid=z(Int32,1))
    if track_minimum
        onebits=reinterpret(UInt32,1f0)
        work=merge(work,(;pair_bound=upload(backend,fill(onebits,capacity)),
            minimum=upload(backend,UInt32[onebits,0,0])))
    end
    return RuntimeCCDQueue(work,scan,z(Int32,1),z(Int32,1),z(Int32,1),z(Int32,1),
        z(Int64,1),Int(capacity),Int(pair_batch))
end

function ccd_queue_prepare!(valid,chunks,total,overflow,prefix_total,invalid,ccd,
        capacity::Int32,pair_batch::Int32)
    KI.get_global_id(Int32).x==1 || return nothing
    @inbounds begin
        n=prefix_total[1]
        overflow[1]=max(n-Int64(capacity),Int64(0))
        safe=0<=n<=capacity && invalid[1]==0
        valid[1]=Int32(safe)
        total[1]=safe ? Int32(n) : Int32(0)
        chunks[1]=safe ? Int32(cld(n,Int64(pair_batch))) : Int32(0)
        if !safe
            # Bit 4 retains the capacity cause across subsequent CCD phases;
            # bit 2 remains the shared sticky "cannot advance" condition.
            failure=n>capacity ? UInt32(6) : UInt32(2)
            Atomix.@atomic ccd.status[1]|=failure
            Atomix.@atomic min(ccd.bound[1],UInt32(0))
        end
    end
    return nothing
end

@inline function ccd_vertex_stationary(ccd,vertex)
    @inbounds p,q=ccd.start[vertex],ccd.finish[vertex]
    return p.hi==q.hi && p.lo==q.lo
end

@inline ccd_triangle_pair_stationary(ccd,a,b)=
    ccd_vertex_stationary(ccd,a[1]) && ccd_vertex_stationary(ccd,a[2]) &&
    ccd_vertex_stationary(ccd,a[3]) && ccd_vertex_stationary(ccd,b[1]) &&
    ccd_vertex_stationary(ccd,b[2]) && ccd_vertex_stationary(ccd,b[3])

function ccd_queue_checks!(work,ccd,total,iteration,pair_batch::Int32,initial_check::Bool)
    lane=KI.get_global_id(Int32).x
    lane<=pair_batch || return nothing
    @inbounds begin
        index=(Int64(iteration[1])-Int64(1))*Int64(pair_batch)+Int64(lane)
        index<=total[1] || return nothing
        query,other=work.pairs[index]
        triangle=ccd.queries[query]
        a,b=ccd.faces[triangle],ccd.faces[other]
        radius=ccd.radius[ccd.owner[triangle]]+ccd.radius[ccd.owner[other]]
        # Only identical stored endpoints certify zero motion here. The fused
        # reference's common-translation test subtracts already-narrowed
        # coordinates: an inclined triangle's sub-ulp motion can disappear and
        # hide a real vertex/face crossing. Disable that shortcut for moving
        # pairs while retaining the unchanged inclusion/refinement tests.
        # Keep the six indices scalar. Iterating (a,b) materializes a dynamically
        # indexed nested private array in SPIR-V. Avoid that lowering in the
        # RADV/ACO store_deref compiler-abort regression; arithmetic is unchanged.
        stationary=ccd_triangle_pair_stationary(ccd,a,b)
        witness=hasproperty(work,:pair_bound) ? (work.pair_bound,index) : nothing
        work.flags[index]=stationary ? UInt32(0) : filtered_swept_triangle_pair!(ccd,a,b,radius,initial_check,witness)
    end
    return nothing
end

function ccd_queue_reports!(work,ccd,total,queries::Int32)
    @inbounds count=total[1]
    swept_pair_reports!(work,ccd,queries,Int32(1),count)
    return nothing
end

# A kernel-local index adapter, never an uploaded/packed argument. It lets the
# unchanged candidate/report kernels address one tile of global scan storage.
struct CCDQueryWindow{A}
    data::A
    offset::Int32
end
@inline Base.getindex(a::CCDQueryWindow,i::Integer)=@inbounds a.data[a.offset+i]
@inline Base.setindex!(a::CCDQueryWindow,v,i::Integer)=(@inbounds a.data[a.offset+i]=v)

@inline function ccd_query_window(work,first)
    return (;count=CCDQueryWindow(work.count,first-Int32(1)),
        offset=CCDQueryWindow(work.offset,first-Int32(1)),
        pairs=work.pairs,flags=work.flags,invalid=work.invalid)
end

function ccd_tiled_candidates!(work,ccd,iteration,batch::Int32,store)
    @inbounds first=(iteration[1]-Int32(1))*batch+Int32(1)
    count=min(batch,Int32(ccd.query_count)-first+Int32(1))
    swept_pair_candidates!(ccd_query_window(work,first),ccd,first,count,store)
    return nothing
end

function ccd_tiled_reports!(work,ccd,total,iteration,batch::Int32)
    @inbounds first=(iteration[1]-Int32(1))*batch+Int32(1)
    count=min(batch,Int32(ccd.query_count)-first+Int32(1))
    ccd_queue_reports!(ccd_query_window(work,first),ccd,total,count)
    return nothing
end

function runtime_ccd_candidates!(target,queue,ccd,store;query_batch,compact_queries)
    n=ccd.query_count
    if compact_queries && is_recording(target)
        Mantle.repeat!(recording_target(target),cld(n,query_batch)) do iteration
            launch!(target,ccd_tiled_candidates!,
                (queue.work,ccd,iteration,Int32(query_batch),store),query_batch;group=128)
        end
    else
        for first in 1:query_batch:n
            count=min(query_batch,n-first+1)
            range=first:(first+count-1)
            sub=merge(queue.work,(;count=view(queue.work.count,range),offset=view(queue.work.offset,range)))
            launch!(target,swept_pair_candidates!,(sub,ccd,Int32(first),Int32(count),store),count;group=128)
        end
    end
    return nothing
end

"Gather all moving candidates and evaluate them without host count readbacks in a graph."
function queued_swept_collisions!(target,queue::RuntimeCCDQueue,ccd;query_batch=2048,compact_queries=true,
        primitive_parallel=false)
    0<query_batch<=typemax(Int32) || throw(ArgumentError("invalid CCD query batch"))
    n=ccd.query_count
    n==0 && return nothing
    # A converged Newton phase skips its conditional producers. Clear the
    # loop count outside that condition, otherwise it replays the previous
    # active phase's entire queue as thousands of idle dispatches.
    launch!(recording_target(target),zero!,(queue.chunks,),1)
    launch!(target,zero!,(queue.work.invalid,),1)
    hasproperty(queue.work,:minimum) && launch!(target,reset_ccd_minimum!,(queue.work.minimum,),1)
    # Each subview maps local lane one to the same global query's count/offset.
    # Pair storage remains global, indexed by the full exclusive scan.
    runtime_ccd_candidates!(target,queue,ccd,Val(false);query_batch,compact_queries)
    prefix_scan!(target,queue.scan,queue.work.count)
    prepare()=launch!(target,ccd_queue_prepare!,
        (queue.valid,queue.chunks,queue.total,queue.overflow,queue.scan.totals[end],
            queue.work.invalid,ccd,Int32(queue.capacity),Int32(queue.pair_batch)),1)
    prepare()
    gated=ConditionalTarget(target,queue.valid)
    runtime_ccd_candidates!(gated,queue,ccd,Val(true);query_batch,compact_queries)
    # Count/gather mismatch is also fail-closed before reading candidate slots.
    prepare()
    if is_recording(target)
        Mantle.repeat!(recording_target(target),cld(queue.capacity,queue.pair_batch),queue.chunks) do iteration
            launch_ccd_pair_checks!(gated,queue.work,ccd,queue.total,iteration,queue.pair_batch;primitive_parallel)
        end
    else
        # Immediate mode is diagnostic; the production Stepper records the
        # device-counted loop above, with no CPU prefix or queue readback.
        chunks=only(Array(queue.chunks))
        for iteration in 1:chunks
            copyto!(queue.iteration,Int32[iteration])
            launch_ccd_pair_checks!(gated,queue.work,ccd,queue.total,queue.iteration,queue.pair_batch;primitive_parallel)
        end
    end
    report_runtime_ccd_queue!(gated,queue,ccd;query_batch,compact_queries)
    return nothing
end

function report_runtime_ccd_queue!(target,queue::RuntimeCCDQueue,ccd;query_batch=2048,compact_queries=true)
    0<query_batch<=typemax(Int32) || throw(ArgumentError("invalid CCD report batch"))
    n=ccd.query_count
    n==0 && return nothing
    if compact_queries && is_recording(target)
        Mantle.repeat!(recording_target(target),cld(n,query_batch)) do iteration
            launch!(target,ccd_tiled_reports!,
                (queue.work,ccd,queue.total,iteration,Int32(query_batch)),query_batch)
        end
        return nothing
    end
    # A report can walk many candidate slots. Bound it like the BVH queries;
    # one dispatch over a multi-million-query scene can exceed the graphics
    # watchdog even though every individual pair check is bounded.
    for first in 1:query_batch:n
        count=min(query_batch,n-first+1)
        range=first:(first+count-1)
        sub=merge(queue.work,(;count=view(queue.work.count,range),offset=view(queue.work.offset,range)))
        launch!(target,ccd_queue_reports!,(sub,ccd,queue.total,Int32(count)),count)
    end
    return nothing
end
