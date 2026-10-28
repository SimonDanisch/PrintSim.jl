# Bounded query-tile ingestion and pair consumption. A single device loop is
# a two-stage queue state machine, avoiding nested Mantle loops: load a tile
# once, consume at most pair_batch pairs per iteration, report, then advance.
# Narrow-phase mathematics and the one-writer-per-query reports are unchanged.
struct StreamingCCDQueue{Q,S,L}
    queue::Q
    state::S
    total::L
    overflow::L
    query_batch::Int
    max_iterations::Int
    primitive_parallel::Bool
end

function StreamingCCDQueue(backend,ccd;query_batch=2048,capacity=1_048_576,pair_batch=8192,
        max_iterations=32768,primitive_parallel::Bool=false,track_minimum::Bool=false)
    0<query_batch<=typemax(Int32) || throw(ArgumentError("invalid streaming CCD tile"))
    max_iterations isa Integer && 0<max_iterations<=typemax(Int32) ||
        throw(ArgumentError("invalid streaming CCD iteration bound"))
    queue=RuntimeCCDQueue(backend,(;query_count=min(ccd.query_count,query_batch));capacity,pair_batch,track_minimum)
    z(T)=KI.zeros(backend,T,1)
    state=(;tile=z(Int32),first=z(Int32),count=z(Int32),retries=z(Int32),
        load=z(Int32),report=z(Int32),running=z(Int32),chunk=z(Int32))
    return StreamingCCDQueue(queue,state,z(Int64),z(Int64),Int(query_batch),Int(max_iterations),primitive_parallel)
end

function ccd_stream_initialize!(s,total,overflow,queries::Int32,batch::Int32)
    KI.get_global_id(Int32).x==1 || return nothing
    @inbounds begin
        s.tile[1]=1;s.first[1]=1;s.count[1]=min(queries,batch);s.retries[1]=0
        s.chunk[1]=1;s.load[1]=Int32(queries>0)
        s.running[1]=Int32(queries>0);s.report[1]=0
        total[1]=0;overflow[1]=0
    end
    return nothing
end

function ccd_stream_candidates!(work,ccd,s,store)
    @inbounds first,count=s.first[1],s.count[1]
    swept_pair_candidates!(work,ccd,first,count,store)
    return nothing
end

# An overflowing multi-query tile has not evaluated any pairs yet. Retry its
# same contiguous prefix at half the query count, without changing CCD status.
# A single query that cannot fit, or a count/gather mismatch, still fails closed.
function ccd_stream_prepare!(valid,chunks,total,overflow,prefix_total,invalid,ccd,
        count,capacity::Int32,pair_batch::Int32)
    KI.get_global_id(Int32).x==1 || return nothing
    @inbounds begin
        n=prefix_total[1]
        overflow[1]=max(n-Int64(capacity),Int64(0))
        safe=0<=n<=capacity && invalid[1]==0
        valid[1]=Int32(safe)
        total[1]=safe ? Int32(n) : Int32(0)
        chunks[1]=safe ? Int32(cld(n,Int64(pair_batch))) : Int32(0)
        retry=n>capacity && invalid[1]==0 && count[1]>1
        if !safe && !retry
            Atomix.@atomic :monotonic ccd.status[1]|=(n>capacity ? UInt32(6) : UInt32(2))
            Atomix.@atomic :monotonic min(ccd.bound[1],UInt32(0))
        end
    end
    return nothing
end

function ccd_stream_report_gate!(report,chunk,chunks,valid)
    KI.get_global_id(Int32).x==1 || return nothing
    @inbounds report[1]=Int32(valid[1]!=0 && chunk[1]>=chunks[1])
    return nothing
end

function ccd_stream_reports!(work,ccd,total,s)
    @inbounds count=s.count[1]
    ccd_queue_reports!(work,ccd,total,count)
    return nothing
end

function ccd_stream_advance!(s,total,overflow,tile_total,tile_overflow,invalid,valid,
        queries::Int32,batch::Int32)
    KI.get_global_id(Int32).x==1 || return nothing
    @inbounds begin
        if s.load[1]!=0 && valid[1]!=0
            total[1]+=Int64(tile_total[1])
        end
        if valid[1]==0
            if tile_overflow[1]>0 && invalid[1]==0 && s.count[1]>1
                s.count[1]=max(Int32(1),s.count[1]÷Int32(2))
                s.retries[1]+=Int32(1);s.chunk[1]=1;s.load[1]=1
            else
                overflow[1]=max(overflow[1],tile_overflow[1])
                total[1]=0;s.running[1]=0;s.load[1]=0
            end
        elseif s.report[1]!=0
            s.tile[1]+=Int32(1);s.chunk[1]=1
            s.first[1]+=s.count[1]
            s.count[1]=min(batch,max(Int32(0),queries-s.first[1]+Int32(1)))
            s.running[1]=Int32(s.first[1]<=queries);s.load[1]=s.running[1]
        else
            s.chunk[1]+=Int32(1);s.load[1]=0
        end
    end
    return nothing
end

function ccd_stream_finished!(running,ccd)
    KI.get_global_id(Int32).x==1 || return nothing
    @inbounds if running[1]!=0
        # Bit 8 distinguishes an exhausted stream schedule from capacity bit 4.
        Atomix.@atomic :monotonic ccd.status[1]|=UInt32(10)
        Atomix.@atomic :monotonic min(ccd.bound[1],UInt32(0))
    end
    return nothing
end

function initialize_ccd_stream!(target,stream::StreamingCCDQueue,ccd)
    ccd.query_count==0 && return nothing
    q,s=stream.queue,stream.state
    batch=stream.query_batch
    # Inactive Newton replay must not retain a previous active loop flag.
    launch!(recording_target(target),zero!,(s.running,),1)
    launch!(target,ccd_stream_initialize!,
        (s,stream.total,stream.overflow,Int32(ccd.query_count),Int32(batch)),1)
    hasproperty(q.work,:minimum) && launch!(target,reset_ccd_minimum!,(q.work.minimum,),1)
    return nothing
end

function ccd_stream_iteration_limit(stream::StreamingCCDQueue,ccd)
    q=stream.queue
    return Int(min(Int64(stream.max_iterations),Int64(ccd.query_count)*
        (max(1,cld(q.capacity,q.pair_batch))+ceil(Int,log2(stream.query_batch)))))
end

"Continue resident queue state without resetting candidates, bounds or progress."
function advance_ccd_stream!(target,stream::StreamingCCDQueue,ccd;
        iterations=ccd_stream_iteration_limit(stream,ccd),primitive_parallel=stream.primitive_parallel)
    ccd.query_count==0 && return nothing
    iterations isa Integer && 0<iterations<=ccd_stream_iteration_limit(stream,ccd) ||
        throw(ArgumentError("positive bounded CCD continuation iteration count required"))
    q,s=stream.queue,stream.state
    batch=stream.query_batch
    body=function ()
        load=ConditionalTarget(target,s.load)
        launch!(load,zero!,(q.work.invalid,),1)
        launch!(load,zero!,(q.work.count,),length(q.work.count))
        launch!(load,ccd_stream_candidates!,(q.work,ccd,s,Val(false)),batch;group=128)
        prefix_scan!(load,q.scan,q.work.count)
        prepare=()->launch!(load,ccd_stream_prepare!,
            (q.valid,q.chunks,q.total,q.overflow,q.scan.totals[end],q.work.invalid,ccd,
                s.count,Int32(q.capacity),Int32(q.pair_batch)),1)
        prepare()
        gather=ConditionalTarget(load,q.valid)
        launch!(gather,ccd_stream_candidates!,(q.work,ccd,s,Val(true)),batch;group=128)
        prepare()
        checks=ConditionalTarget(target,q.valid)
        launch_ccd_pair_checks!(checks,q.work,ccd,q.total,s.chunk,q.pair_batch;primitive_parallel)
        launch!(target,ccd_stream_report_gate!,(s.report,s.chunk,q.chunks,q.valid),1)
        report=ConditionalTarget(target,s.report)
        launch!(report,ccd_stream_reports!,
            (q.work,ccd,q.total,s),batch;group=128)
        launch!(target,ccd_stream_advance!,
            (s,stream.total,stream.overflow,q.total,q.overflow,q.work.invalid,q.valid,
                Int32(ccd.query_count),Int32(batch)),1)
    end
    # Mantle repeats recorded pieces at submission. Bound that schedule too,
    # not only its GPU buffers; exhaustion is explicitly rejected below.
    # Worst case processes one query per tile, with logarithmic split retries.
    # Keep the explicit global work cap: an exhausted schedule is not success.
    limit=Int(iterations)
    limit<=typemax(Int32) || throw(ArgumentError("streaming CCD loop exceeds device iteration range"))
    if is_recording(target)
        Mantle.repeat!(recording_target(target),limit;while_nonzero=s.running) do _
            body()
        end
    else
        for _ in 1:limit
            only(Array(s.running))==0 && break
            body()
        end
    end
    return nothing
end

function queued_swept_collisions!(target,stream::StreamingCCDQueue,ccd;primitive_parallel=stream.primitive_parallel,kw...)
    ccd.query_count==0 && return nothing
    initialize_ccd_stream!(target,stream,ccd)
    advance_ccd_stream!(target,stream,ccd;primitive_parallel)
    launch!(target,ccd_stream_finished!,(stream.state.running,ccd),1)
    return nothing
end
