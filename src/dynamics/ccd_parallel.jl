# Primitive-level parallelism, as in established GPU IPC collision pipelines.
# Candidate generation and narrow-phase arithmetic are unchanged. Splitting VF
# and EE avoids putting fifteen inclusion searches/private queues in one thread.

function ccd_pair_flags_clear!(work,total,iteration,pair_batch::Int32)
    lane=KI.get_global_id(Int32).x
    lane<=pair_batch || return nothing
    @inbounds begin
        index=(Int64(iteration[1])-1)*Int64(pair_batch)+Int64(lane)
        if index<=total[1]
            work.flags[index]=UInt32(0)
            hasproperty(work,:pair_bound) && (work.pair_bound[index]=reinterpret(UInt32,1f0))
        end
    end
    return nothing
end

function ccd_vf_parallel_checks!(work,ccd,total,iteration,pair_batch::Int32)
    lane=KI.get_global_id(Int32).x
    lane<=6pair_batch || return nothing
    local_pair=(lane-Int32(1))÷Int32(6)+Int32(1)
    vertex=(lane-Int32(1))%Int32(6)+Int32(1)
    @inbounds begin
        index=(Int64(iteration[1])-1)*Int64(pair_batch)+Int64(local_pair)
        index<=total[1] || return nothing
        query,other=work.pairs[index]
        triangle=ccd.queries[query]
        a,b=ccd.faces[triangle],ccd.faces[other]
        ccd_triangle_pair_stationary(ccd,a,b) && return nothing
        # Keep the selected triangle scalar/flat, not a dynamically indexed
        # nested tuple of triangles (RADV private-array lowering regression).
        if vertex<=3
            point=a[vertex];face=b
        else
            point=b[vertex-Int32(3)];face=a
        end
        point in face && return nothing
        radius=ccd.radius[ccd.owner[triangle]]+ccd.radius[ccd.owner[other]]
        witness=hasproperty(work,:pair_bound) ? (work.pair_bound,index) : nothing
        flags=filtered_swept_accumulate!(ccd,(point,face...),Val(true),radius,true,witness)
        flags==0 || (Atomix.@atomic work.flags[index]|=flags)
    end
    return nothing
end

function ccd_ee_parallel_checks!(work,ccd,total,iteration,pair_batch::Int32)
    lane=KI.get_global_id(Int32).x
    lane<=9pair_batch || return nothing
    local_pair=(lane-Int32(1))÷Int32(9)+Int32(1)
    edge=(lane-Int32(1))%Int32(9)
    i,j=edge÷Int32(3)+Int32(1),edge%Int32(3)+Int32(1)
    @inbounds begin
        index=(Int64(iteration[1])-1)*Int64(pair_batch)+Int64(local_pair)
        index<=total[1] || return nothing
        query,other=work.pairs[index]
        triangle=ccd.queries[query]
        a,b=ccd.faces[triangle],ccd.faces[other]
        ccd_triangle_pair_stationary(ccd,a,b) && return nothing
        aa,ab=a[i],a[i==3 ? Int32(1) : i+Int32(1)]
        ba,bb=b[j],b[j==3 ? Int32(1) : j+Int32(1)]
        (aa==ba || aa==bb || ab==ba || ab==bb) && return nothing
        radius=ccd.radius[ccd.owner[triangle]]+ccd.radius[ccd.owner[other]]
        witness=hasproperty(work,:pair_bound) ? (work.pair_bound,index) : nothing
        flags=filtered_swept_accumulate!(ccd,(aa,ab,ba,bb),Val(false),radius,true,witness)
        flags==0 || (Atomix.@atomic work.flags[index]|=flags)
    end
    return nothing
end

function launch_ccd_pair_checks!(target,work,ccd,total,iteration,pair_batch;
        primitive_parallel=false)
    pair_batch isa Integer && 0<pair_batch<=typemax(Int32)÷9 ||
        throw(ArgumentError("CCD primitive dispatch exceeds device index range"))
    batch=Int32(pair_batch)
    if primitive_parallel || hasproperty(work,:pair_bound)
        launch!(target,ccd_pair_flags_clear!,(work,total,iteration,batch),pair_batch;group=128)
    end
    if primitive_parallel
        launch!(target,ccd_vf_parallel_checks!,(work,ccd,total,iteration,batch),6pair_batch;group=128)
        launch!(target,ccd_ee_parallel_checks!,(work,ccd,total,iteration,batch),9pair_batch;group=128)
    else
        launch!(target,ccd_queue_checks!,(work,ccd,total,iteration,batch,true),pair_batch;group=128)
    end
    hasproperty(work,:pair_bound) && launch!(target,ccd_pair_minimum!,
        (work,total,iteration,batch),1;group=64)
    return nothing
end

# Single-writer reduction AFTER all primitive lanes. A winning atomic followed
# by separate ID stores would race and could attribute the bound to a different
# pair. This optional diagnostic costs one UInt32 per candidate, not per node.
function ccd_pair_minimum!(work,total,iteration,batch::Int32)
    KI.get_global_id(Int32).x==1 || return nothing
    @inbounds begin
        first=(Int64(iteration[1])-1)*Int64(batch)+1
        last=min(Int64(total[1]),first+Int64(batch)-1)
        best=work.minimum[1]
        winner=Int64(0)
        for index in first:last
            bits=work.pair_bound[index]
            if bits<best
                best=bits
                winner=index
            end
        end
        if winner!=0
            query,other=work.pairs[winner]
            work.minimum[1]=best
            work.minimum[2]=UInt32(query)
            work.minimum[3]=UInt32(other)
        end
    end
    return nothing
end

function reset_ccd_minimum!(minimum)
    KI.get_global_id(Int32).x==1 || return nothing
    @inbounds begin
        minimum[1]=reinterpret(UInt32,1f0);minimum[2]=0;minimum[3]=0
    end
    return nothing
end
