# Hierarchical scan for recorded GPU work queues. Reuses the existing
# AcceleratedKernels-derived block scan in incidence.jl (its MIT notice applies).
# No host counts, device spin-wait/lookback or per-replay allocations are needed.
struct DevicePrefixScan{A}
    offsets::Vector{A}
    totals::Vector{A}
    entries::Int
end

function DevicePrefixScan(backend,entries::Integer;T=Int64)
    0<=entries<=typemax(Int32) && T in (Int32,Int64) ||
        throw(ArgumentError("invalid integer prefix scan shape/type"))
    sizes=Int[max(1,entries)]
    while last(sizes)>256
        push!(sizes,cld(last(sizes),256))
    end
    offsets=[KI.zeros(backend,T,n) for n in sizes]
    totals=[KI.zeros(backend,T,cld(n,256)) for n in sizes]
    return DevicePrefixScan(offsets,totals,Int(entries))
end

function prefix_scan_blocks!(offsets,totals,source,n::Int32)
    lane=KI.get_local_id(Int32).x
    block=KI.get_group_id(Int32).x
    index=(block-Int32(1))*Int32(256)+lane
    shared=KI.localmemory(eltype(offsets),Val((256,)),Val(1))
    @inbounds shared[lane]=index<=n ? source[index] : zero(eltype(offsets))
    prefix,total=contact_block_scan!(shared,lane)
    @inbounds begin
        index<=n && (offsets[index]=prefix)
        lane==256 && (totals[block]=total)
    end
    return nothing
end

function prefix_scan_carry!(offsets,carry,n::Int32)
    index=KI.get_global_id(Int32).x
    index<=n || return nothing
    block=(index-Int32(1))÷Int32(256)+Int32(1)
    @inbounds offsets[index]+=carry[block]
    return nothing
end

"Record/replay an exclusive integer prefix scan; total lives in scan.totals[end][1]."
function prefix_scan!(target,scan::DevicePrefixScan,source)
    length(source)>=scan.entries || throw(DimensionMismatch("prefix source is too short"))
    if scan.entries==0
        launch!(target,zero!,(scan.offsets[1],),1)
        launch!(target,zero!,(scan.totals[1],),1)
        return scan
    end
    for level in eachindex(scan.offsets)
        n=level==1 ? scan.entries : length(scan.totals[level-1])
        values=level==1 ? source : scan.totals[level-1]
        launch!(target,prefix_scan_blocks!,
            (scan.offsets[level],scan.totals[level],values,Int32(n)),256cld(n,256))
    end
    for level in (length(scan.offsets)-1):-1:1
        n=level==1 ? scan.entries : length(scan.totals[level-1])
        launch!(target,prefix_scan_carry!,
            (scan.offsets[level],scan.offsets[level+1],Int32(n)),n)
    end
    return scan
end
