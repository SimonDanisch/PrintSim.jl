# Deterministic on-device incidence lists for contacts whose endpoints change during a run.
# Parallel scans compact contact IDs in their original order, so force gathering never needs
# floating point atomics and a body solve visits its actual contacts rather than every mesh.

# The block scan adapts AcceleratedKernels.jl's block_exclusive_scan! at
# 1b807a3a26f7676a16123313725523c714e0dd8f, src/accumulate/accumulate_1d_gpu.jl.
# Copyright (c) 2024 Andrei-Leonard Nicusan <a.l.nicusan@gmail.com> and contributors
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"Work-efficient exclusive integer scan; every lane of a 256-lane group must participate."
@inline function contact_block_scan!(totals, lane::Int32)
    thread = lane - Int32(1)
    offset, remaining = Int32(1), Int32(128)
    while remaining > 0
        KI.barrier()
        @inbounds if thread < remaining
            a = offset * (Int32(2) * thread + Int32(1))
            b = offset * (Int32(2) * thread + Int32(2))
            totals[b] += totals[a]
        end
        offset <<= 1
        remaining >>= 1
    end
    KI.barrier()
    @inbounds total = totals[256]
    KI.barrier()
    @inbounds lane == 1 && (totals[256] = Int32(0))
    remaining = Int32(1)
    while remaining < 256
        offset >>= 1
        KI.barrier()
        @inbounds if thread < remaining
            a = offset * (Int32(2) * thread + Int32(1))
            b = offset * (Int32(2) * thread + Int32(2))
            previous = totals[a]
            totals[a] = totals[b]
            totals[b] += previous
        end
        remaining <<= 1
    end
    KI.barrier()
    @inbounds return totals[lane], total
end

struct BodyIncidence{I, M}
    a::I
    b::I
    counts::I
    offsets::I
    total::I
    overflow::I
    ids::M
    entries::Int
    chunks::Int
    bodies::Int
    capacity::Int
end
Adapt.@adapt_structure BodyIncidence

function BodyIncidence(backend, entries, bodies; capacity = min(max(entries, 1), 8192))
    chunks = cld(max(entries, 1), 256)
    z(n) = KI.zeros(backend, Int32, max(n, 1))
    return BodyIncidence(z(entries), z(entries), z(chunks * bodies), z(chunks * bodies), z(bodies), z(bodies),
        KI.zeros(backend, Int32, capacity, max(bodies, 1)), entries, chunks, bodies, capacity)
end

function incidence_scan!(inc, fill::Bool)
    lane = KI.get_local_id(Int32).x
    group = KI.get_group_id(Int32).x - Int32(1)
    body = group ÷ Int32(inc.chunks) + Int32(1)
    chunk = group % Int32(inc.chunks)
    entry = Int32(256) * chunk + lane
    sh = KI.localmemory(Int32, Val((256,)), Val(1))
    active = false
    @inbounds if entry <= inc.entries
        active = inc.a[entry] == body || inc.b[entry] == body
    end
    @inbounds sh[lane] = active ? Int32(1) : Int32(0)
    prefix, total = contact_block_scan!(sh, lane)
    @inbounds if fill
        if active
            offset = inc.offsets[group + Int32(1)] + prefix + Int32(1)
            offset <= inc.capacity && (inc.ids[offset, body] = entry)
        end
    elseif lane == 256
        inc.counts[group + Int32(1)] = total
    end
    return nothing
end

function incidence_prefix!(inc)
    body = KI.get_global_id(Int32).x
    body <= inc.bodies || return nothing
    @inbounds begin
        total = Int32(0)
        for chunk in Int32(1):Int32(inc.chunks)
            index = (body - Int32(1)) * Int32(inc.chunks) + chunk
            inc.offsets[index] = total
            total += inc.counts[index]
        end
        inc.total[body] = min(total, Int32(inc.capacity))
        inc.overflow[body] = max(total - Int32(inc.capacity), Int32(0))
    end
    return nothing
end

function compact!(target, inc::BodyIncidence)
    groups = 256inc.chunks * inc.bodies
    launch!(target, incidence_scan!, (inc, false), groups)
    launch!(target, incidence_prefix!, (inc,), inc.bodies)
    launch!(target, incidence_scan!, (inc, true), groups)
    return inc
end

"Sparse particle incidence: integer allocation followed by sorting makes gathering deterministic."
struct VertexIncidence{I, M}
    keys::M
    ids::M
    count::I
    overflow::I
    nodes::Int
    entries::Int
    capacity::Int
end
Adapt.@adapt_structure VertexIncidence
function VertexIncidence(backend, entries, nodes; capacity = 256)
    z(T, dims...) = KI.zeros(backend, T, dims...)
    return VertexIncidence(z(Int32, 4, max(entries, 1)), z(Int32, capacity, max(nodes, 1)),
        z(Int32, max(nodes, 1)), z(Int32, max(nodes, 1)), nodes, entries, capacity)
end

function append_vertex_incidence!(inc)
    entry = KI.get_global_id(Int32).x
    entry <= inc.entries || return nothing
    @inbounds for corner in Int32(1):Int32(4)
        node = inc.keys[corner, entry]
        node == 0 && continue
        duplicate = false
        for previous in Int32(1):(corner - Int32(1))
            duplicate |= inc.keys[previous, entry] == node
        end
        duplicate && continue
        slot = Atomix.@atomic inc.count[node] += Int32(1)
        slot <= inc.capacity && (inc.ids[slot, node] = entry)
    end
    return nothing
end

function sort_vertex_incidence!(inc)
    node = KI.get_global_id(Int32).x
    node <= inc.nodes || return nothing
    @inbounds begin
        count = min(inc.count[node], Int32(inc.capacity))
        inc.overflow[node] = max(inc.count[node] - Int32(inc.capacity), Int32(0))
        inc.count[node] = count
        for i in Int32(2):count
            value = inc.ids[i, node]
            j = i - Int32(1)
            while j >= 1 && inc.ids[j, node] > value
                inc.ids[j + Int32(1), node] = inc.ids[j, node]
                j -= Int32(1)
            end
            inc.ids[j + Int32(1), node] = value
        end
    end
    return nothing
end

function compact!(target, inc::VertexIncidence)
    launch!(target, zero!, (inc.count,), length(inc.count))
    launch!(target, append_vertex_incidence!, (inc,), inc.entries)
    launch!(target, sort_vertex_incidence!, (inc,), inc.nodes)
    return inc
end

# CSR storage for large mapped surfaces. The dense reference above reserves
# capacity*nodes IDs; at most four distinct nodes can occur in each entry.
# Count/scan/fill retains exactly the same per-node capacity and sorted order.
struct SparseIncidenceIDs{I}
    data::I
    offset::I
end
Adapt.@adapt_structure SparseIncidenceIDs
@inline Base.getindex(ids::SparseIncidenceIDs,slot,node)=@inbounds ids.data[ids.offset[node]+slot]
@inline Base.setindex!(ids::SparseIncidenceIDs,value,slot,node)=
    (@inbounds ids.data[ids.offset[node]+slot]=value)

struct SparseVertexIncidence{K,D,I,S}
    keys::K
    ids::D
    count::I
    overflow::I
    cursor::I
    scan::S
    nodes::Int
    entries::Int
    capacity::Int
end
Adapt.@adapt_structure SparseVertexIncidence

function SparseVertexIncidence(backend,entries,nodes;capacity=256)
    entries isa Integer && nodes isa Integer && capacity isa Integer &&
        0<=entries<=typemax(Int32)÷4 && 0<=nodes<=typemax(Int32) &&
        0<capacity<=typemax(Int32) || throw(ArgumentError("invalid sparse contact incidence shape"))
    z(n)=KI.zeros(backend,Int32,max(1,n))
    scan=DevicePrefixScan(backend,nodes;T=Int32)
    return SparseVertexIncidence(KI.zeros(backend,Int32,4,max(1,entries)),
        SparseIncidenceIDs(z(4entries),scan.offsets[1]),z(nodes),z(nodes),z(nodes),scan,
        Int(nodes),Int(entries),Int(capacity))
end

function sparse_vertex_incidence!(inc,fill::Bool)
    entry=KI.get_global_id(Int32).x
    entry<=inc.entries || return nothing
    @inbounds for corner in Int32(1):Int32(4)
        node=inc.keys[corner,entry]
        node==0 && continue
        duplicate=false
        for previous in Int32(1):(corner-Int32(1))
            duplicate |= inc.keys[previous,entry]==node
        end
        duplicate && continue
        if fill
            slot=Atomix.@atomic inc.cursor[node]+=Int32(1)
            slot<=inc.capacity && (inc.ids[slot,node]=entry)
        else
            Atomix.@atomic inc.count[node]+=Int32(1)
        end
    end
    return nothing
end

vertex_incidence_work(inc::VertexIncidence)=inc
vertex_incidence_work(inc::SparseVertexIncidence)=
    (;inc.keys,inc.ids,inc.count,inc.overflow,inc.cursor,inc.nodes,inc.entries,inc.capacity)

function compact!(target,inc::SparseVertexIncidence)
    # The hierarchy's vectors are host declaration state, never kernel args.
    work=vertex_incidence_work(inc)
    launch!(target,zero!,(inc.count,),length(inc.count))
    launch!(target,sparse_vertex_incidence!,(work,false),inc.entries)
    prefix_scan!(target,inc.scan,inc.count)
    launch!(target,zero!,(inc.cursor,),length(inc.cursor))
    launch!(target,sparse_vertex_incidence!,(work,true),inc.entries)
    launch!(target,sort_vertex_incidence!,(work,),inc.nodes)
    return inc
end
