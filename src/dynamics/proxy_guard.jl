# A quadratic patch equals its linear corner interpolant plus
# 4ab*q_ab + 4ac*q_ac + 4bc*q_bc, q_ab=x_ab-(x_a+x_b)/2.
# Along a parameter edge d, midpoint/chord error is
# -(d_a*d_b*q_ab+d_a*d_c*q_ac+d_b*d_c*q_bc).
# Retain the vectors in double-float arithmetic: taking three norms before
# combining them grossly overestimates skinny, nearly aligned parent patches.
struct ProxyCurvature
    q::NTuple{3,Position}
    error::Float32
end
Base.zero(::Type{ProxyCurvature})=ProxyCurvature(ntuple(_->zero(Position),3),0f0)
struct QuadraticProxyGuard{I,M,P,V,U}
    nodes::M
    parent::I
    edges::P
    curvature::V
    maximum::U
    parents::Int
    faces::Int
end
Adapt.@adapt_structure QuadraticProxyGuard

function append_proxy_guard!(nodes,parent,edges,piece,proxy,node_offset)
    first=length(nodes)
    for f in quadratic_boundary_faces(piece)
        push!(nodes,ntuple(k->Int32(f[k]+node_offset),6))
    end
    for p in proxy.parent
        push!(parent,Int32(p+first))
    end
    for uv in proxy.parameters
        push!(edges,ntuple(k->Vec3f(uv[k==3 ? 1 : k+1]-uv[k]),3))
    end
    return nothing
end

function QuadraticProxyGuard(backend,b::ModelBuilder,maps)
    nodes=NTuple{6,Int32}[]
    parent=Int32[]
    edges=NTuple{3,Vec3f}[]
    offset=0
    for piece in b.deformables
        proxy=get(maps,piece.mesh,nothing)
        if proxy!==nothing && hasproperty(proxy,:parameters)
            append_proxy_guard!(nodes,parent,edges,piece,proxy,offset)
        end
        offset+=length(piece.x)
    end
    isempty(parent) && return nothing
    matrix=Matrix{Int32}(undef,6,length(nodes))
    parameters=Matrix{Vec3f}(undef,3,length(edges))
    for i in eachindex(nodes),k in 1:6
        matrix[k,i]=nodes[i][k]
    end
    for i in eachindex(edges),k in 1:3
        parameters[k,i]=edges[i][k]
    end
    up(a)=upload(backend,a)
    curvature=(ab=up(zeros(Position,length(nodes))),ac=up(zeros(Position,length(nodes))),
        bc=up(zeros(Position,length(nodes))),error=up(zeros(Float32,length(nodes))))
    return QuadraticProxyGuard(up(matrix),up(parent),up(parameters),curvature,
        up(zeros(UInt32,length(parent)+1)),length(nodes),length(parent))
end

@inline function proxy_edge_curvature(a::Position,b::Position,mid::Position,one)
    da,dm=difference(b,a,one),difference(mid,a,one)
    q=difference(dm,scaled(da,.5f0,one),one)
    span=norm(da.hi)+norm(da.lo)+norm(dm.hi)+norm(dm.lo)
    # Include absolute coordinate scale too: subtraction of stored low tails
    # can round when crossing a leading-Float32 bin. A span-only error budget
    # would underestimate that rounding for extremely short physical edges.
    coordinate=max(norm(a.hi)+norm(a.lo),norm(b.hi)+norm(b.lo),norm(mid.hi)+norm(mid.lo))
    return q,256f0*eps(Float32)^2*(span+coordinate)*(1f0+16f0*eps(Float32))
end

@inline function proxy_parent_curvature(guard,parent,x,one)
    @inbounds begin
        a,b,c=x[guard.nodes[1,parent]],x[guard.nodes[2,parent]],x[guard.nodes[3,parent]]
        ab,ac,bc=x[guard.nodes[4,parent]],x[guard.nodes[5,parent]],x[guard.nodes[6,parent]]
    end
    qab,eab=proxy_edge_curvature(a,b,ab,one)
    qac,eac=proxy_edge_curvature(a,c,ac,one)
    qbc,ebc=proxy_edge_curvature(b,c,bc,one)
    return ProxyCurvature((qab,qac,qbc),max(eab,eac,ebc))
end

@inline function proxy_curvature_components(curvature::AbstractVector,parent)
    @inbounds q=curvature[parent]
    return q.q[1],q.q[2],q.q[3],q.error
end
@inline function proxy_curvature_components(curvature::NamedTuple,parent)
    @inbounds return curvature.ab[parent],curvature.ac[parent],curvature.bc[parent],curvature.error[parent]
end

@inline function proxy_curvature_term(q,a,b,one)
    high,low=two_prod(a,b,one)
    term=plus(scaled(q,high,one),scaled(q,low,one),one)
    magnitude=abs(high)+abs(low)
    return term,magnitude,magnitude*(norm(q.hi)+norm(q.lo))
end

@inline proxy_finite(x::Float32)=(reinterpret(UInt32,x)&UInt32(0x7f800000))!=UInt32(0x7f800000)
@inline proxy_finite(p::Position)=proxy_finite(p.hi[1])&&proxy_finite(p.hi[2])&&proxy_finite(p.hi[3])&&
    proxy_finite(p.lo[1])&&proxy_finite(p.lo[2])&&proxy_finite(p.lo[3])

@inline function proxy_facet_error(guard,face,curvature,one=1f0)
    @inbounds parent=guard.parent[face]
    qab,qac,qbc,roundoff=proxy_curvature_components(curvature,parent)
    proxy_finite(qab)&&proxy_finite(qac)&&proxy_finite(qbc)&&proxy_finite(roundoff) || return Inf32
    bound=0f0
    @inbounds for k in Int32(1):Int32(3)
        d=guard.edges[k,face]
        proxy_finite(d[1])&&proxy_finite(d[2])&&proxy_finite(d[3]) || return Inf32
        # Exact products of dyadic parameter differences, including low parts.
        ab,wab,sab=proxy_curvature_term(qab,d[1],d[2],one)
        ac,wac,sac=proxy_curvature_term(qac,d[1],d[3],one)
        bc,wbc,sbc=proxy_curvature_term(qbc,d[2],d[3],one)
        value=plus(plus(ab,ac,one),bc,one)
        error=(norm(value.hi)+norm(value.lo)+(wab+wac+wbc)*roundoff+512f0*eps(Float32)^2*(sab+sac+sbc))*(1f0+32f0*eps(Float32))
        bound=max(bound,error)
    end
    return (4f0/3f0)*bound*(1f0+16f0*eps(Float32))
end

function proxy_curvature!(guard,x,one,offset::Int32,count::Int32)
    localid=KI.get_global_id(Int32).x
    localid<=count || return nothing
    parent=localid+offset
    @inbounds begin
        q=proxy_parent_curvature(guard,parent,x,one[1])
        guard.curvature.ab[parent]=q.q[1]
        guard.curvature.ac[parent]=q.q[2]
        guard.curvature.bc[parent]=q.q[3]
        guard.curvature.error[parent]=q.error
    end
    return nothing
end

function proxy_error_bound!(guard,one,offset::Int32,count::Int32)
    localid=KI.get_global_id(Int32).x
    localid<=count || return nothing
    face=localid+offset
    @inbounds bound=proxy_facet_error(guard,face,guard.curvature,one[1])
    # NaNs cannot silently disappear inside max or an unsigned atomic.
    value=proxy_finite(bound) && bound>=0 ? bound : Inf32
    @inbounds guard.maximum[face+Int32(1)]=reinterpret(UInt32,value)
    return nothing
end

function proxy_reduce_error!(guard,offset::Int32,count::Int32)
    localid=KI.get_global_id(Int32).x
    lane=KI.get_local_id(Int32).x
    # Fixed 256-lane tree: one global atomic per block, not per facet.
    shared=KI.localmemory(UInt32,Val((256,)),Val(1))
    @inbounds shared[lane]=localid<=count ? guard.maximum[localid+offset+Int32(1)] : UInt32(0)
    stride=Int32(128)
    while stride>0
        KI.barrier()
        @inbounds if lane<=stride
            shared[lane]=max(shared[lane],shared[lane+stride])
        end
        stride>>=1
    end
    KI.barrier()
    @inbounds if lane==1
        Atomix.@atomic max(guard.maximum[1],shared[1])
    end
    return nothing
end

update_proxy_guard!(target,model,state)=nothing
function update_proxy_guard!(target,model::MappedFEMModel,state)
    guard=model.transfer.guard
    guard===nothing && return nothing
    for offset in 0:65536:guard.parents-1
        count=min(65536,guard.parents-offset)
        launch!(target,proxy_curvature!,(guard,state.x,model.transfer.one,Int32(offset),Int32(count)),count)
    end
    for offset in 0:65536:guard.faces-1
        count=min(65536,guard.faces-offset)
        launch!(target,proxy_error_bound!,(guard,model.transfer.one,Int32(offset),Int32(count)),count)
        launch!(target,proxy_reduce_error!,(guard,Int32(offset),Int32(count)),count)
    end
    return nothing
end

proxy_error_bound(model)=0f0
function proxy_error_bound(model::MappedFEMModel)
    guard=model.transfer.guard
    return guard===nothing ? 0f0 : reinterpret(Float32,only(Array(view(guard.maximum,1:1))))
end

