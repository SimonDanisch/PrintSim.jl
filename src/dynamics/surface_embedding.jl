# High-Order IPC (Ferguson et al., SIGGRAPH 2023): u_surface = W*u_FE,
# f_FE = W'*f_surface, H_FE*p = W'*(H_surface*(W*p)).
# Architecture reference: PolyFEM cc1eac01e10a71dda7f743c8536e5d0e0dfe4653,
# mesh/collision_proxy/CollisionProxy.cpp (MIT). Surface parameter coordinates
# are retained, avoiding the arbitrary-proxy closest-point correspondence step.

"P2 triangular-face basis: corners a,b,c followed by edge nodes ab,ac,bc."
function quadratic_triangle_weights(bary)
    a,b,c=bary
    return (a*(2a-1),b*(2b-1),c*(2c-1),4a*b,4a*c,4b*c)
end

# Rows hold up to ten FE nodes: six for a point on a quadratic boundary face (P2 face
# basis), ten for a point embedded inside a quadratic tetrahedron (T10 basis).
const SURFACE_ROW_WIDTH = 10

"Fixed-width FE-node embedding of collision points and deterministic transpose incidence."
struct SurfaceEmbedding{I,M,F}
    node::M
    weight::F
    first::I
    row::I
    slot::I
    points::Int
    nodes::Int
end
Adapt.@adapt_structure SurfaceEmbedding

"Two Float32 limbs for dyadic P2 basis weights beyond one Float32's mantissa."
struct SurfaceWeight
    hi::Float32
    lo::Float32
end
Base.zero(::Type{SurfaceWeight})=SurfaceWeight(0f0,0f0)
Base.Float64(w::SurfaceWeight)=Float64(w.hi)+Float64(w.lo)
Base.convert(::Type{Float64},w::SurfaceWeight)=Float64(w)
function surface_weight(::Type{SurfaceWeight},value)
    hi=Float32(value)
    lo=Float32(Float64(value)-Float64(hi))
    result=SurfaceWeight(hi,lo)
    Float64(result)==Float64(value) || throw(ArgumentError("surface weight exceeds split-Float32 precision"))
    return result
end
surface_weight(::Type{Float32},value)=Float32(value)
@inline surface_weight_product(w,value)=w*value
@inline surface_weight_product(w::SurfaceWeight,value)=w.hi*value+w.lo*value
@inline surface_weight_square(w)=w^2
@inline surface_weight_square(w::SurfaceWeight)=w.hi*w.hi+2f0*w.hi*w.lo+w.lo*w.lo
@inline surface_weight_scaled(p,w,one)=scaled(p,w,one)
@inline surface_weight_scaled(p,w::SurfaceWeight,one)=plus(scaled(p,w.hi,one),scaled(p,w.lo,one),one)

function surface_weight_type(weights)
    for row in weights,value in row
        Float64(Float32(value))==Float64(value) || return SurfaceWeight
    end
    return Float32
end

function SurfaceEmbedding(backend,rows,weights,nodes::Integer)
    return build_surface_embedding(backend,rows,weights,nodes,surface_weight_type(weights))
end

function build_surface_embedding(backend,rows,weights,nodes::Integer,::Type{W}) where W
    length(rows)==length(weights) || throw(ArgumentError("surface embedding row counts differ"))
    0<=nodes<typemax(Int32) || throw(ArgumentError("invalid FE node count"))
    points=length(rows)
    width=SURFACE_ROW_WIDTH
    indices=zeros(Int32,points,width)
    values=zeros(W,points,width)
    counts=zeros(Int32,nodes)
    for point in eachindex(rows)
        length(rows[point])==length(weights[point])<=width || throw(ArgumentError("surface rows need at most $width entries"))
        row,weight=rows[point],Float64.(weights[point])
        # Preserve the SAME coefficients in geometry and the chain rule.
        # Deep dyadic samples use two limbs; never normalize rounded weights
        # into a different map just to make their sum pass validation.
        all(isfinite,weight) && sum(weight)==1 ||
            throw(ArgumentError("surface weights must form an exact partition of unity"))
        anchor=something(findfirst(!iszero,weight),1)
        n=length(weight)
        for k in 1:n
            weight[k]==0 && continue
            1<=row[k]<=nodes || throw(ArgumentError("surface node outside FE mesh"))
            # Do not capture row/weight in a closure: both are reassigned in
            # the outer loop, which otherwise boxes every tuple/index access.
            for previous in 1:k-1
                weight[previous]!=0 && row[previous]==row[k] &&
                    throw(ArgumentError("duplicate active node in surface row"))
            end
            counts[row[k]]+=1
        end
        # Slot one anchors position evaluation; mid-edge samples often have
        # zero corner weights, so move their first active basis here.
        for slot in 1:n
            k=slot==1 ? anchor : slot==anchor ? 1 : slot
            weight[k]==0 && continue
            indices[point,slot]=row[k]
            values[point,slot]=surface_weight(W,weight[k])
        end
    end
    total=sum(Int64,counts;init=Int64(0))
    total<typemax(Int32) || throw(ArgumentError("surface incidence exceeds Int32 capacity"))
    first=Vector{Int32}(undef,nodes+1)
    first[1]=1
    for i in 1:nodes
        first[i+1]=first[i]+counts[i]
    end
    cursor=copy(first[1:nodes])
    row,slot=Vector{Int32}(undef,total),Vector{Int32}(undef,total)
    for point in 1:points,k in 1:width
        node=indices[point,k]
        node==0 && continue
        at=cursor[node]
        row[at],slot[at]=point,k
        cursor[node]+=1
    end
    up(a)=upload(backend,a)
    return SurfaceEmbedding(up(indices),up(values),up(first),up(row),up(slot),points,Int(nodes))
end

@inline function surface_map_motion(map,point,p)
    value=zero(eltype(p))
    @inbounds for k in Int32(1):Int32(size(map.node,2))
        node=map.node[point,k]
        node==0 || (value+=surface_weight_product(map.weight[point,k],p[node]))
    end
    return value
end

@inline function surface_map_force(map,node,force)
    value=zero(eltype(force))
    @inbounds for at in map.first[node]:(map.first[node+Int32(1)]-Int32(1))
        row,slot=map.row[at],map.slot[at]
        value+=surface_weight_product(map.weight[row,slot],force[row])
    end
    return value
end

@inline function surface_map_position(map,point,x,one)
    @inbounds begin
        anchor=map.node[point,1]
        value=x[anchor]
        for k in Int32(2):Int32(size(map.node,2))
            node=map.node[point,k]
            node==0 && continue
            value=plus(value,surface_weight_scaled(difference(x[node],x[anchor],one),map.weight[point,k],one),one)
        end
    end
    return value
end

function surface_map_forward!(result,p,map)
    point=KI.get_global_id(Int32).x
    point<=map.points || return nothing
    @inbounds result[point]=surface_map_motion(map,point,p)
    return nothing
end

function surface_map_transpose!(result,force,map)
    node=KI.get_global_id(Int32).x
    node<=map.nodes || return nothing
    @inbounds result[node]=surface_map_force(map,node,force)
    return nothing
end

function surface_map_positions!(result,x,map,one)
    point=KI.get_global_id(Int32).x
    point<=map.points || return nothing
    @inbounds result[point]=surface_map_position(map,point,x,one[1])
    return nothing
end

"Proxy-space contact buffers; the FE Hessian remains the composition W' H W."
struct MappedSurfaceContact{E,V,M}
    embedding::E
    direction::V
    force::V
    diagonal::M
end
Adapt.@adapt_structure MappedSurfaceContact

function MappedSurfaceContact(backend,embedding;vector_type=Vec3f)
    n=max(embedding.points,1)
    matrix_type=Mat{3,3,eltype(vector_type),9}
    return MappedSurfaceContact(embedding,KI.zeros(backend,vector_type,n),
        KI.zeros(backend,vector_type,n),KI.zeros(backend,matrix_type,n))
end

function surface_map_transpose_add!(result,force,map,fixed)
    node=KI.get_global_id(Int32).x
    node<=map.nodes || return nothing
    @inbounds if fixed[node]==0x00
        result[node]+=surface_map_force(map,node,force)
    end
    return nothing
end

@inline function surface_map_diagonal(map,node,diagonal)
    block=zero(eltype(diagonal))
    @inbounds for at in map.first[node]:(map.first[node+Int32(1)]-Int32(1))
        row,slot=map.row[at],map.slot[at]
        block+=surface_weight_square(map.weight[row,slot])*diagonal[row]
    end
    return block
end

function surface_map_rhs_add!(r,inverse,force,diagonal,map,fixed)
    node=KI.get_global_id(Int32).x
    node<=map.nodes || return nothing
    @inbounds if fixed[node]==0x00
        r[node]+=surface_map_force(map,node,force)
        # An SPD diagonal PRECONDITIONER approximation only. The physical
        # product below retains every cross-proxy and cross-FE-node term.
        block=preconditioner_inverse(inverse[node])+surface_map_diagonal(map,node,diagonal)
        inverse[node]=preconditioner_inverse(block)
    end
    return nothing
end

"Apply the existing proxy contact operator without expanding each stencil to 24 FE nodes."
function mapped_surface_product!(target,result,p,workspace,fixed,apply_proxy!)
    map=workspace.embedding
    launch!(target,surface_map_forward!,(workspace.direction,p,map),map.points)
    apply_proxy!(target,workspace.force,workspace.direction)
    launch!(target,surface_map_transpose_add!,(result,workspace.force,map,fixed),map.nodes)
    return nothing
end
