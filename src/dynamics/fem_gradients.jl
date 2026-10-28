# Matrix-free storage of the existing seven invariant gradients. A quadrature
# point needs three deformation columns, not seven vectors for each of ten
# nodes. All consumers use the same getindex interface and arithmetic below.
struct CompactFEMGradients{F,G}
    deformation::F
    rest_gradient::G
end
Adapt.@adapt_structure CompactFEMGradients

"Native-FP64 view of the existing split rest gradients; no new geometry."
struct WideFEMRestGradients{H,L}
    hi::H
    lo::L
end
Adapt.@adapt_structure WideFEMRestGradients
@inline Base.getindex(g::WideFEMRestGradients,i::Integer,e::Integer)=
    (@inbounds Vec3d(g.hi[i,e])+Vec3d(g.lo[i,e]))
Base.Array(g::WideFEMRestGradients)=Vec3d.(Array(g.hi)).+Vec3d.(Array(g.lo))

@inline function fem_constraint_gradient(g,a1,a2,a3,constraint)
    root2=sqrt(eltype(g)(2))
    return constraint == 1 ? g[1] * a1 : constraint == 2 ? g[2] * a2 : constraint == 3 ? g[3] * a3 :
        constraint == 4 ? (g[1] * a2 + g[2] * a1) / root2 :
        constraint == 5 ? (g[1] * a3 + g[3] * a1) / root2 :
        constraint == 6 ? (g[2] * a3 + g[3] * a2) / root2 : g[1] * a1 + g[2] * a2 + g[3] * a3
end

@inline function Base.getindex(g::CompactFEMGradients,constraint::Integer,corner::Integer,e::Integer)
    @inbounds a1,a2,a3=g.deformation[1,e],g.deformation[2,e],g.deformation[3,e]
    if corner!=1
        @inbounds rest=g.rest_gradient[corner,e]
        return fem_constraint_gradient(rest,a1,a2,a3,constraint)
    end
    # Keep the dense kernel's summation order for translational invariance,
    # rather than substituting a separately rounded first shape gradient.
    first=zero(eltype(g.deformation))
    @inbounds for node in Int32(2):Int32(10)
        first-=fem_constraint_gradient(g.rest_gradient[node,e],a1,a2,a3,constraint)
    end
    return first
end

@inline store_fem_deformation!(gradients,a1,a2,a3,e)=nothing
@inline function store_fem_deformation!(g::CompactFEMGradients,a1,a2,a3,e)
    @inbounds g.deformation[1,e],g.deformation[2,e],g.deformation[3,e]=a1,a2,a3
    return nothing
end
@inline store_fem_gradient!(g,value,constraint,corner,e)=(@inbounds g[constraint,corner,e]=value;nothing)
@inline store_fem_gradient!(::CompactFEMGradients,value,constraint,corner,e)=nothing

@inline function fem_strain_coefficients(::Type{T},f1,f2,f3,one) where T
    h1,l1=affine_dot(f1,f1,one);h2,l2=affine_dot(f2,f2,one);h3,l3=affine_dot(f3,f3,one)
    h4,l4=affine_dot(f1,f2,one);h5,l5=affine_dot(f1,f3,one);h6,l6=affine_dot(f2,f3,one)
    root2=sqrt(T(2))
    return (((T(h1)-1)+T(l1))/2,((T(h2)-1)+T(l2))/2,((T(h3)-1)+T(l3))/2,
        (T(h4)+T(l4))/root2,(T(h5)+T(l5))/root2,(T(h6)+T(l6))/root2)
end

# Host materialization is diagnostic only; production products and Schwarz
# assembly reconstruct directly on the GPU without allocating this dense view.
Base.size(g::CompactFEMGradients)=(7,10,size(g.deformation,2))
Base.size(g::CompactFEMGradients,dim::Integer)=size(g)[dim]
function Base.Array(g::CompactFEMGradients)
    host=CompactFEMGradients(Array(g.deformation),Array(g.rest_gradient))
    return [host[k,a,e] for k in 1:7,a in 1:10,e in axes(host.deformation,2)]
end

# Only RHS/preconditioner assembly consumes the diagonal quadrature blocks.
# Reconstructing them once per Newton step avoids storing 10 dense 3x3 matrices
# per quadrature point; CG products still use the existing element-wise kernel.
struct ReconstructedFEMBlocks{T,G,K}
    gradients::G
    stiffness::K
end
function Adapt.adapt_structure(to,b::ReconstructedFEMBlocks{T}) where T
    gradients,stiffness=Adapt.adapt(to,b.gradients),Adapt.adapt(to,b.stiffness)
    return ReconstructedFEMBlocks{T,typeof(gradients),typeof(stiffness)}(gradients,stiffness)
end
Base.eltype(::Type{ReconstructedFEMBlocks{T,G,K}}) where {T,G,K}=T
Base.eltype(::ReconstructedFEMBlocks{T}) where {T}=T
Base.size(b::ReconstructedFEMBlocks)=(10,size(b.gradients,3))
@inline function Base.getindex(b::ReconstructedFEMBlocks{T},corner::Integer,e::Integer) where T
    value=zero(T)
    @inbounds for k in Int32(1):Int32(7)
        weighted=sqrt(eltype(T)(b.stiffness[k,e]))*b.gradients[k,corner,e]
        value+=weighted*weighted'
    end
    return value
end
function Base.Array(b::ReconstructedFEMBlocks{T}) where T
    gradients,stiffness=Array(b.gradients),Array(b.stiffness)
    host=ReconstructedFEMBlocks{T,typeof(gradients),typeof(stiffness)}(gradients,stiffness)
    return [host[a,e] for a in 1:10,e in axes(host.stiffness,2)]
end
@inline clear_fem_block!(b,corner,e)=(@inbounds b[corner,e]=zero(eltype(b));nothing)
@inline clear_fem_block!(::ReconstructedFEMBlocks,corner,e)=nothing
@inline function accumulate_fem_block!(b,g,k,corner,e)
    weighted=sqrt(eltype(eltype(b))(k))*g
    @inbounds b[corner,e]+=weighted*weighted'
    return nothing
end
@inline accumulate_fem_block!(::ReconstructedFEMBlocks,g,k,corner,e)=nothing
