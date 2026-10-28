# Keep contact stencils in collision-proxy space. Composition with W and W'
# retains all FE couplings without expanding every four-point contact to a
# potentially 24-node FE stencil.

function require_surface_maps(b::ModelBuilder,surface_maps)
    if surface_maps===nothing
        all(p->length(p.surface_nodes)==length(b.meshes[p.mesh].vertices),b.deformables) ||
            throw(ArgumentError("refined FE collision geometry requires surface_maps"))
    end
    return nothing
end

"Install validated proxy vertex maps after attachments have been built; FE volumes stay unchanged."
function install_surface_proxies!(b::ModelBuilder,proxies;maximum_error=1e-7)
    isfinite(maximum_error) && maximum_error>0 || throw(ArgumentError("invalid proxy error allowance"))
    maps=Dict{Int,Any}()
    replacements=Pair{Int,MeshPiece}[]
    for (part,proxy) in proxies
        1<=part<=length(b.deformables) || throw(ArgumentError("unknown FE part"))
        piece=b.deformables[part]
        old=b.meshes[piece.mesh]
        n=length(proxy.nodes)
        n==length(proxy.weights)==length(proxy.points)==length(proxy.initial) ||
            throw(ArgumentError("proxy point arrays differ"))
        isfinite(proxy.max_error) && 0<=proxy.max_error<=proxy.tolerance<=maximum_error ||
            throw(ArgumentError("proxy exceeds geometric error allowance"))
        all(f->length(unique(f))==3 && all(i->1<=i<=n,f),proxy.faces) ||
            throw(ArgumentError("invalid proxy triangle"))
        boundary=quadratic_boundary_faces(piece)
        length(proxy.parent)==length(proxy.parameters)==length(proxy.faces) ||
            throw(ArgumentError("proxy parameter arrays differ"))
        for t in eachindex(proxy.faces)
            p=proxy.parent[t]
            1<=p<=length(boundary) || throw(ArgumentError("unknown proxy parent face"))
            for (state,positions) in ((piece.x,proxy.points),(piece.initial,proxy.initial))
                quadratic_proxy_oriented(boundary[p],proxy.parameters[t],proxy.faces[t],positions,state) ||
                    throw(ArgumentError("folded or unresolved quadratic proxy facet"))
            end
        end
        closed_surface_topology(proxy.points,proxy.faces)==
            closed_surface_topology(piece.x,[f[1:3] for f in boundary]) ||
            throw(ArgumentError("proxy changed the FE boundary topology"))
        map=SurfaceEmbedding(Array,proxy.nodes,proxy.weights,length(piece.x))
        for (state,positions) in ((piece.x,proxy.points),(piece.initial,proxy.initial)),i in 1:n
            actual=surface_map_motion(map,i,state)
            # Coordinates are evaluated in Float64 on the host. This only
            # validates correspondence, not the independently audited envelope.
            norm(actual-positions[i])<=256eps(Float64)*max(norm(actual),1e-3) ||
                throw(ArgumentError("proxy does not belong to this FE rest/initial state"))
        end
        maps[piece.mesh]=proxy
        push!(replacements,piece.mesh=>MeshPiece(Vec3d.(proxy.initial),NTuple{3,Int32}.(proxy.faces),old.static,old.kinetic,old.stiffness))
    end
    # Transactional with respect to validation failure: do not install half an
    # assembly when a later proxy is incompatible.
    for (mesh,replacement) in replacements
        b.meshes[mesh]=replacement
    end
    return maps
end

surface_transfer_row(piece,::Nothing,i)=(ntuple(k->k==1 ? piece.surface_nodes[i] : Int32(0),SURFACE_ROW_WIDTH),
    ntuple(k->k==1 ? 1.0 : 0.0,SURFACE_ROW_WIDTH))
surface_transfer_row(piece,data,i)=(data.nodes[i],data.weights[i])
surface_transfer_count(piece,::Nothing)=length(piece.surface_nodes)
surface_transfer_count(piece,data)=length(data.nodes)

# Function barrier: maps may be heterogeneous in the assembly dictionary, but
# never dispatch dynamically for each of the millions of proxy vertices.
function append_surface_transfer!(rows,weights,point_row,piece,data,point_offset,node_offset,nvertices)
    n=surface_transfer_count(piece,data)
    n==nvertices || error("collision surface requires a matching FE embedding")
    data===nothing || length(data.weights)==n || throw(ArgumentError("surface embedding row counts differ"))
    for i in 1:n
        row,weight=surface_transfer_row(piece,data,i)
        n=length(weight)
        for k in 1:n
            iszero(weight[k]) || 1<=row[k]<=length(piece.x) ||
                throw(ArgumentError("surface node outside its owning FE part"))
        end
        push!(rows,ntuple(k->k>n || iszero(weight[k]) ? Int32(0) : Int32(row[k]+node_offset),SURFACE_ROW_WIDTH))
        push!(weights,ntuple(k->k>n ? 0.0 : Float64(weight[k]),SURFACE_ROW_WIDTH))
        point_row[point_offset+i]=length(rows)
    end
    return nothing
end

function surface_transfer_data(b::ModelBuilder,surface_maps)
    all(mesh->any(p->p.mesh==mesh,b.deformables),keys(surface_maps)) ||
        throw(ArgumentError("surface embedding supplied for a non-FE mesh"))
    offsets=cumsum([0;length.(getfield.(b.meshes,:vertices))])
    point_row=zeros(Int32,offsets[end])
    rows=NTuple{SURFACE_ROW_WIDTH,Int32}[]
    weights=NTuple{SURFACE_ROW_WIDTH,Float64}[]
    node_offset=0
    for piece in b.deformables
        data=get(surface_maps,piece.mesh,nothing)
        append_surface_transfer!(rows,weights,point_row,piece,data,offsets[piece.mesh],node_offset,
            length(b.meshes[piece.mesh].vertices))
        node_offset+=length(piece.x)
    end
    return (;embedding=SurfaceEmbedding(Array,rows,weights,node_offset),point_row)
end

struct FEMSurfaceTransfer{E,I,F,G}
    embedding::E
    point_row::I
    one::F
    guard::G
end
Adapt.@adapt_structure FEMSurfaceTransfer

function FEMSurfaceTransfer(backend,b::ModelBuilder,surface_maps)
    data=surface_transfer_data(b,surface_maps)
    map=data.embedding
    up(a)=upload(backend,a)
    device=SurfaceEmbedding(up(map.node),up(map.weight),up(map.first),up(map.row),up(map.slot),map.points,map.nodes)
    return FEMSurfaceTransfer(device,up(data.point_row),up(Float32[1]),QuadraticProxyGuard(backend,b,surface_maps))
end

struct MappedFEMModel{M,T}
    base::M
    transfer::T
end
Adapt.@adapt_structure MappedFEMModel
@inline function Base.getproperty(model::MappedFEMModel,name::Symbol)
    name===:base && return getfield(model,:base)
    name===:transfer && return getfield(model,:transfer)
    name===:surface_node && return getfield(model,:transfer).point_row
    return getproperty(getfield(model,:base),name)
end

struct ProxyContactModel{I,U}
    surface_node::I
    fixed::U
    nodes::Int
end
Adapt.@adapt_structure ProxyContactModel

struct MappedFEMContacts{I,W,M}
    incidence::I
    workspace::W
    model::M
end
@inline function Base.getproperty(contacts::MappedFEMContacts,name::Symbol)
    name===:incidence && return getfield(contacts,:incidence)
    name===:workspace && return getfield(contacts,:workspace)
    name===:model && return getfield(contacts,:model)
    return getproperty(getfield(contacts,:incidence),name)
end
compact!(target,contacts::MappedFEMContacts)=compact!(target,contacts.incidence)

function MappedFEMContacts(backend,entries,model::MappedFEMModel;vector_type=Vec3f)
    transfer=model.transfer
    n=transfer.embedding.points
    return MappedFEMContacts(SparseVertexIncidence(backend,entries,n),
        MappedSurfaceContact(backend,transfer.embedding;vector_type),
        ProxyContactModel(transfer.point_row,KI.zeros(backend,UInt8,max(n,1)),n))
end

@inline fem_surface_position(state,model,point)=@inbounds state.x[model.surface_node[point]]
@inline function fem_surface_position(state,model::MappedFEMModel,point)
    transfer=model.transfer
    @inbounds row=transfer.point_row[point]
    return surface_map_position(transfer.embedding,row,state.x,transfer.one[1])
end

@inline function fem_surface_slip(model,state,anchor,point,one)
    @inbounds node=model.surface_node[point]
    return node==0 ? zero(Vec3f) : state.x[node]-anchor[node]
end
@inline function fem_surface_slip(model::MappedFEMModel,state,anchor,point,one)
    transfer=model.transfer
    @inbounds row=transfer.point_row[point]
    row==0 && return zero(Vec3f)
    a=surface_map_position(transfer.embedding,row,state.x,one)
    b=surface_map_position(transfer.embedding,row,anchor,one)
    return a-b
end

function fem_contact_proxy_rhs!(force,diagonal,model,index,C,T,coupling,touches,faces,vertices::Int32)
    node=KI.get_global_id(Int32).x
    node<=model.nodes || return nothing
    value,block=zero(eltype(force)),zero(eltype(diagonal))
    @inbounds for at in Int32(1):index.count[node]
        coefficient,f,H=fem_contact_row(model,C,T,coupling,touches,faces,index.ids[at,node],node,vertices)
        value+=coefficient*f
        block+=coefficient^2*H
    end
    @inbounds force[node],diagonal[node]=value,block
    return nothing
end

fem_contact_index(contacts)=contacts
fem_contact_index(contacts::MappedFEMContacts)=vertex_incidence_work(contacts.incidence)
fem_contact_model(model,contacts)=model
fem_contact_model(model,contacts::MappedFEMContacts)=contacts.model

function world_fem_contact_rhs!(target,world)
    s,b,f=world.sheets,world.bodies,world.deformables
    contacts=world.fem_contacts
    if contacts isa MappedFEMContacts
        w=contacts.workspace
        launch!(target,fem_contact_proxy_rhs!,(w.force,w.diagonal,contacts.model,fem_contact_index(contacts),b.colliders,b.contacts,world.coupling,s.touches,s.faces,Int32(s.n)),w.embedding.points)
        launch!(target,surface_map_rhs_add!,(f.r,f.inverse,w.force,w.diagonal,w.embedding,f.model.fixed),f.model.nodes)
    else
        launch!(target,fem_contact_rhs!,(f.r,f.inverse,f.model,contacts,b.colliders,b.contacts,world.coupling,s.touches,s.faces,Int32(s.n)),f.model.nodes)
    end
    return nothing
end

function world_fem_contact_product!(target,world)
    s,b,f=world.sheets,world.bodies,world.deformables
    contacts=world.fem_contacts
    if contacts isa MappedFEMContacts
        w,model,index=contacts.workspace,contacts.model,fem_contact_index(contacts)
        launch!(target,surface_map_forward!,(w.direction,f.p,w.embedding),w.embedding.points)
        launch!(target,zero!,(w.force,),w.embedding.points)
        launch!(target,fem_contact_apply!,(w.force,w.direction,b.p,s.p,model,b.model,index,b.colliders,b.contacts,world.coupling,s.touches,s.faces,s.model,Int32(s.n)),w.embedding.points)
        launch!(target,surface_map_transpose_add!,(f.q,w.force,w.embedding,f.model.fixed),f.model.nodes)
        launch!(target,body_fem_apply!,(b.q,w.direction,model,b.model,b.contacts,b.colliders),4b.model.bodies)
        launch!(target,sheet_fem_apply!,(s.q,w.direction,model,world.coupling,b.colliders,s.touches,s.faces,s.model),s.n)
    else
        launch!(target,fem_contact_apply!,(f.q,f.p,b.p,s.p,f.model,b.model,contacts,b.colliders,b.contacts,world.coupling,s.touches,s.faces,s.model,Int32(s.n)),f.model.nodes)
        launch!(target,body_fem_apply!,(b.q,f.p,f.model,b.model,b.contacts,b.colliders),4b.model.bodies)
        launch!(target,sheet_fem_apply!,(s.q,f.p,f.model,world.coupling,b.colliders,s.touches,s.faces,s.model),s.n)
    end
    return nothing
end

function world_fem_contact_bound!(target,world)
    s,b,f=world.sheets,world.bodies,world.deformables
    contacts=world.fem_contacts
    model,index,delta,slip=f.model,contacts,f.q,f.p
    if contacts isa MappedFEMContacts
        w=contacts.workspace
        launch!(target,surface_map_forward!,(w.direction,f.q,w.embedding),w.embedding.points)
        launch!(target,surface_map_forward!,(w.force,f.p,w.embedding),w.embedding.points)
        model,index,delta,slip=contacts.model,fem_contact_index(contacts),w.direction,w.force
    end
    launch!(target,fem_step_bound!,(world.step_bound,delta,b.q,s.q,slip,b.p,s.p,model,b.model,index,
        b.colliders,b.contacts,world.coupling,s.touches,s.faces,s.model,world.sheet_base,Int32(s.n),b.one,s.radius,b.roughness),index.entries)
    return nothing
end
