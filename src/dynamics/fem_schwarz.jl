# Connectivity-local block solves plus the global Galerkin correction.
# Architecture reference: StiffGIPC's MAS_Preconditioner assembles local
# subspace matrices from the current Hessian before applying multilevel local
# solves. This is an independent block-Jacobi Schwarz implementation, not a
# claim to reproduce MAS's full adaptive/contact aggregation hierarchy.
# No material constants or physical operator entries are changed.
const SCHWARZ_NODES = 16
const SCHWARZ_DOFS = 3SCHWARZ_NODES

struct FEMSchwarz{I, M, F, S, U}
    node_patch::I
    node_slot::I
    count::I
    nodes::M
    factor::F
    scale::S
    fallback::U
    patches::Int
end
Adapt.@adapt_structure FEMSchwarz

function fem_schwarz_partition(b::ModelBuilder)
    tets = NTuple{10,Int32}[]
    n = 0
    for part in b.deformables
        append!(tets, (t .+ Int32(n) for t in part.tets))
        n += length(part.x)
    end
    start, ids, _ = incidence(n, tets)
    patch, slot = zeros(Int32,n), zeros(Int32,n)
    groups = NTuple{SCHWARZ_NODES,Int32}[]
    counts = Int32[]
    # Deterministic connected breadth-first patches; isolated nodes form their
    # own patch. No spatial proximity can join disconnected printed components.
    for seed in 1:n
        patch[seed] != 0 && continue
        id = Int32(length(groups) + 1)
        group = MVector{SCHWARZ_NODES,Int32}(ntuple(_ -> Int32(0), SCHWARZ_NODES))
        count = 1
        group[1], patch[seed], slot[seed] = seed, id, 1
        cursor = 1
        while cursor <= count && count < SCHWARZ_NODES
            node = group[cursor]
            for at in start[node]:(start[node+1]-1)
                for other in tets[ids[at]]
                    count == SCHWARZ_NODES && break
                    patch[other] != 0 && continue
                    count += 1
                    group[count], patch[other], slot[other] = other, id, count
                end
                count == SCHWARZ_NODES && break
            end
            cursor += 1
        end
        push!(groups, Tuple(group)); push!(counts, Int32(count))
    end
    patches = length(groups)
    nodes = zeros(Int32, max(patches,1), SCHWARZ_NODES)
    for (p, group) in enumerate(groups), j in 1:SCHWARZ_NODES
        nodes[p,j] = group[j]
    end
    return (;node_patch=patch,node_slot=slot,count=counts,nodes,patches)
end

function FEMSchwarz(backend,b::ModelBuilder;scalar_type=Float32)
    tables=fem_schwarz_partition(b)
    patch,slot,counts,nodes,patches=tables.node_patch,tables.node_slot,tables.count,tables.nodes,tables.patches
    up(a) = upload(backend,a)
    # Patch index is contiguous: neighboring invocations read neighboring
    # entries during factorization and application, rather than striding 48².
    return FEMSchwarz(up(patch),up(slot),up(counts),up(nodes),
        up(zeros(scalar_type,max(patches,1),SCHWARZ_DOFS,SCHWARZ_DOFS)),
        up(ones(scalar_type,max(patches,1),SCHWARZ_DOFS)),up(zeros(UInt32,max(patches,1))),patches)
end

function schwarz_assemble_node!(s, inverse, gradients, model, node::Int32)
    @inbounds begin
        patch, localnode = s.node_patch[node], s.node_slot[node]
        row = Int32(3)*(localnode-Int32(1))
        diagonal = model.fixed[node] == 0x00 ? preconditioner_inverse(inverse[node]) : one(eltype(inverse))
        for i in Int32(1):Int32(3), j in Int32(1):Int32(3)
            s.factor[patch,row+i,row+j] = diagonal[i,j]
        end
        model.fixed[node] != 0x00 && return nothing
        # Each invocation owns three whole matrix rows: deterministic gathers,
        # with no contended floating-point scatter atomics.
        for at in model.start[node]:(model.start[node+Int32(1)]-Int32(1))
            e, a = model.incidence[at], model.corner[at]+Int32(1)
            for b in Int32(1):Int32(10)
                other = model.tets[b,e]
                (other == node || model.fixed[other] != 0x00 || s.node_patch[other] != patch) && continue
                col = Int32(3)*(s.node_slot[other]-Int32(1))
                block = zero(eltype(inverse))
                for k in Int32(1):Int32(7)
                    block += (model.stiffness[k,e]*gradients[k,a,e])*gradients[k,b,e]'
                end
                for i in Int32(1):Int32(3), j in Int32(1):Int32(3)
                    s.factor[patch,row+i,col+j] += block[i,j]
                end
            end
        end
    end
    return nothing
end

function schwarz_assemble!(s, inverse, gradients, model)
    node = KI.get_global_id(Int32).x
    node <= model.nodes && schwarz_assemble_node!(s,inverse,gradients,model,node)
    return nothing
end

function schwarz_factor_patch!(s, patch::Int32)
    @inbounds begin
        n = Int32(3)*s.count[patch]
        s.fallback[patch] = 0
        for i in Int32(1):n
            d = s.factor[patch,i,i]
            if !(isfinite(d) && d > 0)
                s.fallback[patch] = 1
                return nothing
            end
            s.scale[patch,i] = sqrt(d)
        end
        for j in Int32(1):n
            d = s.factor[patch,j,j]/s.scale[patch,j]^2
            for k in Int32(1):(j-Int32(1))
                d -= s.factor[patch,j,k]^2
            end
            # Regularize only the PRECONDITIONER's scaled pivot. The coupled
            # FEM/contact operator and material stiffness remain untouched.
            T=eltype(s.factor)
            pivot = sqrt(max(d,T===Float32 ? T(1f-6) : 16eps(T)))
            if !isfinite(pivot)
                s.fallback[patch] = 1
                return nothing
            end
            s.factor[patch,j,j] = pivot
            for i in (j+Int32(1)):n
                value = s.factor[patch,i,j]/s.scale[patch,i]/s.scale[patch,j]
                for k in Int32(1):(j-Int32(1))
                    value -= s.factor[patch,i,k]*s.factor[patch,j,k]
                end
                value /= pivot
                if !isfinite(value)
                    s.fallback[patch] = 1
                    return nothing
                end
                s.factor[patch,i,j] = value
            end
        end
    end
    return nothing
end

function schwarz_factor!(s)
    patch = KI.get_global_id(Int32).x
    patch <= s.patches && schwarz_factor_patch!(s,patch)
    return nothing
end

function schwarz_apply_patch!(z, r, s, model, patch::Int32)
    @inbounds begin
        s.fallback[patch] != 0 && return nothing # retain point-block Jacobi
        n = Int32(3)*s.count[patch]
        work = MVector{SCHWARZ_DOFS,eltype(s.factor)}(undef)
        for i in Int32(1):n
            node = s.nodes[patch,(i-Int32(1))÷Int32(3)+Int32(1)]
            component = (i-Int32(1))%Int32(3)+Int32(1)
            value = r[node][component]/s.scale[patch,i]
            for j in Int32(1):(i-Int32(1))
                value -= s.factor[patch,i,j]*work[j]
            end
            work[i] = value/s.factor[patch,i,i]
        end
        for i in n:-Int32(1):Int32(1)
            value = work[i]
            for j in (i+Int32(1)):n
                value -= s.factor[patch,j,i]*work[j]
            end
            work[i] = value/s.factor[patch,i,i]
        end
        for i in Int32(1):s.count[patch]
            node = s.nodes[patch,i]
            at = Int32(3)*(i-Int32(1))
            V=eltype(z)
            z[node] = model.fixed[node] != 0x00 ? zero(V) : V(
                work[at+1]/s.scale[patch,at+1], work[at+2]/s.scale[patch,at+2],
                work[at+3]/s.scale[patch,at+3])
        end
    end
    return nothing
end

function schwarz_apply!(z,r,s,model)
    patch = KI.get_global_id(Int32).x
    patch <= s.patches && schwarz_apply_patch!(z,r,s,model,patch)
    return nothing
end

fem_prepare_coarse!(target, s, coarse) = nothing

function fem_prepare_preconditioner!(target,s)
    fem_prepare_coarse!(target, s, s.coarse)
    s.schwarz === nothing && return nothing
    launch!(target,zero!,(s.schwarz.factor,),length(s.schwarz.factor))
    launch!(target,schwarz_assemble!,(s.schwarz,s.inverse,s.gradients,s.model),s.model.nodes)
    launch!(target,schwarz_factor!,(s.schwarz,),s.schwarz.patches)
    return nothing
end
