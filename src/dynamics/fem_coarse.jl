# A Galerkin coarse correction containing rigid, strain and cubic bending fields per part.
# This supplies the long bending modes that point Jacobi cannot carry through a thin leaf.
# Its rest matrix is factored in Float64 on the host once for the configured timestep; all
# restrictions, triangular solves and prolongations during a run stay on the device.

struct FEMCoarse{I, F, M}
    node_part::I
    node_first::I
    node_count::I
    mode_first::I
    mode_count::I
    factor::I
    rhs::I
    scratch::F
    basis::M
    parts::Int
    modes::Int
end
Adapt.@adapt_structure FEMCoarse

function FEMCoarse(backend, b, dt;scalar_type=Float32)
    host = FEMModel(Array, b)
    n = host.nodes
    # Consecutive lanes process consecutive nodes in restriction AND
    # prolongation. Keep nodes contiguous in Julia's column-major storage;
    # a (20,n) layout makes both kernels stride through 20 floats per lane.
    basis = zeros(Float64, max(n, 1), 20)
    node_part = zeros(Int32, n)
    firsts, counts, mode_first, mode_count, factors, rhs = (Int32[] for _ in 1:6)
    scratch = scalar_type[]
    offset, element, modes = 0, 0, 0
    for (part, piece) in enumerate(b.deformables)
        count = length(piece.x)
        center = sum(piece.x) / count
        scale = maximum(norm(p - center) for p in piece.x)
        u = [(p - center) / scale for p in piece.x]
        Q = Vector{Float64}[]
        for degree in 0:3, a in 0:degree, c in 0:(degree - a)
            d = degree - a - c
            v = Float64[piece.fixed[i] == 0x00 ? u[i][1]^a * u[i][2]^c * u[i][3]^d : 0 for i in 1:count]
            # Two passes retain affine modes even in very thin or sparsely layered parts.
            for _ in 1:2, q in Q
                v .-= dot(q, v) .* q
            end
            magnitude = norm(v)
            magnitude > 1e-10 * sqrt(count) && push!(Q, v / magnitude)
        end
        r = length(Q)
        push!(firsts, offset + 1); push!(counts, count)
        push!(mode_first, modes + 1); push!(mode_count, r)
        node_part[(offset + 1):(offset + count)] .= part
        if r > 0
            P = reduce(hcat, Q)
            basis[(offset + 1):(offset + count), 1:r] .= P
            K = zeros(Float64, 3r, 3r)
            masses = Float64.(host.mass[(offset + 1):(offset + count)])
            M = P' * (masses .* P) / dt^2
            for j in 1:r, k in 1:r, component in 1:3
                K[3j - 3 + component, 3k - 3 + component] = M[j, k]
            end
            J = zeros(Float64, 7, 3r)
            weighted = similar(J)
            derivatives = zeros(Float64, 3, r)
            for e in (element + 1):(element + 4length(piece.tets))
                fill!(derivatives, 0)
                first_node=host.tets[1,e]-offset
                for corner in 2:10
                    node = host.tets[corner, e] - offset
                    g = Vec3{Float64}(host.gradient_hi[corner, e]) + Vec3{Float64}(host.gradient_lo[corner, e])
                    @inbounds for mode in 1:r, component in 1:3
                        derivatives[component, mode] += (P[node,mode]-P[first_node,mode])*g[component]
                    end
                end
                fill!(J, 0)
                @inbounds for mode in 1:r
                    x, y, z = derivatives[1, mode], derivatives[2, mode], derivatives[3, mode]
                    first = 3mode - 2
                    J[1, first], J[2, first + 1], J[3, first + 2] = x, y, z
                    J[4, first], J[4, first + 1] = y / sqrt(2), x / sqrt(2)
                    J[5, first], J[5, first + 2] = z / sqrt(2), x / sqrt(2)
                    J[6, first + 1], J[6, first + 2] = z / sqrt(2), y / sqrt(2)
                    J[7, first], J[7, first + 1], J[7, first + 2] = x, y, z
                end
                @inbounds for column in 1:3r, constraint in 1:7
                    weighted[constraint, column] = Float64(host.stiffness[constraint, e]) * J[constraint, column]
                end
                mul!(K, transpose(J), weighted, 1.0, 1.0)
            end
            for bond in b.bonds
                a, c, d = bond.nodes
                offset < a <= offset + count || continue
                w = bond.weights
                v = P[a - offset, :] + Float64(w[2]) * (P[c - offset, :] - P[a - offset, :]) +
                    Float64(w[3]) * (P[d - offset, :] - P[a - offset, :])
                for j in 1:r, k in 1:r, component in 1:3
                    K[3j - 3 + component, 3k - 3 + component] += bond.stiffness * v[j] * v[k]
                end
            end
            L = cholesky(Symmetric(K)).L
            push!(factors, length(scratch) + 1)
            append!(scratch, vec(scalar_type.(L)))
            push!(rhs, length(scratch) + 1)
            append!(scratch, zeros(scalar_type, 3r))
        else
            push!(factors, 1); push!(rhs, 1)
        end
        modes += r
        offset += count
        element += 4length(piece.tets)
    end
    up(a) = upload(backend, a)
    return FEMCoarse(up(node_part), up(firsts), up(counts), up(mode_first), up(mode_count), up(factors), up(rhs),
        up(scratch), up(scalar_type.(basis)), length(b.deformables), modes)
end

function fem_coarse_restrict!(coarse, r)
    mode = KI.get_global_id(Int32).x
    mode <= coarse.modes || return nothing
    @inbounds begin
        part = Int32(1)
        while mode >= coarse.mode_first[part] + coarse.mode_count[part]
            part += Int32(1)
        end
        localmode = mode - coarse.mode_first[part] + Int32(1)
        value = zero(Vec3f)
        first, count = coarse.node_first[part], coarse.node_count[part]
        for node in first:(first + count - Int32(1))
            value += coarse.basis[node, localmode] * r[node]
        end
        at = coarse.rhs[part] + Int32(3) * (localmode - Int32(1))
        coarse.scratch[at], coarse.scratch[at + Int32(1)], coarse.scratch[at + Int32(2)] = value[1], value[2], value[3]
    end
    return nothing
end

function fem_coarse_solve!(coarse)
    part = KI.get_global_id(Int32).x
    part <= coarse.parts || return nothing
    @inbounds begin
        n = Int32(3) * coarse.mode_count[part]
        n == 0 && return nothing
        lower_solve!(coarse.scratch, coarse.factor[part], n, coarse.rhs[part])
        upper_solve!(coarse.scratch, coarse.factor[part], n, coarse.rhs[part])
    end
    return nothing
end

function fem_coarse_clear!(coarse)
    part = KI.get_global_id(Int32).x
    part <= coarse.parts || return nothing
    @inbounds for i in Int32(0):(Int32(3) * coarse.mode_count[part] - Int32(1))
        coarse.scratch[coarse.rhs[part] + i] = 0
    end
    return nothing
end

# Restriction must parallelize the nodes as well as the modes: the printed gears
# have millions of nodes, so assigning one invocation to an entire mode leaves
# almost the whole GPU idle and makes every CG preconditioner serial in size.
function fem_coarse_restrict_tiled!(coarse, r, tiles::Int32, partial=nothing)
    group = (KI.get_global_id(Int32).x - Int32(1)) ÷ Int32(256)
    li = KI.get_local_id(Int32).x
    mode, tile = group ÷ tiles + Int32(1), group % tiles
    mode <= coarse.modes || return nothing
    part = Int32(1)
    @inbounds while mode >= coarse.mode_first[part] + coarse.mode_count[part]
        part += Int32(1)
    end
    @inbounds first, count = coarse.node_first[part], coarse.node_count[part]
    # Uniform for the whole workgroup, before any barriers.
    if tile * Int32(256) >= count
        if partial !== nothing && li == 1
            @inbounds partial[group + Int32(1)] = zero(eltype(r))
        end
        return nothing
    end
    localmode = mode - coarse.mode_first[part] + Int32(1)
    node = first + tile * Int32(256) + li - Int32(1)
    sh = KI.localmemory(eltype(r), Val((256,)), Val(1))
    @inbounds sh[li] = node < first + count ? coarse.basis[node, localmode] * r[node] : zero(eltype(r))
    KI.barrier()
    @inbounds li <= 128 && (sh[li] += sh[li + 128])
    KI.barrier()
    @inbounds li <= 64 && (sh[li] += sh[li + 64])
    KI.barrier()
    @inbounds li <= 32 && (sh[li] += sh[li + 32])
    KI.barrier()
    @inbounds li <= 16 && (sh[li] += sh[li + 16])
    KI.barrier()
    @inbounds li <= 8 && (sh[li] += sh[li + 8])
    KI.barrier()
    @inbounds li <= 4 && (sh[li] += sh[li + 4])
    KI.barrier()
    @inbounds li <= 2 && (sh[li] += sh[li + 2])
    KI.barrier()
    @inbounds if li == 1
        value = sh[1] + sh[2]
        if partial === nothing
            at = coarse.rhs[part] + Int32(3) * (localmode - Int32(1))
            Atomix.@atomic :monotonic coarse.scratch[at] += value[1]
            Atomix.@atomic :monotonic coarse.scratch[at + Int32(1)] += value[2]
            Atomix.@atomic :monotonic coarse.scratch[at + Int32(2)] += value[3]
        else
            partial[group + Int32(1)] = value
        end
    end
    return nothing
end

function fem_coarse_prolong!(z, coarse, model)
    node = KI.get_global_id(Int32).x
    node <= model.nodes || return nothing
    @inbounds if model.fixed[node] == 0x00
        part = coarse.node_part[node]
        value = zero(eltype(z))
        for mode in Int32(1):coarse.mode_count[part]
            at = coarse.rhs[part] + Int32(3) * (mode - Int32(1))
            y = eltype(z)(coarse.scratch[at], coarse.scratch[at + Int32(1)], coarse.scratch[at + Int32(2)])
            value += coarse.basis[node, mode] * y
        end
        z[node] += value
    end
    return nothing
end

# FP64 restriction uses deterministic two-pass reduction, not Float64 atomics.
struct WideFEMCoarse{C,P}
    coarse::C
    partial::P
    tiles::Int32
end
Adapt.@adapt_structure WideFEMCoarse

function wide_fem_coarse(backend,coarse,nodes)
    tiles=Int32(cld(nodes,256))
    return WideFEMCoarse(coarse,upload(backend,zeros(Vec3d,Int(tiles)*coarse.modes)),tiles)
end

function fem_coarse_restrict_finish!(coarse,partial,tiles::Int32)
    mode=KI.get_global_id(Int32).x
    mode<=coarse.modes || return nothing
    part=Int32(1)
    @inbounds begin
        while mode>=coarse.mode_first[part]+coarse.mode_count[part]
            part+=Int32(1)
        end
        value=zero(eltype(partial))
        for tile in Int32(1):tiles
            value+=partial[(mode-Int32(1))*tiles+tile]
        end
        at=coarse.rhs[part]+Int32(3)*(mode-coarse.mode_first[part])
        coarse.scratch[at],coarse.scratch[at+Int32(1)],coarse.scratch[at+Int32(2)]=value
    end
    return nothing
end

function fem_apply_coarse!(target,wide::WideFEMCoarse,z,r,model)
    coarse,partial,tiles=wide.coarse,wide.partial,wide.tiles
    launch!(target,fem_coarse_restrict_tiled!,(coarse,r,tiles,partial),Int(tiles)*256*coarse.modes)
    launch!(target,fem_coarse_restrict_finish!,(coarse,partial,tiles),coarse.modes)
    launch!(target,fem_coarse_solve!,(coarse,),coarse.parts;group=1)
    launch!(target,fem_coarse_prolong!,(z,coarse,model),model.nodes)
    return nothing
end

fem_precondition!(target, s) = fem_precondition!(target, s, s.coarse)

function fem_precondition!(target, s, coarse)
    launch!(target, fem_precondition!, (s.z, s.r, s.inverse, s.model), s.model.nodes)
    s.schwarz === nothing || launch!(target,schwarz_apply!,
        (s.z,s.r,s.schwarz,s.model),s.schwarz.patches)
    fem_apply_coarse!(target,s.coarse,s.z,s.r,s.model)
    return nothing
end

function fem_apply_coarse!(target,coarse::FEMCoarse,z,r,model)
    tiles=Int32(cld(model.nodes,256))
    launch!(target,fem_coarse_clear!,(coarse,),coarse.parts)
    launch!(target,fem_coarse_restrict_tiled!,(coarse,r,tiles),Int(tiles)*256*coarse.modes)
    launch!(target,fem_coarse_solve!,(coarse,),coarse.parts;group=1)
    launch!(target,fem_coarse_prolong!,(z,coarse,model),model.nodes)
    return nothing
end
