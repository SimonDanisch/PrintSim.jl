"""
    add_bond!(builder, deformable, body; position, stiffness=1e9)

Glue an affine material point to the nearest triangle on a deformable part's initial
surface. Three noncollinear points attach a rigid pad to its flexible leaf without
assigning PETG elasticity to the pad material. Mesh contact between this glued pair
is excluded; contacts with everything else remain. Returns the bond index.
"""
function add_bond!(b::ModelBuilder, part::Integer, body::Integer; position, stiffness = 1e9)
    1 <= part <= length(b.deformables) && 1 <= body <= length(b.bodies) || throw(ArgumentError("invalid bonded parts"))
    stiffness > 0 || throw(ArgumentError("bond stiffness must be positive"))
    piece, affine = b.deformables[part], b.bodies[body]
    nearest, weights, distance = (Int32(1), Int32(1), Int32(1)), zero(Vec3f), Inf
    p = Position(Vec3{Float64}(position))
    for face in b.meshes[piece.mesh].faces
        a, c, d = (piece.surface_nodes[v] for v in face)
        gap, _, w = triangle_gap(p, Position(piece.initial[a]), Position(piece.initial[c]), Position(piece.initial[d]), 0.0f0, 1.0f0, 1.0f0)
        if abs(gap) < distance
            nearest, weights, distance = (a, c, d), w, abs(gap)
        end
    end
    a, c, d = nearest
    point = piece.initial[a] + Float64(weights[2]) * (piece.initial[c] - piece.initial[a]) + Float64(weights[3]) * (piece.initial[d] - piece.initial[a])
    coordinate = Position(Vec3{Float64}(affine.axes' * (point - affine.center) / affine.scale))
    offset = sum(length(p.x) for p in b.deformables[1:(part - 1)]; init = 0)
    push!(b.bonds, BondPiece(Int32(body), nearest .+ Int32(offset), weights, coordinate, Float32(stiffness)))
    for mesh in affine.meshes
        exclude_contacts!(b, piece.mesh, mesh)
    end
    return length(b.bonds)
end

struct BondModel{I, T, V, P, F}
    body::I
    nodes::T
    weights::V
    coordinate::P
    stiffness::F
    bonds::Int
end
Adapt.@adapt_structure BondModel
function BondModel(backend, b)
    up(a) = upload(backend, a)
    return BondModel(up(Int32[p.body for p in b.bonds]), up(NTuple{3, Int32}[p.nodes for p in b.bonds]),
        up(Vec3f[p.weights for p in b.bonds]), up(Position[p.coordinate for p in b.bonds]),
        up(Float32[p.stiffness for p in b.bonds]), length(b.bonds))
end

struct BondSystem{M, V}
    model::M
    force::V
end
BondSystem(backend, b::ModelBuilder) = BondSystem(BondModel(backend, b), KI.zeros(backend, Vec3f, max(length(b.bonds), 1)))

@inline function bond_body_point(q, model, bond, one)
    @inbounds begin
        body, u = model.body[bond], model.coordinate[bond]
        p = affine_point(q, body, u.hi, one)
        offset = Int32(4) * (body - Int32(1))
        p = plus(p, scaled(q[offset + Int32(2)], u.lo[1], one), one)
        p = plus(p, scaled(q[offset + Int32(3)], u.lo[2], one), one)
        return plus(p, scaled(q[offset + Int32(4)], u.lo[3], one), one)
    end
end

@inline function bond_fem_point(x, model, bond, one)
    @inbounds begin
        a, b, c = model.nodes[bond]
        w = model.weights[bond]
        return plus(plus(x[a], scaled(difference(x[b], x[a], one), w[2], one), one), scaled(difference(x[c], x[a], one), w[3], one), one)
    end
end

@inline function bond_fem_motion(p, model, bond)
    @inbounds begin
        a, b, c = model.nodes[bond]
        w = model.weights[bond]
        return p[a] + w[2] * (p[b] - p[a]) + w[3] * (p[c] - p[a])
    end
end

@inline function bond_fem_weight(model, bond, node)
    @inbounds begin
        a, b, c = model.nodes[bond]
        w = model.weights[bond]
        return node == a ? 1 - w[2] - w[3] : node == b ? w[2] : node == c ? w[3] : 0.0f0
    end
end

function bond_linearize!(force, model, body, fem, one)
    bond = KI.get_global_id(Int32).x
    bond <= model.bonds || return nothing
    @inbounds force[bond] = -model.stiffness[bond] * (bond_body_point(body.q, model, bond, one[1]) - bond_fem_point(fem.x, model, bond, one[1]))
    return nothing
end

function bond_body_rhs!(r, inverse, force, bonds, model)
    i = KI.get_global_id(Int32).x
    i <= 4model.bodies || return nothing
    @inbounds begin
        body, column = (i - Int32(1)) ÷ Int32(4) + Int32(1), (i - Int32(1)) % Int32(4) + Int32(1)
        model.fixed[body] != 0x00 && return nothing
        value, diagonal = zero(Vec3f), preconditioner_inverse(inverse[i])
        for bond in Int32(1):Int32(bonds.bonds)
            bonds.body[bond] == body || continue
            w = affine_weight(approximate(bonds.coordinate[bond]), column)
            value += w * force[bond]
            diagonal += bonds.stiffness[bond] * w^2 * I3
        end
        r[i] += value
        inverse[i] = preconditioner_inverse(diagonal)
    end
    return nothing
end

function bond_fem_rhs!(r, inverse, force, bonds, model)
    i = KI.get_global_id(Int32).x
    i <= model.nodes || return nothing
    @inbounds begin
        model.fixed[i] != 0x00 && return nothing
        value, diagonal = zero(Vec3f), preconditioner_inverse(inverse[i])
        for bond in Int32(1):Int32(bonds.bonds)
            w = bond_fem_weight(bonds, bond, i)
            value -= w * force[bond]
            diagonal += bonds.stiffness[bond] * w^2 * I3
        end
        r[i] += value
        inverse[i] = preconditioner_inverse(diagonal)
    end
    return nothing
end

function bond_body_apply!(result, bp, fp, bonds, model)
    i = KI.get_global_id(Int32).x
    i <= 4model.bodies || return nothing
    @inbounds begin
        body, column = (i - Int32(1)) ÷ Int32(4) + Int32(1), (i - Int32(1)) % Int32(4) + Int32(1)
        model.fixed[body] != 0x00 && return nothing
        value = zero(Vec3f)
        for bond in Int32(1):Int32(bonds.bonds)
            bonds.body[bond] == body || continue
            u = approximate(bonds.coordinate[bond])
            delta = body_motion(bp, body, u) - bond_fem_motion(fp, bonds, bond)
            value += bonds.stiffness[bond] * affine_weight(u, column) * delta
        end
        result[i] += value
    end
    return nothing
end

function bond_fem_apply!(result, fp, bp, bonds, model)
    i = KI.get_global_id(Int32).x
    i <= model.nodes || return nothing
    @inbounds begin
        model.fixed[i] != 0x00 && return nothing
        value = zero(Vec3f)
        for bond in Int32(1):Int32(bonds.bonds)
            w = bond_fem_weight(bonds, bond, i)
            if w != 0
                delta = bond_fem_motion(fp, bonds, bond) - body_motion(bp, bonds.body[bond], approximate(bonds.coordinate[bond]))
                value += bonds.stiffness[bond] * w * delta
            end
        end
        result[i] += value
    end
    return nothing
end
