# The same vertex–triangle / feature contacts, now mapped to tetrahedral surface nodes.
# Each contact contributes one relative motion and equal/opposite forces to all participating
# sheet, affine and finite-element DOFs in the common matrix product.

struct DeformingMeshes{M, F, S, P}
    affine::M
    model::F
    state::S
    anchor::P
end
Adapt.@adapt_structure DeformingMeshes

@inline function fem_point_slip(motion, point, one)
    return fem_surface_slip(motion.model,motion.state,motion.anchor,point,one)
end

@inline function mesh_motion(motion::DeformingMeshes, C, triangle, p, slot, i, one)
    displacement = mesh_motion(motion.affine, C, triangle, p, slot, i, one)
    a, b, c = corners(C, triangle)
    _, _, w = triangle_gap(p, a, b, c, 0.0f0, 1.0f0, one)
    @inbounds begin
        motion.affine.coupling.vertex_barycentric[slot, i] = w
        f = C.faces[triangle]
        return displacement + w[1] * fem_point_slip(motion, f[1], one) +
            w[2] * fem_point_slip(motion, f[2], one) + w[3] * fem_point_slip(motion, f[3], one)
    end
end

@inline function feature_endpoints(C, kind, feature)
    kind == 0x01 && return feature, feature
    triangle = (feature - Int32(1)) ÷ Int32(3) + Int32(1)
    k = feature - Int32(3) * (triangle - Int32(1))
    @inbounds f = C.faces[triangle]
    return ifelse(k == 1, f[1], ifelse(k == 2, f[2], f[3])), ifelse(k == 1, f[2], ifelse(k == 2, f[3], f[1]))
end

@inline function face_motion(motion::DeformingMeshes, C, kind, feature, edge, xa, xb, xc, slot, t, one)
    displacement = face_motion(motion.affine, C, kind, feature, edge, xa, xb, xc, slot, t, one)
    a, b = feature_endpoints(C, kind, feature)
    s = 0.0f0
    @inbounds if kind != 0x01
        b0 = ifelse(edge == 0x01, xa, ifelse(edge == 0x02, xb, xc))
        b1 = ifelse(edge == 0x01, xb, ifelse(edge == 0x02, xc, xa))
        s, _, _ = closest_parameters(C.points[a] - b0, C.points[b] - b0, zero(Vec3f), b1 - b0)
    end
    @inbounds motion.affine.coupling.face_parameter[slot, t] = s
    return displacement + (1 - s) * fem_point_slip(motion, a, one) + s * fem_point_slip(motion, b, one)
end

@inline cache_mesh_contact!(motion::DeformingMeshes, slot, i, force, block) = cache_mesh_contact!(motion.affine, slot, i, force, block)

@inline function body_deformation_slip(motion::DeformingMeshes, C, i, triangle, weights, one)
    @inbounds f = C.faces[triangle]
    return fem_point_slip(motion, i, one) - weights[1] * fem_point_slip(motion, f[1], one) -
        weights[2] * fem_point_slip(motion, f[2], one) - weights[3] * fem_point_slip(motion, f[3], one)
end

function fem_surface!(C, state, model, accepted = nothing)
    accepted === nothing || (@inbounds accepted[1] == 0 || return nothing)
    point = KI.get_global_id(Int32).x
    point <= length(C.point_owner) || return nothing
    @inbounds begin
        node = model.surface_node[point]
        node != 0 && (C.points[point] = fem_surface_position(state,model,point))
    end
    return nothing
end

function fem_surface_normals!(C, model, accepted = nothing)
    accepted === nothing || (@inbounds accepted[1] == 0 || return nothing)
    point = KI.get_global_id(Int32).x
    point <= length(C.point_owner) || return nothing
    @inbounds if model.surface_node[point] != 0
        normal = zero(Vec3f)
        for at in C.point_tri_start[point]:(C.point_tri_start[point + Int32(1)] - Int32(1))
            f = C.faces[C.point_tri[at]]
            k = f[1] == point ? Int32(1) : f[2] == point ? Int32(2) : Int32(3)
            a = C.points[ifelse(k == 1, f[2], ifelse(k == 2, f[3], f[1]))] - C.points[point]
            b = C.points[ifelse(k == 1, f[3], ifelse(k == 2, f[1], f[2]))] - C.points[point]
            angle = acos(clamp(dot(a, b) / max(norm(a) * norm(b), 1.0f-30), -1.0f0, 1.0f0))
            normal += angle * direction(cross(a, b))
        end
        C.point_normal[point] = direction(normal)
    end
    return nothing
end

function fem_contact_endpoints!(index, model, C, T, touches, faces, vertices::Int32)
    entry = KI.get_global_id(Int32).x
    entry <= index.entries || return nothing
    @inbounds begin
        for k in 1:4
            index.keys[k, entry] = 0
        end
        if entry <= T.incidence.entries
            point, slot = (entry - Int32(1)) ÷ Int32(SLOTS) + Int32(1), (entry - Int32(1)) % Int32(SLOTS) + Int32(1)
            if T.mesh[slot, point] != 0
                index.keys[1, entry] = model.surface_node[point]
                f = C.faces[T.triangle[slot, point]]
                index.keys[2, entry], index.keys[3, entry], index.keys[4, entry] = model.surface_node[f[1]], model.surface_node[f[2]], model.surface_node[f[3]]
            end
        elseif entry <= T.incidence.entries + SLOTS * vertices
            id = entry - Int32(T.incidence.entries) - Int32(1)
            point, slot = id ÷ Int32(SLOTS) + Int32(1), id % Int32(SLOTS) + Int32(1)
            if touches.mesh[slot, point] != 0
                f = C.faces[touches.triangle[slot, point]]
                index.keys[1, entry], index.keys[2, entry], index.keys[3, entry] = model.surface_node[f[1]], model.surface_node[f[2]], model.surface_node[f[3]]
            end
        else
            id = entry - Int32(T.incidence.entries) - Int32(SLOTS) * vertices - Int32(1)
            triangle, slot = id ÷ Int32(FACE_SLOTS) + Int32(1), id % Int32(FACE_SLOTS) + Int32(1)
            kind = faces.kind[slot, triangle]
            if kind != 0x00
                a, b = feature_endpoints(C, kind, faces.feature[slot, triangle])
                index.keys[1, entry], index.keys[2, entry] = model.surface_node[a], model.surface_node[b]
            end
        end
    end
    return nothing
end

@inline function fem_triangle_coefficient(model, C, triangle, w, node)
    @inbounds f = C.faces[triangle]
    @inbounds return Float32(model.surface_node[f[1]] == node) * w[1] +
        Float32(model.surface_node[f[2]] == node) * w[2] + Float32(model.surface_node[f[3]] == node) * w[3]
end

@inline function fem_feature_coefficient(model, C, kind, feature, s, node)
    a, b = feature_endpoints(C, kind, feature)
    @inbounds return Float32(model.surface_node[a] == node) * (1 - s) + Float32(model.surface_node[b] == node) * s
end

@inline function fem_contact_row(model, C, T, coupling, touches, faces, entry, node, vertices)
    @inbounds if entry <= T.incidence.entries
        point, slot = (entry - Int32(1)) ÷ Int32(SLOTS) + Int32(1), (entry - Int32(1)) % Int32(SLOTS) + Int32(1)
        coefficient = Float32(model.surface_node[point] == node) - fem_triangle_coefficient(model, C, T.triangle[slot, point], T.barycentric[slot, point], node)
        return coefficient, T.force[slot, point], T.block[slot, point]
    elseif entry <= T.incidence.entries + SLOTS * vertices
        id = entry - Int32(T.incidence.entries) - Int32(1)
        point, slot = id ÷ Int32(SLOTS) + Int32(1), id % Int32(SLOTS) + Int32(1)
        coefficient = -fem_triangle_coefficient(model, C, touches.triangle[slot, point], coupling.vertex_barycentric[slot, point], node)
        return coefficient, coupling.vertex_force[slot, point], coupling.vertex_block[slot, point]
    else
        id = entry - Int32(T.incidence.entries) - Int32(SLOTS) * vertices - Int32(1)
        triangle, slot = id ÷ Int32(FACE_SLOTS) + Int32(1), id % Int32(FACE_SLOTS) + Int32(1)
        coefficient = -fem_feature_coefficient(model, C, faces.kind[slot, triangle], faces.feature[slot, triangle], coupling.face_parameter[slot, triangle], node)
        return coefficient, faces.force[slot, triangle], faces.block[slot, triangle]
    end
end

@inline fem_node_motion(p, node) = @inbounds node == 0 ? zero(eltype(p)) : p[node]

@inline function fem_triangle_motion(p, model, C, triangle, w)
    @inbounds f = C.faces[triangle]
    @inbounds return w[1] * fem_node_motion(p, model.surface_node[f[1]]) +
        w[2] * fem_node_motion(p, model.surface_node[f[2]]) + w[3] * fem_node_motion(p, model.surface_node[f[3]])
end

@inline function fem_feature_motion(p, model, C, kind, feature, s)
    a, b = feature_endpoints(C, kind, feature)
    @inbounds return (1 - s) * fem_node_motion(p, model.surface_node[a]) + s * fem_node_motion(p, model.surface_node[b])
end

@inline function fem_relative_motion(fp, bp, sp, model, affine, C, T, coupling, touches, faces, sheet, entry, vertices)
    @inbounds if entry <= T.incidence.entries
        point, slot = (entry - Int32(1)) ÷ Int32(SLOTS) + Int32(1), (entry - Int32(1)) % Int32(SLOTS) + Int32(1)
        a, b = affine.mesh_body[C.point_owner[point]], affine.mesh_body[T.mesh[slot, point]]
        return body_motion(bp, a, affine.coordinates[point]) - body_motion(bp, b, T.other[slot, point]) +
            fem_node_motion(fp, model.surface_node[point]) - fem_triangle_motion(fp, model, C, T.triangle[slot, point], T.barycentric[slot, point])
    elseif entry <= T.incidence.entries + SLOTS * vertices
        id = entry - Int32(T.incidence.entries) - Int32(1)
        point, slot = id ÷ Int32(SLOTS) + Int32(1), id % Int32(SLOTS) + Int32(1)
        body = affine.mesh_body[touches.mesh[slot, point]]
        return sp[point] - body_motion(bp, body, coupling.vertex_local[slot, point]) -
            fem_triangle_motion(fp, model, C, touches.triangle[slot, point], coupling.vertex_barycentric[slot, point])
    else
        id = entry - Int32(T.incidence.entries) - Int32(SLOTS) * vertices - Int32(1)
        triangle, slot = id ÷ Int32(FACE_SLOTS) + Int32(1), id % Int32(FACE_SLOTS) + Int32(1)
        kind, feature, w = faces.kind[slot, triangle], faces.feature[slot, triangle], faces.weights[slot, triangle]
        f = sheet.tris[triangle]
        body = affine.mesh_body[feature_mesh(C, kind, feature)]
        return w[1] * sp[f[1]] + w[2] * sp[f[2]] + w[3] * sp[f[3]] - body_motion(bp, body, coupling.face_local[slot, triangle]) -
            fem_feature_motion(fp, model, C, kind, feature, coupling.face_parameter[slot, triangle])
    end
end

function fem_contact_rhs!(r, inverse, model, index, C, T, coupling, touches, faces, vertices::Int32)
    node = KI.get_global_id(Int32).x
    node <= model.nodes || return nothing
    @inbounds if model.fixed[node] == 0x00
        force, diagonal = zero(eltype(r)), preconditioner_inverse(inverse[node])
        for at in Int32(1):index.count[node]
            coefficient, f, H = fem_contact_row(model, C, T, coupling, touches, faces, index.ids[at, node], node, vertices)
            force += coefficient * f
            diagonal += coefficient^2 * H
        end
        r[node] += force
        inverse[node] = preconditioner_inverse(diagonal)
    end
    return nothing
end

function fem_contact_apply!(result, fp, bp, sp, model, affine, index, C, T, coupling, touches, faces, sheet, vertices::Int32)
    node = KI.get_global_id(Int32).x
    node <= model.nodes || return nothing
    @inbounds if model.fixed[node] == 0x00
        value = zero(eltype(result))
        for at in Int32(1):index.count[node]
            entry = index.ids[at, node]
            coefficient, _, H = fem_contact_row(model, C, T, coupling, touches, faces, entry, node, vertices)
            value += coefficient * (H * fem_relative_motion(fp, bp, sp, model, affine, C, T, coupling, touches, faces, sheet, entry, vertices))
        end
        result[node] += value
    end
    return nothing
end

function body_fem_apply!(result, fp, model, affine, T, C)
    row = KI.get_global_id(Int32).x
    row <= 4affine.bodies || return nothing
    @inbounds begin
        body, column = (row - Int32(1)) ÷ Int32(4) + Int32(1), (row - Int32(1)) % Int32(4) + Int32(1)
        affine.fixed[body] != 0x00 && return nothing
        value = zero(eltype(fp))
        for at in Int32(1):T.incidence.total[body]
            entry = T.incidence.ids[at, body]
            point, slot = (entry - Int32(1)) ÷ Int32(SLOTS) + Int32(1), (entry - Int32(1)) % Int32(SLOTS) + Int32(1)
            coefficient = body_coefficient(affine, C, T, point, slot, body, column)
            relative = fem_node_motion(fp, model.surface_node[point]) - fem_triangle_motion(fp, model, C, T.triangle[slot, point], T.barycentric[slot, point])
            value += coefficient * (T.block[slot, point] * relative)
        end
        result[row] += value
    end
    return nothing
end

function sheet_fem_apply!(result, fp, model, coupling, C, touches, faces, sheet)
    i = KI.get_global_id(Int32).x
    i <= length(sheet.fixed) || return nothing
    @inbounds if sheet.fixed[i] == 0x00
        value = zero(eltype(fp))
        for slot in 1:SLOTS
            if touches.mesh[slot, i] != 0
                value -= coupling.vertex_block[slot, i] * fem_triangle_motion(fp, model, C, touches.triangle[slot, i], coupling.vertex_barycentric[slot, i])
            end
        end
        for at in sheet.vtri_start[i]:(sheet.vtri_start[i + Int32(1)] - Int32(1))
            t, corner = sheet.vtri[at], sheet.vtri_order[at] + Int32(1)
            for slot in 1:FACE_SLOTS
                kind = faces.kind[slot, t]
                if kind != 0x00
                    w = faces.weights[slot, t]
                    coefficient = ifelse(corner == 1, w[1], ifelse(corner == 2, w[2], w[3]))
                    value -= coefficient * (faces.block[slot, t] * fem_feature_motion(fp, model, C, kind, faces.feature[slot, t], coupling.face_parameter[slot, t]))
                end
            end
        end
        result[i] += value
    end
    return nothing
end

function step_vectors!(delta, slip, x, base, anchor, n::Int32)
    i = KI.get_global_id(Int32).x
    i <= n || return nothing
    @inbounds delta[i], slip[i] = x[i] - base[i], base[i] - anchor[i]
    return nothing
end

function fem_step_bound!(bound, fd, bd, sd, fs, bs, ss, model, affine, index, C, T, coupling, touches, faces, sheet, sheetbase,
        vertices::Int32, one, radius::Float32, σ::Float32)
    entry = KI.get_global_id(Int32).x
    entry <= index.entries || return nothing
    @inbounds begin
        (index.keys[1, entry] == 0 && index.keys[2, entry] == 0 && index.keys[3, entry] == 0 && index.keys[4, entry] == 0) && return nothing
        gap, normal, spring = 0.0f0, zero(Vec3f), zero(Vec3f)
        friction = false
        if entry <= T.incidence.entries
            point, slot = (entry - Int32(1)) ÷ Int32(SLOTS) + Int32(1), (entry - Int32(1)) % Int32(SLOTS) + Int32(1)
            gap, normal, _ = body_gap(C, T.triangle[slot, point], C.points[point], one[1])
            spring = T.spring[slot, point]
            own, other = C.point_owner[point], T.mesh[slot, point]
            friction = T.lagged[slot, point] > 0 &&
                (min(C.grip[1, own], C.grip[1, other]) > 0 || min(C.grip[2, own], C.grip[2, other]) > 0)
        elseif entry <= T.incidence.entries + SLOTS * vertices
            id = entry - Int32(T.incidence.entries) - Int32(1)
            point, slot = id ÷ Int32(SLOTS) + Int32(1), id % Int32(SLOTS) + Int32(1)
            gap, normal, _ = body_gap(C, touches.triangle[slot, point], sheetbase[point], one[1], radius)
            spring = touches.spring[slot, point]
            mesh = touches.mesh[slot, point]
            friction = touches.lagged[slot, point] > 0 && max(C.grip[1, mesh], C.grip[2, mesh]) > 0
        else
            id = entry - Int32(T.incidence.entries) - Int32(SLOTS) * vertices - Int32(1)
            t, slot = id ÷ Int32(FACE_SLOTS) + Int32(1), id % Int32(FACE_SLOTS) + Int32(1)
            triangle = sheet.tris[t]
            gap, normal, _, own = face_gap(C, faces.kind[slot, t], faces.feature[slot, t], faces.edge[slot, t], faces.side[slot, t],
                sheetbase[triangle[1]], sheetbase[triangle[2]], sheetbase[triangle[3]], radius, one[1])
            own || return nothing
            spring = faces.spring[slot, t]
            mesh = feature_mesh(C, faces.kind[slot, t], faces.feature[slot, t])
            friction = faces.lagged[slot, t] > 0 && max(C.grip[1, mesh], C.grip[2, mesh]) > 0
        end
        gap >= RANGE * σ && return nothing
        delta = fem_relative_motion(fd, bd, sd, model, affine, C, T, coupling, touches, faces, sheet, entry, vertices)
        slip = fem_relative_motion(fs, bs, ss, model, affine, C, T, coupling, touches, faces, sheet, entry, vertices)
        stretch = spring + slip
        stretch -= dot(stretch, normal) * normal
        friction || (stretch = zero(Vec3f))
        cap = contact_step_bound(gap, normal, stretch, delta, σ)
        Atomix.@atomic min(bound[1], reinterpret(UInt32, cap))
    end
    return nothing
end
