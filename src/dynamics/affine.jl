# Affine bodies: surface x = t + q1 u1 + q2 u2 + q3 u3, twelve degrees of freedom.
# The three q columns have units of length, as does t. Principal mass coordinates make
# the generalized inertia diagonal without discarding the volume's moments of inertia.

"The mesh's vertices `(3, n)` and faces `(3, k)` on the device."
devicemesh(dev::Mantle.Device, piece::MeshPiece) =
    (Mantle.Buffer(dev, Matrix{Float64}(reinterpret(reshape, Float64, piece.vertices))),
     Mantle.Buffer(dev, Matrix{Int32}(reinterpret(reshape, Int32, piece.faces))))

"""
Volume, center of mass and central second moment per unit volume of an outward closed mesh,
from its volume moments on the device (`GPUMeshing.volumemoments`).
"""
function volume_properties(dev::Mantle.Device, piece::MeshPiece)
    V, F = devicemesh(dev, piece)
    m = GPUMeshing.volumemoments(dev, V, F)
    foreach(Mantle.free!, (V, F))
    volume = m.volume
    isfinite(volume) && volume > 0 || throw(ArgumentError("an affine body needs a closed mesh with outward faces and positive volume"))
    center = Vec3{Float64}(m.first / volume)
    covariance = Symmetric(Matrix(m.second) / volume - center * center')
    minimum(eigvals(covariance)) > 0 || throw(ArgumentError("the mesh has singular volume moments"))
    return volume, center, covariance
end

"The largest distance of a vertex of `pieces` from `center`, on the device."
function radius(dev::Mantle.Device, pieces, center)
    r = 0.0
    c = Mantle.Buffer(dev, Vector{Float64}(center))
    for piece in pieces
        V, F = devicemesh(dev, piece)
        r = max(r, sqrt(maximum(sum((Mantle.storage(V) .- Mantle.storage(c)) .^ 2; dims = 1))))
        foreach(Mantle.free!, (V, F))
    end
    Mantle.free!(c)
    return r
end

"""
    add_body!(dev, builder, mesh; density=1240, mass, young=2e9, velocity, spin, fixed=false, grip...)

A closed triangle mesh in meters as an affine body. Its volume integrals give the center of
mass and inertia. `young` penalizes loss of orthogonality; surface vertices are linear in
twelve degrees of freedom, so contact forces and their reactions use the same Jacobian.
Returns the body's index; its mesh remains in `builder.meshes`. The volume integrals run on
the device `dev`.
"""
function add_body!(dev::Mantle.Device, b::ModelBuilder, mesh; density = 1240.0, mass = nothing, young = 2e9,
        velocity = Vec3(0.0, 0, 0), spin = Vec3(0.0, 0, 0), fixed = false, kw...)
    density > 0 && young > 0 || throw(ArgumentError("density and Young's modulus must be positive"))
    # Validate before adding anything to the builder.
    tmp = ModelBuilder()
    add_mesh!(tmp, mesh; kw...)
    volume, center, covariance = volume_properties(dev, only(tmp.meshes))
    m = mass === nothing ? density * volume : Float64(mass)
    isfinite(m) && m > 0 || throw(ArgumentError("mass must be finite and positive"))
    eigenframe = eigen(covariance)
    axes = Mat{3, 3, Float64, 9}(eigenframe.vectors)
    det(axes) < 0 && (axes = Mat{3, 3, Float64, 9}(hcat(axes[:, 1], axes[:, 2], -axes[:, 3])))
    scale = radius(dev, tmp.meshes, center)
    masses = Vec{4, Float64}(m, (m / scale^2 .* eigenframe.values)...)
    mesh_id = length(b.meshes) + 1
    push!(b.meshes, only(tmp.meshes))
    push!(b.bodies, AffinePiece(mesh_id, [mesh_id], center, axes, scale, masses, young * volume,
        Vec3{Float64}(velocity), Vec3{Float64}(spin), fixed))
    return length(b.bodies)
end

"""
    add_body!(dev, builder, parts::Vector{<:NamedTuple}; young, mass, velocity, spin, fixed)

A pressed/glued rigid assembly, retaining a separate collider and friction material
for each component. Each entry has `mesh` and optionally `density`, `static`, `kinetic`
and `stiffness`. Mass and inertia include every component; keys, bearings and gears
that move relative to each other must remain separate bodies.
"""
function add_body!(dev::Mantle.Device, b::ModelBuilder, parts::AbstractVector{<:NamedTuple}; young = 2e9, mass = nothing,
        velocity = Vec3(0.0, 0, 0), spin = Vec3(0.0, 0, 0), fixed = false)
    isempty(parts) && throw(ArgumentError("a compound body needs components"))
    young > 0 || throw(ArgumentError("Young's modulus must be positive"))
    tmp = ModelBuilder()
    masses, volumes, centers, moments = Float64[], Float64[], Vec3{Float64}[], Matrix{Float64}[]
    for part in parts
        density = get(part, :density, 1240.0)
        density > 0 || throw(ArgumentError("component density must be positive"))
        static = get(part, :static, 0.3)
        mesh = add_mesh!(tmp, part.mesh; static, kinetic = get(part, :kinetic, static), stiffness = get(part, :stiffness, 2e10))
        volume, center, covariance = volume_properties(dev, tmp.meshes[mesh])
        push!(masses, density * volume); push!(volumes, volume); push!(centers, center); push!(moments, Matrix(covariance))
    end
    total = sum(masses)
    center = sum(masses[i] * centers[i] for i in eachindex(parts)) / total
    covariance = sum(masses[i] * (moments[i] + (centers[i] - center) * (centers[i] - center)') for i in eachindex(parts)) / total
    m = mass === nothing ? total : Float64(mass)
    isfinite(m) && m > 0 || throw(ArgumentError("mass must be finite and positive"))
    eigenframe = eigen(Symmetric(covariance))
    axes = Mat{3, 3, Float64, 9}(eigenframe.vectors)
    det(axes) < 0 && (axes = Mat{3, 3, Float64, 9}(hcat(axes[:, 1], axes[:, 2], -axes[:, 3])))
    scale = radius(dev, tmp.meshes, center)
    generalized = Vec{4, Float64}(m, (m / scale^2 .* eigenframe.values)...)
    ids = collect((length(b.meshes) + 1):(length(b.meshes) + length(parts)))
    append!(b.meshes, tmp.meshes)
    push!(b.bodies, AffinePiece(first(ids), ids, center, axes, scale, generalized, young * sum(volumes), Vec3{Float64}(velocity), Vec3{Float64}(spin), fixed))
    return length(b.bodies)
end

"On-device affine rest data, including the linear map from collider vertices to body DOFs."
struct AffineModel{I, V, F, U, R}
    mesh_body::I
    dynamic_mesh::U
    coordinates::V
    normals::V
    mass::F
    scale::F
    scale_squared::R
    orthogonal::F
    fixed::U
    bodies::Int
end
Adapt.@adapt_structure AffineModel

function AffineModel(backend, b::ModelBuilder)
    mesh_body = zeros(Int32, length(b.meshes))
    for (k, body) in enumerate(b.bodies)
        for mesh in body.meshes
            mesh_body[mesh] = k
        end
    end
    dynamic_mesh = UInt8[m == 0 ? 0 : 1 for m in mesh_body]
    for piece in b.deformables
        dynamic_mesh[piece.mesh] = 0x02
    end
    local_coordinates = Vec3f[]
    local_normals = Vec3f[]
    for (m, piece) in enumerate(b.meshes)
        body = mesh_body[m]
        normals = vertex_normals(piece)
        for (i, p) in enumerate(piece.vertices)
            u = body == 0 ? zero(Vec3f) : Vec3f(b.bodies[body].axes' * (p - b.bodies[body].center) / b.bodies[body].scale)
            push!(local_coordinates, u)
            push!(local_normals, body == 0 ? normals[i] : Vec3f(b.bodies[body].axes' * normals[i]))
        end
    end
    up(a) = upload(backend, a)
    return AffineModel(up(mesh_body), up(dynamic_mesh), up(local_coordinates), up(local_normals), up(Float32[x for body in b.bodies for x in body.mass]),
        up(Float32[body.scale for body in b.bodies]), up([float_pair(body.scale^2) for body in b.bodies]), up(Float32[body.orthogonal for body in b.bodies]),
        up(UInt8[body.fixed for body in b.bodies]), length(b.bodies))
end

struct AffineState{P}
    q::P
    v::P
end
Adapt.@adapt_structure AffineState

function AffineState(backend, b::ModelBuilder)
    q, v = Position[], Position[]
    for body in b.bodies
        push!(q, Position(body.center))
        push!(v, Position(body.velocity))
        for j in 1:3
            column = body.scale * body.axes[:, j]
            push!(q, Position(column))
            push!(v, Position(cross(body.spin, column)))
        end
    end
    return AffineState(upload(backend, q), upload(backend, v))
end

"The scalar coefficient of affine column `j` (1: translation)."
@inline affine_weight(u, j) = j == 1 ? 1.0f0 : u[j - Int32(1)]

@inline function affine_point(q, body, u, one)
    o = Int32(4) * (body - Int32(1))
    p = q[o + Int32(1)]
    p = moved(p, scaled(q[o + Int32(2)], u[1], one), one)
    p = moved(p, scaled(q[o + Int32(3)], u[2], one), one)
    return moved(p, scaled(q[o + Int32(4)], u[3], one), one)
end

function affine_surface!(C, state, model, one, accepted = nothing)
    accepted === nothing || (@inbounds accepted[1] == 0 || return nothing)
    i = KI.get_global_id(Int32).x
    i <= length(C.point_owner) || return nothing
    @inbounds begin
        body = model.mesh_body[C.point_owner[i]]
        if body != 0
            C.points[i] = affine_point(state.q, body, model.coordinates[i], one[1])
            o = Int32(4) * (body - Int32(1))
            A = Mat3f(hcat(approximate(state.q[o + Int32(2)]), approximate(state.q[o + Int32(3)]), approximate(state.q[o + Int32(4)])))
            C.point_normal[i] = direction(transpose(inv(A)) * model.normals[i])
        end
    end
    return nothing
end

function affine_predict!(state, anchor, inertia, model, one, gravity::Vec3f, dt::Float32)
    i = KI.get_global_id(Int32).x
    i <= 4model.bodies || return nothing
    @inbounds begin
        p = state.q[i]
        anchor[i] = p
        body = (i - Int32(1)) ÷ Int32(4) + Int32(1)
        if model.fixed[body] == 0x00
            start = moved(p, scaled(state.v[i], dt, one[1]), one[1])
            acceleration = (i - Int32(1)) % Int32(4) == 0 ? gravity : zero(Vec3f)
            inertia[i] = moved(start, dt^2 * acceleration, one[1])
            # Start Newton at the last admissible pose, not beyond a magnetic
            # potential barrier or inside a thin contact after a velocity prediction.
            # The inertial target still contains the full velocity and gravity.
            state.q[i] = p
        else
            inertia[i] = p
        end
    end
    return nothing
end

function affine_velocity!(state, anchor, model, one, dt::Float32)
    i = KI.get_global_id(Int32).x
    i <= 4model.bodies || return nothing
    @inbounds state.v[i] = quotient(difference(state.q[i], anchor[i], one[1]), dt, one[1])
    return nothing
end

"Orthogonality energy's force and positive semidefinite Gauss–Newton blocks, one body."
@inline function affine_dot(p::Position, q::Position, one)
    h, l = dot_pair(p.hi, q, one)
    a, b = dot_pair(p.lo, q, one)
    return two_sum(h, (l + a + b) * one, one)
end

function affine_elastic!(force, blocks, gradients, state, model, one)
    body = KI.get_global_id(Int32).x
    body <= model.bodies || return nothing
    @inbounds begin
        o = Int32(4) * (body - Int32(1))
        invscale = 1 / model.scale[body]
        a1 = approximate(state.q[o + Int32(2)]) * invscale
        a2 = approximate(state.q[o + Int32(3)]) * invscale
        a3 = approximate(state.q[o + Int32(4)]) * invscale
        p1, p2, p3 = state.q[o + Int32(2)], state.q[o + Int32(3)], state.q[o + Int32(4)]
        lh, ll = model.scale_squared[body]
        # No runtime tuple indexing: drivers have miscompiled it in private storage.
        for i in Int32(1):Int32(4)
            force[o + i] = zero(Vec3f)
            for j in Int32(1):Int32(4)
                blocks[j, o + i] = zero(Mat3f)
            end
        end
        stiffness = model.orthogonal[body]
        for constraint in Int32(1):Int32(6)
            g1, g2, g3 = zero(Vec3f), zero(Vec3f), zero(Vec3f)
            c = 0.0f0
            if constraint == 1
                h, l = affine_dot(p1, p1, one[1])
                c, g1 = ((h - lh) + (l - ll)) * invscale^2 / 2, invscale * a1
            elseif constraint == 2
                h, l = affine_dot(p2, p2, one[1])
                c, g2 = ((h - lh) + (l - ll)) * invscale^2 / 2, invscale * a2
            elseif constraint == 3
                h, l = affine_dot(p3, p3, one[1])
                c, g3 = ((h - lh) + (l - ll)) * invscale^2 / 2, invscale * a3
            elseif constraint == 4
                h, l = affine_dot(p1, p2, one[1])
                c, g1, g2 = (h + l) * invscale^2 / sqrt(2.0f0), invscale / sqrt(2.0f0) * a2, invscale / sqrt(2.0f0) * a1
            elseif constraint == 5
                h, l = affine_dot(p1, p3, one[1])
                c, g1, g3 = (h + l) * invscale^2 / sqrt(2.0f0), invscale / sqrt(2.0f0) * a3, invscale / sqrt(2.0f0) * a1
            else
                h, l = affine_dot(p2, p3, one[1])
                c, g2, g3 = (h + l) * invscale^2 / sqrt(2.0f0), invscale / sqrt(2.0f0) * a3, invscale / sqrt(2.0f0) * a2
            end
            for i in Int32(2):Int32(4)
                gi = ifelse(i == 2, g1, ifelse(i == 3, g2, g3))
                gradients[constraint, o + i] = gi
                force[o + i] -= stiffness * c * gi
                for j in Int32(2):Int32(4)
                    gj = ifelse(j == 2, g1, ifelse(j == 3, g2, g3))
                    blocks[j, o + i] += stiffness * (gi * gj')
                end
            end
        end
    end
    return nothing
end

"Contact points on collider vertices, with equal and opposite affine reactions."
struct BodyContacts{I, V, F, U, M, A, O, D}
    mesh::I
    triangle::I
    normal::V
    spring::V
    lagged::F
    sliding::U
    other::V
    barycentric::V
    tangent_stiffness::F
    force::V
    block::M
    area::A
    overflow::O
    incidence::D
end
Adapt.@adapt_structure BodyContacts

function BodyContacts(backend, b)
    n = sum(length(m.vertices) for m in b.meshes; init = 0)
    z(T) = KI.zeros(backend, T, SLOTS, max(n, 1))
    area = Float32[]
    for mesh in b.meshes
        a = zeros(Float32, length(mesh.vertices))
        for f in mesh.faces
            weight = norm(cross(mesh.vertices[f[2]] - mesh.vertices[f[1]], mesh.vertices[f[3]] - mesh.vertices[f[1]])) / 6
            for v in f
                a[v] += weight
            end
        end
        append!(area, a)
    end
    return BodyContacts(z(Int32), z(Int32), z(Vec3f), z(Vec3f), z(Float32), z(UInt8), z(Vec3f), z(Vec3f), z(Float32), z(Vec3f), z(Mat3f),
        upload(backend, area), KI.zeros(backend, UInt8, max(n, 1)), BodyIncidence(backend, SLOTS * n, length(b.bodies)))
end

function body_endpoints!(T, C, model)
    entry = KI.get_global_id(Int32).x
    entry <= T.incidence.entries || return nothing
    point, slot = (entry - Int32(1)) ÷ Int32(SLOTS) + Int32(1), (entry - Int32(1)) % Int32(SLOTS) + Int32(1)
    @inbounds begin
        mesh = T.mesh[slot, point]
        T.incidence.a[entry] = mesh == 0 ? Int32(0) : model.mesh_body[C.point_owner[point]]
        T.incidence.b[entry] = mesh == 0 ? Int32(0) : model.mesh_body[mesh]
    end
    return nothing
end

function index_body_contacts!(target, T, C, model)
    launch!(target, body_endpoints!, (T, C, model), T.incidence.entries)
    compact!(target, T.incidence)
    return nothing
end

"Signed distance to a closed mesh, using the adjacent faces' pseudonormal at edges and corners."
@inline function body_gap(C, triangle, p, one, radius = 0.0f0)
    a, b, c = corners(C, triangle)
    gap, normal, weights = triangle_gap(p, a, b, c, radius, 1.0f0, one)
    w, _, face = projection(a - p, b - p, c - p)
    if minimum(w) < 0
        d = minus(minus(difference(p, a, one), scaled(difference(b, a, one), weights[2], one), one),
            scaled(difference(c, a, one), weights[3], one), one)
        length = length_pair(d, one)
        outward = approximate(d) / max(length, 1.0f-30)
        @inbounds f = C.faces[triangle]
        pseudo = face
        if maximum(weights) > 1 - 1.0f-5
            vertex = ifelse(weights[1] >= weights[2] && weights[1] >= weights[3], f[1], ifelse(weights[2] >= weights[3], f[2], f[3]))
            @inbounds pseudo = C.point_normal[vertex]
        else
            edge = weights[3] < 1.0f-5 ? Int32(1) : weights[1] < 1.0f-5 ? Int32(2) : Int32(3)
            @inbounds other = C.other_face[edge, triangle]
            if other != 0
                oa, ob, oc = corners(C, other)
                pseudo += direction(cross(ob - oa, oc - oa))
            end
        end
        h, l = dot_pair(pseudo, d, one)
        return h + l >= 0 ? (stretch(d, (radius, 0.0f0), one), outward, weights) : (-length - radius, -outward, weights)
    end
    return gap, normal, weights
end

function find_body_contacts!(T, C, model, state, anchor, one, σ::Float32, reach = state.q,
        first::Int32=Int32(1),count::Int32=Int32(length(C.point_owner)))
    lane = KI.get_global_id(Int32).x
    lane<=count || return nothing
    i = first+lane-Int32(1)
    i <= length(C.point_owner) || return nothing
    @inbounds begin
        own = C.point_owner[i]
        body = model.mesh_body[own]
        point = C.points[i]
        # Preserve friction memory by mesh identity, irrespective of the new slot ordering.
        oldmesh = MVector{SLOTS, Int32}(undef)
        oldspring = MVector{SLOTS, Vec3f}(undef)
        oldnormal = MVector{SLOTS, Vec3f}(undef)
        oldlagged = MVector{SLOTS, Float32}(undef)
        oldsliding = MVector{SLOTS, UInt8}(undef)
        for slot in 1:SLOTS
            oldmesh[slot], oldspring[slot], oldnormal[slot] = T.mesh[slot, i], T.spring[slot, i], T.normal[slot, i]
            oldlagged[slot], oldsliding[slot] = T.lagged[slot, i], T.sliding[slot, i]
            T.mesh[slot, i] = 0
            T.lagged[slot, i] = 0
        end
        count = Int32(0)
        T.overflow[i] = 0x00
        for mesh in Int32(1):Int32(C.meshes)
            other = model.mesh_body[mesh]
            (mesh == own || C.excluded[own, mesh] != 0 || (model.dynamic_mesh[own] == 0x00 && model.dynamic_mesh[mesh] == 0x00) || (body != 0 && other == body)) && continue
            travel = body == 0 ? 0.0f0 : norm(point - affine_point(anchor, body, model.coordinates[i], one[1])) +
                norm(affine_point(reach, body, model.coordinates[i], one[1]) - affine_point(anchor, body, model.coordinates[i], one[1]))
            if other != 0
                o = Int32(4) * (other - Int32(1))
                for column in Int32(1):Int32(4)
                    travel += norm(state.q[o + column] - anchor[o + column])
                    travel += norm(reach[o + column] - anchor[o + column])
                end
            end
            radius = RANGE * σ + 2travel + 1.0f-4
            box_distance2(approximate(point), C.lo[C.root[mesh]], C.hi[C.root[mesh]]) > radius^2 && continue
            t1, d1, t2, d2 = near_triangles(C, mesh, point, radius)
            closest_gap = Inf32
            if t1 != 0
                closest_gap, _, _ = body_gap(C, t1, point, one[1])
            end
            first_normal = zero(Vec3f)
            for candidate in Int32(1):Int32(2)
                triangle, distance = ifelse(candidate == 1, t1, t2), ifelse(candidate == 1, d1, d2)
                triangle == 0 && continue
                gap, normal, _ = body_gap(C, triangle, point, one[1])
                gap >= RANGE * σ && continue
                # A local signed distance is meaningful at the closest feature,
                # not at an arbitrary second face of a concave solid. Across a gear root
                # the latter falsely reported 162 μm penetration beside an 11 μm contact.
                # A second face can support an outside corner, but cannot add a distant
                # negative half-space constraint through the solid's opposite flank.
                candidate == 2 && gap < 0 && (closest_gap >= 0 || distance > d1 + σ) && continue
                candidate == 2 && dot(first_normal, normal) > 0.9999f0 && continue
                # Coplanar end faces of two spur gears face the same way. They must not
                # replace the opposing tooth-face contact with a spurious axial repulsion.
                dot(C.point_normal[i], normal) >= -1.0f-5 && continue
                candidate == 1 && (first_normal = normal)
                count += Int32(1)
                if count > SLOTS
                    T.overflow[i] = 0x01
                    continue
                end
                T.mesh[count, i], T.triangle[count, i], T.normal[count, i] = mesh, triangle, normal
                T.spring[count, i], T.sliding[count, i] = zero(Vec3f), 0x00
                for old in 1:SLOTS
                    if oldmesh[old] == mesh && dot(oldnormal[old], normal) > 0.9063f0
                        T.spring[count, i], T.sliding[count, i], T.lagged[count, i] = oldspring[old], oldsliding[old], oldlagged[old]
                        break
                    end
                end
            end
        end
    end
    return nothing
end

"One contact's geometry, slip and material, evaluated at the current affine positions."
@inline body_deformation_slip(::StaticMeshes, C, i, triangle, weights, one) = zero(Vec3f)

@inline function body_contact(T, C, model, state, anchor, i, slot, one, σ, tangential, motion)
    @inbounds begin
        mesh = T.mesh[slot, i]
        triangle = T.triangle[slot, i]
        f = C.faces[triangle]
        gap, normal, weights = body_gap(C, triangle, C.points[i], one)
        T.barycentric[slot, i] = weights
        own = C.point_owner[i]
        body, other = model.mesh_body[own], model.mesh_body[mesh]
        u = model.coordinates[i]
        v = weights[1] * model.coordinates[f[1]] + weights[2] * model.coordinates[f[2]] + weights[3] * model.coordinates[f[3]]
        slip = zero(Vec3f)
        body != 0 && (slip += affine_point(state.q, body, u, one) - affine_point(anchor, body, u, one))
        other != 0 && (slip -= affine_point(state.q, other, v, one) - affine_point(anchor, other, v, one))
        slip += body_deformation_slip(motion, C, i, triangle, weights, one)
        Pt = I3 - normal * normal'
        # Symmetric material combination; both directed samples share the stiffness.
        k = min(C.grip[3, own], C.grip[3, mesh]) * T.area[i]
        μs, μk = min(C.grip[1, own], C.grip[1, mesh]), min(C.grip[2, own], C.grip[2, mesh])
        λ, κ = normal_force(gap, k, σ)
        gap >= RANGE * σ && (λ = κ = 0.0f0)
        return gap, normal, Pt, Pt * slip, v, λ, κ, μs, μk
    end
end

function body_contact_forces!(T, C, model, state, anchor, one, σ::Float32, tangential::Float32, update::Bool, motion = StaticMeshes())
    i = KI.get_global_id(Int32).x
    i <= length(C.point_owner) || return nothing
    @inbounds for slot in 1:SLOTS
        T.force[slot, i], T.block[slot, i] = zero(Vec3f), zero(Mat3f)
        if T.mesh[slot, i] != 0
            gap, n, Pt, slip, v, λ, κ, μs, μk = body_contact(T, C, model, state, anchor, i, slot, one[1], σ, tangential, motion)
            ξ = Pt * T.spring[slot, i]
            if update
                T.spring[slot, i], T.sliding[slot, i] = friction_state(ξ, T.sliding[slot, i], slip, λ, tangential * κ, μs, μk)
                T.lagged[slot, i] = λ
            else
                T.tangent_stiffness[slot, i] = tangential * κ
                ft, Ht = friction_force(Pt, ξ, T.sliding[slot, i], slip, T.lagged[slot, i], tangential * κ, μs, μk)
                T.force[slot, i], T.block[slot, i] = λ * n + ft, κ * (n * n') + Ht
                T.other[slot, i], T.normal[slot, i] = v, n
            end
        end
    end
    return nothing
end

@inline function body_coefficient(model, C, T, point, slot, body, column)
    @inbounds begin
        own = model.mesh_body[C.point_owner[point]]
        other = model.mesh_body[T.mesh[slot, point]]
        w = own == body ? affine_weight(model.coordinates[point], column) : 0.0f0
        return w - (other == body ? affine_weight(T.other[slot, point], column) : 0.0f0)
    end
end

@inline function body_motion(p, body, u)
    body == 0 && return zero(Vec3f)
    o = Int32(4) * (body - Int32(1))
    @inbounds return p[o + Int32(1)] + u[1] * p[o + Int32(2)] + u[2] * p[o + Int32(3)] + u[3] * p[o + Int32(4)]
end

function body_rhs!(r, inverse, state, inertia, elastic, model, C, T, dt::Float32)
    row = KI.get_global_id(Int32).x
    row <= 4model.bodies || return nothing
    @inbounds begin
        body, column = (row - Int32(1)) ÷ Int32(4) + Int32(1), (row - Int32(1)) % Int32(4) + Int32(1)
        mass = model.mass[row] / dt^2
        force = mass * (inertia[row] - state.q[row]) + elastic[row]
        diagonal = mass * I3
        for at in Int32(1):T.incidence.total[body]
            entry = T.incidence.ids[at, body]
            point, slot = (entry - Int32(1)) ÷ Int32(SLOTS) + Int32(1), (entry - Int32(1)) % Int32(SLOTS) + Int32(1)
            if T.mesh[slot, point] != 0
                w = body_coefficient(model, C, T, point, slot, body, column)
                if w != 0
                    force += w * T.force[slot, point]
                    diagonal += w^2 * T.block[slot, point]
                end
            end
        end
        r[row] = model.fixed[body] == 0x00 ? force : zero(Vec3f)
        inverse[row] = preconditioner_inverse(diagonal)
    end
    return nothing
end

function body_apply!(result, p, gradients, model, C, T, dt::Float32)
    row = KI.get_global_id(Int32).x
    row <= 4model.bodies || return nothing
    @inbounds begin
        body, column = (row - Int32(1)) ÷ Int32(4) + Int32(1), (row - Int32(1)) % Int32(4) + Int32(1)
        o = Int32(4) * (body - Int32(1))
        value = model.mass[row] / dt^2 * p[row]
        for constraint in 1:6
            mh, ml = 0.0f0, 0.0f0
            for j in Int32(2):Int32(4)
                h, l = affine_dot(Position(gradients[constraint, o + j]), Position(p[o + j]), 1.0f0)
                mh, error = two_sum(mh, h, 1.0f0)
                ml += error + l
            end
            motion = mh + ml
            value += model.orthogonal[body] * motion * gradients[constraint, row]
        end
        for at in Int32(1):T.incidence.total[body]
            entry = T.incidence.ids[at, body]
            point, slot = (entry - Int32(1)) ÷ Int32(SLOTS) + Int32(1), (entry - Int32(1)) % Int32(SLOTS) + Int32(1)
            if T.mesh[slot, point] != 0
                w = body_coefficient(model, C, T, point, slot, body, column)
                if w != 0
                    a, b = model.mesh_body[C.point_owner[point]], model.mesh_body[T.mesh[slot, point]]
                    motion = body_motion(p, a, model.coordinates[point]) - body_motion(p, b, T.other[slot, point])
                    value += w * (T.block[slot, point] * motion)
                end
            end
        end
        result[row] = model.fixed[body] == 0x00 ? value : p[row]
    end
    return nothing
end

"Woodbury's dense body solve: diagonal contact blocks plus all six orthogonality rows."
function body_precondition!(z, r, inverse, gradients, model)
    body = KI.get_global_id(Int32).x
    body <= model.bodies || return nothing
    @inbounds begin
        o = Int32(4) * (body - Int32(1))
        L = MVector{36, Float32}(undef)
        y = MVector{6, Float32}(undef)
        for i in 1:6
            rhs = 0.0f0
            for column in Int32(2):Int32(4)
                rhs += dot(gradients[i, o + column], inverse[o + column] * r[o + column])
            end
            y[i] = rhs
            for j in 1:6
                value = i == j ? 1 / model.orthogonal[body] : 0.0f0
                for column in Int32(2):Int32(4)
                    value += dot(gradients[i, o + column], inverse[o + column] * gradients[j, o + column])
                end
                L[(i - 1) * 6 + j] = value
            end
        end
        for i in 1:6
            for j in 1:i
                value = L[(i - 1) * 6 + j]
                for k in 1:(j - 1)
                    value -= L[(i - 1) * 6 + k] * L[(j - 1) * 6 + k]
                end
                L[(i - 1) * 6 + j] = i == j ? sqrt(max(value, 1.0f-30)) : value / L[(j - 1) * 6 + j]
            end
        end
        for i in 1:6
            value = y[i]
            for j in 1:(i - 1)
                value -= L[(i - 1) * 6 + j] * y[j]
            end
            y[i] = value / L[(i - 1) * 6 + i]
        end
        for i in 6:-1:1
            value = y[i]
            for j in (i + 1):6
                value -= L[(j - 1) * 6 + i] * y[j]
            end
            y[i] = value / L[(i - 1) * 6 + i]
        end
        for column in Int32(1):Int32(4)
            force = r[o + column]
            for i in 1:6
                force -= gradients[i, o + column] * y[i]
            end
            z[o + column] = model.fixed[body] == 0x00 ? inverse[o + column] * force : zero(Vec3f)
        end
    end
    return nothing
end

struct BodySolver{B, M, C, T, P, V, H, G, D, F, J, R, MM, MS, U}
    backend::B
    model::M
    colliders::C
    contacts::T
    anchor::P
    inertia::P
    elastic::V
    blocks::H
    gradients::G
    inverse::D
    r::V
    z::V
    p::V
    q::V
    scalars::F
    one::F
    joints::J
    joint_rows::R
    motors::MM
    motor_state::MS
    load_force::V
    load_torque::V
    base::P
    step_bound::U
    trial::P
    base_force::V
    energy::V
    energy_ones::V
    energy_sum::F
    accepted::U
    gravity::Vec3f
    roughness::Float32
    tangential::Float32
    newton::Int
    iterations::Int
end

function BodySolver(backend, b::ModelBuilder; roughness = 1e-6, tangential = 0.5, newton = 8, iterations = 24, allow_empty = false)
    isempty(b.bodies) && !allow_empty && throw(ArgumentError("BodySolver needs at least one affine body"))
    n = max(4length(b.bodies), 1)
    energy_count = max(length(b.bodies) + SLOTS * sum(length(m.vertices) for m in b.meshes; init = 0), 1)
    z(T, dims...) = KI.zeros(backend, T, dims...)
    return BodySolver(backend, AffineModel(backend, b), Colliders(backend, b), BodyContacts(backend, b),
        z(Position, n), z(Position, n), z(Vec3f, n), z(Mat3f, 4, n), z(Vec3f, 6, n), z(Mat3f, n),
        z(Vec3f, n), z(Vec3f, n), z(Vec3f, n), z(Vec3f, n), z(Float32, 2iterations + 2),
        upload(backend, Float32[1]), JointModel(backend, b), JointRows(backend, b), MotorModel(backend, b), MotorState(backend, b),
        z(Vec3f, max(length(b.bodies), 1)), z(Vec3f, max(length(b.bodies), 1)), z(Position, n), upload(backend, UInt32[reinterpret(UInt32, 1.0f0)]),
        z(Position, n), z(Vec3f, n), z(Vec3f, energy_count), upload(backend, [Vec3f(1, 0, 0)]), z(Float32, 4), KI.zeros(backend, UInt32, 1),
        b.gravity[], Float32(roughness), Float32(tangential), newton, iterations)
end

function prime!(s::BodySolver, state::AffineState;on_phase=nothing)
    s.colliders.meshes == 0 && return s
    n = length(s.colliders.point_owner)
    launch!(s.backend, affine_surface!, (s.colliders, state, s.model, s.one), n)
    on_phase===nothing || on_phase(:body_surface)
    refit!(s.backend, s.colliders)
    on_phase===nothing || on_phase(:body_refit)
    copyto!(s.anchor, state.q)
    launch_indexed_batches!(s.backend, find_body_contacts!, (s.contacts, s.colliders, s.model, state, s.anchor, s.one, s.roughness,state.q), n)
    on_phase===nothing || on_phase(:body_contact_search)
    index_body_contacts!(s.backend, s.contacts, s.colliders, s.model)
    on_phase===nothing || on_phase(:body_contact_incidence)
    launch!(s.backend, body_contact_forces!, (s.contacts, s.colliders, s.model, state, s.anchor, s.one, s.roughness, s.tangential, true), n)
    on_phase===nothing || on_phase(:body_contact_force)
    return s
end

function step!(target, s::BodySolver, state::AffineState, dt)
    dt = Float32(dt)
    n = 4s.model.bodies
    points = length(s.colliders.point_owner)
    groups = cld(n, 256) * 256
    launch!(target, motor_clock!, (s.motor_state, s.motors, dt), s.motors.motors)
    launch!(target, affine_predict!, (state, s.anchor, s.inertia, s.model, s.one, s.gravity, dt), n)
    launch!(target, affine_surface!, (s.colliders, state, s.model, s.one), points)
    refit!(target, s.colliders)
    for _ in 1:s.newton
        # The closest feature can change during a stiff tooth impact. A signed distance
        # to yesterday's local triangle is not a signed distance to a concave solid.
        launch_indexed_batches!(target, find_body_contacts!, (s.contacts, s.colliders, s.model, state, s.anchor, s.one, s.roughness, s.inertia), points)
        index_body_contacts!(target, s.contacts, s.colliders, s.model)
        launch!(target, copy_into!, (s.base, state.q), n)
        launch!(target, affine_elastic!, (s.elastic, s.blocks, s.gradients, state, s.model, s.one), s.model.bodies)
        launch!(target, body_contact_forces!, (s.contacts, s.colliders, s.model, state, s.anchor, s.one, s.roughness, s.tangential, false), points)
        launch!(target, body_rhs!, (s.r, s.inverse, state, s.inertia, s.elastic, s.model, s.colliders, s.contacts, dt), n)
        launch!(target, joint_linearize!, (s.joint_rows, s.joints, state, s.anchor, s.one, dt), s.joints.joints)
        launch!(target, joint_rhs!, (s.r, s.inverse, s.joint_rows, s.joints, s.model), s.joints.joints > 0 ? n : 0)
        launch!(target, motor_linearize!, (s.motor_state, s.motors, state), s.motors.motors)
        launch!(target, motor_rhs!, (s.r, s.inverse, s.motor_state, s.motors, s.model), s.motors.motors > 0 ? n : 0)
        launch!(target, body_load_rhs!, (s.r, s.load_force, s.load_torque, s.anchor, s.model), n)
        launch!(target, copy_into!, (s.base_force, s.r), n)
        body_energy!(target, s, state, dt, Int32(1))
        launch!(target, zero!, (s.scalars,), length(s.scalars))
        launch!(target, body_precondition!, (s.z, s.r, s.inverse, s.gradients, s.model), s.model.bodies)
        launch!(target, dot_into!, (s.scalars, s.r, s.z, Int32(n), Int32(1)), groups)
        launch!(target, cg_direction!, (s.p, s.z, s.scalars, Int32(0)), n)
        for k in 1:s.iterations
            launch!(target, body_apply!, (s.q, s.p, s.gradients, s.model, s.colliders, s.contacts, dt), n)
            launch!(target, joint_apply!, (s.q, s.p, s.joint_rows, s.joints, s.model), s.joints.joints > 0 ? n : 0)
            launch!(target, motor_apply!, (s.q, s.p, s.motor_state, s.motors, s.model), s.motors.motors > 0 ? n : 0)
            launch!(target, dot_into!, (s.scalars, s.p, s.q, Int32(n), Int32(2k)), groups)
            launch!(target, cg_step!, (state.q, s.r, s.p, s.q, s.scalars, s.one, Int32(k)), n)
            launch!(target, body_precondition!, (s.z, s.r, s.inverse, s.gradients, s.model), s.model.bodies)
            launch!(target, dot_into!, (s.scalars, s.r, s.z, Int32(n), Int32(2k + 1)), groups)
            launch!(target, cg_direction!, (s.p, s.z, s.scalars, Int32(k)), n)
        end
        launch!(target, reset_step_bound!, (s.step_bound,), 1)
        launch!(target, motor_step_bound!, (s.step_bound, state.q, s.base, s.motor_state, s.motors), s.motors.motors)
        launch!(target, body_step_bound!, (s.step_bound, state.q, s.base, s.anchor, s.model, s.contacts, s.colliders, s.one, s.roughness), points)
        launch!(target, bounded_step!, (state.q, s.base, s.step_bound, s.one), n)
        body_line_search!(target, s, state, dt)
        launch!(target, affine_surface!, (s.colliders, state, s.model, s.one), points)
        refit!(target, s.colliders)
    end
    launch!(target, affine_velocity!, (state, s.anchor, s.model, s.one, dt), n)
    launch!(target, motor_observe!, (s.motor_state, s.motors, state), s.motors.motors)
    launch!(target, body_contact_forces!, (s.contacts, s.colliders, s.model, state, s.anchor, s.one, s.roughness, s.tangential, true), points)
    return state
end

step!(s::BodySolver, state::AffineState, dt) = step!(s.backend, s, state, dt)

"The physical center and affine matrix (host diagnostics)."
function body_pose(state::AffineState, b::ModelBuilder, body::Integer)
    q = Array(state.q)
    precise(p) = Vec3{Float64}(p.hi) + Vec3{Float64}(p.lo)
    o = 4(body - 1)
    A = hcat((precise(q[o + j]) / b.bodies[body].scale for j in 2:4)...)
    return precise(q[o + 1]), A * b.bodies[body].axes'
end
