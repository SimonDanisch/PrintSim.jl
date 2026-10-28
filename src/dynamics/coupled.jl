# One Newton/PCG system for sheets and affine bodies. The sheet multigrid hierarchy and
# dense body solves precondition their diagonal blocks; every contact cross block is applied
# in the common matrix product, and both subsystems share the same CG step length.

struct ContactCoupling{V, M, D, F}
    vertex_local::V
    vertex_force::V
    vertex_block::M
    face_local::V
    vertex_barycentric::V
    face_parameter::F
    incidence::D
end
Adapt.@adapt_structure ContactCoupling

function ContactCoupling(backend, b)
    v = KI.zeros(backend, Vec3f, SLOTS, max(length(b.x), 1))
    f = KI.zeros(backend, Vec3f, FACE_SLOTS, max(length(b.tris), 1))
    return ContactCoupling(v, KI.zeros(backend, Vec3f, size(v)...), KI.zeros(backend, Mat3f, size(v)...), f,
        KI.zeros(backend, Vec3f, size(v)...), KI.zeros(backend, Float32, size(f)...),
        BodyIncidence(backend, SLOTS * length(b.x) + FACE_SLOTS * length(b.tris), length(b.bodies)))
end

function coupled_endpoints!(coupling, model, C, touches, faces, vertices::Int32)
    entry = KI.get_global_id(Int32).x
    entry <= coupling.incidence.entries || return nothing
    @inbounds begin
        body = Int32(0)
        if entry <= SLOTS * vertices
            point, slot = (entry - Int32(1)) ÷ Int32(SLOTS) + Int32(1), (entry - Int32(1)) % Int32(SLOTS) + Int32(1)
            mesh = touches.mesh[slot, point]
            mesh != 0 && (body = model.mesh_body[mesh])
        else
            id = entry - Int32(SLOTS) * vertices - Int32(1)
            triangle, slot = id ÷ Int32(FACE_SLOTS) + Int32(1), id % Int32(FACE_SLOTS) + Int32(1)
            kind = faces.kind[slot, triangle]
            kind != 0x00 && (body = model.mesh_body[feature_mesh(C, kind, faces.feature[slot, triangle])])
        end
        coupling.incidence.a[entry], coupling.incidence.b[entry] = Int32(0), body
    end
    return nothing
end

struct MovingMeshes{M, S, P, C}
    model::M
    state::S
    anchor::P
    coupling::C
end
Adapt.@adapt_structure MovingMeshes

@inline function mesh_motion(motion::MovingMeshes, C, triangle, p, slot, i, one)
    @inbounds begin
        body = motion.model.mesh_body[C.owner[triangle]]
        u = zero(Vec3f)
        displacement = zero(Vec3f)
        if body != 0
            a, b, c = corners(C, triangle)
            _, _, w = triangle_gap(p, a, b, c, 0.0f0, 1.0f0, one)
            f = C.faces[triangle]
            u = w[1] * motion.model.coordinates[f[1]] + w[2] * motion.model.coordinates[f[2]] + w[3] * motion.model.coordinates[f[3]]
            displacement = affine_point(motion.state.q, body, u, one) - affine_point(motion.anchor, body, u, one)
        end
        motion.coupling.vertex_local[slot, i] = u
        return displacement
    end
end

@inline function face_motion(motion::MovingMeshes, C, kind, feature, edge, xa, xb, xc, slot, t, one)
    @inbounds begin
        body = motion.model.mesh_body[feature_mesh(C, kind, feature)]
        u, displacement = zero(Vec3f), zero(Vec3f)
        if body != 0
            if kind == 0x01
                u = motion.model.coordinates[feature]
            else
                triangle = (feature - Int32(1)) ÷ Int32(3) + Int32(1)
                k = feature - Int32(3) * (triangle - Int32(1))
                f = C.faces[triangle]
                ia = ifelse(k == 1, f[1], ifelse(k == 2, f[2], f[3]))
                ib = ifelse(k == 1, f[2], ifelse(k == 2, f[3], f[1]))
                b0 = ifelse(edge == 0x01, xa, ifelse(edge == 0x02, xb, xc))
                b1 = ifelse(edge == 0x01, xb, ifelse(edge == 0x02, xc, xa))
                s, _, _ = closest_parameters(C.points[ia] - b0, C.points[ib] - b0, zero(Vec3f), b1 - b0)
                u = (1 - s) * motion.model.coordinates[ia] + s * motion.model.coordinates[ib]
            end
            displacement = affine_point(motion.state.q, body, u, one) - affine_point(motion.anchor, body, u, one)
        end
        motion.coupling.face_local[slot, t] = u
        return displacement
    end
end

@inline function cache_mesh_contact!(motion::MovingMeshes, slot, i, force, block)
    @inbounds begin
        motion.coupling.vertex_force[slot, i] = force
        motion.coupling.vertex_block[slot, i] = block
    end
    return nothing
end

function coupled_body_rhs!(r, inverse, model, coupling, C, touches, faces, triangles::Int32, vertices::Int32)
    row = KI.get_global_id(Int32).x
    row <= 4model.bodies || return nothing
    @inbounds begin
        body, column = (row - Int32(1)) ÷ Int32(4) + Int32(1), (row - Int32(1)) % Int32(4) + Int32(1)
        model.fixed[body] != 0x00 && return nothing
        force, diagonal = zero(Vec3f), preconditioner_inverse(inverse[row])
        for at in Int32(1):coupling.incidence.total[body]
            entry = coupling.incidence.ids[at, body]
            if entry <= SLOTS * vertices
                i, slot = (entry - Int32(1)) ÷ Int32(SLOTS) + Int32(1), (entry - Int32(1)) % Int32(SLOTS) + Int32(1)
                w = affine_weight(coupling.vertex_local[slot, i], column)
                force -= w * coupling.vertex_force[slot, i]
                diagonal += w^2 * coupling.vertex_block[slot, i]
            else
                id = entry - Int32(SLOTS) * vertices - Int32(1)
                t, slot = id ÷ Int32(FACE_SLOTS) + Int32(1), id % Int32(FACE_SLOTS) + Int32(1)
                w = affine_weight(coupling.face_local[slot, t], column)
                force -= w * faces.force[slot, t]
                diagonal += w^2 * faces.block[slot, t]
            end
        end
        r[row] += force
        inverse[row] = preconditioner_inverse(diagonal)
    end
    return nothing
end

function coupled_body_apply!(result, bp, sp, model, coupling, C, touches, faces, tris, triangles::Int32, vertices::Int32)
    row = KI.get_global_id(Int32).x
    row <= 4model.bodies || return nothing
    @inbounds begin
        body, column = (row - Int32(1)) ÷ Int32(4) + Int32(1), (row - Int32(1)) % Int32(4) + Int32(1)
        model.fixed[body] != 0x00 && return nothing
        value = zero(Vec3f)
        for at in Int32(1):coupling.incidence.total[body]
            entry = coupling.incidence.ids[at, body]
            if entry <= SLOTS * vertices
                i, slot = (entry - Int32(1)) ÷ Int32(SLOTS) + Int32(1), (entry - Int32(1)) % Int32(SLOTS) + Int32(1)
                u = coupling.vertex_local[slot, i]
                w = affine_weight(u, column)
                value += w * (coupling.vertex_block[slot, i] * (body_motion(bp, body, u) - sp[i]))
            else
                id = entry - Int32(SLOTS) * vertices - Int32(1)
                t, slot = id ÷ Int32(FACE_SLOTS) + Int32(1), id % Int32(FACE_SLOTS) + Int32(1)
                u, w = coupling.face_local[slot, t], faces.weights[slot, t]
                f = tris[t]
                sheet_motion = w[1] * sp[f[1]] + w[2] * sp[f[2]] + w[3] * sp[f[3]]
                value += affine_weight(u, column) * (faces.block[slot, t] * (body_motion(bp, body, u) - sheet_motion))
            end
        end
        result[row] += value
    end
    return nothing
end

function coupled_sheet_apply!(result, bp, model, coupling, C, touches, faces, sheet)
    i = KI.get_global_id(Int32).x
    i <= length(sheet.fixed) || return nothing
    @inbounds if sheet.fixed[i] == 0x00
        value = zero(Vec3f)
        for slot in 1:SLOTS
            mesh = touches.mesh[slot, i]
            if mesh != 0
                body = model.mesh_body[mesh]
                value -= coupling.vertex_block[slot, i] * body_motion(bp, body, coupling.vertex_local[slot, i])
            end
        end
        for e in sheet.vtri_start[i]:(sheet.vtri_start[i + Int32(1)] - Int32(1))
            t, c = sheet.vtri[e], sheet.vtri_order[e] + Int32(1)
            for slot in 1:FACE_SLOTS
                kind = faces.kind[slot, t]
                if kind != 0x00
                    body = model.mesh_body[feature_mesh(C, kind, faces.feature[slot, t])]
                    w = faces.weights[slot, t]
                    wi = ifelse(c == 1, w[1], ifelse(c == 2, w[2], w[3]))
                    value -= wi * (faces.block[slot, t] * body_motion(bp, body, coupling.face_local[slot, t]))
                end
            end
        end
        result[i] += value
    end
    return nothing
end

"The evolving sheet vertices and affine DOFs of one coupled world."
struct WorldState{S, B, F}
    sheets::S
    bodies::B
    deformables::F
end
WorldState(backend, b::ModelBuilder) = WorldState(State(backend, b), AffineState(backend, b), FEMState(backend, b))

struct EmptySheets{M, T, F, V, P, C}
    model::M
    touches::T
    faces::F
    p::V
    q::V
    anchor::P
    inertia::P
    colliders::C
    n::Int
    triangles::Int
    radius::Float32
    roughness::Float32
    tangential::Float32
end
function EmptySheets(backend, b, colliders, roughness, tangential)
    z(T) = KI.zeros(backend, T, 1)
    return EmptySheets(Model(backend, b), Touches(backend, 1), FaceTouches(backend, 0), z(Vec3f), z(Vec3f),
        z(Position), z(Position), colliders, 0, 0, 0.5f-4, Float32(roughness), Float32(tangential))
end

struct WorldSolver{B, S, A, C, P, U, F, I, E, J, D, Q}
    backend::B
    sheets::S
    bodies::A
    coupling::C
    deformables::F
    fem_contacts::I
    sheet_base::P
    body_base::P
    fem_base::P
    step_bound::U
    energy::E
    bonds::J
    ccd::D
    ccd_pairs::Q
    cg_tolerance::Float32
    newton_tolerance::Float32
    newton_motion_limit::Float32
    newton::Int
    iterations::Int
end

function WorldSolver(backend, b::ModelBuilder; newton = 8, iterations = isempty(b.deformables) ? 48 : 400,
        roughness = 1e-6, tangential = 0.5, timestep = 1e-3, cg_tolerance = 1e-8, progress = nothing,
        continuous_collisions = false, elastic_schwarz = false, newton_tolerance = 0,
        newton_motion_limit=0,
        elastic_coarse_builder = FEMCoarse,surface_maps=nothing,
        ccd_pair_capacity=nothing,ccd_pair_batch=8192,ccd_query_tile=0,
        ccd_max_iterations=32768,ccd_primitive_parallel::Bool=false,compact_gradients=false,
        elastic_krylov_precision=:float32,matrix_free_blocks=false,kw...)
    require_surface_maps(b,surface_maps)
    ccd_query_tile isa Integer && 0<=ccd_query_tile<=typemax(Int32) ||
        throw(ArgumentError("invalid CCD query tile"))
    !ccd_primitive_parallel || (continuous_collisions && ccd_query_tile>0) ||
        throw(ArgumentError("primitive-parallel CCD requires a streaming collision queue"))
    ccd_max_iterations isa Integer && 0<ccd_max_iterations<=typemax(Int32) ||
        throw(ArgumentError("invalid CCD stream work bound"))
    0 < cg_tolerance < 1 || throw(ArgumentError("CG tolerance must lie between zero and one"))
    isfinite(newton_tolerance) && newton_tolerance>=0 || throw(ArgumentError("nonnegative finite Newton tolerance required"))
    isfinite(newton_motion_limit) && 0<=newton_motion_limit<=floatmax(Float32) ||
        throw(ArgumentError("nonnegative finite trial-motion limit required"))
    bodies = BodySolver(backend, b; newton, iterations, roughness, tangential, allow_empty = true)
    progress === nothing || progress(:affine_ready)
    sheets = isempty(b.sheets) ? EmptySheets(backend, b, bodies.colliders, roughness, tangential) :
        SheetSolver(backend, b; newton, iterations, roughness, tangential, colliders = bodies.colliders, kw...)
    progress === nothing || progress(:sheets_ready)
    fem = FEMSolver(backend, b; newton, iterations, timestep, progress,
        schwarz = elastic_schwarz, coarse_builder = elastic_coarse_builder,surface_maps,compact_gradients,
        krylov_precision=elastic_krylov_precision,matrix_free_blocks)
    entries = bodies.contacts.incidence.entries + SLOTS * length(b.x) + FACE_SLOTS * length(b.tris)
    index = surface_maps===nothing ? VertexIncidence(backend, fem.model.nodes > 0 ? entries : 0, fem.model.nodes) :
        MappedFEMContacts(backend,fem.model.nodes>0 ? entries : 0,fem.model;vector_type=eltype(fem.r))
    z(n) = KI.zeros(backend, Position, max(n, 1))
    ccd=continuous_collisions ? SweptCollision(backend,b;sheet_radius=sheets.radius) : nothing
    ccd_pairs=ccd===nothing ? nothing : ccd_query_tile==0 ?
        RuntimeCCDQueue(backend,ccd;capacity=ccd_pair_capacity,pair_batch=ccd_pair_batch) :
        StreamingCCDQueue(backend,ccd;query_batch=ccd_query_tile,
            capacity=ccd_pair_capacity===nothing ? 1_048_576 : ccd_pair_capacity,
            pair_batch=ccd_pair_batch,max_iterations=ccd_max_iterations,primitive_parallel=ccd_primitive_parallel)
    return WorldSolver(backend, sheets, bodies, ContactCoupling(backend, b), fem, index, z(length(b.x)),
        z(4length(b.bodies)), z(fem.model.nodes), upload(backend, UInt32[reinterpret(UInt32, 1.0f0)]),
        WorldEnergy(backend,b;newton,iterations,krylov_precision=elastic_krylov_precision),BondSystem(backend, b),
        ccd,ccd_pairs,
        Float32(cg_tolerance), Float32(newton_tolerance), Float32(newton_motion_limit), newton, iterations)
end

@inline function contact_step_bound(gap, normal, stretch, delta, σ)
    bound = 1.0f0
    normal_step = dot(normal, delta)
    # Closing contacts need no cap: the barrier is infinite at contact and the CCD filter keeps
    # every trial collision-free (IPC's filtered line search). A Newton linearization of kinetic friction has no curvature along slip. Stop before
    # it reverses the spring; the next linearization can then enter the sticking branch.
    along = dot(stretch, delta - normal_step * normal)
    if along < -dot(stretch, stretch) && dot(stretch, stretch) > 1.0f-4 * σ^2
        bound = min(bound, 0.9f0 * dot(stretch, stretch) / -along)
    end
    return bound
end

function reset_step_bound!(bound)
    KI.get_global_id(Int32).x == 1 && (@inbounds bound[1] = reinterpret(UInt32, 1.0f0))
    return nothing
end

function sheet_step_bound!(bound, x, base, anchor, q, qbase, qanchor, model, coupling, C, T, one, radius::Float32, σ::Float32)
    i = KI.get_global_id(Int32).x
    i <= length(x) || return nothing
    @inbounds for slot in 1:SLOTS
        mesh = T.mesh[slot, i]
        if mesh != 0
            model.dynamic_mesh[mesh] == 0x02 && continue
            triangle = T.triangle[slot, i]
            gap, normal, _ = body_gap(C, triangle, base[i], one[1], radius)
            gap >= RANGE * σ && continue
            body = model.mesh_body[mesh]
            u = coupling.vertex_local[slot, i]
            delta, slip = x[i] - base[i], base[i] - anchor[i]
            if body != 0
                old = affine_point(qbase, body, u, one[1])
                delta -= affine_point(q, body, u, one[1]) - old
                slip -= old - affine_point(qanchor, body, u, one[1])
            end
            Pt = I3 - normal * normal'
            stretch = T.lagged[slot, i] > 0 && max(C.grip[1, mesh], C.grip[2, mesh]) > 0 ? Pt * (T.spring[slot, i] + slip) : zero(Vec3f)
            cap = contact_step_bound(gap, normal, stretch, delta, σ)
            Atomix.@atomic min(bound[1], reinterpret(UInt32, cap))
        end
    end
    return nothing
end

function face_step_bound!(bound, x, base, anchor, q, qbase, qanchor, model, coupling, C, F, tris, one, radius::Float32, σ::Float32)
    t = KI.get_global_id(Int32).x
    t <= length(tris) || return nothing
    @inbounds begin
        ia, ib, ic = tris[t]
        for slot in 1:FACE_SLOTS
            kind = F.kind[slot, t]
            if kind != 0x00
                model.dynamic_mesh[feature_mesh(C, kind, F.feature[slot, t])] == 0x02 && continue
                gap, normal, w, own = face_gap(C, kind, F.feature[slot, t], F.edge[slot, t], F.side[slot, t], base[ia], base[ib], base[ic], radius, one[1])
                own || continue
                gap >= RANGE * σ && continue
                body = model.mesh_body[feature_mesh(C, kind, F.feature[slot, t])]
                u = coupling.face_local[slot, t]
                delta = w[1] * (x[ia] - base[ia]) + w[2] * (x[ib] - base[ib]) + w[3] * (x[ic] - base[ic])
                slip = w[1] * (base[ia] - anchor[ia]) + w[2] * (base[ib] - anchor[ib]) + w[3] * (base[ic] - anchor[ic])
                if body != 0
                    old = affine_point(qbase, body, u, one[1])
                    delta -= affine_point(q, body, u, one[1]) - old
                    slip -= old - affine_point(qanchor, body, u, one[1])
                end
                Pt = I3 - normal * normal'
                mesh = feature_mesh(C, kind, F.feature[slot, t])
                stretch = F.lagged[slot, t] > 0 && max(C.grip[1, mesh], C.grip[2, mesh]) > 0 ? Pt * (F.spring[slot, t] + slip) : zero(Vec3f)
                cap = contact_step_bound(gap, normal, stretch, delta, σ)
                Atomix.@atomic min(bound[1], reinterpret(UInt32, cap))
            end
        end
    end
    return nothing
end

function bounded_step!(x, base, bound, one)
    i = KI.get_global_id(Int32).x
    i <= length(x) || return nothing
    @inbounds begin
        α = reinterpret(Float32, bound[1])
        x[i] = moved(base[i], scaled(difference(x[i], base[i], one[1]), α, one[1]), one[1])
    end
    return nothing
end

function body_step_bound!(bound, q, base, anchor, model, T, C, one, σ::Float32)
    i = KI.get_global_id(Int32).x
    i <= length(C.point_owner) || return nothing
    @inbounds for slot in 1:SLOTS
        mesh = T.mesh[slot, i]
        if mesh != 0
            (model.dynamic_mesh[mesh] == 0x02 || model.dynamic_mesh[C.point_owner[i]] == 0x02) && continue
            gap, normal, _ = body_gap(C, T.triangle[slot, i], C.points[i], one[1])
            gap >= RANGE * σ && continue
            source, target = model.mesh_body[C.point_owner[i]], model.mesh_body[mesh]
            u, v = model.coordinates[i], T.other[slot, i]
            delta, slip = zero(Vec3f), zero(Vec3f)
            if source != 0
                old = affine_point(base, source, u, one[1])
                delta += affine_point(q, source, u, one[1]) - old
                slip += old - affine_point(anchor, source, u, one[1])
            end
            if target != 0
                old = affine_point(base, target, v, one[1])
                delta -= affine_point(q, target, v, one[1]) - old
                slip -= old - affine_point(anchor, target, v, one[1])
            end
            Pt = I3 - normal * normal'
            own = C.point_owner[i]
            friction = min(C.grip[1, own], C.grip[1, mesh]) > 0 || min(C.grip[2, own], C.grip[2, mesh]) > 0
            stretch = friction && T.lagged[slot, i] > 0 ? Pt * (T.spring[slot, i] + slip) : zero(Vec3f)
            cap = contact_step_bound(gap, normal, stretch, delta, σ)
            Atomix.@atomic min(bound[1], reinterpret(UInt32, cap))
        end
    end
    return nothing
end

function prime!(s::WorldSolver, state::WorldState;on_phase=nothing,
        on_ccd_prepared=nothing,on_ccd_batch=nothing,ccd_batch=128,
        initial_ccd_validator=validate_initial_collisions!)
    if s.deformables.model.nodes > 0
        launch!(s.backend, fem_surface!, (s.bodies.colliders, state.deformables, s.deformables.model), length(s.bodies.colliders.point_owner))
        launch!(s.backend, fem_surface_normals!, (s.bodies.colliders, s.deformables.model), length(s.bodies.colliders.point_owner))
        update_proxy_guard!(s.backend,s.deformables.model,state.deformables)
    end
    on_phase===nothing || on_phase(:elastic_surface)
    prime!(s.bodies, state.bodies;on_phase)
    s.sheets.n > 0 && prime!(s.sheets, state.sheets)
    on_phase===nothing || on_phase(:sheet_contacts)
    initial_ccd_validator(s,state;batch=ccd_batch,on_prepared=on_ccd_prepared,on_batch=on_ccd_batch)
    on_phase===nothing || on_phase(:initial_ccd)
    return s
end

function world_surface!(target, world, state; refit = true, accepted = nothing)
    b, f = world.bodies, world.deformables
    points = b.colliders.meshes > 0 ? length(b.colliders.point_owner) : 0
    launch!(target, affine_surface!, (b.colliders, state.bodies, b.model, b.one, accepted), points)
    if f.model.nodes > 0
        launch!(target, fem_surface!, (b.colliders, state.deformables, f.model, accepted), points)
        launch!(target, fem_surface_normals!, (b.colliders, f.model, accepted), points)
        update_proxy_guard!(target,f.model,state.deformables)
    end
    refit && refit!(target, b.colliders)
    return nothing
end

"""
Fixed-order, preallocated two-pass dot reduction for coupled PCG. The existing
atomic final sums can change summation order between separate recorded graphs.
This uses the same partial-buffer / final-group architecture as Julia's
AcceleratedKernels mapreduce_1d_gpu, without its immediate-launch wrapper (this
path must declare Mantle passes and support device-gated Newton/CG phases).
"""
function world_dot!(target,world,a,b,n::Int32,slot::Int32)
    n>0 || return nothing
    blocks=cld(n,Int32(256))
    partial=world.energy.cg_partial
    launch!(target,dot_into!,(partial,a,b,n,Int32(1),Val(1)),Int(blocks)*256)
    launch!(target,dot_into!,(world.energy.cg_scalars,partial,nothing,blocks,slot,Val(2)),256)
    return nothing
end

function world_precondition!(target, world, slot::Int32)
    s, b, f = world.sheets, world.bodies, world.deformables
    ns, nb, nf = s.n, 4b.model.bodies, f.model.nodes
    if ns > 0
        fine = s.mg.levels[1]
        vcycle!(target, s.mg)
        world_dot!(target,world,fine.r,fine.z,Int32(ns),slot)
    end
    launch!(target, body_precondition!, (b.z, b.r, b.inverse, b.gradients, b.model), b.model.bodies)
    world_dot!(target,world,b.r,b.z,Int32(nb),slot)
    if nf > 0
        fem_precondition!(target, f)
        world_dot!(target,world,f.r,f.z,Int32(nf),slot)
    end
    return nothing
end

function world_direction!(target, world, k::Int32)
    s, b, f = world.sheets, world.bodies, world.deformables
    s.n > 0 && launch!(target, cg_direction!, (s.p, s.mg.levels[1].z, world.energy.cg_scalars, k), s.n)
    launch!(target, cg_direction!, (b.p, b.z, world.energy.cg_scalars, k), 4b.model.bodies)
    launch!(target, cg_direction!, (f.p, f.z, world.energy.cg_scalars, k), f.model.nodes)
    return nothing
end

function world_cg_iteration!(target, world, state, dt, k::Int32)
    s, b, f = world.sheets, world.bodies, world.deformables
    ss, bs, fs = state.sheets, state.bodies, state.deformables
    ns, nb, nf = s.n, 4b.model.bodies, f.model.nodes
    m = s.model
    ns > 0 && launch!(target, level_apply!, (s.q, s.p, s.mg.levels[1]), ns)
    launch!(target, body_apply!, (b.q, b.p, b.gradients, b.model, b.colliders, b.contacts, dt), nb)
    launch!(target, joint_apply!, (b.q, b.p, b.joint_rows, b.joints, b.model), b.joints.joints > 0 ? nb : 0)
    launch!(target, motor_apply!, (b.q, b.p, b.motor_state, b.motors, b.model), b.motors.motors > 0 ? nb : 0)
    launch!(target, coupled_body_apply!, (b.q, b.p, s.p, b.model, world.coupling, b.colliders, s.touches, s.faces, m.tris, Int32(s.triangles), Int32(ns)), ns > 0 ? nb : 0)
    launch!(target, coupled_sheet_apply!, (s.q, b.p, b.model, world.coupling, b.colliders, s.touches, s.faces, m), ns)
    if nf > 0
        fem_product!(target, f, dt)
        world_fem_contact_product!(target,world)
    end
    if world.bonds.model.bonds > 0
        launch!(target, bond_body_apply!, (b.q, b.p, f.p, world.bonds.model, b.model), nb)
        launch!(target, bond_fem_apply!, (f.q, f.p, b.p, world.bonds.model, f.model), nf)
    end
    world_dot!(target,world,s.p,s.q,Int32(ns),Int32(2k))
    world_dot!(target,world,b.p,b.q,Int32(nb),Int32(2k))
    world_dot!(target,world,f.p,f.q,Int32(nf),Int32(2k))
    ns > 0 && launch!(target, cg_step!, (ss.x, s.mg.levels[1].r, s.p, s.q, world.energy.cg_scalars, m.one, k), ns)
    launch!(target, cg_step!, (bs.q, b.r, b.p, b.q, world.energy.cg_scalars, b.one, k), nb)
    launch!(target, cg_step!, (fs.x, f.r, f.p, f.q, world.energy.cg_scalars, f.one, k), nf)
    world_precondition!(target, world, Int32(2k + 1))
    world_direction!(target, world, k)
    return nothing
end

function step!(target, world::WorldSolver, state::WorldState, dt; compact_cg = true, solve_history = nothing,
        on_linearized=nothing,on_cg_initial=nothing,on_predicted=nothing,
        phase=:all,newton_phases=1:world.newton,dynamic_newton=false,compact_linesearch=true)
    phase in (:all,:predict,:newton,:newton_linear,:newton_trial,:finish) || throw(ArgumentError("invalid world-step phase"))
    all(i->i isa Integer && 1<=i<=world.newton,newton_phases) ||
        throw(ArgumentError("invalid Newton phase range"))
    dynamic_newton && (!(phase in (:newton,:newton_linear,:newton_trial)) || length(newton_phases)!=1) &&
        throw(ArgumentError("dynamic Newton indexing requires one reusable Newton phase"))
    (on_linearized===nothing && on_cg_initial===nothing && on_predicted===nothing) || !is_recording(target) ||
        throw(ArgumentError("host diagnostic callbacks require immediate execution, not graph recording"))
    solve_history === nothing || size(solve_history) == (4, world.newton) ||
        throw(ArgumentError("solve history must have size (4, $(world.newton))"))
    dt = Float32(dt)
    s, b, f = world.sheets, world.bodies, world.deformables
    ss, bs, fs = state.sheets, state.bodies, state.deformables
    m = s.model
    ns, nb, nf = s.n, 4b.model.bodies, f.model.nodes
    nf > 0 && dt != f.timestep && throw(ArgumentError("construct WorldSolver with timestep=$dt for this step size"))
    points = b.colliders.meshes > 0 ? length(b.colliders.point_owner) : 0
    motion = DeformingMeshes(MovingMeshes(b.model, bs, b.anchor, world.coupling), f.model, fs, f.anchor)
  if phase===:all || phase===:predict
    launch!(target,initialize_newton_progress!, (world.energy,solve_history),1)
    if world.ccd !== nothing
        launch!(target, zero!, (world.ccd.status,), 1)
        launch!(target, zero!, (world.ccd.blocking,), world.ccd.query_count)
        launch!(target, zero!, (world.ccd.blocking_reason,), world.ccd.query_count)
        launch!(target, ccd_capture!, (world.ccd, b.colliders, ss.x, true), length(world.ccd.start))
    end
    launch!(target, zero!, (world.energy.cg_iterations,), 2)
    launch!(target, motor_clock!, (b.motor_state, b.motors, dt), b.motors.motors)
    launch!(target, affine_predict!, (bs, b.anchor, b.inertia, b.model, b.one, b.gravity, dt), nb)
    launch!(target, fem_predict!, (fs, f.anchor, f.inertia, f.model, f.one, f.gravity, dt), nf)
    on_predicted===nothing || on_predicted(world,state,dt)
    world_surface!(target, world, state)
    if ns > 0
        predict_sheet!(target, s, ss, dt)
    end
    world_ccd_filter!(target, world, state, s.anchor, b.anchor, f.anchor)
    if ns > 0
        launch!(target, coupled_endpoints!, (world.coupling, b.model, b.colliders, s.touches, s.faces, Int32(ns)), world.coupling.incidence.entries)
        compact!(target, world.coupling.incidence)
    end
  end
  if phase in (:all,:newton,:newton_linear,:newton_trial)
    for newton in newton_phases
      # Immediate execution stops once the device flags convergence, rather than issuing
      # gated iterations (and host factorizations) that do nothing.
      !is_recording(target) && world.newton_tolerance>0 && newton>first(newton_phases) &&
          first(Array(world.energy.newton_status))==0 && break
      let target = world.newton_tolerance>0 ? ConditionalTarget(target,world.energy.newton_status) : target
        E = world.energy
       if phase!==:newton_trial
        # Newton corrections can cross a mesh feature. Refresh the owning triangles
        # and their affine/FE incidence together, keeping the friction history that
        # find_body_contacts! transfers between matching contacts.
        launch_indexed_batches!(target, find_body_contacts!, (b.contacts, b.colliders, b.model, bs, b.anchor, b.one, b.roughness, b.inertia), points)
        index_body_contacts!(target, b.contacts, b.colliders, b.model)
        if nf > 0
            launch!(target, fem_contact_endpoints!, (fem_contact_index(world.fem_contacts), fem_contact_model(f.model,world.fem_contacts), b.colliders, b.contacts, s.touches, s.faces, Int32(ns)), world.fem_contacts.entries)
            compact!(target, world.fem_contacts)
        end
        launch!(target, copy_into!, (world.sheet_base, ss.x), ns)
        launch!(target, copy_into!, (world.body_base, bs.q), nb)
        launch!(target, copy_into!, (world.fem_base, fs.x), nf)
        world.ccd === nothing || launch!(target, ccd_capture!,
            (world.ccd, b.colliders, ss.x, true), length(world.ccd.start))
        ns > 0 && linearize_sheet!(target, s, ss, dt, motion)
        launch!(target, affine_elastic!, (b.elastic, b.blocks, b.gradients, bs, b.model, b.one), b.model.bodies)
        launch!(target, body_contact_forces!, (b.contacts, b.colliders, b.model, bs, b.anchor, b.one, b.roughness, b.tangential, false, motion), points)
        launch!(target, body_rhs!, (b.r, b.inverse, bs, b.inertia, b.elastic, b.model, b.colliders, b.contacts, dt), nb)
        launch!(target, joint_linearize!, (b.joint_rows, b.joints, bs, b.anchor, b.one, dt), b.joints.joints)
        launch!(target, joint_rhs!, (b.r, b.inverse, b.joint_rows, b.joints, b.model), b.joints.joints > 0 ? nb : 0)
        launch!(target, motor_linearize!, (b.motor_state, b.motors, bs), b.motors.motors)
        launch!(target, motor_rhs!, (b.r, b.inverse, b.motor_state, b.motors, b.model), b.motors.motors > 0 ? nb : 0)
        launch!(target, body_load_rhs!, (b.r, b.load_force, b.load_torque, b.anchor, b.model), nb)
        launch!(target, coupled_body_rhs!, (b.r, b.inverse, b.model, world.coupling, b.colliders, s.touches, s.faces, Int32(s.triangles), Int32(ns)), ns > 0 ? nb : 0)
        if nf > 0
            fem_linearize!(target, f, fs)
            fem_rhs!(target, f, fs, dt)
            world_fem_contact_rhs!(target,world)
        end
        if world.bonds.model.bonds > 0
            launch!(target, bond_linearize!, (world.bonds.force, world.bonds.model, bs, fs, b.one), world.bonds.model.bonds)
            launch!(target, bond_body_rhs!, (b.r, b.inverse, world.bonds.force, world.bonds.model, b.model), nb)
            launch!(target, bond_fem_rhs!, (f.r, f.inverse, world.bonds.force, world.bonds.model, f.model), nf)
        end
        fem_prepare_preconditioner!(target,f)
        on_linearized===nothing || on_linearized(newton,world,state)
        E = world.energy
        launch!(target, zero!, (E.accepted,), 1)
        ns > 0 && launch!(target, copy_into!, (E.sheet_force, s.mg.levels[1].r), ns)
        launch!(target, copy_into!, (E.body_force, b.r), nb)
        launch!(target, copy_into!, (E.fem_force, f.r), nf)
        if ns > 0
            launch!(target, freeze_vertex_energy!, (E, ss.x, m, b.colliders, s.touches, s.partners, b.one, s.radius, s.roughness, s.tangential), ns)
            launch!(target, freeze_face_energy!, (E, ss.x, m, b.colliders, s.faces, b.one, s.radius, s.roughness, s.tangential), s.triangles)
        end
        world_energy!(target, world, state, motion, dt, Int32(1))
        launch!(target, zero!, (E.cg_scalars,), length(E.cg_scalars))
        world_precondition!(target, world, Int32(1))
        on_cg_initial===nothing || on_cg_initial(newton,world,state)
        world_direction!(target, world, Int32(0))
        launch!(target, world_cg_start!, (E.cg_active, E.cg_initial, E.cg_iterations, E.cg_scalars), 1)
        if is_recording(target) && compact_cg
            Mantle.repeat!(recording_target(target), world.iterations; while_nonzero = E.cg_active) do _
                world_cg_iteration!(target, world, state, dt, Int32(1))
                launch!(target, world_cg_check!, (E.cg_active, E.cg_initial, E.cg_iterations, E.cg_scalars, Int32(1), world.cg_tolerance), 1)
                launch!(target, world_cg_roll!, (E.cg_scalars,), 1)
            end
        elseif is_recording(target)
            for k in 1:world.iterations
                # Reference expansion for numerical comparisons.
                Mantle.repeat!(recording_target(target), 1; while_nonzero = E.cg_active) do _
                    world_cg_iteration!(target, world, state, dt, Int32(k))
                    launch!(target, world_cg_check!, (E.cg_active, E.cg_initial, E.cg_iterations, E.cg_scalars, Int32(k), world.cg_tolerance), 1)
                end
            end
        else
            # Immediate execution reads the convergence flag after every iteration and stops
            # where the recorded loop stops.
            for _ in 1:world.iterations
                world_cg_iteration!(target, world, state, dt, Int32(1))
                launch!(target, world_cg_check!, (E.cg_active, E.cg_initial, E.cg_iterations, E.cg_scalars, Int32(1), world.cg_tolerance), 1)
                launch!(target, world_cg_roll!, (E.cg_scalars,), 1)
                only(Array(E.cg_active)) == 0 && break
            end
        end
        if solve_history!==nothing
            if dynamic_newton
                launch!(target,world_cg_record_dynamic!,
                    (solve_history,E.cg_initial,E.cg_iterations,E.cg_scalars,E.newton_status,
                        compact_cg || !is_recording(target),
                        true,Int32(world.iterations),world.cg_tolerance),1)
            else
                launch!(target, world_cg_record!,
                    (solve_history, E.cg_initial, E.cg_iterations, E.cg_scalars, Int32(newton),
                        compact_cg || !is_recording(target), true,
                        Int32(world.iterations), world.cg_tolerance), 1)
            end
        end
        # Measure the UNCLIPPED Newton correction. A contact-limited or rejected
        # step can be tiny despite a large remaining nonlinear residual.
        launch!(target,zero!,(E.newton_step,),1)
        launch!(target,newton_step_max!, (E.newton_step,ss.x,world.sheet_base),ns)
        launch!(target,newton_step_max!, (E.newton_step,fs.x,world.fem_base),nf)
        launch!(target,affine_newton_step_max!,
            (E.newton_step,bs.q,world.body_base,b.model,b.colliders,b.one),points)
        launch!(target, reset_step_bound!, (world.step_bound,), 1)
        world.newton_motion_limit>0 && launch!(target,newton_motion_bound!,
            (world.step_bound,E.newton_step,world.newton_motion_limit),1)
        launch!(target, motor_step_bound!, (world.step_bound, bs.q, world.body_base, b.motor_state, b.motors), b.motors.motors)
        launch!(target, body_step_bound!, (world.step_bound, bs.q, world.body_base, b.anchor, b.model, b.contacts, b.colliders, b.one, b.roughness), points)
        launch!(target, sheet_step_bound!, (world.step_bound, ss.x, world.sheet_base, s.anchor, bs.q, world.body_base, b.anchor,
            b.model, world.coupling, b.colliders, s.touches, m.one, s.radius, s.roughness), ns)
        launch!(target, face_step_bound!, (world.step_bound, ss.x, world.sheet_base, s.anchor, bs.q, world.body_base, b.anchor,
            b.model, world.coupling, b.colliders, s.faces, m.tris, m.one, s.radius, s.roughness), s.triangles)
        ns > 0 && launch!(target, stack_step_bound!, (world.step_bound, ss.x, world.sheet_base, s.anchor, s.partners, m.one, s.radius, s.roughness, Int32(ns)), ns)
        if nf > 0
            launch!(target, step_vectors!, (f.q, f.p, fs.x, world.fem_base, f.anchor, Int32(nf)), nf)
            launch!(target, step_vectors!, (b.q, b.p, bs.q, world.body_base, b.anchor, Int32(nb)), nb)
            launch!(target, step_vectors!, (s.q, s.p, ss.x, world.sheet_base, s.anchor, Int32(ns)), ns)
            world_fem_contact_bound!(target,world)
        end
        launch!(target, bounded_step!, (ss.x, world.sheet_base, world.step_bound, m.one), ns)
        launch!(target, bounded_step!, (bs.q, world.body_base, world.step_bound, b.one), nb)
        launch!(target, bounded_step!, (fs.x, world.fem_base, world.step_bound, f.one), nf)
       end
       if phase!==:newton_linear
        world_ccd_filter!(target, world, state, world.sheet_base, world.body_base, world.fem_base)
        world_line_search!(target, world, state, motion, dt;compact=compact_linesearch)
        world_surface!(target, world, state)
        # Deactivate only after every accepted-state update has completed.
        if dynamic_newton
            launch!(target,record_dynamic_newton_progress!, (E,world.newton_tolerance),1)
        else
            launch!(target,record_newton_progress!, (E,Int32(newton),world.newton_tolerance),1)
        end
       end
      end
    end
  end
  if phase===:all || phase===:finish
    launch!(target, affine_velocity!, (bs, b.anchor, b.model, b.one, dt), nb)
    launch!(target, motor_observe!, (b.motor_state, b.motors, bs), b.motors.motors)
    launch!(target, fem_velocity!, (fs, f.anchor, f.model, f.one, dt), nf)
    launch!(target, body_contact_forces!, (b.contacts, b.colliders, b.model, bs, b.anchor, b.one, b.roughness, b.tangential, true, motion), points)
    if ns > 0
        launch!(target, velocity!, (ss.v, s.acceleration, ss.x, s.anchor, s.touches, s.partners, m.one, dt), ns)
        launch!(target, mesh_friction_update!, (ss.x, s.anchor, s.colliders, s.touches, m.area, m.one, s.radius, s.roughness, s.tangential, motion), ns)
        launch!(target, face_friction_update!, (ss.x, s.anchor, m.tris, s.colliders, s.faces, m.area, m.one, s.radius, s.roughness, s.tangential, motion), s.triangles)
        launch!(target, sheet_friction_update!, (s.partners, ss.x, s.anchor, m.area, m.one, s.radius, s.roughness, s.tangential), ns)
    end
  end
    return state
end

step!(s::WorldSolver, state::WorldState, dt) = step!(s.backend, s, state, dt)
