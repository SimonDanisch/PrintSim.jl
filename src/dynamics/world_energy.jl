# One incremental potential and one backtracking factor for all three kinds of DOFs.
# Friction's normal, interpolation and tangent stiffness are frozen at the Newton
# linearization, matching the force law whose normal-stiffness derivative is lagged.
struct WorldEnergy{P, V, N, F, R, U, H, C}
    sheet_trial::P
    body_trial::P
    fem_trial::P
    sheet_force::V
    body_force::V
    fem_force::V
    values::V
    ones::V
    vertex_normal::N
    vertex_kt::F
    face_kt::F
    stack_kt::R
    sum::R
    bound::U
    accepted::U
    cg_active::U
    cg_initial::C
    cg_scalars::C
    cg_iterations::U
    newton_step::U
    newton_status::U
    newton_progress::H
    cg_partial::C
    body_count::Int32
    sheet_count::Int32
    springs::Int32
    hinges::Int32
    triangles::Int32
    fem_nodes::Int32
    fem_elements::Int32
    bonds::Int32
    count::Int32
end
Adapt.@adapt_structure WorldEnergy

function WorldEnergy(backend, b; newton = 8,iterations=400,krylov_precision=:float32)
    scalar_type=krylov_precision===:float64 ? Float64 : Float32
    ns, nb = length(b.x), 4length(b.bodies)
    nf = sum(length(p.x) for p in b.deformables; init = 0)
    nq = 4sum(length(p.tets) for p in b.deformables; init = 0)
    body_count = length(b.bodies) + SLOTS * sum(length(m.vertices) for m in b.meshes; init = 0)
    count = body_count + ns + length(b.springs) + length(b.hinges) + length(b.tris) + nf + nq + length(b.bonds)
    z(T, dims...) = KI.zeros(backend, T, max.(dims, 1)...)
    return WorldEnergy(z(Position, ns), z(Position, nb), z(Position, nf), z(Vec3f, ns), z(Vec3f, nb), z(Vec3f, nf),
        z(Vec3f, count), upload(backend, [Vec3f(1, 0, 0)]), z(Vec3f, SLOTS, ns),
        z(Float32, SLOTS, ns), z(Float32, FACE_SLOTS, length(b.tris)), z(Float32, ns), z(Float32, 4),
        upload(backend, UInt32[reinterpret(UInt32, 1.0f0)]), z(UInt32, 1), z(UInt32, 1), z(scalar_type, 1),
        z(scalar_type,2iterations+2),z(UInt32, 2),
        z(UInt32,1), z(UInt32,2), z(Float32,5,newton),
        z(scalar_type,cld(max(ns,nb,nf),256)), Int32(body_count), Int32(ns),
        Int32(length(b.springs)), Int32(length(b.hinges)), Int32(length(b.tris)), Int32(nf), Int32(nq), Int32(length(b.bonds)), Int32(count))
end

function world_cg_start!(active, initial, iterations, scalars)
    KI.get_global_id(Int32).x == 1 || return nothing
    @inbounds begin
        value = scalars[1]
        initial[1] = value
        iterations[1] = 0
        # Keep a sticky failure bit in the total count: a later Newton iteration
        # must not conceal a nonfinite or non-SPD solve earlier in this step.
        (!isfinite(value) || value < 0) && (iterations[2] |= UInt32(0x80000000))
        active[1] = isfinite(value) && value > 1.0f-35 ? UInt32(1) : UInt32(0)
    end
    return nothing
end

function world_cg_check!(active, initial, iterations, scalars, k::Int32, relative::Float32)
    KI.get_global_id(Int32).x == 1 || return nothing
    @inbounds begin
        value = scalars[2k + Int32(1)]
        iterations[1] += UInt32(1)
        iterations[2] += UInt32(1)
        (!isfinite(value) || value < 0) && (iterations[2] |= UInt32(0x80000000))
        ratio=typeof(value)(relative)
        active[1] = isfinite(value) && value > max(ratio^2 * initial[1], typeof(value)(1.0f-35)) ? UInt32(1) : UInt32(0)
    end
    return nothing
end

# A recorded CG loop reuses three reduction slots instead of declaring a distinct
# set of passes and arguments for every iteration. The direction update has
# already consumed both residual products before these slots are recycled.
function world_cg_roll!(scalars)
    KI.get_global_id(Int32).x == 1 || return nothing
    @inbounds begin
        scalars[1] = scalars[3]
        scalars[2] = 0
        scalars[3] = 0
    end
    return nothing
end

"Record each Newton solve before the shared CG reductions are reused."
function world_cg_record!(history, initial, iterations, scalars, newton::Int32,
        compact::Bool, tracked::Bool, limit::Int32, relative::Float32)
    KI.get_global_id(Int32).x == 1 || return nothing
    @inbounds begin
        count = tracked ? Int32(iterations[1]) : limit
        slot = compact ? Int32(1) : 2count + Int32(1)
        first, last = initial[1], scalars[slot]
        history[1, newton] = first
        history[2, newton] = last
        history[3, newton] = Float32(count)
        history[4, newton] = isfinite(first) && first >= 0 && isfinite(last) && last >= 0 &&
            last <= max(typeof(first)(relative)^2 * first, typeof(first)(1.0f-35)) ? 1.0f0 : 0.0f0
    end
    return nothing
end

function world_cg_record_dynamic!(history,initial,iterations,scalars,status,
        compact::Bool,tracked::Bool,limit::Int32,relative::Float32)
    KI.get_global_id(Int32).x==1 || return nothing
    @inbounds iteration=Int32(status[2])+Int32(1)
    world_cg_record!(history,initial,iterations,scalars,iteration,compact,tracked,limit,relative)
    return nothing
end

function freeze_vertex_energy!(E, x, model, C, T, P, one, r::Float32, σ::Float32, tangential::Float32)
    i = KI.get_global_id(Int32).x
    i <= E.sheet_count || return nothing
    @inbounds begin
        E.stack_kt[i] = tangential * P.link_kn[1, i]
        for slot in 1:SLOTS
            E.vertex_kt[slot, i], E.vertex_normal[slot, i] = 0.0f0, zero(Vec3f)
            mesh = T.mesh[slot, i]
            if mesh != 0
                gap, normal = touch_gap(C, T.triangle[slot, i], x[i], r, one[1])
                _, stiffness = normal_force(gap, model.area[i] * C.grip[3, mesh], σ)
                E.vertex_normal[slot, i] = normal
                E.vertex_kt[slot, i] = gap < RANGE * σ ? tangential * stiffness : 0.0f0
            end
        end
    end
    return nothing
end

function freeze_face_energy!(E, x, model, C, F, one, r::Float32, σ::Float32, tangential::Float32)
    t = KI.get_global_id(Int32).x
    t <= E.triangles || return nothing
    @inbounds begin
        a, b, c = model.tris[t]
        for slot in 1:FACE_SLOTS
            E.face_kt[slot, t] = 0
            kind = F.kind[slot, t]
            if kind != 0x00
                gap, _, _, own = face_gap(C, kind, F.feature[slot, t], F.edge[slot, t], F.side[slot, t], x[a], x[b], x[c], r, one[1])
                mesh = feature_mesh(C, kind, F.feature[slot, t])
                _, stiffness = normal_force(gap, face_area(model.area, a, b, c) * C.grip[3, mesh], σ)
                E.face_kt[slot, t] = own && gap < RANGE * σ ? tangential * stiffness : 0.0f0
            end
        end
    end
    return nothing
end

@inline function frozen_mesh_slip(motion, C, T, slot, i, one)
    @inbounds begin
        affine = motion.affine
        coupling = affine.coupling
        mesh = T.mesh[slot, i]
        body = affine.model.mesh_body[mesh]
        u = coupling.vertex_local[slot, i]
        slip = body == 0 ? zero(Vec3f) : affine_point(affine.state.q, body, u, one) - affine_point(affine.anchor, body, u, one)
        f = C.faces[T.triangle[slot, i]]
        w = coupling.vertex_barycentric[slot, i]
        return slip + w[1] * fem_point_slip(motion, f[1], one) + w[2] * fem_point_slip(motion, f[2], one) + w[3] * fem_point_slip(motion, f[3], one)
    end
end

@inline function frozen_feature_slip(motion, C, F, slot, t, one)
    @inbounds begin
        affine = motion.affine
        kind, feature = F.kind[slot, t], F.feature[slot, t]
        body = affine.model.mesh_body[feature_mesh(C, kind, feature)]
        u = affine.coupling.face_local[slot, t]
        slip = body == 0 ? zero(Vec3f) : affine_point(affine.state.q, body, u, one) - affine_point(affine.anchor, body, u, one)
        a, b = feature_endpoints(C, kind, feature)
        s = affine.coupling.face_parameter[slot, t]
        return slip + (1 - s) * fem_point_slip(motion, a, one) + s * fem_point_slip(motion, b, one)
    end
end

@inline function frozen_friction_energy(spring, normal, slip, kt, λ, sliding, μs, μk)
    stretch = spring + slip
    stretch -= dot(stretch, normal) * normal
    return spring_potential(stretch, kt, λ, sliding, μs, μk)
end

function sheet_vertex_energy!(E, x, inertia, anchor, model, C, T, P, motion, one, dt::Float32, r::Float32, σ::Float32)
    @inbounds E.accepted[1] != 0 && return nothing
    i = KI.get_global_id(Int32).x
    i <= E.sheet_count || return nothing
    @inbounds begin
        d = x[i] - inertia[i]
        value = model.mass[i] / (2dt^2) * dot(d, d)
        if model.fixed[i] == 0x00
            for slot in 1:SLOTS
                mesh = T.mesh[slot, i]
                if mesh != 0
                    gap, _ = touch_gap(C, T.triangle[slot, i], x[i], r, one[1])
                    value += normal_potential(gap, model.area[i] * C.grip[3, mesh], σ)
                    slip = (x[i] - anchor[i]) - frozen_mesh_slip(motion, C, T, slot, i, one[1])
                    value += frozen_friction_energy(T.spring[slot, i], E.vertex_normal[slot, i], slip, E.vertex_kt[slot, i],
                        T.lagged[slot, i], T.sliding[slot, i], C.grip[1, mesh], C.grip[2, mesh])
                end
            end
            if P.partner[1, i] != 0
                gap, _, _, _, _, _, found = partner_gap(P, x, x[i], i, Int32(1), r, one[1])
                if found
                    sheet = P.sheet[i]
                    value += normal_potential(gap, model.area[i] * P.friction[3, sheet], σ)
                    if E.stack_kt[i] > 0
                        a, b, c = P.link_triangle[1, i]
                        w = P.link_weights[1, i]
                        slip = (x[i] - anchor[i]) - moved_by(x, anchor, a, b, c, w)
                        value += frozen_friction_energy(P.spring[1, i], P.link_normal[1, i], slip, E.stack_kt[i], P.lagged[1, i],
                            P.sliding[1, i], P.friction[1, sheet], P.friction[2, sheet])
                    end
                end
            end
        end
        E.values[E.body_count + i] = Vec3f(value, 0, 0)
    end
    return nothing
end

function sheet_elastic_energy!(E, x, anchor, model, one)
    @inbounds E.accepted[1] != 0 && return nothing
    i = KI.get_global_id(Int32).x
    i <= E.springs + E.hinges || return nothing
    @inbounds begin
        value = 0.0f0
        if i <= E.springs
            a, b = model.springs[i]
            d = difference(x[b], x[a], one[1])
            stretch0 = stretch(d, model.spring_rest[i], one[1])
            stretch1 = stretch(d, difference(anchor[b], anchor[a], one[1]), one[1])
            value = model.spring_stiffness[i] / 2 * (stretch0^2 + model.spring_damping[i] * stretch1^2)
        else
            h = i - E.springs
            a, b, c, d = model.hinges[h]
            angle = hinge_angle(zero(Vec3f), x[b] - x[a], x[c] - x[a], x[d] - x[a])
            previous = hinge_angle(zero(Vec3f), anchor[b] - anchor[a], anchor[c] - anchor[a], anchor[d] - anchor[a])
            value = model.hinge_stiffness[h] / 2 * ((angle - model.hinge_rest[h])^2 + model.hinge_damping[h] * (angle - previous)^2)
        end
        E.values[E.body_count + E.sheet_count + i] = Vec3f(value, 0, 0)
    end
    return nothing
end

function sheet_face_energy!(E, x, anchor, model, C, F, motion, one, r::Float32, σ::Float32)
    @inbounds E.accepted[1] != 0 && return nothing
    t = KI.get_global_id(Int32).x
    t <= E.triangles || return nothing
    @inbounds begin
        a, b, c = model.tris[t]
        area = face_area(model.area, a, b, c)
        value = 0.0f0
        for slot in 1:FACE_SLOTS
            kind = F.kind[slot, t]
            if kind != 0x00
                gap, _, _, own = face_gap(C, kind, F.feature[slot, t], F.edge[slot, t], F.side[slot, t], x[a], x[b], x[c], r, one[1])
                if own
                    mesh = feature_mesh(C, kind, F.feature[slot, t])
                    value += normal_potential(gap, area * C.grip[3, mesh], σ)
                    w = F.weights[slot, t]
                    slip = w[1] * (x[a] - anchor[a]) + w[2] * (x[b] - anchor[b]) + w[3] * (x[c] - anchor[c]) -
                        frozen_feature_slip(motion, C, F, slot, t, one[1])
                    value += frozen_friction_energy(F.spring[slot, t], F.normal[slot, t], slip, E.face_kt[slot, t], F.lagged[slot, t],
                        F.sliding[slot, t], C.grip[1, mesh], C.grip[2, mesh])
                end
            end
        end
        E.values[E.body_count + E.sheet_count + E.springs + E.hinges + t] = Vec3f(value, 0, 0)
    end
    return nothing
end

@inline function fem_potential(state, model, e, one,::Val{W}=Val(false)) where W
    @inbounds begin
        first = model.tets[1, e]
        f1, f2, f3 = Position(Vec3f(1, 0, 0)), Position(Vec3f(0, 1, 0)), Position(Vec3f(0, 0, 1))
        for k in Int32(2):Int32(10)
            node = model.tets[k, e]
            du = minus(difference(state.x[node], state.x[first], one), difference(model.rest[node], model.rest[first], one), one)
            gh, gl = model.gradient_hi[k, e], model.gradient_lo[k, e]
            f1 = plus(f1, plus(scaled(du, gh[1], one), scaled(du, gl[1], one), one), one)
            f2 = plus(f2, plus(scaled(du, gh[2], one), scaled(du, gl[2], one), one), one)
            f3 = plus(f3, plus(scaled(du, gh[3], one), scaled(du, gl[3], one), one), one)
        end
        # StVK energy is finite under inversion. The line search must exclude
        # inverted deformation gradients independently of energy decrease.
        dot(approximate(f1), cross(approximate(f2), approximate(f3))) > 0 || return Inf32
        c1,c2,c3,c4,c5,c6=fem_strain_coefficients(W ? Float64 : Float32,f1,f2,f3,one)
        return model.stiffness[1, e] / 2 * (c1^2 + c2^2 + c3^2 + c4^2 + c5^2 + c6^2) + model.stiffness[7, e] / 2 * (c1 + c2 + c3)^2
    end
end

function fem_energy_values!(E, state, inertia, anchor, external, model, one, dt::Float32,
        offset::Int32 = Int32(0), count::Int32 = typemax(Int32))
    @inbounds E.accepted[1] != 0 && return nothing
    localid = KI.get_global_id(Int32).x
    localid <= count || return nothing
    i = localid + offset
    i <= E.fem_nodes + E.fem_elements || return nothing
    @inbounds begin
        value = 0.0f0
        if i <= E.fem_nodes
            delta = state.x[i] - inertia[i]
            value = model.mass[i] / (2dt^2) * dot(delta, delta) - dot(external[i], state.x[i] - anchor[i])
        else
            value = fem_potential(state, model, i - E.fem_nodes, one[1],Val(eltype(external)===Vec3d))
        end
        E.values[E.body_count + E.sheet_count + E.springs + E.hinges + E.triangles + i] = Vec3f(value, 0, 0)
    end
    return nothing
end

function world_energy!(target, world, state, motion, dt, slot)
    s, b, f, E = world.sheets, world.bodies, world.deformables, world.energy
    launch!(target, body_energy_values!, (E.values, state.bodies, b.inertia, b.anchor, b.model, b.colliders, b.contacts,
        b.load_force, b.load_torque, b.joints, b.joint_rows, b.motors, b.motor_state, b.one, dt, b.roughness, motion, E.accepted), E.body_count)
    if s.n > 0
        launch!(target, sheet_vertex_energy!, (E, state.sheets.x, s.inertia, s.anchor, s.model, b.colliders, s.touches,
            s.partners, motion, b.one, dt, s.radius, s.roughness), s.n)
        launch!(target, sheet_elastic_energy!, (E, state.sheets.x, s.anchor, s.model, b.one), E.springs + E.hinges)
        launch!(target, sheet_face_energy!, (E, state.sheets.x, s.anchor, s.model, b.colliders, s.faces, motion, b.one, s.radius, s.roughness), s.triangles)
    end
    for offset in 0:262144:(E.fem_nodes + E.fem_elements - 1)
        count = min(262144, E.fem_nodes + E.fem_elements - offset)
        launch!(target, fem_energy_values!, (E, state.deformables, f.inertia, f.anchor, f.external,
            f.model, b.one, dt, Int32(offset), Int32(count)), count)
    end
    launch!(target, bond_energy_values!, (E, world.bonds.model, state.bodies, state.deformables, b.one), E.bonds)
    launch!(target, reset_energy_slot!, (E.sum, slot), 1)
    launch!(target, dot_into!, (E.sum, E.values, RepeatedEnergyMask(E.ones), E.count, slot), cld(E.count, 256) * 256)
    return nothing
end

function bond_energy_values!(E, model, body, fem, one)
    @inbounds E.accepted[1] != 0 && return nothing
    bond = KI.get_global_id(Int32).x
    bond <= model.bonds || return nothing
    @inbounds begin
        delta = bond_body_point(body.q, model, bond, one[1]) - bond_fem_point(fem.x, model, bond, one[1])
        E.values[E.count - E.bonds + bond] = Vec3f(model.stiffness[bond] / 2 * dot(delta, delta), 0, 0)
    end
    return nothing
end

function world_work_values!(values, force, trial, base, n::Int32, offset::Int32)
    i = KI.get_global_id(Int32).x
    i <= n || return nothing
    @inbounds values[offset + i] = Vec3f(dot(force[i], trial[i] - base[i]), 0, 0)
    return nothing
end

"Sequential CG/line-search activity flag: make discarded trials visible to Mantle."
function line_search_pending!(active,accepted)
    KI.get_global_id(Int32).x==1 || return nothing
    @inbounds active[1]=accepted[1]==0 ? UInt32(1) : UInt32(0)
    return nothing
end

function finalize_energy_step!(bound, accepted)
    KI.get_global_id(Int32).x == 1 || return nothing
    @inbounds accepted[1] == 0 && (bound[1] = reinterpret(UInt32, 0.0f0))
    return nothing
end

function stack_step_bound!(bound, x, base, anchor, P, one, r::Float32, σ::Float32, n::Int32)
    i = KI.get_global_id(Int32).x
    i <= n || return nothing
    @inbounds begin
        P.partner[1, i] == 0 && return nothing
        gap, normal, a, b, c, w, found = partner_gap(P, base, base[i], i, Int32(1), r, one[1])
        found || return nothing
        delta = (x[i] - base[i]) - moved_by(x, base, a, b, c, w)
        closing = dot(delta, normal)
        cap = 1.0f0
        if gap >= RANGE * σ
            closing < 0 && gap + closing < 2σ && (cap = clamp((gap - 2σ) / -closing, 0.0f0, 1.0f0))
        else
            sheet = P.sheet[i]
            slip = (base[i] - anchor[i]) - moved_by(base, anchor, a, b, c, w)
            stretch = P.spring[1, i] + slip
            stretch -= dot(stretch, normal) * normal
            (P.lagged[1, i] <= 0 || max(P.friction[1, sheet], P.friction[2, sheet]) <= 0) && (stretch = zero(Vec3f))
            cap = contact_step_bound(gap, normal, stretch, delta, σ)
            closing < 0 && (cap = min(cap, clamp(0.9f0 * (gap + r) / -closing, 0.0f0, 1.0f0)))
        end
        Atomix.@atomic min(bound[1], reinterpret(UInt32, cap))
    end
    return nothing
end

function world_line_search!(target, world, state, motion, dt;compact=is_recording(target))
    s, b, f, E = world.sheets, world.bodies, world.deformables, world.energy
    ns, nb, nf = s.n, 4b.model.bodies, f.model.nodes
    launch!(target, copy_into!, (E.sheet_trial, state.sheets.x), ns)
    launch!(target, copy_into!, (E.body_trial, state.bodies.q), nb)
    launch!(target, copy_into!, (E.fem_trial, state.deformables.x), nf)
    launch!(target, world_work_values!, (E.values, E.sheet_force, E.sheet_trial, world.sheet_base, Int32(ns), Int32(0)), ns)
    launch!(target, world_work_values!, (E.values, E.body_force, E.body_trial, world.body_base, Int32(nb), Int32(ns)), nb)
    launch!(target, world_work_values!, (E.values, E.fem_force, E.fem_trial, world.fem_base, Int32(nf), Int32(ns + nb)), nf)
    launch!(target, reset_energy_slot!, (E.sum, Int32(3)), 1)
    launch!(target, dot_into!, (E.sum, E.values, RepeatedEnergyMask(E.ones), Int32(ns + nb + nf), Int32(3)), cld(ns + nb + nf, 256) * 256)
    launch!(target, reset_step_bound!, (E.bound,), 1)
    trial=function ()
        launch!(target, trial_step!, (state.sheets.x, world.sheet_base, E.sheet_trial, E.bound, b.one, E.accepted), ns)
        launch!(target, trial_step!, (state.bodies.q, world.body_base, E.body_trial, E.bound, b.one, E.accepted), nb)
        launch!(target, trial_step!, (state.deformables.x, world.fem_base, E.fem_trial, E.bound, b.one, E.accepted), nf)
        world_surface!(target, world, state; refit = false, accepted = E.accepted)
        world_energy!(target, world, state, motion, dt, Int32(2))
        launch!(target, accept_energy_step!, (E.bound, E.accepted, E.sum), 1)
    end
    if compact && is_recording(target)
        # A line-search trial is the same resident work on every backtrack.
        # This loop is sequential with, not nested in, the CG and CCD loops.
        # Reuse the finished CG activity flag. An explicit device gate is
        # necessary: internal accepted[] early-outs alone let zero timings of
        # discarded trials merge future ACTIVE trials into one submission.
        launch!(target,line_search_pending!,(E.cg_active,E.accepted),1)
        Mantle.repeat!(recording_target(target),12;while_nonzero=E.cg_active) do _
            trial()
            launch!(target,line_search_pending!,(E.cg_active,E.accepted),1)
        end
    else
        for _ in 1:12
            trial()
        end
    end
    launch!(target, finalize_energy_step!, (E.bound, E.accepted), 1)
    launch!(target, trial_step!, (state.sheets.x, world.sheet_base, E.sheet_trial, E.bound, b.one), ns)
    launch!(target, trial_step!, (state.bodies.q, world.body_base, E.body_trial, E.bound, b.one), nb)
    launch!(target, trial_step!, (state.deformables.x, world.fem_base, E.fem_trial, E.bound, b.one), nf)
    return nothing
end
