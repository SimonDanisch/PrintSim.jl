# Globalize affine Newton steps with their incremental potential. In particular, a linear
# solve can be exact while its finite rotation leaves the rigid-motion tangent plane and
# creates large strain. Backtracking prevents that step from injecting elastic energy.

@inline function affine_energy(q, model, body, one)
    @inbounds begin
        o = Int32(4) * (body - Int32(1))
        a, b, c = q[o + Int32(2)], q[o + Int32(3)], q[o + Int32(4)]
        # Orthogonality energy alone also vanishes for a reflection. Reject
        # orientation reversal in the line search rather than accepting an
        # inverted affine body with a deceptively small elastic energy.
        dot(approximate(a), cross(approximate(b), approximate(c))) > 0 || return Inf32
        lh, ll = model.scale_squared[body]
        inverse = 1 / model.scale[body]^2
        h, l = affine_dot(a, a, one); c1 = ((h - lh) + (l - ll)) * inverse / 2
        h, l = affine_dot(b, b, one); c2 = ((h - lh) + (l - ll)) * inverse / 2
        h, l = affine_dot(c, c, one); c3 = ((h - lh) + (l - ll)) * inverse / 2
        h, l = affine_dot(a, b, one); c4 = (h + l) * inverse / sqrt(2.0f0)
        h, l = affine_dot(a, c, one); c5 = (h + l) * inverse / sqrt(2.0f0)
        h, l = affine_dot(b, c, one); c6 = (h + l) * inverse / sqrt(2.0f0)
        return model.orthogonal[body] / 2 * (c1^2 + c2^2 + c3^2 + c4^2 + c5^2 + c6^2)
    end
end

@inline function joint_energy(q, anchor, joints, rows, joint, dt, one)
    @inbounds begin
        a, b = joints.a[joint], joints.b[joint]
        pa = affine_point(q, a, joints.anchor_a[joint], one)
        pb = b == 0 ? Position(joints.anchor_b[joint]) : affine_point(q, b, joints.anchor_b[joint], one)
        da, db = affine_direction(q, a, joints.axis_a[joint]), affine_direction(q, b, joints.axis_b[joint])
        position = pa - pb
        if joints.kind[joint] == 0x02
            axis = normalize(db)
            position -= dot(position, axis) * axis
        end
        value = joints.linear[joint] / 2 * dot(position, position) + joints.angular[joint] / 2 * norm(cross(da, db))^2
        if joints.kind[joint] != 0x01
            ra, rb = affine_direction(q, a, joints.reference_a[joint]), affine_direction(q, b, joints.reference_b[joint])
            value += joints.angular[joint] / 2 * norm(cross(ra, rb))^2
        end
        if joints.damping[joint] > 0
            for k in Int32(1):Int32(9)
                row, motion = Int32(9) * (joint - Int32(1)) + k, 0.0f0
                for column in Int32(1):Int32(4)
                    index = Int32(4) * (a - Int32(1)) + column
                    motion += dot(rows.a[column, row], q[index] - anchor[index])
                    if b != 0
                        index = Int32(4) * (b - Int32(1)) + column
                        motion += dot(rows.b[column, row], q[index] - anchor[index])
                    end
                end
                value += joints.damping[joint] / (2dt) * motion^2
            end
        end
        return value
    end
end

@inline function magnetic_energy(q, motors, control, motor)
    @inbounds begin
        control.remaining[motor] <= 0 && return 0.0f0
        ra = affine_direction(q, motors.rotor[motor], motors.reference_a[motor])
        rb = affine_direction(q, motors.housing[motor], motors.reference_b[motor])
        axis = normalize(affine_direction(q, motors.housing[motor], motors.axis_b[motor]))
        angle = atan(dot(axis, cross(rb, ra)), dot(rb, ra))
        poles = Float32(motors.steps[motor]) / 4
        phase = Float32(control.phase[motor]) * (Float32(π) / 2) - poles * angle
        return motors.torque[motor] / poles * (1 - cos(phase))
    end
end

@inline function spring_potential(stretch, kt, λ, sliding, μs, μk)
    kt <= 0 && return 0.0f0
    distance = norm(stretch)
    cap = ifelse(sliding != 0x00, μk, μs) * λ / kt
    distance <= cap && return kt / 2 * distance^2
    return kt / 2 * cap^2 + μk * λ * (distance - cap)
end

function body_energy_values!(energy, state, inertia, anchor, model, C, T, force, torque, joints, rows, motors, control, one, dt::Float32, σ::Float32,
        motion = StaticMeshes(), accepted = nothing)
    accepted === nothing || (@inbounds accepted[1] == 0 || return nothing)
    index = KI.get_global_id(Int32).x
    index <= model.bodies + T.incidence.entries || return nothing
    @inbounds begin
        value = 0.0f0
        if index <= model.bodies
            body = index
            o = Int32(4) * (body - Int32(1))
            value = affine_energy(state.q, model, body, one[1])
            for column in Int32(1):Int32(4)
                i = o + column
                displacement = state.q[i] - inertia[i]
                value += model.mass[i] / (2dt^2) * dot(displacement, displacement)
                load = column == 1 ? force[body] : cross(torque[body], approximate(anchor[i])) / (2model.scale[body]^2)
                value -= dot(load, state.q[i] - anchor[i])
            end
            for joint in Int32(1):Int32(joints.joints)
                joints.a[joint] == body && (value += joint_energy(state.q, anchor, joints, rows, joint, dt, one[1]))
            end
            for motor in Int32(1):Int32(motors.motors)
                motors.rotor[motor] == body && (value += magnetic_energy(state.q, motors, control, motor))
            end
        else
            entry = index - Int32(model.bodies)
            point, slot = (entry - Int32(1)) ÷ Int32(SLOTS) + Int32(1), (entry - Int32(1)) % Int32(SLOTS) + Int32(1)
            mesh = T.mesh[slot, point]
            if mesh != 0
                own = C.point_owner[point]
                gap, _, _ = body_gap(C, T.triangle[slot, point], C.points[point], one[1])
                k = min(C.grip[3, own], C.grip[3, mesh]) * T.area[point]
                value = normal_potential(gap, k, σ)
                a, b = model.mesh_body[own], model.mesh_body[mesh]
                u, v = model.coordinates[point], T.other[slot, point]
                slip = zero(Vec3f)
                a != 0 && (slip += affine_point(state.q, a, u, one[1]) - affine_point(anchor, a, u, one[1]))
                b != 0 && (slip -= affine_point(state.q, b, v, one[1]) - affine_point(anchor, b, v, one[1]))
                slip += body_deformation_slip(motion, C, point, T.triangle[slot, point], T.barycentric[slot, point], one[1])
                normal = T.normal[slot, point]
                stretch = T.spring[slot, point] + slip
                stretch -= dot(stretch, normal) * normal
                value += spring_potential(stretch, T.tangent_stiffness[slot, point], T.lagged[slot, point], T.sliding[slot, point],
                    min(C.grip[1, own], C.grip[1, mesh]), min(C.grip[2, own], C.grip[2, mesh]))
            end
        end
        energy[index] = Vec3f(value, 0, 0)
    end
    return nothing
end

function reset_energy_slot!(energy, slot::Int32)
    KI.get_global_id(Int32).x == 1 && (@inbounds energy[slot] = 0)
    return nothing
end

function body_energy!(target, s, state, dt, slot)
    count = s.model.bodies + s.contacts.incidence.entries
    launch!(target, body_energy_values!, (s.energy, state, s.inertia, s.anchor, s.model, s.colliders, s.contacts,
        s.load_force, s.load_torque, s.joints, s.joint_rows, s.motors, s.motor_state, s.one, dt, s.roughness), count)
    launch!(target, reset_energy_slot!, (s.energy_sum, slot), 1)
    launch!(target, dot_into!, (s.energy_sum, s.energy, RepeatedEnergyMask(s.energy_ones), Int32(count), slot), cld(count, 256) * 256)
    return nothing
end

function body_energy_work!(energy, force, trial, base, n::Int32)
    i = KI.get_global_id(Int32).x
    i <= n || return nothing
    @inbounds energy[i] = Vec3f(dot(force[i], trial[i] - base[i]), 0, 0)
    return nothing
end

function trial_step!(q, base, trial, bound, one, accepted = nothing)
    accepted === nothing || (@inbounds accepted[1] == 0 || return nothing)
    i = KI.get_global_id(Int32).x
    i <= length(q) || return nothing
    @inbounds q[i] = moved(base[i], scaled(difference(trial[i], base[i], one[1]), reinterpret(Float32, bound[1]), one[1]), one[1])
    return nothing
end

function accept_energy_step!(bound, accepted, energy)
    KI.get_global_id(Int32).x == 1 || return nothing
    @inbounds if accepted[1] == 0
        α = reinterpret(Float32, bound[1])
        limit = energy[1] - 1.0f-4 * α * max(energy[3], 0.0f0) + 1.0f-7 * max(abs(energy[1]), 1.0f-20)
        if isfinite(energy[2]) && energy[2] <= limit
            accepted[1] = 1
        else
            bound[1] = reinterpret(UInt32, α / 2)
        end
    end
    return nothing
end

function body_line_search!(target, s, state, dt)
    n = 4s.model.bodies
    points = length(s.colliders.point_owner)
    launch!(target, copy_into!, (s.trial, state.q), n)
    launch!(target, body_energy_work!, (s.energy, s.base_force, s.trial, s.base, Int32(n)), n)
    launch!(target, reset_energy_slot!, (s.energy_sum, Int32(3)), 1)
    launch!(target, dot_into!, (s.energy_sum, s.energy, RepeatedEnergyMask(s.energy_ones), Int32(n), Int32(3)), cld(n, 256) * 256)
    launch!(target, zero!, (s.accepted,), 1)
    launch!(target, reset_step_bound!, (s.step_bound,), 1)
    for _ in 1:12
        launch!(target, trial_step!, (state.q, s.base, s.trial, s.step_bound, s.one), n)
        launch!(target, affine_surface!, (s.colliders, state, s.model, s.one), points)
        body_energy!(target, s, state, dt, Int32(2))
        launch!(target, accept_energy_step!, (s.step_bound, s.accepted, s.energy_sum), 1)
    end
    launch!(target, finalize_energy_step!, (s.step_bound, s.accepted), 1)
    launch!(target, trial_step!, (state.q, s.base, s.trial, s.step_bound, s.one), n)
    return nothing
end
