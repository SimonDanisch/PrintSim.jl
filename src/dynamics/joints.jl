# Joints use the same generalized force, matrix product and preconditioner as contacts.
# They do not prescribe a body trajectory: a motor housing can hold a shaft's axis while
# its rotation, load and reaction remain degrees of freedom in the common solve.

"""
    add_joint!(builder, a, b=0; position, axis, kind=:hinge, linear=1e7, angular=1e3, damping=0)

Connect affine bodies at a world-space `position` and `axis`; body 0 is the world. A hinge
allows rotation about the axis, a prismatic joint translation along it, a fixed joint neither.
Use modeled mesh contacts for bearings and keys whose geometry exists in the assembly.
"""
function add_joint!(b::ModelBuilder, a::Integer, other::Integer = 0; position = b.bodies[a].center,
        axis = Vec3(1.0, 0, 0), kind = :hinge, linear = 1e7, angular = 1e3, damping = 0.0)
    1 <= a <= length(b.bodies) && 0 <= other <= length(b.bodies) && a != other || throw(ArgumentError("invalid joint bodies"))
    kind in (:hinge, :prismatic, :fixed) || throw(ArgumentError("unknown joint kind $kind"))
    linear > 0 && angular > 0 && damping >= 0 || throw(ArgumentError("joint stiffness must be positive and damping nonnegative"))
    axis = normalize(Vec3{Float64}(axis))
    all(isfinite, axis) || throw(ArgumentError("a joint needs a nonzero finite axis"))
    seed = abs(axis[1]) < 0.8 ? Vec3(1.0, 0, 0) : Vec3(0.0, 1, 0)
    reference = normalize(cross(axis, seed))
    point(k) = k == 0 ? Vec3f(position) : Vec3f(b.bodies[k].axes' * (position - b.bodies[k].center) / b.bodies[k].scale)
    direction(k, v) = k == 0 ? Vec3f(v) : Vec3f(b.bodies[k].axes' * v / b.bodies[k].scale)
    push!(b.joints, JointPiece(Int32(a), Int32(other), point(a), point(other), direction(a, axis), direction(other, axis),
        direction(a, reference), direction(other, reference), kind == :hinge ? 0x01 : kind == :prismatic ? 0x02 : 0x03,
        Float32(linear), Float32(angular), Float32(damping)))
    return length(b.joints)
end

struct JointModel{I, V, U, F}
    a::I
    b::I
    anchor_a::V
    anchor_b::V
    axis_a::V
    axis_b::V
    reference_a::V
    reference_b::V
    kind::U
    linear::F
    angular::F
    damping::F
    joints::Int
end
Adapt.@adapt_structure JointModel

function JointModel(backend, b)
    up(a) = upload(backend, a)
    return JointModel(up(Int32[j.a for j in b.joints]), up(Int32[j.b for j in b.joints]),
        up(Vec3f[j.anchor_a for j in b.joints]), up(Vec3f[j.anchor_b for j in b.joints]),
        up(Vec3f[j.axis_a for j in b.joints]), up(Vec3f[j.axis_b for j in b.joints]),
        up(Vec3f[j.reference_a for j in b.joints]), up(Vec3f[j.reference_b for j in b.joints]),
        up(UInt8[j.kind for j in b.joints]), up(Float32[j.linear for j in b.joints]),
        up(Float32[j.angular for j in b.joints]), up(Float32[j.damping for j in b.joints]), length(b.joints))
end

struct JointRows{V, F}
    a::V
    b::V
    force::F
    stiffness::F
end
Adapt.@adapt_structure JointRows
function JointRows(backend, b)
    n = max(9length(b.joints), 1)
    return JointRows(KI.zeros(backend, Vec3f, 4, n), KI.zeros(backend, Vec3f, 4, n),
        KI.zeros(backend, Float32, n), KI.zeros(backend, Float32, n))
end

@inline function affine_direction(q, body, u)
    body == 0 && return u
    o = Int32(4) * (body - Int32(1))
    @inbounds return u[1] * approximate(q[o + Int32(2)]) + u[2] * approximate(q[o + Int32(3)]) + u[3] * approximate(q[o + Int32(4)])
end

@inline function row_vector(A, row)
    # Explicit fixed indices: dynamic indexing of a private StaticArray inside the
    # constraint loop has compiled to zero angular rows in the device compiler.
    return row == 1 ? Vec3f(A[1, 1], A[1, 2], A[1, 3]) :
        row == 2 ? Vec3f(A[2, 1], A[2, 2], A[2, 3]) : Vec3f(A[3, 1], A[3, 2], A[3, 3])
end

function joint_linearize!(rows, joints, state, anchor, one, dt::Float32)
    joint = KI.get_global_id(Int32).x
    joint <= joints.joints || return nothing
    @inbounds begin
        a, b = joints.a[joint], joints.b[joint]
        ua, ub = joints.anchor_a[joint], joints.anchor_b[joint]
        pa = affine_point(state.q, a, ua, one[1])
        pb = b == 0 ? Position(ub) : affine_point(state.q, b, ub, one[1])
        position = pa - pb
        da, db = affine_direction(state.q, a, joints.axis_a[joint]), affine_direction(state.q, b, joints.axis_b[joint])
        ra, rb = affine_direction(state.q, a, joints.reference_a[joint]), affine_direction(state.q, b, joints.reference_b[joint])
        projection = joints.kind[joint] == 0x02 ? I3 - normalize(db) * normalize(db)' : I3
        for localrow in Int32(1):Int32(9)
            row = Int32(9) * (joint - Int32(1)) + localrow
            component = (localrow - Int32(1)) % Int32(3) + Int32(1)
            residual = 0.0f0
            k = localrow <= 3 ? joints.linear[joint] : joints.angular[joint]
            if localrow <= 3
                residual = (projection * position)[component]
                axisrow = row_vector(projection, component)
                for column in Int32(1):Int32(4)
                    rows.a[column, row] = affine_weight(ua, column) * axisrow
                    rows.b[column, row] = b == 0 ? zero(Vec3f) : -affine_weight(ub, column) * axisrow
                end
            else
                va, vb = localrow <= 6 ? da : ra, localrow <= 6 ? db : rb
                ca, cb = localrow <= 6 ? joints.axis_a[joint] : joints.reference_a[joint], localrow <= 6 ? joints.axis_b[joint] : joints.reference_b[joint]
                enabled = localrow <= 6 || joints.kind[joint] != 0x01
                residual = enabled ? cross(va, vb)[component] : 0.0f0
                JA, JB = -skew(vb), skew(va)
                rows.a[1, row], rows.b[1, row] = zero(Vec3f), zero(Vec3f)
                for column in Int32(2):Int32(4)
                    rows.a[column, row] = enabled ? ca[column - Int32(1)] * row_vector(JA, component) : zero(Vec3f)
                    rows.b[column, row] = enabled && b != 0 ? cb[column - Int32(1)] * row_vector(JB, component) : zero(Vec3f)
                end
            end
            motion = 0.0f0
            for column in Int32(1):Int32(4)
                motion += dot(rows.a[column, row], state.q[Int32(4) * (a - Int32(1)) + column] - anchor[Int32(4) * (a - Int32(1)) + column])
                if b != 0
                    motion += dot(rows.b[column, row], state.q[Int32(4) * (b - Int32(1)) + column] - anchor[Int32(4) * (b - Int32(1)) + column])
                end
            end
            rows.force[row] = -k * residual - joints.damping[joint] / dt * motion
            rows.stiffness[row] = k + joints.damping[joint] / dt
        end
    end
    return nothing
end

@inline function joint_gradient(rows, joints, body, column, row)
    joint = (row - Int32(1)) ÷ Int32(9) + Int32(1)
    @inbounds return joints.a[joint] == body ? rows.a[column, row] : joints.b[joint] == body ? rows.b[column, row] : zero(Vec3f)
end

function joint_rhs!(r, inverse, rows, joints, model)
    index = KI.get_global_id(Int32).x
    index <= 4model.bodies || return nothing
    @inbounds begin
        body, column = (index - Int32(1)) ÷ Int32(4) + Int32(1), (index - Int32(1)) % Int32(4) + Int32(1)
        model.fixed[body] != 0x00 && return nothing
        diagonal = preconditioner_inverse(inverse[index])
        force = zero(Vec3f)
        for row in Int32(1):Int32(9joints.joints)
            g = joint_gradient(rows, joints, body, column, row)
            force += rows.force[row] * g
            diagonal += rows.stiffness[row] * (g * g')
        end
        r[index] += force
        inverse[index] = preconditioner_inverse(diagonal)
    end
    return nothing
end

function joint_apply!(result, p, rows, joints, model)
    index = KI.get_global_id(Int32).x
    index <= 4model.bodies || return nothing
    @inbounds begin
        body, column = (index - Int32(1)) ÷ Int32(4) + Int32(1), (index - Int32(1)) % Int32(4) + Int32(1)
        model.fixed[body] != 0x00 && return nothing
        value = zero(Vec3f)
        for row in Int32(1):Int32(9joints.joints)
            joint = (row - Int32(1)) ÷ Int32(9) + Int32(1)
            a, b = joints.a[joint], joints.b[joint]
            (a != body && b != body) && continue
            motion = 0.0f0
            for c in Int32(1):Int32(4)
                motion += dot(rows.a[c, row], p[Int32(4) * (a - Int32(1)) + c])
                b != 0 && (motion += dot(rows.b[c, row], p[Int32(4) * (b - Int32(1)) + c]))
            end
            value += rows.stiffness[row] * motion * joint_gradient(rows, joints, body, column, row)
        end
        result[index] += value
    end
    return nothing
end
