# A stepper's electrical phase advances in full steps; the rotor stays a mechanical DOF.
# A periodic magnetic potential supplies finite torque. When the required load exceeds it,
# the commanded phase keeps advancing and the rotor falls behind instead of being prescribed.

"""
    add_stepper!(builder, rotor, housing=0; axis, torque=0.0373, steps=2048)

Attach a full-step motor to a rotor and its housing (0: world). The 28BYJ-48 defaults use the
output shaft's 2048 full steps per revolution and the feeder's 37.3 mN·m measurement at
333 steps/s. `torque` is the available torque at the commanded speed, not an invented speed
curve. A separate housing joint defines the shaft's bearings.
"""
function add_stepper!(b::ModelBuilder, rotor::Integer, housing::Integer = 0;
        axis = Vec3(1.0, 0, 0), torque = 0.0373, steps = 2048)
    1 <= rotor <= length(b.bodies) && 0 <= housing <= length(b.bodies) && rotor != housing || throw(ArgumentError("invalid motor bodies"))
    torque > 0 && steps > 0 && steps % 4 == 0 || throw(ArgumentError("positive torque and a step count divisible by four are required"))
    axis = normalize(Vec3{Float64}(axis))
    all(isfinite, axis) || throw(ArgumentError("a motor needs a nonzero finite axis"))
    seed = abs(axis[1]) < 0.8 ? Vec3(1.0, 0, 0) : Vec3(0.0, 1, 0)
    reference = normalize(cross(axis, seed))
    direction(k, v) = k == 0 ? Vec3f(v) : Vec3f(b.bodies[k].axes' * v / b.bodies[k].scale)
    push!(b.motors, StepperPiece(Int32(rotor), Int32(housing), direction(rotor, axis), direction(housing, axis),
        direction(rotor, reference), direction(housing, reference), Float32(torque), Int32(steps)))
    return length(b.motors)
end

struct MotorModel{I, V, F}
    rotor::I
    housing::I
    axis_a::V
    axis_b::V
    reference_a::V
    reference_b::V
    torque::F
    steps::I
    motors::Int
end
Adapt.@adapt_structure MotorModel
function MotorModel(backend, b)
    up(a) = upload(backend, a)
    return MotorModel(up(Int32[m.rotor for m in b.motors]), up(Int32[m.housing for m in b.motors]),
        up(Vec3f[m.axis_a for m in b.motors]), up(Vec3f[m.axis_b for m in b.motors]),
        up(Vec3f[m.reference_a for m in b.motors]), up(Vec3f[m.reference_b for m in b.motors]),
        up(Float32[m.torque for m in b.motors]), up(Int32[m.steps for m in b.motors]), length(b.motors))
end

struct MotorState{I, F, V}
    remaining::I
    direction::I
    phase::I
    commanded::I
    rpm::F
    fraction::F
    angle::F
    raw_angle::F
    torque::F
    stiffness::F
    a::V
    b::V
end
Adapt.@adapt_structure MotorState
function MotorState(backend, b)
    n = max(length(b.motors), 1)
    zi() = KI.zeros(backend, Int32, n)
    zf() = KI.zeros(backend, Float32, n)
    zv() = KI.zeros(backend, Vec3f, 4, n)
    return MotorState(zi(), zi(), zi(), zi(), zf(), zf(), zf(), zf(), zf(), zf(), zv(), zv())
end

function motor_command!(state, id::Int32, rpm::Float32, steps::Int32)
    KI.get_global_id(Int32).x == 1 || return nothing
    @inbounds begin
        state.remaining[id] = abs(steps)
        state.direction[id] = steps >= 0 ? Int32(1) : -Int32(1)
        state.rpm[id] = rpm
        state.fraction[id] = 0
    end
    return nothing
end

function motor_clock!(state, model, dt::Float32)
    i = KI.get_global_id(Int32).x
    i <= model.motors || return nothing
    @inbounds if state.remaining[i] > 0
        count = state.fraction[i] + dt * state.rpm[i] * Float32(model.steps[i]) / 60
        steps = min(floor(Int32, count), state.remaining[i])
        state.fraction[i] = count - Float32(steps)
        state.remaining[i] -= steps
        state.commanded[i] += steps * state.direction[i]
        state.phase[i] = (state.phase[i] + steps * state.direction[i] + Int32(4) * (steps + Int32(1))) % Int32(4)
    end
    return nothing
end

function motor_linearize!(output, motors, state)
    i = KI.get_global_id(Int32).x
    i <= motors.motors || return nothing
    @inbounds begin
        rotor, housing = motors.rotor[i], motors.housing[i]
        ra = affine_direction(state.q, rotor, motors.reference_a[i])
        rb = affine_direction(state.q, housing, motors.reference_b[i])
        db = affine_direction(state.q, housing, motors.axis_b[i])
        axis = normalize(db)
        sine, cosine = dot(axis, cross(rb, ra)), dot(rb, ra)
        denominator = max(sine^2 + cosine^2, 1.0f-20)
        angle = atan(sine, cosine)
        ga = (cosine * cross(axis, rb) - sine * rb) / denominator
        gb = (cosine * cross(ra, axis) - sine * ra) / denominator
        gd = (I3 - axis * axis') * (cosine / denominator * cross(rb, ra)) / norm(db)
        output.a[1, i], output.b[1, i] = zero(Vec3f), zero(Vec3f)
        for column in Int32(2):Int32(4)
            output.a[column, i] = motors.reference_a[i][column - Int32(1)] * ga
            output.b[column, i] = housing == 0 ? zero(Vec3f) :
                motors.reference_b[i][column - Int32(1)] * gb + motors.axis_b[i][column - Int32(1)] * gd
        end
        poles = Float32(motors.steps[i]) / 4
        phase = Float32(output.phase[i]) * (Float32(π) / 2) - poles * angle
        on = output.remaining[i] > 0
        output.torque[i] = on ? motors.torque[i] * sin(phase) : 0.0f0
        output.stiffness[i] = on ? max(motors.torque[i] * poles * cos(phase), 0.0f0) : 0.0f0
    end
    return nothing
end

"Track actual output rotation independently of the electrical command and its four phases."
function motor_observe!(output, motors, state)
    i = KI.get_global_id(Int32).x
    i <= motors.motors || return nothing
    @inbounds begin
        ra = affine_direction(state.q, motors.rotor[i], motors.reference_a[i])
        rb = affine_direction(state.q, motors.housing[i], motors.reference_b[i])
        axis = normalize(affine_direction(state.q, motors.housing[i], motors.axis_b[i]))
        raw = atan(dot(axis, cross(rb, ra)), dot(rb, ra))
        delta = raw - output.raw_angle[i]
        output.angle[i] += atan(sin(delta), cos(delta))
        output.raw_angle[i] = raw
    end
    return nothing
end

@inline function motor_gradient(rows, motors, body, column, motor)
    @inbounds return motors.rotor[motor] == body ? rows.a[column, motor] : motors.housing[motor] == body ? rows.b[column, motor] : zero(Vec3f)
end

function motor_rhs!(r, inverse, output, motors, model)
    index = KI.get_global_id(Int32).x
    index <= 4model.bodies || return nothing
    @inbounds begin
        body, column = (index - Int32(1)) ÷ Int32(4) + Int32(1), (index - Int32(1)) % Int32(4) + Int32(1)
        model.fixed[body] != 0x00 && return nothing
        diagonal, force = preconditioner_inverse(inverse[index]), zero(Vec3f)
        for motor in Int32(1):Int32(motors.motors)
            g = motor_gradient(output, motors, body, column, motor)
            force += output.torque[motor] * g
            diagonal += output.stiffness[motor] * (g * g')
        end
        r[index] += force
        inverse[index] = preconditioner_inverse(diagonal)
    end
    return nothing
end

function motor_apply!(result, p, output, motors, model)
    index = KI.get_global_id(Int32).x
    index <= 4model.bodies || return nothing
    @inbounds begin
        body, column = (index - Int32(1)) ÷ Int32(4) + Int32(1), (index - Int32(1)) % Int32(4) + Int32(1)
        model.fixed[body] != 0x00 && return nothing
        value = zero(Vec3f)
        for motor in Int32(1):Int32(motors.motors)
            a, b = motors.rotor[motor], motors.housing[motor]
            (a != body && b != body) && continue
            motion = 0.0f0
            for c in Int32(1):Int32(4)
                motion += dot(output.a[c, motor], p[Int32(4) * (a - Int32(1)) + c])
                b != 0 && (motion += dot(output.b[c, motor], p[Int32(4) * (b - Int32(1)) + c]))
            end
            value += output.stiffness[motor] * motion * motor_gradient(output, motors, body, column, motor)
        end
        result[index] += value
    end
    return nothing
end

function body_load!(force, torque, body::Int32, f::Vec3f, τ::Vec3f)
    KI.get_global_id(Int32).x == 1 || return nothing
    @inbounds force[body], torque[body] = f, τ
    return nothing
end

function body_load_rhs!(r, force, torque, state, model)
    i = KI.get_global_id(Int32).x
    i <= 4model.bodies || return nothing
    @inbounds begin
        body, column = (i - Int32(1)) ÷ Int32(4) + Int32(1), (i - Int32(1)) % Int32(4) + Int32(1)
        model.fixed[body] != 0x00 && return nothing
        r[i] += column == 1 ? force[body] : cross(torque[body], approximate(state[i])) / (2model.scale[body]^2)
    end
    return nothing
end

function command!(s::BodySolver, motor::Integer; rpm = 9.75, steps::Integer)
    1 <= motor <= s.motors.motors || throw(ArgumentError("unknown motor $motor"))
    isfinite(rpm) && rpm >= 0 || throw(ArgumentError("rpm must be finite and nonnegative"))
    launch!(s.backend, motor_command!, (s.motor_state, Int32(motor), Float32(rpm), Int32(steps)), 1)
    return s
end

function set_load!(s::BodySolver, body::Integer; force = Vec3f(0), torque = Vec3f(0))
    1 <= body <= s.model.bodies || throw(ArgumentError("unknown body $body"))
    launch!(s.backend, body_load!, (s.load_force, s.load_torque, Int32(body), Vec3f(force), Vec3f(torque)), 1)
    return s
end

function motor_step_bound!(bound, q, base, output, motors)
    i = KI.get_global_id(Int32).x
    i <= motors.motors || return nothing
    @inbounds begin
        a, b = motors.rotor[i], motors.housing[i]
        delta = 0.0f0
        for column in Int32(2):Int32(4)
            index = Int32(4) * (a - Int32(1)) + column
            delta += dot(output.a[column, i], q[index] - base[index])
            if b != 0
                index = Int32(4) * (b - Int32(1)) + column
                delta += dot(output.b[column, i], q[index] - base[index])
            end
        end
        # Resolve the magnetic wells even when inertia alone would cross several of them.
        limit = Float32(π) / Float32(motors.steps[i])
        cap = min(1.0f0, limit / max(abs(delta), 1.0f-30))
        Atomix.@atomic :monotonic min(bound[1], reinterpret(UInt32, cap))
    end
    return nothing
end

command!(s::WorldSolver, motor::Integer; kw...) = (command!(s.bodies, motor; kw...); s)
set_load!(s::WorldSolver, body::Integer; kw...) = (set_load!(s.bodies, body; kw...); s)
