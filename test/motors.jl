using PrintSim, Test, LinearAlgebra
import PrintSim as PS
using GeometryBasics: Vec3, Vec3f
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

function driven_shaft(backend; load = 0.0, steps = 250, dt = 1e-3)
    b = ModelBuilder(; gravity = Vec3f(0))
    axes = (Vec3(1.0, 0, 0), Vec3(0.0, 1, 0), Vec3(0.0, 0, 1))
    rotor = add_body!(TEST_DEVICE, b, box_mesh(Vec3(0.0, 0, 0), axes, (0.01, 0.006, 0.006)); mass = 0.01)
    add_joint!(b, rotor; axis = Vec3(1.0, 0, 0))
    motor = add_stepper!(b, rotor; axis = Vec3(1.0, 0, 0), torque = 1e-3)
    solver = BodySolver(backend, b; newton = 8, iterations = 32)
    state = AffineState(backend, b)
    PS.prime!(solver, state)
    command!(solver, motor; rpm = 3.0, steps = 100)
    set_load!(solver, rotor; torque = Vec3f(-load, 0, 0))
    for _ in 1:steps
        PS.step!(solver, state, dt)
    end
    angle = Float64(Array(solver.motor_state.angle)[motor])
    command = Array(solver.motor_state.commanded)[motor] * 2π / 2048
    return (; angle, command, torque = Array(solver.motor_state.torque)[motor])
end

@testset "a stepper follows phase and loses steps under overload" begin
    light = driven_shaft(BACKEND; load = 0.2e-3)
    @test abs(light.angle - light.command) < 2 * 2π / 2048
    heavy = driven_shaft(BACKEND; load = 2e-3)
    @test heavy.command - heavy.angle > 4 * 2π / 2048
    @test abs(heavy.torque) <= 1.001e-3
end
