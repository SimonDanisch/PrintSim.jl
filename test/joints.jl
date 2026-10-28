using PrintSim, Test, LinearAlgebra
import PrintSim as PS
using GeometryBasics: Vec3, Vec3f
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

function shaft_in_housing(backend; kind = :hinge, steps = 50)
    b = ModelBuilder()
    axes = (Vec3(1.0, 0, 0), Vec3(0.0, 1, 0), Vec3(0.0, 0, 1))
    id = add_body!(TEST_DEVICE, b, box_mesh(Vec3(0.0, 0, 0), axes, (0.015, 0.003, 0.003)); spin = Vec3(2.0, 0, 0))
    add_joint!(b, id; position = Vec3(0.0, 0, 0), axis = Vec3(1.0, 0, 0), kind, linear = 1e6)
    solver = BodySolver(backend, b; iterations = 48)
    state = AffineState(backend, b)
    PS.prime!(solver, state)
    for _ in 1:steps
        PS.step!(solver, state, 1e-3)
    end
    return body_pose(state, b, id), b, solver, state
end

@testset "hinge angular Jacobian survives device matrix indexing" begin
    axes = (Vec3(1.0, 0, 0), Vec3(0.0, 1, 0), Vec3(0.0, 0, 1))
    for axis in axes
        b = ModelBuilder(; gravity = Vec3f(0))
        add_body!(TEST_DEVICE, b, box_mesh(Vec3(0.0, 0, 0), axes, (0.002, 0.016, 0.016)))
        add_joint!(b, 1; axis)
        solver, state = BodySolver(BACKEND, b), AffineState(BACKEND, b)
        q = Array(state.q)
        θ = 0.001
        rotation = [cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
        for k in 2:4
            p = Vec3{Float64}(q[k].hi) + Vec3{Float64}(q[k].lo)
            q[k] = Position(Vec3{Float64}(rotation * p))
        end
        copyto!(state.q, q)
        copyto!(solver.anchor, q)
        PS.launch!(BACKEND, PS.joint_linearize!, (solver.joint_rows, solver.joints, state, solver.anchor, solver.one, 0.001f0), 1)
        actual = Array(solver.joint_rows.a)
        localaxis = Array(solver.joints.axis_a)[1]
        worldaxis = Array(solver.joints.axis_b)[1]
        derivative = -PS.skew(worldaxis)
        for row in 1:3, column in 2:4
            expected = localaxis[column - 1] * Vec3f(derivative[row, 1], derivative[row, 2], derivative[row, 3])
            @test actual[column, row + 3] ≈ expected atol = 1e-6
        end
    end
end

@testset "a shaft in its unmodeled motor housing" begin
    (position, A), b, solver, state = shaft_in_housing(BACKEND)
    @test norm(position) < 1e-7
    @test norm(A * Vec3(1.0, 0, 0) - Vec3(1.0, 0, 0)) < 1e-4
    @test atan(A[3, 2], A[2, 2]) ≈ 0.1 atol = 0.002
    @test norm(A' * A - I) < 1e-5
end
