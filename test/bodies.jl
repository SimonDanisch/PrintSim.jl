using PrintSim, Test, LinearAlgebra
import PrintSim as PS
using GeometryBasics: Vec3, Vec3f
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

const BODY_AXES = (Vec3(1.0, 0, 0), Vec3(0.0, 1, 0), Vec3(0.0, 0, 1))

function falling_body(backend; steps = 20, dt = 1e-3, spin = Vec3(0.0, 0, 0))
    b = ModelBuilder()
    add_body!(TEST_DEVICE, b, box_mesh(Vec3(0.02, 0.03, 0.05), BODY_AXES, (0.01, 0.02, 0.03)); spin)
    solver = BodySolver(backend, b)
    state = AffineState(backend, b)
    PS.prime!(solver, state)
    for _ in 1:steps
        PS.step!(solver, state, dt)
    end
    return body_pose(state, b, 1), b, solver, state
end

function body_on_slope(backend; θdeg = 0.0, steps = 80, dt = 1e-3, kw...)
    θ = deg2rad(θdeg)
    up, across, normal = Vec3(cos(θ), 0, sin(θ)), Vec3(0.0, 1, 0), Vec3(-sin(θ), 0, cos(θ))
    b = ModelBuilder()
    add_mesh!(b, box_mesh(-0.005normal, (up, across, normal), (0.2, 0.2, 0.005)); static = 0.5, kinetic = 0.4)
    center = (0.002 + 3e-6) * normal
    id = add_body!(TEST_DEVICE, b, box_mesh(center, (up, across, normal), (0.01, 0.01, 0.002)); static = 0.5, kinetic = 0.4)
    solver = BodySolver(backend, b; kw...)
    state = AffineState(backend, b)
    PS.prime!(solver, state)
    slide, load = Float64[], Float64[]
    weight = b.bodies[id].mass[1] * 9.81 * cos(θ)
    for _ in 1:steps
        PS.step!(solver, state, dt)
        p, _ = body_pose(state, b, id)
        push!(slide, dot(center - p, up))
        push!(load, sum(Array(solver.contacts.lagged)) / weight)
    end
    return slide, load, solver, state, b
end

@testset "affine mass from a closed mesh" begin
    b = ModelBuilder()
    add_body!(TEST_DEVICE, b, box_mesh(Vec3(0.1, 0.2, 0.3), BODY_AXES, (0.01, 0.02, 0.03)); density = 1240)
    volume, center, covariance = PS.volume_properties(TEST_DEVICE, only(b.meshes))
    @test volume ≈ 8 * 0.01 * 0.02 * 0.03
    @test center ≈ [0.1, 0.2, 0.3]
    @test Matrix(covariance) ≈ Diagonal([0.01^2, 0.02^2, 0.03^2] ./ 3)
    @test b.bodies[1].mass[1] ≈ 1240volume
end

@testset "compound body retains constituent surfaces and exact inertia" begin
    b = ModelBuilder()
    left = box_mesh(Vec3(-0.002, 0.0, 0), BODY_AXES, (0.001, 0.001, 0.001))
    right = box_mesh(Vec3(0.002, 0.0, 0), BODY_AXES, (0.001, 0.001, 0.001))
    id = add_body!(TEST_DEVICE, b, [(; mesh = left, density = 1000.0, static = 0.3),
        (; mesh = right, density = 2000.0, static = 1.2)])
    body = b.bodies[id]
    @test body.mass[1] ≈ 24e-6
    @test body.center ≈ [0.002 / 3, 0, 0]
    @test [m.static for m in b.meshes] == [0.3, 1.2]
    @test body.meshes == [1, 2]
    model = PS.AffineModel(BACKEND, b)
    @test Array(model.mesh_body) == [1, 1]
    state = AffineState(BACKEND, b)
    solver = BodySolver(BACKEND, b)
    PS.prime!(solver, state)
    PS.step!(solver, state, 1e-3)
    @test all(iszero, Array(solver.contacts.mesh))
    @test body_pose(state, b, id)[1][3] ≈ -9.81e-6 atol = 1e-10
end

@testset "an affine body in free fall" begin
    (center, A), b, solver, state = falling_body(BACKEND)
    @test center ≈ [0.02, 0.03, 0.05 - 9.81 * 1e-6 * 20 * 21 / 2] atol = 1e-9
    @test norm(A' * A - I) < 1e-6
    @test all(iszero, Array(solver.contacts.overflow))
end

@testset "an affine block on a meshed slope" begin
    _, load, _, _, _ = body_on_slope(BACKEND)
    @test maximum(abs(l - 1) for l in load[40:end]) < 0.002
    slide, _, _, _, _ = body_on_slope(BACKEND; θdeg = 20.0)
    # Initial settling can rotate the block in the plane; its friction memory then holds it.
    @test abs(slide[end]) < 20e-6
    @test abs(slide[end] - slide[end - 10]) < 1e-9
    slide, load, _, _, _ = body_on_slope(BACKEND; θdeg = 35.0)
    rate = ((slide[80] - slide[79]) - (slide[60] - slide[59])) / 20e-6
    θ = deg2rad(35.0)
    @test isapprox(rate, 9.81 * (sin(θ) - 0.4cos(θ)); rtol = 0.002)
    @test maximum(abs(l-1) for l in load[40:end])<0.002
    # Four Newton iterations previously left a 30% acceleration error and a
    # spurious normal load. Require refinement agreement, not just finiteness.
    refined, _, _, _, _ = body_on_slope(BACKEND; θdeg = 35.0,newton=32)
    refined_rate=((refined[80]-refined[79])-(refined[60]-refined[59]))/20e-6
    @test isapprox(rate,refined_rate;rtol=2e-5)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for θ in (0.0, 20.0, 35.0)
        slide, load, solver, state, b = body_on_slope(BACKEND; θdeg = θ)
        println((θ = θ, slide = slide[end], load = extrema(load[40:end]), pose = body_pose(state, b, 1)))
    end
end
