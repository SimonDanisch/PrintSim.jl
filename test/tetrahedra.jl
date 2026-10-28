using PrintSim, Test, LinearAlgebra
import PrintSim as PS
using GeometryBasics: Vec3, Vec3f
isdefined(@__MODULE__, :BACKEND) || include("backend.jl")

@testset "tetrahedral mass and stress-free motion" begin
    b = ModelBuilder()
    points, tets = tet_box((0.01, 0.02, 0.03), (3, 3, 3))
    add_deformable!(b, points, tets; density = 1270)
    model = FEMModel(Array, b)
    @test sum(model.mass) ≈ 1270 * 0.01 * 0.02 * 0.03 rtol = 1e-6
    @test length(b.meshes[1].faces) == 192
    solver = FEMSolver(BACKEND, b; iterations = 80)
    state = FEMState(BACKEND, b)
    for _ in 1:10
        PS.step!(solver, state, 1e-3)
    end
    actual = [Vec3{Float64}(p.hi) + Vec3{Float64}(p.lo) for p in Array(state.x)]
    expected = b.deformables[1].x .+ Ref(Vec3(0.0, 0, -9.81 * 1e-6 * 10 * 11 / 2))
    @test maximum(norm.(actual .- expected)) < 1e-9
    @test maximum(norm, Array(solver.strain)) < 1e-8
end

"A PETG cantilever under 0.9 N tip load, compared with 3 E I / L³."
function petg_leaf(backend; dims = (9, 3, 2), steps = 20, dt = 0.005, iterations = 400, poisson = 0.35)
    L, W, T, E, force = 0.033, 0.030, 0.0012, 1.46e9, 0.9
    b = ModelBuilder(; gravity = Vec3f(0))
    points, tets = tet_box((L, W, T), dims)
    add_deformable!(b, points, tets; young = E, poisson, fixed = p -> p[1] < 1e-10)
    points = b.deformables[1].x
    solver = FEMSolver(backend, b; iterations, newton = 4, timestep = dt)
    state = FEMState(backend, b)
    tip = findall(p -> p[1] > L - 1e-10, points)
    load = zeros(Vec3f, length(points))
    load[tip] .= Ref(Vec3f(0, 0, -force / length(tip)))
    copyto!(solver.external, load)
    for _ in 1:steps
        PS.step!(solver, state, dt)
    end
    x = [Vec3{Float64}(p.hi) + Vec3{Float64}(p.lo) for p in Array(state.x)]
    sag = sum(points[i][3] - x[i][3] for i in tip) / length(tip)
    exact = force * L^3 / (3E * W * T^3 / 12)
    return (; dims, sag, exact, ratio = sag / exact, residual = norm(Array(solver.r)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    @show petg_leaf(BACKEND)
end

@testset "quadratic PETG leaf against the beam reference" begin
    # Euler–Bernoulli's uniaxial beam omits the transverse Poisson constraint of the
    # feeder's wide (30 mm / 33 mm) leaf. First test that actual beam limit, then
    # retain the physical ν=0.35 case separately rather than changing its modulus.
    beam = petg_leaf(BACKEND; poisson = 0.0)
    @test beam.sag ≈ beam.exact rtol = 0.01
    @test beam.residual < 1e-5
    plate = petg_leaf(BACKEND)
    @test 0.8 < plate.ratio < 0.95
    @test plate.residual < 1e-5
end
